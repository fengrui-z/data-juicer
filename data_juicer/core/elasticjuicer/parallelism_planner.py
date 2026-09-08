"""Throughput-aware actor planning for ElasticJuicer auto-parallelism.

Pure planning logic: every GPU stage starts with one actor, then actors
are repeatedly assigned to the current bottleneck (lowest probed
throughput) until the next assignment would violate any constraint:

- cluster CPU budget;
- per-device Ray GPU fraction;
- measured per-device memory (probed peak + headroom);
- per-device replica limit;
- available batches for the stage;
- capacity reserved by explicitly configured operators;
- the economics of one more actor: each extra actor must own enough
  batches to amortize its startup and must reduce the stage's batch
  round count (a necessary condition for a real makespan gain).

If the minimum plan (one actor per stage) does not fit, planning fails
fast instead of hiding the problem through overcommitment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .gpu_probe import atomic_write_json

RUNTIME_PLAN_SCHEMA_VERSION = 1
RUNTIME_PLAN_FILENAME = "runtime_plan.json"

DEFAULT_MIN_BATCHES_PER_ACTOR = 4
DEFAULT_STARTUP_AMORTIZATION_RATIO = 0.10
DEFAULT_MIN_MARGINAL_MAKESPAN_GAIN = 0.10


class PlanInfeasibleError(RuntimeError):
    """Raised when even the minimum one-actor-per-stage plan cannot fit."""


@dataclass(frozen=True)
class ClusterSpec:
    """Driver-observed cluster resources."""

    total_cpus: float
    num_devices: int
    device_memory_bytes: int
    device_memory_budget_fraction: float = 0.90
    gpu_fraction_budget: float = 1.0

    def __post_init__(self):
        if self.total_cpus <= 0:
            raise ValueError("total_cpus must be positive")
        if self.num_devices < 1:
            raise ValueError("num_devices must be >= 1")
        if self.device_memory_bytes <= 0:
            raise ValueError("device_memory_bytes must be positive")
        if not 0.0 < self.device_memory_budget_fraction <= 1.0:
            raise ValueError("device_memory_budget_fraction must be in (0, 1]")
        if not 0.0 < self.gpu_fraction_budget <= 1.0:
            raise ValueError("gpu_fraction_budget must be in (0, 1]")

    @property
    def device_memory_budget_bytes(self) -> int:
        return int(self.device_memory_bytes * self.device_memory_budget_fraction)


@dataclass
class StageDemand:
    """Planning input for one GPU stage."""

    stage_key: str
    op_name: str
    probed_throughput_rps: float
    effective_peak_bytes: int
    batch_size: int
    available_batches: int
    num_gpus: float = 1.0
    num_cpus: float = 1.0
    oom_upper_bound: Optional[int] = None
    explicit: bool = False
    explicit_actors: Optional[int] = None
    actor_startup_seconds: float = 0.0
    resident_bytes: int = 0
    memory_headroom_factor: float = 1.0

    def __post_init__(self):
        if not self.stage_key:
            raise ValueError("stage_key must not be empty")
        if self.probed_throughput_rps <= 0:
            raise ValueError("probed_throughput_rps must be positive")
        if self.effective_peak_bytes <= 0:
            raise ValueError("effective_peak_bytes must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.available_batches < 1:
            raise ValueError("available_batches must be >= 1")
        if self.num_gpus <= 0:
            raise ValueError("num_gpus must be positive")
        if self.num_cpus <= 0:
            raise ValueError("num_cpus must be positive")
        if self.explicit and (self.explicit_actors or 0) < 1:
            raise ValueError("explicit stages need explicit_actors >= 1")
        if self.actor_startup_seconds < 0:
            raise ValueError("actor_startup_seconds must be non-negative")
        if not 0 <= self.resident_bytes <= self.effective_peak_bytes:
            raise ValueError("resident_bytes must be within the effective peak")
        if not math.isfinite(self.memory_headroom_factor) or self.memory_headroom_factor < 1:
            raise ValueError("memory_headroom_factor must be finite and >= 1")

    @property
    def per_row_vram_bytes(self) -> float:
        return (self.effective_peak_bytes - self.resident_bytes) / float(self.batch_size)


@dataclass
class StagePlan:
    stage_key: str
    op_name: str
    planned_actors: int
    device_ids: List[int] = field(default_factory=list)
    per_actor_vram_budget_bytes: int = 0
    batch_hard_cap: int = 0
    fixed: bool = False


@dataclass
class RuntimePlan:
    stages: Dict[str, StagePlan] = field(default_factory=dict)
    cluster: Optional[ClusterSpec] = None
    generated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    max_replicas_per_device: int = 1
    planning_gates: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "schema_version": RUNTIME_PLAN_SCHEMA_VERSION,
            "generated_at_ms": self.generated_at_ms,
            "max_replicas_per_device": self.max_replicas_per_device,
            "planning_gates": dict(self.planning_gates),
            "cluster": asdict(self.cluster) if self.cluster else None,
            "stages": {key: asdict(value) for key, value in self.stages.items()},
        }


def save_runtime_plan(path: str, plan: RuntimePlan) -> None:
    atomic_write_json(path, plan.to_dict())


def plan_envelope_fingerprint(plan: RuntimePlan) -> str:
    """Fingerprint of the resource-semantics fields of a runtime plan.

    Binds cross-restart stage profiles (contract C3's execution-plan
    dimension) to the envelope they were learned under: actors, placement,
    per-actor VRAM budget, batch hard caps, replica limits and the cluster
    they were planned against.  Any change to these fields must reject
    previously learned profiles instead of silently seeding a different
    envelope.  Advisory throughput numbers and gate bookkeeping are
    deliberately excluded: they do not change what an actor owns.
    """

    cluster_payload = None
    if plan.cluster is not None:
        cluster_payload = {
            "num_devices": plan.cluster.num_devices,
            "device_memory_bytes": plan.cluster.device_memory_bytes,
            "device_memory_budget_fraction": plan.cluster.device_memory_budget_fraction,
        }
    payload = {
        "cluster": cluster_payload,
        "max_replicas_per_device": plan.max_replicas_per_device,
        "stages": {
            key: {
                "planned_actors": stage.planned_actors,
                "device_ids": sorted(stage.device_ids),
                "per_actor_vram_budget_bytes": stage.per_actor_vram_budget_bytes,
                "batch_hard_cap": stage.batch_hard_cap,
            }
            for key, stage in sorted(plan.stages.items())
        },
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def device_replica_counts(stages: Dict[str, StagePlan]) -> Dict[int, int]:
    """Total replicas placed on each device across all stages."""

    counts: Dict[int, int] = {}
    for stage_plan in stages.values():
        for device_id in stage_plan.device_ids:
            counts[device_id] = counts.get(device_id, 0) + 1
    return counts


def colocated_actor_budget(cluster: ClusterSpec, stages: Dict[str, StagePlan], device_id: int) -> int:
    """Per-actor VRAM envelope: the device budget shared over colocated replicas.

    The envelope never drops below the stage's own probed peak, so a stage
    that already fits is never told to shrink below what it measured.
    """

    replicas = max(1, device_replica_counts(stages).get(device_id, 0))
    return max(1, cluster.device_memory_budget_bytes // replicas)


def fractional_actor_entitlement(cluster: ClusterSpec, num_gpus: float) -> int:
    """Memory an actor is honestly entitled to under fractional-GPU packing.

    Ray schedules actors by their num_gpus fraction and may colocate up to
    1/num_gpus replicas of a stage on one device regardless of where the
    plan lists them, so an actor's guaranteed share is its fraction of the
    device budget -- not the whole device divided over the replicas the
    plan happens to place there.  Full-GPU actors keep the whole-device
    share; multi-GPU actors scale linearly.  The envelope floor stays the
    probed peak (see colocated_actor_budget): a stage that measured more
    than its entitlement is never told to shrink below the measurement.
    """

    return max(1, int(cluster.device_memory_budget_bytes * num_gpus))


class _DeviceState:
    __slots__ = ("memory_used", "gpu_fraction_used", "replicas")

    def __init__(self):
        self.memory_used = 0
        self.gpu_fraction_used = 0.0
        self.replicas = 0


def _batch_hard_cap(
    demand: StageDemand, per_actor_budget_bytes: int, observed_bytes_per_row: Optional[float] = None
) -> int:
    """Reserve resident memory, then budget the batch's working memory."""

    per_row = demand.per_row_vram_bytes
    if observed_bytes_per_row is not None:
        per_row = max(per_row, observed_bytes_per_row * demand.memory_headroom_factor)
    if per_row <= 0:
        return demand.batch_size
    cap = int(max(0, per_actor_budget_bytes - demand.resident_bytes) // per_row)
    return max(1, cap)


def _rounds(available_batches: int, actors: int) -> int:
    """Batch rounds needed to drain ``available_batches`` with ``actors``."""

    return max(1, math.ceil(available_batches / max(1, actors)))


def _per_actor_batch_floor(
    demand: StageDemand,
    min_batches_per_actor: int,
    startup_amortization_ratio: float,
) -> int:
    """Minimum batches the (n+1)-th actor must own to justify its startup.

    Two additive terms:

    - structural: ``min(min_batches_per_actor, B // 2)`` so every stage can
      always reach at least two actors when batches allow, while a short
      stage (e.g. 32 batches) never gets one-actor-per-batch plans;
    - startup: batches whose compute covers ``ratio * actor_startup_seconds``.
      The probed batch rate overestimates the pipeline rate (dedicated
      device, in-memory samples, no input pipeline), so this term is
      deliberately advisory; the structural floor is the default guard.
    """

    structural = max(1, min(min_batches_per_actor, demand.available_batches // 2))
    if demand.actor_startup_seconds <= 0 or demand.probed_throughput_rps <= 0:
        return structural
    batch_seconds = demand.batch_size / demand.probed_throughput_rps
    if batch_seconds <= 0:
        return structural
    startup_floor = max(
        1, math.ceil(startup_amortization_ratio * demand.actor_startup_seconds / batch_seconds)
    )
    return max(structural, startup_floor)


def _marginal_makespan_gain(available_batches: int, actors: int) -> float:
    """Round-count reduction fraction from adding one actor to the stage.

    Reducing a stage's round count is a necessary condition for reducing
    the job makespan (the makespan is the max over stages of rounds * batch
    time), so gating on per-stage round gain never over-promises.
    """

    before = _rounds(available_batches, actors)
    after = _rounds(available_batches, actors + 1)
    return (before - after) / before


def plan_parallelism(
    demands: List[StageDemand],
    cluster: ClusterSpec,
    *,
    max_replicas_per_device: int = 1,
    reserved_cpus: float = 0.0,
    min_batches_per_actor: int = DEFAULT_MIN_BATCHES_PER_ACTOR,
    startup_amortization_ratio: float = DEFAULT_STARTUP_AMORTIZATION_RATIO,
    min_marginal_makespan_gain: float = DEFAULT_MIN_MARGINAL_MAKESPAN_GAIN,
) -> RuntimePlan:
    """Assign actors to GPU stages under every resource constraint.

    Deterministic: stages are processed in list order and ties resolve to
    the earliest stage, so identical inputs always produce identical plans.
    """

    if not demands:
        return RuntimePlan(stages={}, cluster=cluster, max_replicas_per_device=max_replicas_per_device)
    if max_replicas_per_device < 1:
        raise ValueError("max_replicas_per_device must be >= 1")
    if reserved_cpus < 0:
        raise ValueError("reserved_cpus must be non-negative")
    if min_batches_per_actor < 1:
        raise ValueError("min_batches_per_actor must be >= 1")
    if not 0.0 <= startup_amortization_ratio <= 1.0:
        raise ValueError("startup_amortization_ratio must be in [0, 1]")
    if not 0.0 <= min_marginal_makespan_gain < 1.0:
        raise ValueError("min_marginal_makespan_gain must be in [0, 1)")

    devices = [_DeviceState() for _ in range(cluster.num_devices)]
    cpu_used = reserved_cpus
    plan = RuntimePlan(stages={}, cluster=cluster, max_replicas_per_device=max_replicas_per_device)

    def _place(demand: StageDemand, device_id: int) -> None:
        device = devices[device_id]
        device.memory_used += demand.effective_peak_bytes
        device.gpu_fraction_used += demand.num_gpus
        device.replicas += 1

    def _candidate_device(demand: StageDemand) -> Optional[int]:
        best = None
        for device_id, device in enumerate(devices):
            if device.replicas >= max_replicas_per_device:
                continue
            if device.gpu_fraction_used + demand.num_gpus > cluster.gpu_fraction_budget + 1e-9:
                continue
            if device.memory_used + demand.effective_peak_bytes > cluster.device_memory_budget_bytes:
                continue
            if best is None or devices[best].memory_used > device.memory_used:
                best = device_id
        return best

    def _fits_extra(demand: StageDemand, stage_plan: StagePlan) -> Optional[int]:
        if stage_plan.planned_actors + 1 > demand.available_batches:
            return None
        if cpu_used + demand.num_cpus > cluster.total_cpus + 1e-9:
            return None
        return _candidate_device(demand)

    # Phase 0: explicit stages keep their configured counts (capacity
    # reserved by user configuration always wins).
    for demand in demands:
        if not demand.explicit:
            continue
        actors = demand.explicit_actors or 1
        stage_plan = StagePlan(stage_key=demand.stage_key, op_name=demand.op_name, planned_actors=0, fixed=True)
        for _ in range(actors):
            device_id = _candidate_device(demand)
            if device_id is None or cpu_used + demand.num_cpus > cluster.total_cpus + 1e-9:
                raise PlanInfeasibleError(
                    f"explicit stage {demand.op_name!r} requests {actors} actors but the "
                    "cluster cannot host them under the configured constraints"
                )
            _place(demand, device_id)
            cpu_used += demand.num_cpus
            stage_plan.planned_actors += 1
            stage_plan.device_ids.append(device_id)
        plan.stages[demand.stage_key] = stage_plan

    # Phase 1: minimum feasible plan -- one actor per planned stage.
    for demand in demands:
        if demand.explicit:
            continue
        device_id = _candidate_device(demand)
        if device_id is None or cpu_used + demand.num_cpus > cluster.total_cpus + 1e-9:
            raise PlanInfeasibleError(
                f"stage {demand.op_name!r} cannot host one actor under the configured "
                "constraints (memory, GPU fraction, replica limit, or CPU budget); "
                "refusing to overcommit"
            )
        _place(demand, device_id)
        cpu_used += demand.num_cpus
        plan.stages[demand.stage_key] = StagePlan(
            stage_key=demand.stage_key,
            op_name=demand.op_name,
            planned_actors=1,
            device_ids=[device_id],
        )

    # Phase 2: repeatedly add one actor to the bottleneck stage, but only
    # when the extra actor is economically justified:
    #   (G1) its round-count (makespan) gain clears the marginal-gain gate;
    #   (G2) it can still own enough batches to amortize its startup;
    #   (G3) it fits the CPU / GPU / memory / replica constraints.
    demand_by_key = {demand.stage_key: demand for demand in demands}
    batch_floor_by_key = {
        demand.stage_key: _per_actor_batch_floor(demand, min_batches_per_actor, startup_amortization_ratio)
        for demand in demands
    }
    while True:
        candidates = []
        for demand in demands:
            if demand.explicit:
                continue
            stage_plan = plan.stages[demand.stage_key]
            device_id = _fits_extra(demand, stage_plan)
            if device_id is None:
                continue
            actors = stage_plan.planned_actors
            if _rounds(demand.available_batches, actors + 1) < batch_floor_by_key[demand.stage_key]:
                continue
            if _marginal_makespan_gain(demand.available_batches, actors) < min_marginal_makespan_gain:
                continue
            candidates.append((demand.probed_throughput_rps, demand.stage_key, device_id))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1]))
        _rps, stage_key, device_id = candidates[0]
        demand = demand_by_key[stage_key]
        _place(demand, device_id)
        cpu_used += demand.num_cpus
        stage_plan = plan.stages[stage_key]
        stage_plan.planned_actors += 1
        stage_plan.device_ids.append(device_id)

    plan.planning_gates = {
        "min_batches_per_actor": min_batches_per_actor,
        "startup_amortization_ratio": startup_amortization_ratio,
        "min_marginal_makespan_gain": min_marginal_makespan_gain,
        "per_stage": {
            demand.stage_key: {
                "actor_startup_seconds": demand.actor_startup_seconds,
                "per_actor_batch_floor": batch_floor_by_key[demand.stage_key],
                "per_actor_entitlement_bytes": fractional_actor_entitlement(cluster, demand.num_gpus),
            }
            for demand in demands
        },
    }

    # Phase 3: per-actor VRAM budgets and batch hard caps.  The budget is
    # the device budget divided over colocated replicas (capped by the free
    # space left on the device AND by the actor's fractional entitlement --
    # Ray packs fractional-GPU actors up to 1/num_gpus per device no matter
    # where the plan lists them), so drift detection later compares apples
    # to apples against the envelope every actor actually owns.  Because a
    # stage's actors all share the same budget, it is derived from the most
    # replicated device the stage occupies (conservative for every actor).
    device_replicas = device_replica_counts(plan.stages)
    for stage_key, stage_plan in plan.stages.items():
        demand = demand_by_key[stage_key]
        device_ids = stage_plan.device_ids
        entitlement = fractional_actor_entitlement(cluster, demand.num_gpus)
        per_actor_budget = entitlement
        if device_ids:
            per_device_budget = {}
            for device_id in set(device_ids):
                device = devices[device_id]
                free_on_device = (
                    cluster.device_memory_budget_bytes - (device.memory_used - demand.effective_peak_bytes)
                )
                shared_budget = max(
                    demand.effective_peak_bytes,
                    min(
                        cluster.device_memory_budget_bytes // max(1, device_replicas.get(device_id, 1)),
                        entitlement,
                    ),
                )
                per_device_budget[device_id] = max(
                    demand.effective_peak_bytes, min(free_on_device, shared_budget)
                )
            per_actor_budget = min(per_device_budget.values())
        stage_plan.per_actor_vram_budget_bytes = int(per_actor_budget)
        cap = _batch_hard_cap(demand, stage_plan.per_actor_vram_budget_bytes)
        if demand.oom_upper_bound is not None:
            cap = min(cap, max(1, demand.oom_upper_bound - 1))
        stage_plan.batch_hard_cap = cap

    return plan


def plan_from_probe_results(
    probe_results,
    ops_info: List[Dict],
    cluster: ClusterSpec,
    *,
    max_replicas_per_device: int = 1,
    rows_per_partition: int,
    reserved_cpus: float = 0.0,
    min_batches_per_actor: int = DEFAULT_MIN_BATCHES_PER_ACTOR,
    startup_amortization_ratio: float = DEFAULT_STARTUP_AMORTIZATION_RATIO,
    min_marginal_makespan_gain: float = DEFAULT_MIN_MARGINAL_MAKESPAN_GAIN,
) -> RuntimePlan:
    """Build StageDemand rows from probe results + executor op metadata."""

    demands: List[StageDemand] = []
    for info in ops_info:
        stage_key = info["stage_key"]
        result = probe_results.get(stage_key)
        if result is None:
            raise ValueError(f"no probe result for stage {stage_key}")
        available_batches = max(1, rows_per_partition // info["batch_size"])
        demands.append(
            StageDemand(
                stage_key=stage_key,
                op_name=info["op_name"],
                probed_throughput_rps=max(result.throughput_rows_per_sec, 1e-9),
                effective_peak_bytes=max(result.peak_bytes_with_headroom, 1),
                resident_bytes=result.resident_bytes_with_headroom,
                memory_headroom_factor=result.memory_headroom_factor,
                batch_size=info["batch_size"],
                available_batches=available_batches,
                num_gpus=info.get("num_gpus", 1.0),
                num_cpus=info.get("num_cpus", 1.0),
                oom_upper_bound=result.oom_upper_bound,
                explicit=bool(info.get("explicit", False)),
                explicit_actors=info.get("explicit_actors"),
                actor_startup_seconds=float(info.get("actor_startup_seconds", 0.0) or 0.0),
            )
        )
    return plan_parallelism(
        demands,
        cluster,
        max_replicas_per_device=max_replicas_per_device,
        reserved_cpus=reserved_cpus,
        min_batches_per_actor=min_batches_per_actor,
        startup_amortization_ratio=startup_amortization_ratio,
        min_marginal_makespan_gain=min_marginal_makespan_gain,
    )
