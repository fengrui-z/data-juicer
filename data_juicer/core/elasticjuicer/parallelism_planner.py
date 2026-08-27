"""Throughput-aware actor planning for ElasticJuicer auto-parallelism.

Pure planning logic: every GPU stage starts with one actor, then actors
are repeatedly assigned to the current bottleneck (lowest probed
throughput) until the next assignment would violate any constraint:

- cluster CPU budget;
- per-device Ray GPU fraction;
- measured per-device memory (probed peak + headroom);
- per-device replica limit;
- available batches for the stage;
- capacity reserved by explicitly configured operators.

If the minimum plan (one actor per stage) does not fit, planning fails
fast instead of hiding the problem through overcommitment.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .gpu_probe import atomic_write_json

RUNTIME_PLAN_SCHEMA_VERSION = 1
RUNTIME_PLAN_FILENAME = "runtime_plan.json"


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

    @property
    def per_row_vram_bytes(self) -> float:
        return self.effective_peak_bytes / float(self.batch_size)


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

    def to_dict(self) -> Dict:
        return {
            "schema_version": RUNTIME_PLAN_SCHEMA_VERSION,
            "generated_at_ms": self.generated_at_ms,
            "max_replicas_per_device": self.max_replicas_per_device,
            "cluster": asdict(self.cluster) if self.cluster else None,
            "stages": {key: asdict(value) for key, value in self.stages.items()},
        }


def save_runtime_plan(path: str, plan: RuntimePlan) -> None:
    atomic_write_json(path, plan.to_dict())


class _DeviceState:
    __slots__ = ("memory_used", "gpu_fraction_used", "replicas")

    def __init__(self):
        self.memory_used = 0
        self.gpu_fraction_used = 0.0
        self.replicas = 0


def _batch_hard_cap(demand: StageDemand, per_actor_budget_bytes: int) -> int:
    """Largest batch whose probed per-row VRAM fits the per-actor budget."""

    per_row = demand.per_row_vram_bytes
    if per_row <= 0:
        return demand.batch_size
    cap = int(per_actor_budget_bytes // per_row)
    return max(1, cap)


def plan_parallelism(
    demands: List[StageDemand],
    cluster: ClusterSpec,
    *,
    max_replicas_per_device: int = 1,
    reserved_cpus: float = 0.0,
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

    # Phase 2: repeatedly add one actor to the bottleneck stage.
    demand_by_key = {demand.stage_key: demand for demand in demands}
    while True:
        candidates = []
        for demand in demands:
            if demand.explicit:
                continue
            stage_plan = plan.stages[demand.stage_key]
            device_id = _fits_extra(demand, stage_plan)
            if device_id is None:
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

    # Phase 3: per-actor VRAM budgets and batch hard caps.
    for stage_key, stage_plan in plan.stages.items():
        demand = demand_by_key[stage_key]
        device_ids = stage_plan.device_ids
        per_actor_budget = cluster.device_memory_budget_bytes
        if device_ids:
            device = devices[device_ids[0]]
            free_on_device = cluster.device_memory_budget_bytes - (device.memory_used - demand.effective_peak_bytes)
            per_actor_budget = max(demand.effective_peak_bytes, free_on_device)
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
                batch_size=info["batch_size"],
                available_batches=available_batches,
                num_gpus=info.get("num_gpus", 1.0),
                num_cpus=info.get("num_cpus", 1.0),
                oom_upper_bound=result.oom_upper_bound,
                explicit=bool(info.get("explicit", False)),
                explicit_actors=info.get("explicit_actors"),
            )
        )
    return plan_parallelism(
        demands,
        cluster,
        max_replicas_per_device=max_replicas_per_device,
        reserved_cpus=reserved_cpus,
    )
