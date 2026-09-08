"""Partition-boundary actor replanning (the outer throughput loop).

Between partitions the driver aggregates actor metrics events and applies
two deterministic rules per stage:

- drift-down: observed per-row VRAM moved above the probed baseline past
  ``drift_threshold`` and the per-actor budget is under pressure, so one
  replica is removed from every device currently at the stage's per-device
  maximum (a device-level envelope step, e.g. 4->3 replicas per device) and
  the freed VRAM budget is redistributed to the remaining actors (batch
  hard cap recomputed).  When the stage already runs one replica per
  device, dropping actors cannot change the envelope, so the decision is
  recorded as a no-op instead of shrinking only the actor count;
- scale-up gate: per-actor throughput fell far below the probed steady
  rate, so one replica is added behind a marginal-gain gate -- if the
  next partition's goodput gain is below ``marginal_gain_epsilon`` the
  addition is reverted and further performance exploration is frozen.
  Safety tightening continues during freeze, cooldown and gain evaluation.

Batch size remains owned by the actor-local controller; this module only
changes replica counts and the batch hard-cap envelope.  Every action
records its full resource envelope before/after (placement, per-device
replica maxima, colocated maxima, per-actor budget, hard cap) plus an
``effective`` flag that is true only when the envelope actually moved;
shrinking the actor list alone (e.g. 32->31 with the per-device maximum
unchanged) is recorded as a no-op.  All decisions are appended to
``plan_events.jsonl`` for auditability.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from .parallelism_planner import (
    ClusterSpec,
    RuntimePlan,
    StageDemand,
    device_replica_counts,
    fractional_actor_entitlement,
)

PLAN_EVENTS_FILENAME = "plan_events.jsonl"

OOM_ERROR_MARKER = "outofmemory"


@dataclass(frozen=True)
class ReplanConfig:
    drift_threshold: float = 1.3
    marginal_gain_epsilon: float = 0.03
    cooldown_partitions: int = 1
    min_actors: int = 1
    underperformance_ratio: float = 0.6

    def validate(self) -> None:
        if self.drift_threshold <= 1.0:
            raise ValueError("elastic_juicer_drift_threshold must be > 1.0")
        if not 0.0 <= self.marginal_gain_epsilon <= 1.0:
            raise ValueError("elastic_juicer_marginal_gain_epsilon must be in [0, 1]")
        if self.cooldown_partitions < 0:
            raise ValueError("elastic_juicer_parallelism_cooldown_partitions must be >= 0")
        if self.min_actors < 1:
            raise ValueError("min_actors must be >= 1")
        if not 0.0 < self.underperformance_ratio <= 1.0:
            raise ValueError("underperformance_ratio must be in (0, 1]")


@dataclass
class StageObservation:
    """Aggregated metrics for one stage in one partition."""

    stage_key: str
    succeeded_batches: int = 0
    oom_batches: int = 0
    rows_per_sec: float = 0.0
    bytes_per_row: Optional[float] = None
    sample_count: int = 0


def _event_attr(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def aggregate_stage_observations(
    events: Iterable[Any],
    partition_id: Optional[int] = None,
) -> Dict[str, StageObservation]:
    """Fold metrics-sink events into per-stage observations for a partition.

    Accepts ActorMetricsEvent instances or plain dicts (tests).  Events
    from other partitions are ignored; per-row VRAM is derived from CUDA
    peak allocations so drift detection needs no cost column.
    """

    observations: Dict[str, StageObservation] = {}
    throughputs: Dict[str, List[float]] = {}
    bytes_per_row_values: Dict[str, List[float]] = {}
    for event in events:
        event_partition = _event_attr(event, "partition_id", None)
        if partition_id is not None and event_partition != partition_id:
            continue
        stage_key = _event_attr(event, "stage_id") or _event_attr(event, "op_name")
        if not stage_key:
            continue
        snapshot = _event_attr(event, "snapshot")
        if snapshot is None:
            continue
        observation = observations.setdefault(stage_key, StageObservation(stage_key=stage_key))
        error_type = _event_attr(snapshot, "error_type", None) or ""
        succeeded = bool(_event_attr(snapshot, "succeeded", True))
        if not succeeded and OOM_ERROR_MARKER in str(error_type).lower():
            observation.oom_batches += 1
            continue
        if not succeeded:
            continue
        observation.succeeded_batches += 1
        throughput = _event_attr(snapshot, "throughput", 0.0)
        if throughput and throughput > 0:
            throughputs.setdefault(stage_key, []).append(float(throughput))
        cuda = _event_attr(snapshot, "cuda", None)
        batch_size = _event_attr(snapshot, "batch_size", 0)
        if cuda is not None and batch_size:
            peak_bytes = float(_event_attr(cuda, "peak_allocated_mb", 0.0)) * 1024 * 1024
            if peak_bytes > 0:
                bytes_per_row_values.setdefault(stage_key, []).append(peak_bytes / float(batch_size))
        observation.sample_count += 1

    for stage_key, observation in observations.items():
        values = throughputs.get(stage_key)
        if values:
            observation.rows_per_sec = sorted(values)[len(values) // 2]
        per_row = bytes_per_row_values.get(stage_key)
        if per_row:
            observation.bytes_per_row = sorted(per_row)[len(per_row) // 2]
    return observations


@dataclass
class PlanChange:
    partition_id: int
    stage_key: str
    op_name: str
    action: str
    reason: str
    actors_before: int
    actors_after: int
    batch_hard_cap_before: int
    batch_hard_cap_after: int
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    device_ids_before: List[int] = field(default_factory=list)
    device_ids_after: List[int] = field(default_factory=list)
    stage_max_replicas_per_device_before: int = 0
    stage_max_replicas_per_device_after: int = 0
    plan_max_colocated_replicas_before: int = 0
    plan_max_colocated_replicas_after: int = 0
    per_actor_vram_budget_before: int = 0
    per_actor_vram_budget_after: int = 0
    effective: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "stage_key": self.stage_key,
            "op_name": self.op_name,
            "action": self.action,
            "reason": self.reason,
            "actors_before": self.actors_before,
            "actors_after": self.actors_after,
            "batch_hard_cap_before": self.batch_hard_cap_before,
            "batch_hard_cap_after": self.batch_hard_cap_after,
            "device_ids_before": list(self.device_ids_before),
            "device_ids_after": list(self.device_ids_after),
            "stage_max_replicas_per_device_before": self.stage_max_replicas_per_device_before,
            "stage_max_replicas_per_device_after": self.stage_max_replicas_per_device_after,
            "plan_max_colocated_replicas_before": self.plan_max_colocated_replicas_before,
            "plan_max_colocated_replicas_after": self.plan_max_colocated_replicas_after,
            "per_actor_vram_budget_before": self.per_actor_vram_budget_before,
            "per_actor_vram_budget_after": self.per_actor_vram_budget_after,
            "effective": self.effective,
            "timestamp_ms": self.timestamp_ms,
        }


class PartitionReplanController:
    """Deterministic outer loop; one instance per job."""

    def __init__(
        self,
        plan: RuntimePlan,
        demands: Dict[str, StageDemand],
        cluster: ClusterSpec,
        config: ReplanConfig,
        *,
        baseline_bytes_per_row: Optional[Dict[str, float]] = None,
        events_path: Optional[str] = None,
        max_replicas_per_device: Optional[int] = None,
    ):
        config.validate()
        self.plan = plan
        self.demands = dict(demands)
        self.cluster = cluster
        self.config = config
        self.events_path = events_path
        self.max_replicas_per_device = (
            max_replicas_per_device
            if max_replicas_per_device is not None
            else plan.max_replicas_per_device
        )
        self.baseline_bytes_per_row = dict(baseline_bytes_per_row or {})
        self._last_change_partition: Dict[str, int] = {}
        self._frozen_stages: set = set()
        self._pending_scale_up: Dict[str, Dict[str, Any]] = {}
        self._device_memory_used: Dict[int, int] = {}
        for stage_plan in plan.stages.values():
            demand = self.demands.get(stage_plan.stage_key)
            if demand is None:
                continue
            for device_id in stage_plan.device_ids:
                self._device_memory_used[device_id] = (
                    self._device_memory_used.get(device_id, 0) + demand.effective_peak_bytes
                )

    # ------------------------------------------------------------------
    # constraint helpers (mirror the planner, evaluated against live state)
    # ------------------------------------------------------------------

    def _cpu_free(self) -> float:
        used = 0.0
        for stage_plan in self.plan.stages.values():
            demand = self.demands.get(stage_plan.stage_key)
            if demand is not None:
                used += stage_plan.planned_actors * demand.num_cpus
        return self.cluster.total_cpus - used

    def _device_candidates(self, demand: StageDemand) -> List[int]:
        replicas: Dict[int, int] = {}
        gpu_fraction_used: Dict[int, float] = {}
        for stage_plan in self.plan.stages.values():
            placed_demand = self.demands.get(stage_plan.stage_key)
            if placed_demand is None:
                # Unknown reservations cannot safely admit another actor.
                return []
            for device_id in stage_plan.device_ids:
                replicas[device_id] = replicas.get(device_id, 0) + 1
                gpu_fraction_used[device_id] = (
                    gpu_fraction_used.get(device_id, 0.0) + placed_demand.num_gpus
                )
        candidates = []
        for device_id in range(self.cluster.num_devices):
            if replicas.get(device_id, 0) >= self.max_replicas_per_device:
                continue
            if gpu_fraction_used.get(device_id, 0.0) + demand.num_gpus > self.cluster.gpu_fraction_budget + 1e-9:
                continue
            used = self._device_memory_used.get(device_id, 0)
            if used + demand.effective_peak_bytes > self.cluster.device_memory_budget_bytes:
                continue
            candidates.append(device_id)
        candidates.sort(key=lambda d: (self._device_memory_used.get(d, 0), d))
        return candidates

    def _recompute_budget_and_cap(
        self, stage_key: str, observed_bytes_per_row: Optional[float] = None, *, tighten_only: bool = False
    ) -> int:
        stage_plan = self.plan.stages[stage_key]
        demand = self.demands[stage_key]
        device_ids = stage_plan.device_ids
        # Same honesty rule as the planner: the budget is bounded by the
        # actor's fractional entitlement (Ray packs fractional-GPU actors
        # up to 1/num_gpus per device), never above the free device space,
        # and never below the probed peak the stage already measured.
        entitlement = fractional_actor_entitlement(self.cluster, demand.num_gpus)
        budget = entitlement
        if device_ids:
            device_replicas = device_replica_counts(self.plan.stages)
            per_device_budget = {}
            for device_id in set(device_ids):
                used = self._device_memory_used.get(device_id, 0)
                free_after_release = self.cluster.device_memory_budget_bytes - (
                    used - demand.effective_peak_bytes
                )
                replicas = max(1, device_replicas.get(device_id, 1))
                shared_budget = max(
                    demand.effective_peak_bytes,
                    min(
                        self.cluster.device_memory_budget_bytes // replicas,
                        entitlement,
                    ),
                )
                per_device_budget[device_id] = max(
                    demand.effective_peak_bytes, min(free_after_release, shared_budget)
                )
            # The stage's actors share one budget, so it is derived from the
            # most replicated device the stage occupies (conservative for
            # every actor, mirroring the planner).
            budget = min(per_device_budget.values())
        stage_plan.per_actor_vram_budget_bytes = int(budget)
        # Drift only tightens: an observed per-row cost above the probed
        # value replaces it (the probe sampled the dataset head; the
        # observation measured the region the actors actually ran).
        # Relaxation stays the scale-up path's job, never noise's.
        per_row = demand.per_row_vram_bytes
        if observed_bytes_per_row is not None and observed_bytes_per_row > per_row:
            per_row = observed_bytes_per_row
        cap = max(1, int(budget // max(per_row, 1e-9)))
        if demand.oom_upper_bound is not None:
            cap = min(cap, max(1, demand.oom_upper_bound - 1))
        if tighten_only and stage_plan.batch_hard_cap > 0:
            cap = min(cap, stage_plan.batch_hard_cap)
        stage_plan.batch_hard_cap = cap
        return cap

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    def on_partition_complete(
        self,
        partition_id: int,
        observations: Dict[str, StageObservation],
        goodput_rows_per_sec: float,
    ) -> List[PlanChange]:
        """Apply the replan rules once per finished partition."""

        changes: List[PlanChange] = []
        for stage_key in sorted(self.plan.stages):
            stage_plan = self.plan.stages[stage_key]
            demand = self.demands.get(stage_key)
            if demand is None:
                continue

            # Safety is evaluated before performance freeze, cooldown and
            # gain checks. A resource-pressure window cannot validate a
            # throughput experiment, so discard its pending verdict.
            observation = observations.get(stage_key)
            baseline = self.baseline_bytes_per_row.get(stage_key)
            if (
                observation is not None
                and observation.sample_count > 0
                and baseline
                and observation.bytes_per_row is not None
                and observation.bytes_per_row > baseline * self.config.drift_threshold
                and self._budget_pressure(stage_key, observation)
            ):
                self._pending_scale_up.pop(stage_key, None)
                changes.append(self._scale(
                    stage_key,
                    partition_id,
                    -1,
                    action="scale_down",
                    observed_bytes_per_row=observation.bytes_per_row,
                    reason=(
                        f"per-row VRAM {observation.bytes_per_row:.0f}B > baseline "
                        f"{baseline:.0f}B x {self.config.drift_threshold} "
                        f"(oom_batches={observation.oom_batches}); safety tightening"
                    ),
                ))
                continue

            if stage_plan.fixed:
                continue

            # Performance-only decisions may be frozen or rate-limited.
            pending = self._pending_scale_up.pop(stage_key, None)
            if pending is not None:
                previous_goodput = pending["goodput_before"]
                gain = 0.0
                if previous_goodput > 0:
                    gain = (goodput_rows_per_sec - previous_goodput) / previous_goodput
                if gain < self.config.marginal_gain_epsilon:
                    changes.append(self._scale(
                        stage_key,
                        partition_id,
                        -1,
                        action="revert_scale_up",
                        reason=(
                            f"marginal goodput gain {gain:.3f} < epsilon "
                            f"{self.config.marginal_gain_epsilon}; stage frozen"
                        ),
                        revert_device=pending.get("device_id"),
                    ))
                    self._frozen_stages.add(stage_key)
                    continue
                # Gain met: treat the resolution as a change of this
                # partition so the cooldown applies before the next move.
                self._last_change_partition[stage_key] = partition_id
                continue

            if stage_key in self._frozen_stages:
                continue
            if observation is None or observation.sample_count == 0:
                continue

            last_change = self._last_change_partition.get(stage_key)
            if last_change is not None and partition_id - last_change < self.config.cooldown_partitions:
                continue

            expected_rps = demand.probed_throughput_rps * self.config.underperformance_ratio
            if (
                observation.rows_per_sec > 0
                and observation.rows_per_sec < expected_rps
                and stage_plan.planned_actors < demand.available_batches
                and self._cpu_free() >= demand.num_cpus
            ):
                candidates = self._device_candidates(demand)
                if candidates:
                    device_id = candidates[0]
                    change = self._scale(stage_key, partition_id, +1, action="scale_up", reason=(
                        f"per-actor throughput {observation.rows_per_sec:.1f} rows/s below "
                        f"{self.config.underperformance_ratio:.0%} of probed "
                        f"{demand.probed_throughput_rps:.1f} rows/s"
                    ))
                    self._pending_scale_up[stage_key] = {
                        "goodput_before": goodput_rows_per_sec,
                        "device_id": device_id,
                    }
                    changes.append(change)

        if changes and self.events_path:
            self._append_events(changes)
        return changes

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _budget_pressure(self, stage_key: str, observation: StageObservation) -> bool:
        stage_plan = self.plan.stages[stage_key]
        if observation.oom_batches > 0:
            return True
        if observation.bytes_per_row is None:
            return False
        return observation.bytes_per_row * self.demands[stage_key].batch_size > stage_plan.per_actor_vram_budget_bytes

    def _stage_max_replicas(self, stage_key: str) -> int:
        """Largest number of this stage's replicas on any single device."""

        counts: Dict[int, int] = {}
        for device_id in self.plan.stages[stage_key].device_ids:
            counts[device_id] = counts.get(device_id, 0) + 1
        return max(counts.values()) if counts else 0

    def _plan_max_colocated_replicas(self) -> int:
        """Largest total replicas (any stage) colocated on any single device."""

        counts = device_replica_counts(self.plan.stages)
        return max(counts.values()) if counts else 0

    def _remove_device_replicas(self, stage_key: str, *, device_ids: List[int]) -> int:
        """Remove one replica of the stage from each given device."""

        stage_plan = self.plan.stages[stage_key]
        demand = self.demands[stage_key]
        removed = 0
        for device_id in device_ids:
            if device_id not in stage_plan.device_ids:
                continue
            stage_plan.device_ids.remove(device_id)
            self._device_memory_used[device_id] = max(
                0, self._device_memory_used.get(device_id, 0) - demand.effective_peak_bytes
            )
            removed += 1
        stage_plan.planned_actors -= removed
        return removed

    def _scale_down_targets(self, stage_key: str) -> List[int]:
        """Devices to shed: one replica from every device at the stage maximum.

        All-or-nothing: the step only makes sense if the per-device maximum
        actually drops, and never below ``min_actors`` total.  Returns an
        empty list when the stage already runs one replica per device (a
        pure actor-count reduction would leave the envelope untouched).
        """

        stage_plan = self.plan.stages[stage_key]
        if stage_plan.fixed:
            return []
        counts: Dict[int, int] = {}
        for device_id in stage_plan.device_ids:
            counts[device_id] = counts.get(device_id, 0) + 1
        if not counts or max(counts.values()) < 2:
            return []
        stage_max = max(counts.values())
        targets = sorted(device_id for device_id, count in counts.items() if count == stage_max)
        if stage_plan.planned_actors - len(targets) < self.config.min_actors:
            return []
        return targets

    def _scale(
        self,
        stage_key: str,
        partition_id: int,
        delta: int,
        *,
        action: str,
        reason: str,
        revert_device: Optional[int] = None,
        observed_bytes_per_row: Optional[float] = None,
    ) -> PlanChange:
        stage_plan = self.plan.stages[stage_key]
        demand = self.demands[stage_key]
        actors_before = stage_plan.planned_actors
        cap_before = stage_plan.batch_hard_cap
        budget_before = stage_plan.per_actor_vram_budget_bytes
        devices_before = list(stage_plan.device_ids)
        stage_max_before = self._stage_max_replicas(stage_key)
        colocated_before = self._plan_max_colocated_replicas()

        if delta > 0:
            candidates = self._device_candidates(demand)
            if not candidates:
                raise RuntimeError(f"cannot scale up stage {stage_key}: no device has capacity")
            device_id = candidates[0]
            stage_plan.device_ids.append(device_id)
            self._device_memory_used[device_id] = (
                self._device_memory_used.get(device_id, 0) + demand.effective_peak_bytes
            )
            stage_plan.planned_actors += 1
        else:
            if revert_device is not None:
                # Exact undo of the probe scale-up: shed the replica that was
                # added on the recorded device.
                targets = [revert_device]
            else:
                # Device-level envelope step: one replica off every device
                # currently at the stage's per-device maximum.
                targets = self._scale_down_targets(stage_key)
            if (
                not targets
                or (revert_device is not None and revert_device not in stage_plan.device_ids)
            ):
                # No device-level replica step is possible (e.g. one replica
                # per device): dropping actors cannot change the envelope.
                # Drift still has one honest lever left -- tighten the budget
                # and batch hard cap against the observed per-row cost so
                # the next partition's co-located actors do not collide at
                # the probed-head cap (the W2' floor OOMs).
                self._last_change_partition[stage_key] = partition_id
                cap_after = self._recompute_budget_and_cap(
                    stage_key, observed_bytes_per_row=observed_bytes_per_row, tighten_only=action == "scale_down"
                )
                budget_after = stage_plan.per_actor_vram_budget_bytes
                envelope_tightened = cap_after != cap_before or budget_after != budget_before
                return PlanChange(
                    partition_id=partition_id,
                    stage_key=stage_key,
                    op_name=stage_plan.op_name,
                    action=action,
                    reason=(
                        reason
                        + (
                            "; envelope tightened to the observed per-row VRAM"
                            if envelope_tightened
                            else "; no-op: no device-level replica step available and "
                            "the envelope is already tight at the observed per-row VRAM "
                            "(actor-count-only reductions are not recorded as adaptations)"
                        )
                    ),
                    actors_before=actors_before,
                    actors_after=actors_before,
                    batch_hard_cap_before=cap_before,
                    batch_hard_cap_after=cap_after,
                    device_ids_before=devices_before,
                    device_ids_after=devices_before,
                    stage_max_replicas_per_device_before=stage_max_before,
                    stage_max_replicas_per_device_after=stage_max_before,
                    plan_max_colocated_replicas_before=colocated_before,
                    plan_max_colocated_replicas_after=colocated_before,
                    per_actor_vram_budget_before=budget_before,
                    per_actor_vram_budget_after=budget_after,
                    effective=envelope_tightened,
                )
            self._remove_device_replicas(stage_key, device_ids=targets)

        cap_after = self._recompute_budget_and_cap(
            stage_key, observed_bytes_per_row=observed_bytes_per_row, tighten_only=action == "scale_down"
        )
        self._last_change_partition[stage_key] = partition_id
        stage_max_after = self._stage_max_replicas(stage_key)
        colocated_after = self._plan_max_colocated_replicas()
        budget_after = stage_plan.per_actor_vram_budget_bytes
        envelope_changed = (
            stage_max_before != stage_max_after
            or colocated_before != colocated_after
            or budget_before != budget_after
            or cap_before != cap_after
        )
        # A scale-up always extends the physical footprint (one more actor
        # process plus its reserved peak memory on a device); a scale-down
        # only counts when the safety envelope itself moved.
        effective = envelope_changed or delta > 0
        return PlanChange(
            partition_id=partition_id,
            stage_key=stage_key,
            op_name=stage_plan.op_name,
            action=action,
            reason=reason,
            actors_before=actors_before,
            actors_after=stage_plan.planned_actors,
            batch_hard_cap_before=cap_before,
            batch_hard_cap_after=cap_after,
            device_ids_before=devices_before,
            device_ids_after=list(stage_plan.device_ids),
            stage_max_replicas_per_device_before=stage_max_before,
            stage_max_replicas_per_device_after=stage_max_after,
            plan_max_colocated_replicas_before=colocated_before,
            plan_max_colocated_replicas_after=colocated_after,
            per_actor_vram_budget_before=budget_before,
            per_actor_vram_budget_after=budget_after,
            effective=effective,
        )

    def _append_events(self, changes: List[PlanChange]) -> None:
        try:
            directory = os.path.dirname(os.path.abspath(self.events_path))
            os.makedirs(directory, exist_ok=True)
            with open(self.events_path, "a") as handle:
                for change in changes:
                    handle.write(json.dumps(change.to_dict(), sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            # Auditability loss must never fail the data path.
            pass
