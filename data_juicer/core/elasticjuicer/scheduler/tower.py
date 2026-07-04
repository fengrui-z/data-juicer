"""Global macro-scheduler for ElasticJuicer.

Tower computes cluster-wide resource plans.  In shadow mode those plans are
advisory; in closed-loop mode they are also delivered to registered Captains.
"""

import json
import os
import tempfile
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

from ..contracts import (
    AllocationPlan,
    ClusterState,
    ResourceQuota,
    ResourceQuotaSnapshot,
    StageMetrics,
    TopologyMode,
    TowerMode,
)


class Tower:
    """Plan and optionally apply resource allocations across pipeline stages."""

    def __init__(
        self,
        cluster_state: ClusterState,
        target_queue_depth: int = 100,
        sla_latency_ms: float = 5000.0,
        update_interval_sec: float = 5.0,
        history_window: int = 20,
        mode: TowerMode | str = TowerMode.SHADOW,
        clock=time.time,
    ):
        if target_queue_depth < 0:
            raise ValueError("target_queue_depth must be non-negative")
        if sla_latency_ms <= 0:
            raise ValueError("sla_latency_ms must be positive")
        if update_interval_sec < 0:
            raise ValueError("update_interval_sec must be non-negative")
        if history_window < 1:
            raise ValueError("history_window must be at least 1")

        self.cluster = cluster_state
        self.target_queue_depth = target_queue_depth
        self.sla_latency_ms = sla_latency_ms
        self.update_interval = update_interval_sec
        self.history_window = history_window
        self.mode = TowerMode(mode)
        self._clock = clock

        self.stages: Dict[str, StageMetrics] = {}
        self.quotas: Dict[str, ResourceQuota] = {}
        self._captain_to_stage: Dict[str, str] = {}
        self._stage_to_captain: Dict[str, str] = {}
        self._controllers: Dict[str, object] = {}
        self.metrics_history: Dict[str, deque] = {}

        self.last_allocation_time = 0.0
        self.last_plan: Optional[AllocationPlan] = None
        self._generation = 0
        self.sla_violations = 0
        self.total_requests = 0

    def register_stage(self, stage_name: str, initial_parallelism: int = 1) -> str:
        """Register a stage once and return its stable Captain identifier."""

        if not stage_name:
            raise ValueError("stage_name must not be empty")
        if initial_parallelism < 1:
            raise ValueError("initial_parallelism must be at least 1")
        if stage_name in self._stage_to_captain:
            return self._stage_to_captain[stage_name]

        captain_id = f"captain_{stage_name}_{len(self.stages):04d}"
        self.stages[stage_name] = StageMetrics(
            stage_name=stage_name,
            current_parallelism=initial_parallelism,
        )
        self.metrics_history[stage_name] = deque(maxlen=self.history_window)
        self._captain_to_stage[captain_id] = stage_name
        self._stage_to_captain[stage_name] = captain_id
        self._rebalance_registered_quotas()
        return captain_id

    def register_controller(self, captain_id: str, controller: object):
        """Attach a local controller that accepts ``set_quota(ResourceQuota)``."""

        if captain_id not in self._captain_to_stage:
            raise ValueError(f"Captain {captain_id} not registered")
        setter = getattr(controller, "set_quota", None)
        if not callable(setter):
            raise TypeError("controller must provide a callable set_quota")
        self._controllers[captain_id] = controller

    def update_cluster_state(self, cluster_state: ClusterState):
        """Replace the capacity snapshot and conserve quotas immediately."""

        self.cluster = cluster_state
        self._rebalance_registered_quotas()

    def update_stage_metrics(self, stage_name: str, metrics: StageMetrics) -> bool:
        """Record a fresh stage metric, rejecting mismatched or stale samples."""

        if stage_name not in self.stages:
            raise ValueError(f"Stage {stage_name} not registered")
        if metrics.stage_name != stage_name:
            raise ValueError("metrics.stage_name must match stage_name")
        current = self.stages[stage_name]
        if metrics.last_update <= current.last_update:
            return False

        self.stages[stage_name] = replace(metrics)
        self.metrics_history[stage_name].append(
            {
                "timestamp": metrics.last_update,
                "queue_depth": metrics.queue_depth,
                "throughput": metrics.throughput,
                "latency_ms": metrics.avg_latency_ms,
                "cpu_util": metrics.cpu_utilization,
                "memory_util": metrics.memory_utilization,
            }
        )
        self.total_requests += 1
        if metrics.avg_latency_ms > self.sla_latency_ms:
            self.sla_violations += 1
        return True

    def plan_allocation(self) -> AllocationPlan:
        """Compute an immutable allocation plan without changing live quotas."""

        bottlenecks = self._identify_bottlenecks()
        stage_parallelism = {
            stage_name: self._compute_target_parallelism(
                metrics,
                is_bottleneck=stage_name in bottlenecks,
            )
            for stage_name, metrics in self.stages.items()
        }
        total_parallelism = max(1, sum(stage_parallelism.values()))

        snapshots = []
        for stage_name, target_parallelism in stage_parallelism.items():
            captain_id = self._stage_to_captain[stage_name]
            weight = target_parallelism / total_parallelism
            metrics = self.stages[stage_name]
            snapshots.append(
                ResourceQuotaSnapshot(
                    captain_id=captain_id,
                    stage_name=stage_name,
                    target_parallelism=target_parallelism,
                    cpu_quota=self.cluster.available_cpu_cores * weight,
                    memory_quota_mb=self.cluster.available_memory_mb * weight,
                    gpu_quota=self.cluster.available_gpus * weight,
                    target_throughput=self._compute_target_throughput(metrics),
                    topology_mode=self._decide_topology(stage_name, metrics),
                )
            )

        self._generation += 1
        return AllocationPlan(
            generation=self._generation,
            created_at=self._clock(),
            mode=self.mode,
            bottlenecks=tuple(bottlenecks),
            quotas=tuple(snapshots),
        )

    def apply_plan(self, plan: AllocationPlan):
        """Validate and atomically install a plan, then notify controllers."""

        expected_ids = set(self._captain_to_stage)
        plan_ids = {snapshot.captain_id for snapshot in plan.quotas}
        if plan_ids != expected_ids:
            raise ValueError("allocation plan Captain IDs do not match registered stages")
        for snapshot in plan.quotas:
            expected_stage = self._captain_to_stage[snapshot.captain_id]
            if snapshot.stage_name != expected_stage:
                raise ValueError("allocation plan stage does not match Captain ID")

        quotas = plan.quota_map()
        self.quotas = quotas
        for captain_id, controller in self._controllers.items():
            controller.set_quota(quotas[captain_id])

    def rebalance(self, force: bool = False) -> AllocationPlan:
        """Pull local metrics, plan, and apply only in closed-loop mode."""

        self._collect_controller_metrics()
        now = self._clock()
        if not force and self.last_plan is not None and now - self.last_allocation_time < self.update_interval:
            return self.last_plan

        plan = self.plan_allocation()
        self.last_plan = plan
        self.last_allocation_time = now
        if self.mode is TowerMode.CLOSED_LOOP:
            self.apply_plan(plan)
        return plan

    def allocate_resources(self) -> Dict[str, ResourceQuota]:
        """Compatibility API: compute and install quotas without broadcasting."""

        now = self._clock()
        if self.last_plan is not None and now - self.last_allocation_time < self.update_interval:
            return self.quotas

        plan = self.plan_allocation()
        self.last_plan = plan
        self.last_allocation_time = now
        self.quotas = plan.quota_map()
        return self.quotas

    def save_plan(self, path: str | os.PathLike, plan: Optional[AllocationPlan] = None):
        """Persist a plan with an atomic replace."""

        selected = plan or self.last_plan
        if selected is None:
            raise ValueError("no allocation plan available")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(selected.to_dict(), stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    def _collect_controller_metrics(self):
        for captain_id, controller in self._controllers.items():
            metrics = getattr(controller, "metrics", None)
            if not isinstance(metrics, StageMetrics):
                continue
            stage_name = self._captain_to_stage[captain_id]
            self.update_stage_metrics(stage_name, metrics)

    def _rebalance_registered_quotas(self):
        if not self.stages:
            self.quotas = {}
            return
        total_parallelism = sum(metrics.current_parallelism for metrics in self.stages.values())
        quotas = {}
        for stage_name, metrics in self.stages.items():
            captain_id = self._stage_to_captain[stage_name]
            weight = metrics.current_parallelism / total_parallelism
            quotas[captain_id] = ResourceQuota(
                captain_id=captain_id,
                target_parallelism=metrics.current_parallelism,
                cpu_quota=self.cluster.available_cpu_cores * weight,
                memory_quota_mb=self.cluster.available_memory_mb * weight,
                gpu_quota=self.cluster.available_gpus * weight,
                target_throughput=10.0,
                topology_mode=TopologyMode.ADAPTIVE,
            )
        self.quotas = quotas

    def _identify_bottlenecks(self) -> List[str]:
        bottlenecks = []
        for stage_name, metrics in self.stages.items():
            queue_pressure = metrics.queue_depth > self.target_queue_depth
            latency_pressure = metrics.avg_latency_ms > self.sla_latency_ms * 0.8
            throughput_declining = False
            history = self.metrics_history.get(stage_name)
            if history and len(history) >= 3:
                recent = list(history)[-3:]
                throughput_declining = recent[-1]["throughput"] < recent[0]["throughput"] * 0.9
            if queue_pressure or latency_pressure or throughput_declining:
                bottlenecks.append(stage_name)
        return bottlenecks

    def _compute_target_parallelism(
        self,
        metrics: StageMetrics,
        is_bottleneck: bool,
    ) -> int:
        current = metrics.current_parallelism
        if is_bottleneck:
            if metrics.throughput > 0:
                queue_drain_time = metrics.queue_depth / metrics.throughput
                sla_seconds = self.sla_latency_ms / 1000.0
                if queue_drain_time > sla_seconds:
                    scale_factor = min(2.0, queue_drain_time / sla_seconds)
                    target = max(current + 1, int(current * scale_factor))
                else:
                    target = current + 1
            else:
                target = current + 1
        elif metrics.queue_depth < self.target_queue_depth * 0.5 and current > 1:
            target = current - 1
        else:
            target = current
        return max(1, min(target, self._estimate_max_parallelism()))

    def _compute_target_throughput(self, metrics: StageMetrics) -> float:
        remaining_time = max(
            0.1,
            self.sla_latency_ms / 1000.0 - metrics.avg_latency_ms / 1000.0,
        )
        target = metrics.queue_depth / remaining_time if metrics.queue_depth > 0 else metrics.throughput
        return max(1.0, target)

    def _decide_topology(
        self,
        stage_name: str,
        metrics: StageMetrics,
    ) -> TopologyMode:
        del stage_name
        bottleneck_count = sum(
            utilization > 80
            for utilization in (
                metrics.cpu_utilization,
                metrics.memory_utilization,
                metrics.gpu_utilization,
            )
        )
        if bottleneck_count >= 2:
            return TopologyMode.DISTRIBUTED
        if bottleneck_count == 0:
            return TopologyMode.CO_LOCATION
        return TopologyMode.ADAPTIVE

    def _estimate_max_parallelism(self) -> int:
        cpu_limit = int(self.cluster.available_cpu_cores)
        memory_limit = int(self.cluster.available_memory_mb / 1024)
        return max(1, min(cpu_limit, memory_limit))

    def _get_stage_from_captain(self, captain_id: str) -> str:
        return self._captain_to_stage.get(captain_id, captain_id)

    def _captain_id_for_stage(self, stage_name: str) -> Optional[str]:
        return self._stage_to_captain.get(stage_name)

    def get_sla_compliance_rate(self) -> float:
        if self.total_requests == 0:
            return 100.0
        return ((self.total_requests - self.sla_violations) / self.total_requests) * 100.0

    def get_global_stats(self) -> Dict:
        return {
            "mode": self.mode.value,
            "generation": self._generation,
            "total_stages": len(self.stages),
            "total_parallelism": sum(quota.target_parallelism for quota in self.quotas.values()),
            "sla_compliance_rate": self.get_sla_compliance_rate(),
            "total_requests": self.total_requests,
            "sla_violations": self.sla_violations,
            "cluster_cpu_util": (
                (self.cluster.total_cpu_cores - self.cluster.available_cpu_cores) / self.cluster.total_cpu_cores * 100
                if self.cluster.total_cpu_cores > 0
                else 0
            ),
            "cluster_memory_util": (
                (self.cluster.total_memory_mb - self.cluster.available_memory_mb) / self.cluster.total_memory_mb * 100
                if self.cluster.total_memory_mb > 0
                else 0
            ),
        }
