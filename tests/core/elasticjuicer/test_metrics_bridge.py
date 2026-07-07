import sys

import pytest

from data_juicer.core.elasticjuicer.contracts import ClusterState, TowerMode
from data_juicer.core.elasticjuicer.elastic_juicer import ElasticJuicer
from data_juicer.core.elasticjuicer.runtime.metrics_bridge import MetricsBridge
from data_juicer.core.elasticjuicer.scheduler.captain import CaptainConfig


class _SnapshotRemote:
    def __init__(self, snapshots):
        self._snapshots = list(snapshots)

    def remote(self):
        if len(self._snapshots) == 1:
            return self._snapshots[0]
        return self._snapshots.pop(0)


class _Collector:
    def __init__(self, snapshots):
        self.snapshot = _SnapshotRemote(snapshots)


def _cluster():
    return ClusterState(
        total_cpu_cores=8,
        total_memory_mb=16000,
        total_gpu_count=0,
        available_cpu_cores=8,
        available_memory_mb=16000,
        available_gpus=0,
    )


def _metrics(rows, calls=1, slices=1, ooms=0, failures=0, batch_size=4):
    return {
        "stage": {
            "calls": calls,
            "outer_rows": rows,
            "successful_slices": slices,
            "oom_retries": ooms,
            "failures": failures,
            "current_batch_size": batch_size,
        }
    }


def test_metrics_bridge_uses_counter_deltas(monkeypatch):
    ray = sys.modules["ray"]
    monkeypatch.setattr(ray, "get", lambda value: value, raising=False)

    now = [100.0]
    coordinator = ElasticJuicer(
        cluster_state=_cluster(),
        tower_mode=TowerMode.CLOSED_LOOP,
    )
    captain = coordinator.register_captain(
        CaptainConfig(stage_name="stage", initial_batch_size=4, max_batch_size=16)
    )
    coordinator.tower.stages["stage"].last_update = 0.0
    collector = _Collector([
        _metrics(rows=10, calls=2, slices=3, ooms=1, batch_size=2),
        _metrics(rows=10, calls=2, slices=3, ooms=1, batch_size=2),
        _metrics(rows=15, calls=3, slices=4, ooms=1, batch_size=3),
    ])
    bridge = MetricsBridge(coordinator, collector, poll_interval=1.0, clock=lambda: now[0])

    bridge._last_poll_time = 99.0
    bridge.bridge_cycle()

    assert captain.samples_processed == 10
    assert captain.metrics.throughput == pytest.approx(10.0)
    assert captain.metrics.oom_count == 1
    assert captain.micro_scheduler.controller.current_batch_size == 2

    now[0] = 101.0
    bridge.bridge_cycle()
    assert captain.samples_processed == 10

    now[0] = 102.0
    bridge.bridge_cycle()
    assert captain.samples_processed == 15
    assert captain.metrics.throughput == pytest.approx(5.0)
    assert captain.micro_scheduler.controller.current_batch_size == 3
