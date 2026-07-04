import threading
from types import SimpleNamespace

import pytest

from data_juicer.core.elasticjuicer.contracts import (
    ClusterState,
    TowerMode,
)
from data_juicer.core.elasticjuicer.elastic_juicer import (
    ElasticJuicer,
    detect_cluster_state,
)
from data_juicer.core.elasticjuicer.scheduler.captain import CaptainConfig


def _cluster():
    return ClusterState(
        total_cpu_cores=8,
        total_memory_mb=16000,
        total_gpu_count=0,
        available_cpu_cores=8,
        available_memory_mb=16000,
        available_gpus=0,
    )


def test_detect_cluster_state_uses_available_host_capacity(monkeypatch):
    memory = SimpleNamespace(total=16 * 1024**3, available=6 * 1024**3)
    monkeypatch.setattr(
        "data_juicer.core.elasticjuicer.elastic_juicer.psutil.virtual_memory",
        lambda: memory,
    )
    monkeypatch.setattr(
        "data_juicer.core.elasticjuicer.elastic_juicer.psutil.cpu_count",
        lambda logical: 12,
    )

    state = detect_cluster_state()

    assert state.total_cpu_cores == 12
    assert state.available_cpu_cores == 12
    assert state.total_memory_mb == pytest.approx(16384)
    assert state.available_memory_mb == pytest.approx(6144)


def test_register_captain_is_idempotent():
    coordinator = ElasticJuicer(cluster_state=_cluster())
    config = CaptainConfig(stage_name="mapper")

    first = coordinator.register_captain(config)
    second = coordinator.register_captain(config)

    assert first is second
    assert len(coordinator.tower.stages) == 1
    assert len(coordinator.tower._controllers) == 1


def test_shadow_tick_returns_plan_without_applying_quota():
    coordinator = ElasticJuicer(
        cluster_state=_cluster(),
        tower_mode=TowerMode.SHADOW,
    )
    captain = coordinator.register_captain(CaptainConfig(stage_name="mapper"))
    captain.metrics.queue_depth = 1000
    captain.metrics.throughput = 1

    plan = coordinator.tick()

    assert plan.mode is TowerMode.SHADOW
    assert plan.quotas[0].stage_name == "mapper"
    assert captain.quota is None


def test_closed_loop_tick_applies_quota_to_captain():
    coordinator = ElasticJuicer(
        cluster_state=_cluster(),
        tower_mode=TowerMode.CLOSED_LOOP,
    )
    captain = coordinator.register_captain(CaptainConfig(stage_name="mapper"))
    captain.metrics.queue_depth = 1000
    captain.metrics.throughput = 1

    plan = coordinator.tick()

    assert plan.mode is TowerMode.CLOSED_LOOP
    assert captain.quota is not None
    assert captain.quota.captain_id == plan.quotas[0].captain_id
    assert captain.metrics.current_parallelism == plan.quotas[0].target_parallelism


def test_status_reports_tower_and_captain_state():
    coordinator = ElasticJuicer(cluster_state=_cluster())
    coordinator.register_captain(CaptainConfig(stage_name="mapper"))

    status = coordinator.get_status()

    assert status["is_running"] is False
    assert status["tower"]["total_stages"] == 1
    assert status["captains"]["mapper"]["stage_name"] == "mapper"


def test_background_lifecycle_ticks_and_stops():
    coordinator = ElasticJuicer(
        cluster_state=_cluster(),
        rebalance_interval_sec=0.01,
    )
    ticked = threading.Event()
    original_tick = coordinator.tick

    def recording_tick(force=True):
        ticked.set()
        return original_tick(force=force)

    coordinator.tick = recording_tick
    coordinator.start()
    assert ticked.wait(timeout=1)
    coordinator.stop()

    assert coordinator.is_running is False
