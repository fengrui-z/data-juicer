import json
from dataclasses import FrozenInstanceError

import pytest

from data_juicer.core.elasticjuicer.contracts import (
    ClusterState,
    StageMetrics,
    TopologyMode,
    TowerMode,
)
from data_juicer.core.elasticjuicer.scheduler.tower import Tower


def _cluster(cpu=8, mem_mb=16000, gpu=2):
    return ClusterState(
        total_cpu_cores=cpu,
        total_memory_mb=mem_mb,
        total_gpu_count=gpu,
        available_cpu_cores=cpu,
        available_memory_mb=mem_mb,
        available_gpus=gpu,
    )


def test_register_stage_returns_unique_captain_ids():
    tower = Tower(cluster_state=_cluster())
    id1 = tower.register_stage("stage_a")
    id2 = tower.register_stage("stage_b")
    assert id1 != id2


def test_register_same_stage_twice_returns_same_id():
    tower = Tower(cluster_state=_cluster())
    id1 = tower.register_stage("stage_a")
    id2 = tower.register_stage("stage_a")
    assert id1 == id2


def test_resource_conservation_cpu():
    tower = Tower(cluster_state=_cluster(cpu=8, mem_mb=16000, gpu=2), update_interval_sec=0)
    tower.register_stage("s1", initial_parallelism=2)
    tower.register_stage("s2", initial_parallelism=3)

    for name, metrics in tower.stages.items():
        metrics.queue_depth = 200
        metrics.throughput = 10.0

    quotas = tower.allocate_resources()
    total_cpu = sum(q.cpu_quota for q in quotas.values())
    assert total_cpu <= _cluster(cpu=8).available_cpu_cores + 1e-6


def test_resource_conservation_memory():
    tower = Tower(cluster_state=_cluster(cpu=8, mem_mb=16000, gpu=2), update_interval_sec=0)
    tower.register_stage("s1", initial_parallelism=1)
    tower.register_stage("s2", initial_parallelism=1)
    tower.register_stage("s3", initial_parallelism=1)

    for name in tower.stages:
        tower.stages[name].queue_depth = 200
        tower.stages[name].throughput = 5.0

    quotas = tower.allocate_resources()
    total_mem = sum(q.memory_quota_mb for q in quotas.values())
    assert total_mem <= 16000 + 1e-6


def test_resource_conservation_gpu():
    tower = Tower(cluster_state=_cluster(cpu=8, mem_mb=16000, gpu=4), update_interval_sec=0)
    tower.register_stage("s1", initial_parallelism=2)
    tower.register_stage("s2", initial_parallelism=2)

    for name in tower.stages:
        tower.stages[name].queue_depth = 500
        tower.stages[name].throughput = 1.0

    quotas = tower.allocate_resources()
    total_gpu = sum(q.gpu_quota for q in quotas.values())
    assert total_gpu <= 4 + 1e-6


def test_stale_metrics_handled():
    tower = Tower(cluster_state=_cluster(), update_interval_sec=0)
    tower.register_stage("s1")

    tower.stages["s1"].last_update = 0.0
    tower.stages["s1"].throughput = 0.0
    tower.stages["s1"].queue_depth = 0

    quotas = tower.allocate_resources()
    assert "s1" in [tower._get_stage_from_captain(cid) for cid in quotas]


def test_scale_up_on_bottleneck():
    tower = Tower(cluster_state=_cluster(cpu=16, mem_mb=32000, gpu=4), update_interval_sec=0)
    tower.register_stage("s1", initial_parallelism=1)

    tower.stages["s1"].queue_depth = 1000
    tower.stages["s1"].throughput = 5.0
    tower.stages["s1"].avg_latency_ms = 4500

    quotas = tower.allocate_resources()
    q = list(quotas.values())[0]
    assert q.target_parallelism >= 1


def test_scale_down_when_idle():
    tower = Tower(cluster_state=_cluster(cpu=8, mem_mb=16000, gpu=2), update_interval_sec=0)
    tower.register_stage("s1", initial_parallelism=4)

    tower.stages["s1"].queue_depth = 0
    tower.stages["s1"].throughput = 100.0
    tower.stages["s1"].avg_latency_ms = 1.0

    quotas = tower.allocate_resources()
    q = list(quotas.values())[0]
    assert q.target_parallelism <= 4


def test_sla_compliance_rate_initial():
    tower = Tower(cluster_state=_cluster())
    assert tower.get_sla_compliance_rate() == 100.0


def test_sla_violation_tracked():
    tower = Tower(cluster_state=_cluster(), sla_latency_ms=100)
    tower.register_stage("s1")

    metrics = StageMetrics(stage_name="s1", avg_latency_ms=200)
    tower.update_stage_metrics("s1", metrics)

    assert tower.sla_violations == 1
    assert tower.get_sla_compliance_rate() == 0.0


def test_topology_decision_distributed_on_multi_bottleneck():
    tower = Tower(cluster_state=_cluster())
    tower.register_stage("s1")

    metrics = StageMetrics(
        stage_name="s1",
        cpu_utilization=90,
        memory_utilization=90,
        gpu_utilization=10,
    )
    mode = tower._decide_topology("s1", metrics)
    assert mode == TopologyMode.DISTRIBUTED


def test_topology_decision_co_location_no_pressure():
    tower = Tower(cluster_state=_cluster())
    tower.register_stage("s1")

    metrics = StageMetrics(
        stage_name="s1",
        cpu_utilization=10,
        memory_utilization=10,
        gpu_utilization=10,
    )
    mode = tower._decide_topology("s1", metrics)
    assert mode == TopologyMode.CO_LOCATION


def test_global_stats():
    tower = Tower(cluster_state=_cluster())
    tower.register_stage("s1")
    tower.register_stage("s2")

    stats = tower.get_global_stats()
    assert stats["total_stages"] == 2
    assert stats["total_parallelism"] >= 2


def test_rate_limiting_prevents_thrashing():
    tower = Tower(cluster_state=_cluster(), update_interval_sec=9999)
    tower.register_stage("s1")

    q1 = tower.allocate_resources()
    tower.stages["s1"].queue_depth = 99999
    q2 = tower.allocate_resources()

    assert q1 is q2


def test_registration_conserves_initial_resources():
    tower = Tower(cluster_state=_cluster(cpu=12, mem_mb=24000, gpu=3))
    tower.register_stage("s1", initial_parallelism=1)
    tower.register_stage("s2", initial_parallelism=2)

    assert sum(q.cpu_quota for q in tower.quotas.values()) == pytest.approx(12)
    assert sum(q.memory_quota_mb for q in tower.quotas.values()) == pytest.approx(24000)
    assert sum(q.gpu_quota for q in tower.quotas.values()) == pytest.approx(3)


def test_update_stage_metrics_rejects_stale_sample():
    tower = Tower(cluster_state=_cluster())
    tower.register_stage("s1")
    current_timestamp = tower.stages["s1"].last_update
    stale = StageMetrics(
        stage_name="s1",
        queue_depth=999,
        last_update=current_timestamp - 1,
    )

    assert tower.update_stage_metrics("s1", stale) is False
    assert tower.stages["s1"].queue_depth == 0
    assert tower.total_requests == 0


def test_allocation_plan_is_immutable():
    tower = Tower(cluster_state=_cluster())
    tower.register_stage("s1")
    plan = tower.plan_allocation()

    with pytest.raises(FrozenInstanceError):
        plan.generation = 99
    with pytest.raises(FrozenInstanceError):
        plan.quotas[0].cpu_quota = 0


def test_shadow_rebalance_does_not_mutate_live_quota():
    class Controller:
        def __init__(self):
            self.calls = []

        def set_quota(self, quota):
            self.calls.append(quota)

    tower = Tower(
        cluster_state=_cluster(),
        mode=TowerMode.SHADOW,
        update_interval_sec=0,
    )
    captain_id = tower.register_stage("s1")
    controller = Controller()
    tower.register_controller(captain_id, controller)
    tower.stages["s1"].queue_depth = 1000
    tower.stages["s1"].throughput = 1
    original_parallelism = tower.quotas[captain_id].target_parallelism

    plan = tower.rebalance(force=True)

    assert plan.quotas[0].target_parallelism > original_parallelism
    assert tower.quotas[captain_id].target_parallelism == original_parallelism
    assert controller.calls == []


def test_closed_loop_rebalance_broadcasts_quota():
    class Controller:
        def __init__(self):
            self.calls = []

        def set_quota(self, quota):
            self.calls.append(quota)

    tower = Tower(
        cluster_state=_cluster(),
        mode=TowerMode.CLOSED_LOOP,
        update_interval_sec=0,
    )
    captain_id = tower.register_stage("s1")
    controller = Controller()
    tower.register_controller(captain_id, controller)
    tower.stages["s1"].queue_depth = 1000
    tower.stages["s1"].throughput = 1

    plan = tower.rebalance(force=True)

    assert controller.calls == [tower.quotas[captain_id]]
    assert tower.quotas[captain_id].target_parallelism == (plan.quotas[0].target_parallelism)


def test_save_plan_writes_complete_json(tmp_path):
    tower = Tower(cluster_state=_cluster())
    tower.register_stage("s1")
    plan = tower.rebalance(force=True)
    destination = tmp_path / "plans" / "latest.json"

    tower.save_plan(destination)

    assert json.loads(destination.read_text()) == plan.to_dict()
    assert list(destination.parent.glob("*.tmp")) == []
