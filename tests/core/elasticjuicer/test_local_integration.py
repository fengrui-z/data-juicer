import pytest

from data_juicer.core.elasticjuicer.contracts import (
    ClusterState,
    MemoryState,
    StageMetrics,
)
from data_juicer.core.elasticjuicer.scheduler.captain import Captain, CaptainConfig
from data_juicer.core.elasticjuicer.scheduler.tower import Tower
from data_juicer.core.elasticjuicer.profiler.resource_monitor import ResourceMonitor


class FakeClock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        self.now += 1.0
        return self.now


def _mem_state(available_mb=8000):
    return MemoryState(
        timestamp=0.0,
        total_memory_mb=16000,
        used_memory_mb=16000 - available_mb,
        available_memory_mb=available_mb,
        memory_percent=(16000 - available_mb) / 160,
    )


def test_captain_processes_all_samples():
    config = CaptainConfig(
        stage_name="integration_op",
        initial_batch_size=4,
        enable_micro_scheduler=False,
        enable_prediction=False,
    )
    captain = Captain(config)
    samples = list(range(20))
    captain.enqueue_samples(samples)

    results = []

    def op(batch):
        results.extend(batch)
        return batch

    while captain.queue:
        captain.process_batch(op)

    assert sorted(results) == list(range(20))


def test_tower_captain_loop():
    cluster = ClusterState(
        total_cpu_cores=4,
        total_memory_mb=8000,
        total_gpu_count=0,
        available_cpu_cores=4,
        available_memory_mb=8000,
        available_gpus=0,
    )
    tower = Tower(cluster_state=cluster, update_interval_sec=0)
    reported_metrics = []

    def callback(metrics):
        reported_metrics.append(metrics)

    cid = tower.register_stage("op_a", initial_parallelism=1)
    config = CaptainConfig(
        stage_name="op_a",
        initial_batch_size=2,
        report_interval_sec=0,
        enable_micro_scheduler=False,
        enable_prediction=False,
    )
    captain = Captain(config, tower_callback=callback)

    captain.enqueue_samples(list(range(10)))

    def op(batch):
        return batch

    while captain.queue:
        captain.process_batch(op)

    assert captain.samples_processed == 10
    assert len(reported_metrics) > 0


def test_end_to_end_no_data_loss():
    config = CaptainConfig(
        stage_name="e2e_op",
        initial_batch_size=3,
        enable_micro_scheduler=True,
        enable_prediction=False,
    )

    mem_provider = lambda: _mem_state(8000)
    captain = Captain(config)
    captain.micro_scheduler.controller._memory_state_provider = mem_provider

    samples = list(range(50))
    captain.enqueue_samples(samples)

    processed = []

    def op(batch):
        processed.extend(batch)
        return batch

    while captain.queue:
        captain.process_batch(op)

    assert sorted(processed) == list(range(50))


def test_monitor_and_captain_consistent():
    monitor = ResourceMonitor(enabled=True)
    config = CaptainConfig(
        stage_name="mon_op",
        initial_batch_size=5,
        enable_micro_scheduler=False,
        enable_prediction=False,
    )
    captain = Captain(config)
    captain.monitor = monitor

    captain.enqueue_samples(list(range(15)))

    def op(batch):
        return batch

    while captain.queue:
        captain.process_batch(op)

    stats = monitor.get_stats("mon_op")
    assert stats is not None
    assert stats.total_samples == captain.samples_processed
