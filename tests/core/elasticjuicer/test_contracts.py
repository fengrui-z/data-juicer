import pytest

from data_juicer.core.elasticjuicer.contracts import (
    BatchDecision,
    BatchObservation,
    ClusterState,
    MemoryState,
    ResourceQuota,
    StageMetrics,
)


def test_memory_state_selects_requested_device():
    state = MemoryState(
        timestamp=1.0,
        total_memory_mb=100,
        used_memory_mb=40,
        available_memory_mb=60,
        memory_percent=40,
        gpu_total_mb=20,
        gpu_used_mb=5,
        gpu_available_mb=15,
        gpu_percent=25,
    )

    assert state.get_available_memory() == 60
    assert state.get_available_memory(use_gpu=True) == 15


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BatchDecision(batch_size=0, reason="invalid"),
        lambda: BatchObservation(
            stage_name="op",
            batch_size=1,
            latency_ms=-1,
            throughput=1,
            memory_peak_mb=1,
        ),
        lambda: StageMetrics(stage_name="op", current_parallelism=0),
        lambda: ResourceQuota(
            captain_id="captain",
            target_parallelism=1,
            cpu_quota=-1,
            memory_quota_mb=1,
        ),
        lambda: ClusterState(
            total_cpu_cores=4,
            total_memory_mb=100,
            total_gpu_count=0,
            available_cpu_cores=5,
            available_memory_mb=50,
            available_gpus=0,
        ),
    ],
)
def test_invalid_contracts_fail_fast(factory):
    with pytest.raises(ValueError):
        factory()
