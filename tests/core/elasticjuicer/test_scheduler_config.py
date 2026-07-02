import pytest

from data_juicer.core.elasticjuicer.scheduler.scheduler_config import SchedulerConfig


def test_config_presets_are_valid_and_ordered_by_risk():
    conservative = SchedulerConfig.conservative()
    default = SchedulerConfig()
    aggressive = SchedulerConfig.aggressive()

    assert conservative.target_memory_utilization < default.target_memory_utilization
    assert default.target_memory_utilization < aggressive.target_memory_utilization
    assert conservative.safety_buffer_mb > aggressive.safety_buffer_mb


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_batch_size": 0},
        {"initial_batch_size": 0},
        {"initial_batch_size": 11, "max_batch_size": 10},
        {"target_memory_utilization": 1.0},
        {"oom_backoff_ratio": 0.0},
        {"predictor_min_samples": 101, "predictor_window_size": 100},
        {"pid_kp": -1},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SchedulerConfig(**kwargs)
