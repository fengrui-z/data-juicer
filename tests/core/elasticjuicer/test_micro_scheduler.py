from data_juicer.core.elasticjuicer.contracts import MemoryState
from data_juicer.core.elasticjuicer.scheduler.micro_scheduler import (
    BatchSizeController,
    PIDController,
)


class FakeClock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        self.now += 1.0
        return self.now


def make_memory_state(available_mb=6000):
    return MemoryState(
        timestamp=1.0,
        total_memory_mb=10000,
        used_memory_mb=10000 - available_mb,
        available_memory_mb=available_mb,
        memory_percent=(10000 - available_mb) / 100,
    )


def test_pid_uses_injected_clock_and_respects_limits():
    controller = PIDController(
        kp=1,
        ki=0,
        kd=0,
        setpoint=10,
        output_limits=(1, 5),
        clock=FakeClock(),
    )

    assert controller.update(current_value=0) == 5
    assert controller.update(current_value=20) == 1


def test_batch_controller_uses_injected_memory_provider():
    calls = []

    def provider():
        calls.append(True)
        return make_memory_state()

    controller = BatchSizeController(
        initial_batch_size=10,
        min_batch_size=1,
        max_batch_size=100,
        safety_buffer_mb=1000,
        memory_state_provider=provider,
        clock=FakeClock(),
    )

    next_batch_size = controller.update_batch_size()

    assert calls == [True]
    assert 1 <= next_batch_size <= 100
    assert controller.batch_size_history[-1]["timestamp"] == 11.0
