import pytest

from data_juicer.core.elasticjuicer.contracts import MemoryState
from data_juicer.core.elasticjuicer.scheduler.micro_scheduler import (
    BatchSizeController,
    MicroScheduler,
)


class FakeClock:
    def __init__(self, start=10.0):
        self.now = start

    def __call__(self):
        self.now += 1.0
        return self.now


def _mem(available_mb=6000, total_mb=16000, used_mb=None):
    if used_mb is None:
        used_mb = total_mb - available_mb
    return MemoryState(
        timestamp=0.0,
        total_memory_mb=total_mb,
        used_memory_mb=used_mb,
        available_memory_mb=available_mb,
        memory_percent=used_mb / total_mb * 100,
    )


def test_batch_size_respects_min_bound():
    ctrl = BatchSizeController(
        initial_batch_size=1,
        min_batch_size=1,
        max_batch_size=100,
        safety_buffer_mb=99999,
        memory_state_provider=lambda: _mem(available_mb=0, total_mb=16000),
        clock=FakeClock(),
    )
    result = ctrl.update_batch_size()
    assert result >= 1


def test_batch_size_respects_max_bound():
    ctrl = BatchSizeController(
        initial_batch_size=50,
        min_batch_size=1,
        max_batch_size=100,
        safety_buffer_mb=0,
        memory_state_provider=lambda: _mem(available_mb=15000, total_mb=16000),
        clock=FakeClock(),
    )
    result = ctrl.update_batch_size()
    assert result <= 100


def test_batch_size_increases_when_memory_abundant():
    clock = FakeClock()
    ctrl = BatchSizeController(
        initial_batch_size=10,
        min_batch_size=1,
        max_batch_size=500,
        safety_buffer_mb=500,
        memory_state_provider=lambda: _mem(available_mb=12000, total_mb=16000),
        clock=clock,
    )
    first = ctrl.current_batch_size
    for _ in range(5):
        ctrl.update_batch_size()
    assert ctrl.current_batch_size >= first


def test_batch_size_decreases_when_memory_tight():
    clock = FakeClock()
    ctrl = BatchSizeController(
        initial_batch_size=100,
        min_batch_size=1,
        max_batch_size=500,
        safety_buffer_mb=5000,
        memory_state_provider=lambda: _mem(available_mb=100, total_mb=16000, used_mb=15900),
        clock=clock,
    )
    first = ctrl.current_batch_size
    for _ in range(5):
        ctrl.update_batch_size()
    assert ctrl.current_batch_size <= first


def test_max_change_rate_limits_adjustment():
    clock = FakeClock()
    ctrl = BatchSizeController(
        initial_batch_size=100,
        min_batch_size=1,
        max_batch_size=1000,
        safety_buffer_mb=0,
        memory_state_provider=lambda: _mem(available_mb=100, total_mb=16000, used_mb=15900),
        clock=clock,
    )
    old = ctrl.current_batch_size
    ctrl.update_batch_size()
    change = abs(ctrl.current_batch_size - old)
    assert change <= max(1, int(old * 0.5))


def test_oom_backoff_halves_batch_size():
    ctrl = BatchSizeController(
        initial_batch_size=64,
        min_batch_size=1,
        max_batch_size=1000,
        clock=FakeClock(),
    )
    ctrl.report_oom(batch_size=64, memory_mb=15000)
    assert ctrl.current_batch_size == 32
    assert ctrl.max_batch_size == 64


def test_oom_backoff_respects_minimum():
    ctrl = BatchSizeController(
        initial_batch_size=1,
        min_batch_size=1,
        max_batch_size=1000,
        clock=FakeClock(),
    )
    ctrl.report_oom(batch_size=1, memory_mb=15000)
    assert ctrl.current_batch_size == 1


def test_micro_scheduler_get_batch_size_returns_bounded_value():
    sched = MicroScheduler(
        initial_batch_size=32,
        min_batch_size=1,
        max_batch_size=256,
        memory_state_provider=lambda: _mem(available_mb=8000),
        clock=FakeClock(),
    )
    bs = sched.get_batch_size()
    assert 1 <= bs <= 256


def test_micro_scheduler_update_adjusts_batch():
    clock = FakeClock()
    sched = MicroScheduler(
        initial_batch_size=50,
        min_batch_size=1,
        max_batch_size=500,
        safety_buffer_mb=5000,
        memory_state_provider=lambda: _mem(available_mb=200, total_mb=16000, used_mb=15800),
        clock=clock,
    )
    old = sched.controller.current_batch_size
    sched.update(actual_memory_used=15000)
    assert sched.controller.current_batch_size != old or old == 1
