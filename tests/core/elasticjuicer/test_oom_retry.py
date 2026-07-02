import pytest
from unittest.mock import MagicMock

from data_juicer.core.elasticjuicer.scheduler.captain import Captain, CaptainConfig
from data_juicer.core.elasticjuicer.contracts import MemoryState


class FakeClock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        self.now += 1.0
        return self.now


def _make_captain():
    config = CaptainConfig(
        stage_name="test_op",
        initial_batch_size=4,
        enable_micro_scheduler=True,
        enable_prediction=False,
    )
    captain = Captain(config)
    return captain


def test_oom_requeues_samples_in_order():
    captain = _make_captain()
    samples = [1, 2, 3, 4, 5]
    captain.enqueue_samples(samples)

    def failing_op(batch):
        raise MemoryError("OOM")

    with pytest.raises(MemoryError):
        captain.process_batch(failing_op)

    queue_contents = list(captain.queue)
    assert queue_contents == [1, 2, 3, 4, 5]


def test_oom_no_data_loss_after_multiple_retries():
    captain = _make_captain()
    samples = list(range(10))
    captain.enqueue_samples(samples)

    call_count = 0
    processed_samples = []

    def flaky_op(batch):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise MemoryError("OOM")
        processed_samples.extend(batch)
        return batch

    with pytest.raises(MemoryError):
        captain.process_batch(flaky_op)
    with pytest.raises(MemoryError):
        captain.process_batch(flaky_op)

    result = captain.process_batch(flaky_op)
    assert result is not None
    assert captain.samples_processed > 0

    remaining = set(captain.queue)
    done = set(processed_samples)
    assert remaining | done == set(range(10))
    assert remaining & done == set()


def test_oom_does_not_duplicate_samples():
    captain = _make_captain()
    samples = [10, 20, 30]
    captain.enqueue_samples(samples)

    def failing_op(batch):
        raise MemoryError("OOM")

    with pytest.raises(MemoryError):
        captain.process_batch(failing_op)

    all_items = list(captain.queue)
    assert len(all_items) == len(set(all_items))
    assert sorted(all_items) == [10, 20, 30]


def test_oom_preserves_order_after_requeue():
    captain = _make_captain()
    samples = ["a", "b", "c", "d"]
    captain.enqueue_samples(samples)

    def failing_op(batch):
        raise MemoryError("OOM")

    for _ in range(3):
        with pytest.raises(MemoryError):
            captain.process_batch(failing_op)

    assert list(captain.queue) == ["a", "b", "c", "d"]


def test_runtime_error_also_triggers_requeue():
    captain = _make_captain()
    samples = [1, 2, 3]
    captain.enqueue_samples(samples)

    def cuda_oom(batch):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        captain.process_batch(cuda_oom)

    assert list(captain.queue) == [1, 2, 3]


def test_successful_batch_does_not_requeue():
    captain = _make_captain()
    samples = [1, 2, 3, 4]
    captain.enqueue_samples(samples)

    def ok_op(batch):
        return [x * 2 for x in batch]

    result = captain.process_batch(ok_op)
    assert result is not None
    assert len(captain.queue) < 4


def test_oom_increments_oom_counter():
    captain = _make_captain()
    captain.enqueue_samples([1, 2])

    def failing_op(batch):
        raise MemoryError("OOM")

    for _ in range(3):
        with pytest.raises(MemoryError):
            captain.process_batch(failing_op)

    assert captain.oom_events == 3
    assert captain.metrics.oom_count == 3
