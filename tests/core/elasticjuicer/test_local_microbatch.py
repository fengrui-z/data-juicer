from types import SimpleNamespace

import pytest

from data_juicer.core.elasticjuicer.runtime.local_microbatch import (
    AdaptiveMicrobatchExecutor,
)


class DeterministicScheduler:
    def __init__(self, batch_size=4, min_batch_size=1):
        self.controller = SimpleNamespace(
            current_batch_size=batch_size,
            min_batch_size=min_batch_size,
        )
        self.oom_batches = []
        self.updates = 0

    def report_oom(self, batch_size, memory_mb):
        self.oom_batches.append(batch_size)
        self.controller.current_batch_size = max(
            self.controller.min_batch_size,
            batch_size // 2,
        )

    def update(self, actual_memory_used):
        self.updates += 1


def test_list_batches_preserve_order():
    scheduler = DeterministicScheduler(batch_size=3)
    executor = AdaptiveMicrobatchExecutor(
        lambda values: [value * 2 for value in values],
        scheduler,
    )

    result = executor(list(range(8)))

    assert result == [value * 2 for value in range(8)]
    assert executor.successful_slices == 3


def test_mapping_batches_are_split_and_merged():
    scheduler = DeterministicScheduler(batch_size=2)

    def process(batch):
        return {
            "text": [value.upper() for value in batch["text"]],
            "score": [value + 1 for value in batch["score"]],
        }

    executor = AdaptiveMicrobatchExecutor(process, scheduler)
    result = executor({"text": ["a", "b", "c"], "score": [1, 2, 3]})

    assert result == {"text": ["A", "B", "C"], "score": [2, 3, 4]}


def test_oom_retries_same_slice_without_loss_or_duplication():
    scheduler = DeterministicScheduler(batch_size=8)
    processed = []

    def process(values):
        if len(values) > 2:
            raise RuntimeError("CUDA out of memory")
        processed.extend(values)
        return list(values)

    executor = AdaptiveMicrobatchExecutor(process, scheduler)
    result = executor(list(range(9)))

    assert result == list(range(9))
    assert processed == list(range(9))
    assert scheduler.oom_batches == [8, 4]
    assert executor.oom_retries == 2


def test_non_oom_runtime_error_is_propagated():
    scheduler = DeterministicScheduler(batch_size=4)

    def broken(_values):
        raise RuntimeError("bad schema")

    executor = AdaptiveMicrobatchExecutor(broken, scheduler)

    with pytest.raises(RuntimeError, match="bad schema"):
        executor([1, 2, 3])

    assert scheduler.oom_batches == []


def test_oom_at_minimum_batch_is_propagated():
    scheduler = DeterministicScheduler(batch_size=1)

    def out_of_memory(_values):
        raise MemoryError("OOM")

    executor = AdaptiveMicrobatchExecutor(out_of_memory, scheduler)

    with pytest.raises(MemoryError):
        executor([1])


def test_empty_batch_is_delegated_once():
    scheduler = DeterministicScheduler(batch_size=4)
    calls = []

    def process(values):
        calls.append(values)
        return values

    executor = AdaptiveMicrobatchExecutor(process, scheduler)

    assert executor([]) == []
    assert calls == [[]]
