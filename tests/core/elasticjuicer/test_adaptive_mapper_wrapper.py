"""Stage 1 lossless OOM retry tests."""

from contextlib import contextmanager

import pytest

from data_juicer.core.elasticjuicer.adaptive_mapper import (
    AdaptiveBatchContractError,
    OOMSafeAdaptiveMapper,
)
from data_juicer.core.elasticjuicer.batch_controller import AdaptiveBatchController


class FakeCudaOutOfMemoryError(RuntimeError):
    pass


class RecordingSampler:
    def __init__(self):
        self.batch_sizes = []

    @contextmanager
    def measure(self, batch_size):
        self.batch_sizes.append(batch_size)
        yield


def test_failed_mutation_is_not_replayed_into_a_smaller_retry():
    batch = {"id": [0, 1, 2, 3], "stats": [{"count": 0} for _ in range(4)]}

    def mapper(part):
        for stats in part["stats"]:
            stats["count"] += 1
        if len(part["id"]) > 2:
            raise FakeCudaOutOfMemoryError("CUDA out of memory")
        return part

    result = OOMSafeAdaptiveMapper(mapper, AdaptiveBatchController(4, max_batch_size=4))(batch)

    assert result["id"] == list(range(4))
    assert result["stats"] == [{"count": 1}] * 4
    assert batch["stats"] == [{"count": 0}] * 4


def test_same_slice_retries_smaller_batches_without_loss_or_duplicates():
    calls = []
    sampler = RecordingSampler()

    def mapper(batch):
        calls.append(list(batch))
        if len(batch) > 2:
            raise FakeCudaOutOfMemoryError("CUDA out of memory")
        return [value * 2 for value in batch]

    wrapper = OOMSafeAdaptiveMapper(
        mapper,
        AdaptiveBatchController(8, max_batch_size=8),
        sampler=sampler,
    )
    result = wrapper(list(range(8)))

    assert result == [value * 2 for value in range(8)]
    assert calls[:3] == [list(range(8)), list(range(4)), [0, 1]]
    assert sampler.batch_sizes[:3] == [8, 4, 2]
    assert wrapper.controller.next_batch_size(100) == 2


def test_persistent_minimum_batch_oom_propagates_after_bounded_waits():
    calls = []

    def mapper(batch):
        calls.append(list(batch))
        raise FakeCudaOutOfMemoryError("CUDA out of memory")

    wrapper = OOMSafeAdaptiveMapper(
        mapper,
        AdaptiveBatchController(1, max_batch_size=1),
        max_floor_retries=2,
        floor_retry_backoff_sec=0,
    )

    with pytest.raises(FakeCudaOutOfMemoryError, match="CUDA out of memory"):
        wrapper([1])
    assert calls == [[1], [1], [1]]


def test_invalid_output_rows_is_a_contract_error():
    wrapper = OOMSafeAdaptiveMapper(
        lambda batch: batch[:-1],
        AdaptiveBatchController(2, max_batch_size=2),
    )

    with pytest.raises(AdaptiveBatchContractError, match="returned 1 rows"):
        wrapper([1, 2])
