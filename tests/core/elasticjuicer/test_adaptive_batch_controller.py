"""Stage 1 backoff controller tests."""

import pytest

from data_juicer.core.elasticjuicer.batch_controller import (
    AdaptiveBatchController,
    MinimumBatchSizeOOM,
)


def test_controller_validates_static_bounds():
    with pytest.raises(ValueError, match="min_batch_size"):
        AdaptiveBatchController(initial_batch_size=1, min_batch_size=0)
    with pytest.raises(ValueError, match="max_batch_size"):
        AdaptiveBatchController(initial_batch_size=4, min_batch_size=8, max_batch_size=4)
    with pytest.raises(ValueError, match="initial_batch_size"):
        AdaptiveBatchController(initial_batch_size=16, min_batch_size=1, max_batch_size=8)


def test_oom_reduces_next_slice_and_success_never_grows_it():
    controller = AdaptiveBatchController(initial_batch_size=16, min_batch_size=1, max_batch_size=16)

    assert controller.observe_oom(16) == 8
    controller.observe_success(8)
    controller.observe_success(8)

    assert controller.next_batch_size(100) == 8
    assert controller.state.oom_upper_bound == 16
    assert controller.state.success_lower_bound == 8


def test_backoff_keeps_the_largest_proven_safe_size():
    controller = AdaptiveBatchController(initial_batch_size=16, min_batch_size=1, max_batch_size=16)

    controller.observe_oom(16)
    controller.observe_success(8)
    assert controller.observe_oom(12) == 8
    assert controller.state.oom_upper_bound == 12
    assert controller.next_batch_size(100) == 8


def test_oom_at_minimum_batch_size_is_terminal():
    controller = AdaptiveBatchController(initial_batch_size=1, min_batch_size=1, max_batch_size=8)

    with pytest.raises(MinimumBatchSizeOOM, match="minimum batch size 1"):
        controller.observe_oom(1)
