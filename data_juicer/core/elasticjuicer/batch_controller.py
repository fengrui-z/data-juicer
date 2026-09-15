"""Bounded actor-local batch-size control for lossless OOM retries."""

from dataclasses import dataclass
from typing import Optional


class MinimumBatchSizeOOM(RuntimeError):
    """Raised when even the configured minimum batch size cannot execute."""


@dataclass(frozen=True)
class BatchControllerState:
    """Immutable diagnostic snapshot of the retry controller state."""

    current_batch_size: int
    min_batch_size: int
    max_batch_size: int
    success_lower_bound: Optional[int]
    oom_upper_bound: Optional[int]
    success_events: int
    oom_events: int


class AdaptiveBatchController:
    """Reduce only after OOM; successful slices never increase batch size.

    The Stage 1 controller intentionally has no capacity-recovery or automatic
    growth policy. Its local state only records a successful floor and failed
    ceiling so retries preserve the failed input slice while using a smaller
    microbatch.
    """

    def __init__(
        self,
        initial_batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: int = 1000,
    ):
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be at least 1")
        if max_batch_size < min_batch_size:
            raise ValueError("max_batch_size must be >= min_batch_size")
        if not min_batch_size <= initial_batch_size <= max_batch_size:
            raise ValueError("initial_batch_size must be within configured bounds")

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.success_lower_bound: Optional[int] = None
        self.oom_upper_bound: Optional[int] = None
        self.success_events = 0
        self.oom_events = 0

    @property
    def state(self) -> BatchControllerState:
        return BatchControllerState(
            current_batch_size=self.current_batch_size,
            min_batch_size=self.min_batch_size,
            max_batch_size=self.max_batch_size,
            success_lower_bound=self.success_lower_bound,
            oom_upper_bound=self.oom_upper_bound,
            success_events=self.success_events,
            oom_events=self.oom_events,
        )

    def next_batch_size(self, remaining_samples: int) -> int:
        """Return the current bounded size; a final slice may be smaller."""
        if remaining_samples < 0:
            raise ValueError("remaining_samples must be non-negative")
        if remaining_samples == 0:
            return 0
        return min(self.current_batch_size, remaining_samples)

    def observe_success(self, batch_size: int) -> int:
        """Record successful work without expanding the next microbatch."""
        self._validate_observation(batch_size)
        self.success_events += 1
        if batch_size >= self.current_batch_size:
            if self.success_lower_bound is None:
                self.success_lower_bound = batch_size
            else:
                self.success_lower_bound = max(self.success_lower_bound, batch_size)
        return self.current_batch_size

    def observe_oom(self, batch_size: int) -> int:
        """Make a failing size an exclusive ceiling and reduce the next slice."""
        self._validate_observation(batch_size)
        self.oom_events += 1
        self.oom_upper_bound = batch_size if self.oom_upper_bound is None else min(self.oom_upper_bound, batch_size)
        if batch_size <= self.min_batch_size:
            self.current_batch_size = self.min_batch_size
            raise MinimumBatchSizeOOM(f"OOM at minimum batch size {self.min_batch_size}; no smaller retry is allowed")

        reduced = max(self.min_batch_size, batch_size // 2)
        if self.success_lower_bound is not None:
            reduced = max(reduced, self.success_lower_bound)
        self.current_batch_size = min(reduced, self.oom_upper_bound - 1)
        return self.current_batch_size

    def _validate_observation(self, batch_size: int) -> None:
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError("observed batch_size must be within static batch-size bounds")
