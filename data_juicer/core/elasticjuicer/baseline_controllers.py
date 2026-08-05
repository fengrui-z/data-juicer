"""Baseline batch-size policies for drift/OOM experiments.

Classic adaptive-batching controllers used as comparison arms against the
ElasticJuicer success-driven controller (AdaptiveBatchController). All
policies implement the SAME interface consumed by OOMSafeAdaptiveMapper
(next_batch_size / observe_success / observe_oom / state / seed_bounds /
set_hard_limit), so they run inside the identical lossless slice-retry
harness: an OOM retries the same slice smaller, offset only advances on
success. The only variable isolated is the batch-size decision policy.

Policies:
- fixed:            constant size; OOM propagates (static config arms)
- halving:          OOM halves; x2 growth after N consecutive successes
- bisect:           binary search between proven-safe and failing sizes
- proportional:     batch scaled by measured VRAM headroom toward a target
                    utilization (memory-metric-driven, the classic contrast
                    to EJ's success-driven policy)
"""

from dataclasses import dataclass
from typing import Callable, Optional

from .batch_controller import MinimumBatchSizeOOM


@dataclass
class BaselineControllerState:
    """Diagnostic snapshot with the fields the harness observability reads."""

    current_batch_size: int
    min_batch_size: int
    max_batch_size: int
    hard_limit: int
    success_lower_bound: Optional[int]
    oom_upper_bound: Optional[int]
    consecutive_successes: int = 0
    cooldown_remaining: int = 0
    success_events: int = 0
    oom_events: int = 0
    probe_oom_events: int = 0
    oom_reprobe_events: int = 0
    successes_since_oom: int = 0
    capacity_recovery_hint_pending: bool = False
    capacity_recovery_hint_expires_at_ms: Optional[int] = None
    capacity_recovery_hints: int = 0


class BaselineBatchController:
    """Common scaffolding and harness-facing surface for baseline policies."""

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
        self.hard_limit = max_batch_size
        self.success_lower_bound: Optional[int] = None
        self.oom_upper_bound: Optional[int] = None
        self.consecutive_successes = 0
        self.success_events = 0
        self.oom_events = 0

    # -- harness surface ------------------------------------------------------

    @property
    def state(self) -> BaselineControllerState:
        return BaselineControllerState(
            current_batch_size=self.current_batch_size,
            min_batch_size=self.min_batch_size,
            max_batch_size=self.max_batch_size,
            hard_limit=self.hard_limit,
            success_lower_bound=self.success_lower_bound,
            oom_upper_bound=self.oom_upper_bound,
            consecutive_successes=self.consecutive_successes,
            success_events=self.success_events,
            oom_events=self.oom_events,
        )

    def next_batch_size(self, remaining_samples: int) -> int:
        if remaining_samples <= 0:
            return 0
        return min(self.current_batch_size, self._effective_maximum(), remaining_samples)

    def set_hard_limit(self, hard_limit: int) -> int:
        if not self.min_batch_size <= hard_limit <= self.max_batch_size:
            raise ValueError("hard_limit must be within static batch-size bounds")
        self.hard_limit = hard_limit
        self.current_batch_size = self._clamp(self.current_batch_size)
        return self.current_batch_size

    def seed_bounds(
        self,
        safe_batch_size: Optional[int] = None,
        oom_upper_bound: Optional[int] = None,
    ) -> int:
        """Adopt prior learning as an initial prior (same semantics as EJ)."""
        if oom_upper_bound is not None and oom_upper_bound > self.min_batch_size:
            self.oom_upper_bound = (
                oom_upper_bound
                if self.oom_upper_bound is None
                else min(self.oom_upper_bound, oom_upper_bound)
            )
        if safe_batch_size is not None and safe_batch_size >= self.min_batch_size:
            if self.success_lower_bound is None:
                self.success_lower_bound = safe_batch_size
            else:
                self.success_lower_bound = max(self.success_lower_bound, safe_batch_size)
        self.current_batch_size = self._clamp(self._initial_from_seed(safe_batch_size))
        return self.current_batch_size

    def _initial_from_seed(self, safe_batch_size: Optional[int]) -> int:
        if safe_batch_size is not None and safe_batch_size >= self.min_batch_size:
            return safe_batch_size
        return self.current_batch_size

    def record_capacity_recovery_hint(self, expires_at_ms: int, now_ms: Optional[int] = None) -> bool:
        # Baselines do not consume Captain hints; keep the interface inert.
        return False

    def reset_oom_bound(self) -> int:
        self.oom_upper_bound = None
        return self.current_batch_size

    # -- shared helpers -------------------------------------------------------

    def _effective_maximum(self) -> int:
        return min(self.hard_limit, self.max_batch_size)

    def _clamp(self, size: int) -> int:
        upper = self._effective_maximum()
        if self.oom_upper_bound is not None:
            upper = min(upper, max(self.min_batch_size, self.oom_upper_bound - 1))
        return max(self.min_batch_size, min(size, upper))

    def _record_success(self, batch_size: int) -> None:
        self.success_events += 1
        self.consecutive_successes += 1
        if self.success_lower_bound is None:
            self.success_lower_bound = batch_size
        else:
            self.success_lower_bound = max(self.success_lower_bound, batch_size)

    def _record_oom(self, batch_size: int) -> None:
        self.oom_events += 1
        self.consecutive_successes = 0
        self.oom_upper_bound = (
            batch_size if self.oom_upper_bound is None else min(self.oom_upper_bound, batch_size)
        )
        if self.oom_upper_bound <= self.min_batch_size:
            raise MinimumBatchSizeOOM(
                f"OOM at minimum batch size {self.min_batch_size}"
            )


class FixedBatchController(BaselineBatchController):
    """Static configuration arm: constant size, OOM propagates as misconfig."""

    def observe_success(self, batch_size: int) -> int:
        self._record_success(batch_size)
        return self.current_batch_size

    def observe_oom(self, batch_size: int) -> int:
        self._record_oom(batch_size)
        # Fixed configs have no fallback: surface the failure like a static
        # config that picked an infeasible batch.
        raise MinimumBatchSizeOOM(f"fixed batch {self.current_batch_size} hit OOM")


class HalvingController(BaselineBatchController):
    """Classic reactive halving: OOM halves; growth resumes after a streak."""

    def __init__(self, *args, grow_after_successes: int = 3, growth_factor: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        if grow_after_successes < 1:
            raise ValueError("grow_after_successes must be at least 1")
        if growth_factor <= 1:
            raise ValueError("growth_factor must be > 1")
        self.grow_after_successes = grow_after_successes
        self.growth_factor = growth_factor

    def observe_success(self, batch_size: int) -> int:
        self._record_success(batch_size)
        if self.consecutive_successes >= self.grow_after_successes:
            grown = int(batch_size * self.growth_factor)
            self.current_batch_size = self._clamp(max(grown, batch_size + 1))
            self.consecutive_successes = 0
        else:
            self.current_batch_size = self._clamp(batch_size)
        return self.current_batch_size

    def observe_oom(self, batch_size: int) -> int:
        self._record_oom(batch_size)
        self.current_batch_size = self._clamp(max(self.min_batch_size, batch_size // 2))
        return self.current_batch_size


class BisectController(BaselineBatchController):
    """Binary search between the proven-safe lower and failing upper size."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Start aggressive: the first OOM establishes the upper bound.
        self._safe_lower = self.min_batch_size - 1  # nothing proven yet
        self._fail_upper = self._effective_maximum() + 1

    def observe_success(self, batch_size: int) -> int:
        self._record_success(batch_size)
        self._safe_lower = max(self._safe_lower, batch_size)
        self.current_batch_size = self._clamp(self._midpoint())
        return self.current_batch_size

    def observe_oom(self, batch_size: int) -> int:
        self._record_oom(batch_size)
        self._fail_upper = min(self._fail_upper, batch_size)
        self.current_batch_size = self._clamp(self._midpoint())
        return self.current_batch_size

    def _midpoint(self) -> int:
        mid = (max(self._safe_lower, self.min_batch_size - 1) + self._fail_upper) // 2
        return max(self.min_batch_size, mid)


class MemoryProportionalController(BaselineBatchController):
    """Scale batch by measured VRAM headroom toward a target utilization.

    The classic memory-metric-driven policy: after each successful slice,
    batch_new = batch * (target_bytes / peak_bytes), with bounded growth.
    ``peak_memory_probe`` must return peak bytes since the last call and
    reset the accounting (actor passes a torch-based probe).
    """

    def __init__(
        self,
        *args,
        peak_memory_probe: Optional[Callable[[], int]] = None,
        target_fraction: float = 0.85,
        total_bytes: Optional[int] = None,
        max_growth_factor: float = 1.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0 < target_fraction <= 1:
            raise ValueError("target_fraction must be in (0, 1]")
        if max_growth_factor <= 1:
            raise ValueError("max_growth_factor must be > 1")
        self.peak_memory_probe = peak_memory_probe
        self.target_fraction = target_fraction
        self.total_bytes = total_bytes
        self.max_growth_factor = max_growth_factor

    def observe_success(self, batch_size: int) -> int:
        self._record_success(batch_size)
        peak_bytes = self.peak_memory_probe() if self.peak_memory_probe is not None else 0
        if peak_bytes > 0 and self.total_bytes:
            target_bytes = self.total_bytes * self.target_fraction
            scaled = batch_size * (target_bytes / float(peak_bytes))
            scaled = min(scaled, batch_size * self.max_growth_factor)
            self.current_batch_size = self._clamp(max(int(scaled), self.min_batch_size))
        else:
            # No memory signal: fall back to cautious additive growth.
            self.current_batch_size = self._clamp(batch_size + 1)
        return self.current_batch_size

    def observe_oom(self, batch_size: int) -> int:
        self._record_oom(batch_size)
        self.current_batch_size = self._clamp(max(self.min_batch_size, batch_size // 2))
        return self.current_batch_size


def make_baseline_controller(
    policy: str,
    initial_batch_size: int,
    min_batch_size: int = 1,
    max_batch_size: Optional[int] = None,
    peak_memory_probe: Optional[Callable[[], int]] = None,
    total_bytes: Optional[int] = None,
) -> BaselineBatchController:
    """Construct a baseline policy controller by name."""
    max_batch_size = initial_batch_size if max_batch_size is None else max_batch_size
    common = dict(
        initial_batch_size=initial_batch_size,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )
    if policy == "fixed":
        return FixedBatchController(**common)
    if policy == "halving":
        return HalvingController(**common)
    if policy == "bisect":
        return BisectController(**common)
    if policy == "proportional":
        if total_bytes is None:
            raise ValueError("proportional policy requires total_bytes")
        return MemoryProportionalController(
            peak_memory_probe=peak_memory_probe, total_bytes=total_bytes, **common
        )
    raise ValueError(f"unknown baseline controller policy: {policy!r}")
