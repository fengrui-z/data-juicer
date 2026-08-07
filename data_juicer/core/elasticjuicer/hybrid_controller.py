"""Hybrid batch-size controller v3: model-driven shrink, success-driven grow.

Synthesis of the directional findings from the phase-3 experiments:
- Rising cost (drift into bigger inputs): the cost model is the best SHRINK
  authority -- it predicts the safe batch before the OOM wall (phase 3b:
  0 probes, fastest adaptive arm on small->big).
- Falling cost (drift into cheaper inputs): success-driven streak growth is
  the best GROW authority -- model-based expansion is damped by fit decay
  and growth caps and trails behind (phase 3b: costaware slowest on
  big->small).

Policy:
- observe_batch_cost: same trend-extrapolated cost observation as the
  cost-aware controller.
- observe_success: keep fitting the memory model. If the model predicts a
  SMALLER batch than the current one (cost rising), the model decides --
  immediate pre-shrink, no OOM needed. Otherwise growth is success-driven
  (streaks expand multiplicatively) but CEILED by the model prediction when
  the model is reliable: streaks set the pace, the model sets the ceiling.
  Without the ceiling, streak growth overshoots the boundary every cycle
  (observed: OOM ping-pong on rising drift).
- observe_oom: model prediction (with safety margin) or halving fallback,
  same as the cost-aware controller; bounds are updated so growth never
  re-probes a known failing size.
- Bound validity is cost-scoped: an OOM upper bound learned at one cost
  level is stale once cost falls materially below it (the wall moves), so
  it is released when the current cost drops under 90% of the cost where
  the OOM happened. Without this, one cold-start OOM pins the batch at the
  minimum forever on falling drift.
"""

from typing import Optional

from .cost_aware_controller import CostAwareController


class HybridController(CostAwareController):
    """Model-driven shrink + success-driven grow."""

    def __init__(
        self,
        *args,
        successes_before_growth: int = 3,
        growth_factor: float = 1.5,
        bound_release_ratio: float = 0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if successes_before_growth < 1:
            raise ValueError("successes_before_growth must be at least 1")
        if growth_factor <= 1:
            raise ValueError("growth_factor must be > 1")
        if not 0 < bound_release_ratio <= 1:
            raise ValueError("bound_release_ratio must be in (0, 1]")
        self.successes_before_growth = successes_before_growth
        self.growth_factor = growth_factor
        self.bound_release_ratio = bound_release_ratio
        self._oom_bound_cost: Optional[float] = None

    def observe_batch_cost(self, per_row_cost: float) -> None:
        super().observe_batch_cost(per_row_cost)
        # Cost-scoped bounds: an OOM wall learned at a higher cost no longer
        # applies once cost has fallen materially below it.
        if (
            self.oom_upper_bound is not None
            and self._oom_bound_cost is not None
            and self._pending_cost is not None
            and self._pending_cost < self._oom_bound_cost * self.bound_release_ratio
        ):
            self.reset_oom_bound()
            self._oom_bound_cost = None

    def observe_oom(self, batch_size: int) -> int:
        self._oom_bound_cost = self._pending_cost
        next_size = super().observe_oom(batch_size)
        self.consecutive_successes = 0
        return next_size

    def observe_success(self, batch_size: int) -> int:
        self._record_success(batch_size)
        # Keep the shrink model fresh; its only authority is shrinking.
        peak_bytes = self.peak_memory_probe()
        cost = self._pending_cost
        if peak_bytes > 0 and cost is not None:
            self._fit_sample(batch_size * cost, float(peak_bytes))

        predicted = self._predicted_batch_size()
        if predicted is not None and predicted < batch_size:
            # Model-driven shrink: cost is rising toward the OOM wall.
            self.model_preemptions += 1
            candidate = predicted
            self.consecutive_successes = 0
        elif self.consecutive_successes >= self.successes_before_growth:
            # Success-driven growth: streaks expand, model stays silent --
            # except as a ceiling when it is reliable, which stops streak
            # growth from ping-ponging into the OOM boundary.
            grown = int(batch_size * self.growth_factor)
            candidate = max(grown, batch_size + 1)
            if predicted is not None:
                candidate = min(candidate, max(predicted, batch_size))
            self.consecutive_successes = 0
        else:
            candidate = batch_size
        self.current_batch_size = self._clamp(candidate)
        return self.current_batch_size
