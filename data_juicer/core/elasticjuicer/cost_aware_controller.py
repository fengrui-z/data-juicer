"""Cost-aware batch-size controller (phase-3 elastic upgrade).

Hybrid of a learned memory model and the success/OOM safety net:

- Online linear model of peak VRAM: ``peak_bytes ~= a + b * (batch_size *
  per_row_cost)`` fitted incrementally from successful slices.
- Prediction: ``batch = (budget - a) / (b * per_row_cost)``, damped growth,
  undamped shrink — as per-row cost rises the controller pre-shrinks BEFORE
  the OOM wall instead of probing into it.
- Safety net: every observation still updates the success lower bound and the
  OOM upper bound; predictions are clamped below the known failing size and
  growth never exceeds the success-driven streak rule. If the model is
  unreliable (too few samples / degenerate fit), it degrades to cautious
  additive growth — the lossless harness absorbs any residual error.

The controller implements the standard harness interface (next_batch_size /
observe_success / observe_oom / state / seed_bounds / set_hard_limit) plus
``observe_batch_cost`` fed by OOMSafeAdaptiveMapper's batch_cost_fn hook.
"""

from typing import Callable, Optional

from .baseline_controllers import BaselineBatchController


class CostAwareController(BaselineBatchController):
    """Predict batch size from a learned per-row-cost memory model."""

    def __init__(
        self,
        *args,
        peak_memory_probe: Optional[Callable[[], int]] = None,
        total_bytes: Optional[int] = None,
        target_fraction: float = 0.85,
        max_growth_factor: float = 1.5,
        min_model_samples: int = 3,
        max_trend_factor: float = 1.10,
        fit_decay: float = 0.94,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if peak_memory_probe is None:
            raise ValueError("costaware controller requires peak_memory_probe")
        if not total_bytes:
            raise ValueError("costaware controller requires total_bytes")
        if not 0 < target_fraction <= 1:
            raise ValueError("target_fraction must be in (0, 1]")
        if max_growth_factor <= 1:
            raise ValueError("max_growth_factor must be > 1")
        if min_model_samples < 1:
            raise ValueError("min_model_samples must be at least 1")
        if max_trend_factor < 1:
            raise ValueError("max_trend_factor must be >= 1")
        if not 0 < fit_decay <= 1:
            raise ValueError("fit_decay must be in (0, 1]")
        self.fit_decay = fit_decay
        self.peak_memory_probe = peak_memory_probe
        self.budget_bytes = float(total_bytes) * target_fraction
        self.max_growth_factor = max_growth_factor
        self.min_model_samples = min_model_samples

        # Recursive least-squares state for peak = a + b * total_cost,
        # kept as running sums (all observations weighted equally).
        self._samples = 0
        self._sum_x = 0.0  # total_cost = batch_size * per_row_cost
        self._sum_y = 0.0  # peak bytes
        self._sum_xx = 0.0
        self._sum_xy = 0.0

        self._pending_cost: Optional[float] = None
        # EMA of per-block cost ratios: extrapolates drift direction so the
        # prediction prices in the NEXT block's cost, not the last one's.
        self._cost_trend = 1.0
        self._last_block_cost: Optional[float] = None
        self.max_trend_factor = max_trend_factor
        self.model_predictions = 0
        self.model_preemptions = 0  # predictions below the current size

    # -- harness surface ------------------------------------------------------

    def observe_batch_cost(self, per_row_cost: float) -> None:
        """Receive the per-row cost of the incoming block (harness hook)."""
        if per_row_cost is None or per_row_cost <= 0:
            return
        if self._last_block_cost is not None and self._last_block_cost > 0:
            ratio = per_row_cost / self._last_block_cost
            # Damped EMA, bounded so a single outlier block cannot skew us.
            ratio = max(0.5, min(ratio, 2.0))
            self._cost_trend = 0.5 * self._cost_trend + 0.5 * ratio
        self._last_block_cost = float(per_row_cost)
        self._pending_cost = float(per_row_cost)

    def observe_success(self, batch_size: int) -> int:
        self._record_success(batch_size)
        peak_bytes = self.peak_memory_probe()
        cost = self._pending_cost
        if peak_bytes > 0 and cost is not None:
            self._fit_sample(batch_size * cost, float(peak_bytes))

        predicted = self._predicted_batch_size()
        if predicted is not None:
            self.model_predictions += 1
            if predicted < batch_size:
                self.model_preemptions += 1
            candidate = self._damp(batch_size, predicted, allow_shrink=True)
        else:
            # Not enough evidence yet: cautious additive growth.
            candidate = batch_size + 1
        self.current_batch_size = self._clamp(candidate)
        return self.current_batch_size

    def observe_oom(self, batch_size: int) -> int:
        self._record_oom(batch_size)
        # Consume the peak window: it is contaminated by the failed attempt's
        # partial allocations, and must not leak into the retry's sample.
        try:
            self.peak_memory_probe()
        except Exception:  # noqa: BLE001 - probe failure must not break recovery
            pass
        # Prefer the learned model (with an extra safety margin) over blind
        # halving; fall back to halving while the model is still unreliable.
        predicted = self._predicted_batch_size(safety_factor=self._safety_factor())
        if predicted is not None:
            candidate = self._damp(batch_size, predicted, allow_shrink=True)
        else:
            candidate = batch_size // 2
        self.current_batch_size = self._clamp(max(self.min_batch_size, candidate))
        return self.current_batch_size

    # -- model ----------------------------------------------------------------

    def _fit_sample(self, total_cost: float, peak_bytes: float) -> None:
        # Recency-weighted least squares: decay old evidence so the model
        # tracks the current cost regime on drift workloads (the VRAM-per-cost
        # ratio itself can shift across sections, e.g. compression effects).
        d = self.fit_decay
        self._sum_x *= d
        self._sum_y *= d
        self._sum_xx *= d
        self._sum_xy *= d
        self._samples = self._samples * d + 1.0
        self._sum_x += total_cost
        self._sum_y += peak_bytes
        self._sum_xx += total_cost * total_cost
        self._sum_xy += total_cost * peak_bytes

    def _model_coefficients(self):
        """Return (a, b) for peak = a + b * total_cost, or None if unusable."""
        n = self._samples
        if n < self.min_model_samples:
            return None
        denom = n * self._sum_xx - self._sum_x * self._sum_x
        if denom <= 0:
            return None
        b = (n * self._sum_xy - self._sum_x * self._sum_y) / denom
        if b <= 0:
            return None
        a = (self._sum_y - b * self._sum_x) / n
        return a, b

    def _predicted_batch_size(self, safety_factor: float = 1.0) -> Optional[int]:
        coefficients = self._model_coefficients()
        cost = self._pending_cost
        if coefficients is None or cost is None or cost <= 0:
            return None
        a, b = coefficients
        usable = self.budget_bytes - a
        if usable <= 0:
            return self.min_batch_size
        # Price in drift direction: use the extrapolated next-block cost.
        trend = max(1.0, min(self._cost_trend, self.max_trend_factor))
        predicted = usable / (b * cost * trend) * safety_factor
        return max(1, int(predicted))

    def _safety_factor(self) -> float:
        # OOM-based prediction keeps extra margin below the implied boundary.
        return 0.75

    def _damp(self, current: int, target: int, allow_shrink: bool) -> int:
        if target >= current:
            capped = current * self.max_growth_factor
            return int(min(target, capped))
        if allow_shrink:
            return target
        return current
