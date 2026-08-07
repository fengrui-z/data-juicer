"""SOTA-style batch-size controllers for phase-4 comparison (Trident family).

Two learning-based policies approximating the SOTA design space that
Trident occupies (GP throughput/memory model + memory-constrained safe
exploration, and bandit arm selection), implemented honestly at experiment
scale:

- SafeBatchOptimizationController ("safebo"): memory model with an explicit
  uncertainty estimate (residual-based, shrinks with evidence). The batch
  is the largest size whose predicted peak PLUS a confidence margin stays
  inside the VRAM budget -- the "safe set" rule of safe Bayesian
  optimization, with cost-trend extrapolation inherited from the
  cost-aware controller. Cold start is maximally conservative (large
  initial uncertainty); evidence converts exploration into exploitation.

- BanditController ("bandit"): discrete batch sizes as arms, but arm
  feasibility posteriors are CONTEXTUAL -- keyed by a log-spaced cost
  bucket, because a pure batch-arm bandit cannot adapt to drift (an arm
  that was feasible at low cost is not evidence for high cost; Trident's
  online model is conditioned on workload features for exactly this
  reason). Beta posteriors per (bucket, arm), recency-decayed; Thompson
  sampling plays the largest arm whose sampled feasibility clears a
  threshold; an OOM penalizes the arm in that bucket and hard-excludes
  arms at/above it until a success streak. This is the constrained
  contextual-bandit formulation, not a throughput-optimized one (the
  harness does not expose slice latency to the controller).
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from .cost_aware_controller import CostAwareController


class SafeBatchOptimizationController(CostAwareController):
    """Largest batch inside the predicted-safe set (model + uncertainty)."""

    def __init__(
        self,
        *args,
        initial_uncertainty_bytes: float = 1.0e9,
        uncertainty_decay: float = 0.85,
        safety_multiplier: float = 2.0,
        successes_before_growth: int = 2,
        max_growth_factor: float = 1.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if initial_uncertainty_bytes < 0:
            raise ValueError("initial_uncertainty_bytes must be non-negative")
        if not 0 < uncertainty_decay <= 1:
            raise ValueError("uncertainty_decay must be in (0, 1]")
        if safety_multiplier < 0:
            raise ValueError("safety_multiplier must be non-negative")
        if successes_before_growth < 1:
            raise ValueError("successes_before_growth must be at least 1")
        if max_growth_factor <= 1:
            raise ValueError("max_growth_factor must be > 1")
        self.safety_multiplier = safety_multiplier
        self.successes_before_growth = successes_before_growth
        self.max_growth_factor = max_growth_factor
        # Residual EWMA stands in for the GP posterior standard deviation.
        self._residual_ema: Optional[float] = None
        self._uncertainty = float(initial_uncertainty_bytes)
        self._uncertainty_decay = uncertainty_decay

    def observe_success(self, batch_size: int) -> int:
        self._record_success(batch_size)
        peak_bytes = self.peak_memory_probe()
        cost = self._pending_cost
        if peak_bytes > 0 and cost is not None:
            total_cost = batch_size * cost
            previous = self._predicted_peak(total_cost)
            self._fit_sample(total_cost, float(peak_bytes))
            if previous is not None:
                residual = abs(float(peak_bytes) - previous)
                if self._residual_ema is None:
                    self._residual_ema = residual
                else:
                    self._residual_ema = (
                        0.7 * self._residual_ema + 0.3 * residual
                    )
        self._uncertainty = self._current_uncertainty()

        predicted = self._safe_batch_size()
        if predicted is not None and predicted < batch_size:
            # The safe set shrinks: preempt before the OOM wall.
            self.model_preemptions += 1
            self.consecutive_successes = 0
            candidate = predicted
        elif self.consecutive_successes >= self.successes_before_growth:
            # Grow inside the safe set: the prediction is the ceiling, the
            # step is capped so a noisy fit cannot launch a wild jump.
            grown = int(batch_size * self.max_growth_factor)
            candidate = max(grown, batch_size + 1)
            if predicted is not None:
                candidate = min(candidate, max(predicted, batch_size))
            self.consecutive_successes = 0
        else:
            candidate = batch_size
        self.current_batch_size = self._clamp(candidate)
        return self.current_batch_size

    def observe_oom(self, batch_size: int) -> int:
        self._record_oom(batch_size)
        try:
            self.peak_memory_probe()
        except Exception:  # noqa: BLE001 - contaminated window, discard
            pass
        # An OOM is evidence the uncertainty was too small: widen it.
        self._uncertainty = max(
            self._current_uncertainty(), self.budget_bytes * 0.10
        )
        predicted = self._safe_batch_size()
        if predicted is not None:
            # Extra margin on the OOM path: the safe-set estimate sits right
            # at the boundary and cost keeps drifting, so a plain boundary
            # retry OOMs again.
            candidate = int(predicted * 0.75)
            candidate = self._damp(batch_size, candidate, allow_shrink=True)
        else:
            candidate = batch_size // 2
        # The retry must be strictly below the failing size.
        candidate = min(candidate, max(self.min_batch_size, batch_size - 1))
        self.consecutive_successes = 0
        self.current_batch_size = self._clamp(max(self.min_batch_size, candidate))
        return self.current_batch_size

    # -- safe set ---------------------------------------------------------------

    def _predicted_peak(self, total_cost: float) -> Optional[float]:
        coefficients = self._model_coefficients()
        if coefficients is None:
            return None
        a, b = coefficients
        return a + b * total_cost

    def _current_uncertainty(self) -> float:
        if self._residual_ema is None:
            return self._uncertainty
        return max(self._residual_ema * self.safety_multiplier, 1.0)

    def _safe_batch_size(self) -> Optional[int]:
        coefficients = self._model_coefficients()
        cost = self._pending_cost
        if coefficients is None or cost is None or cost <= 0:
            return None
        a, b = coefficients
        usable = self.budget_bytes - a - self._current_uncertainty()
        if usable <= 0:
            return self.min_batch_size
        trend = max(1.0, min(self._cost_trend, self.max_trend_factor))
        return max(1, int(usable / (b * cost * trend)))


class BanditController(CostAwareController):
    """Contextual Thompson sampling over batch arms (context = cost bucket)."""

    def __init__(
        self,
        *args,
        arm_candidates: Optional[List[int]] = None,
        feasibility_threshold: float = 0.5,
        count_decay: float = 0.97,
        max_bucket: int = 16,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0 < feasibility_threshold < 1:
            raise ValueError("feasibility_threshold must be in (0, 1)")
        if not 0 < count_decay <= 1:
            raise ValueError("count_decay must be in (0, 1]")
        upper = self._effective_maximum()
        default_arms = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64]
        arms = arm_candidates if arm_candidates is not None else default_arms
        arms = sorted({int(a) for a in arms if self.min_batch_size <= a <= upper})
        if not arms:
            arms = [self.min_batch_size]
        self.arms = arms
        self.feasibility_threshold = feasibility_threshold
        self.count_decay = count_decay
        self.max_bucket = max_bucket
        self._rng = random.Random(seed)
        # Beta posteriors per (cost bucket, arm); the first observed cost
        # anchors bucket 0, each doubling of cost moves one bucket up.
        self._cost_base: Optional[float] = None
        self._alpha: Dict[Tuple[int, int], float] = {}
        self._beta: Dict[Tuple[int, int], float] = {}
        # Hard exclusion per bucket after an OOM: posteriors alone are too
        # slow to suppress the failing arm, so arms at/above it are barred
        # until a streak of successes proves the retreat stable.
        self._excluded_above: Dict[int, int] = {}
        self._exclusion_successes: Dict[int, int] = {}
        self._exclusion_release_successes = 3
        self._retreat_size: Optional[int] = None

    def _bucket_of(self, cost: float) -> int:
        if self._cost_base is None:
            self._cost_base = cost
        if cost <= 0 or self._cost_base <= 0:
            return 0
        ratio = cost / self._cost_base
        if ratio <= 0:
            return 0
        return max(0, min(self.max_bucket, int(math.log2(ratio))))

    def _current_bucket(self) -> int:
        if self._pending_cost is None:
            return 0
        return self._bucket_of(self._pending_cost)

    def _posterior(self, bucket: int, arm: int) -> Tuple[float, float]:
        key = (bucket, arm)
        return self._alpha.get(key, 1.0), self._beta.get(key, 1.0)

    def next_batch_size(self, remaining_samples: int) -> int:
        if remaining_samples <= 0:
            return 0
        if self._retreat_size is not None:
            # Deliver the hard OOM retreat once, then hand control back to
            # posterior-driven selection.
            size = min(self._retreat_size, remaining_samples)
            self._retreat_size = None
            return max(self.min_batch_size, size)
        bucket = self._current_bucket()
        # Thompson sampling inside the bucket: draw feasibility per arm,
        # play the largest arm clearing the threshold; fall back to the
        # most-feasible arm.
        arms = self.arms
        excluded = self._excluded_above.get(bucket)
        if excluded is not None:
            barred = [arm for arm in arms if arm < excluded]
            if barred:
                arms = barred
        draws = {
            arm: self._rng.betavariate(*self._posterior(bucket, arm)) for arm in arms
        }
        feasible = [arm for arm in arms if draws[arm] >= self.feasibility_threshold]
        if feasible:
            choice = max(feasible)
        else:
            choice = max(arms, key=lambda arm: draws[arm])
        choice = min(choice, remaining_samples)
        return max(self.min_batch_size, choice)

    def observe_success(self, batch_size: int) -> int:
        self._record_success(batch_size)
        self.peak_memory_probe()  # keep the window accounting clean
        self._update_posterior(batch_size, success=True)
        bucket = self._current_bucket()
        if bucket in self._excluded_above:
            self._exclusion_successes[bucket] = self._exclusion_successes.get(bucket, 0) + 1
            if self._exclusion_successes[bucket] >= self._exclusion_release_successes:
                del self._excluded_above[bucket]
                del self._exclusion_successes[bucket]
        self.current_batch_size = self._clamp(batch_size)
        return self.current_batch_size

    def observe_oom(self, batch_size: int) -> int:
        self._record_oom(batch_size)
        try:
            self.peak_memory_probe()
        except Exception:  # noqa: BLE001 - discard contaminated window
            pass
        self._update_posterior(batch_size, success=False)
        bucket = self._current_bucket()
        previous = self._excluded_above.get(bucket)
        self._excluded_above[bucket] = (
            batch_size if previous is None else min(previous, batch_size)
        )
        self._exclusion_successes[bucket] = 0
        # The retry is never sampled: a fresh bucket has uniform posteriors,
        # so Thompson selection can pick an arm infeasible at the current
        # cost (observed: retry FAIL). Hard retreat to half the failing size;
        # posterior-driven selection climbs back afterwards.
        self._retreat_size = max(self.min_batch_size, batch_size // 2)
        self.current_batch_size = self._clamp(self._retreat_size)
        return self.current_batch_size

    # -- posteriors ---------------------------------------------------------------

    def _nearest_arm(self, batch_size: int) -> int:
        return min(self.arms, key=lambda arm: abs(arm - batch_size))

    def _update_posterior(self, batch_size: int, success: bool) -> None:
        bucket = self._current_bucket()
        # Recency decay inside the bucket: even within one cost regime the
        # boundary can shift, so stale evidence is discounted.
        for arm in self.arms:
            key = (bucket, arm)
            self._alpha[key] = 1.0 + (self._alpha.get(key, 1.0) - 1.0) * self.count_decay
            self._beta[key] = 1.0 + (self._beta.get(key, 1.0) - 1.0) * self.count_decay
        arm = self._nearest_arm(batch_size)
        key = (bucket, arm)
        if success:
            self._alpha[key] = self._alpha.get(key, 1.0) + 1.0
        else:
            self._beta[key] = self._beta.get(key, 1.0) + 2.0  # safety weight
            # A failure invalidates most of the arm's historical success
            # evidence in this bucket (observed OOM ping-pong otherwise).
            self._alpha[key] = 1.0 + (self._alpha.get(key, 1.0) - 1.0) * 0.3
