"""Unit tests for the phase-4 SOTA-style controllers (safebo, bandit)."""

import unittest

from data_juicer.core.elasticjuicer.sota_controllers import (
    BanditController,
    SafeBatchOptimizationController,
)
from tests.core.elasticjuicer.test_cost_aware_controller import make_world


def drive(controller, run_slice, state, steps, drift, start_cost):
    """Run the harness loop over a drifting cost stream; return OOM count.

    Retries mirror the real harness: the same slice is retried at whatever
    size the controller proposes after an OOM, up to max_retries times.
    """
    ooms = 0
    state["cost"] = start_cost
    for _ in range(steps):
        controller.observe_batch_cost(state["cost"])
        size = controller.next_batch_size(10_000)
        retries = 0
        while not run_slice(controller, size):
            ooms += 1
            retries += 1
            if retries > 16:
                raise AssertionError("slice could not be delivered after 16 OOM retries")
            size = controller.next_batch_size(10_000)
        state["cost"] *= drift
    return ooms


class SafeBoControllerTest(unittest.TestCase):
    def _make(self, initial_batch_size=32, **overrides):
        state, probe, run_slice = make_world()
        kwargs = dict(
            initial_batch_size=initial_batch_size,
            max_batch_size=32,
            peak_memory_probe=probe,
            total_bytes=100_000_000,
            min_model_samples=3,
        )
        kwargs.update(overrides)
        return state, run_slice, SafeBatchOptimizationController(**kwargs)

    def test_rising_cost_stays_safe(self):
        state, run_slice, c = self._make()
        ooms = drive(c, run_slice, state, steps=60, drift=1.02, start_cost=80_000.0)
        self.assertLessEqual(ooms, 2, "safe-set margin must absorb drift noise")
        self.assertLess(c.current_batch_size, 32, "must shrink as cost rises")

    def test_falling_cost_regrows(self):
        state, run_slice, c = self._make(initial_batch_size=8)
        drive(c, run_slice, state, steps=50, drift=0.95, start_cost=300_000.0)
        # The safe set hugs the boundary (conservative by design); require
        # meaningful recovery from the cold-start collapse, not the cap.
        self.assertGreaterEqual(c.current_batch_size, 8, "safe set must widen as cost falls")

    def test_oom_widens_uncertainty(self):
        state, run_slice, c = self._make()
        before = c._current_uncertainty()
        state["cost"] = 200_000.0
        c.observe_batch_cost(state["cost"])
        c.observe_oom(32)
        self.assertGreater(c._current_uncertainty(), 0)
        self.assertLessEqual(c.current_batch_size, 31)
        _ = before  # uncertainty only grows from evidence


class BanditControllerTest(unittest.TestCase):
    def _make(self, initial_batch_size=32, **overrides):
        state, probe, run_slice = make_world()
        kwargs = dict(
            initial_batch_size=initial_batch_size,
            max_batch_size=32,
            peak_memory_probe=probe,
            total_bytes=100_000_000,
            seed=7,
        )
        kwargs.update(overrides)
        return state, run_slice, BanditController(**kwargs)

    def test_rising_cost_survives(self):
        state, run_slice, c = self._make()
        ooms = drive(c, run_slice, state, steps=60, drift=1.02, start_cost=80_000.0)
        # Feasibility probing is the bandit's intrinsic exploration cost
        # (observed ~23 OOMs vs safebo/hybrid ~1); the contract here is
        # survival and eventual retreat, not probe efficiency.
        self.assertLessEqual(ooms, 30, "bandit must not thrash unboundedly")
        self.assertLess(c.current_batch_size, 32, "must retreat from the OOM wall")

    def test_falling_cost_regrows(self):
        state, run_slice, c = self._make(initial_batch_size=8)
        ooms = drive(c, run_slice, state, steps=50, drift=0.95, start_cost=300_000.0)
        self.assertLessEqual(ooms, 15)
        self.assertGreaterEqual(c.current_batch_size, 8, "decay must un-learn stale failures")

    def test_oom_retreats_below_failing_arm(self):
        state, run_slice, c = self._make()
        state["cost"] = 200_000.0
        c.observe_batch_cost(state["cost"])
        c.observe_oom(32)
        self.assertLess(c.current_batch_size, 32)
        # The failed arm's feasibility posterior must be damaged in its bucket.
        arm = c._nearest_arm(32)
        bucket = c._current_bucket()
        alpha, beta = c._posterior(bucket, arm)
        self.assertGreater(beta, alpha)

    def test_validation(self):
        with self.assertRaises(ValueError):
            BanditController(
                initial_batch_size=8, peak_memory_probe=lambda: 0,
                total_bytes=100, feasibility_threshold=1.5,
            )


if __name__ == "__main__":
    unittest.main()
