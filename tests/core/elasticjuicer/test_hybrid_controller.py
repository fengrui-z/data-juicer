"""Unit tests for the hybrid controller v3 (model shrink + success grow)."""

import unittest

from data_juicer.core.elasticjuicer.hybrid_controller import HybridController
from tests.core.elasticjuicer.test_cost_aware_controller import make_world


class HybridControllerTest(unittest.TestCase):
    def _make(self, **overrides):
        state, probe, run_slice = make_world()
        kwargs = dict(
            initial_batch_size=32,
            max_batch_size=32,
            peak_memory_probe=probe,
            total_bytes=100_000_000,
            min_model_samples=3,
        )
        kwargs.update(overrides)
        return state, run_slice, HybridController(**kwargs)

    def test_rising_cost_preempts_oom(self):
        """Shrink side: model must pre-shrink as cost rises (<=1 cold OOM)."""
        state, run_slice, c = self._make()
        ooms = 0
        state["cost"] = 80_000.0
        for _ in range(60):
            c.observe_batch_cost(state["cost"])
            size = c.next_batch_size(10_000)
            if not run_slice(c, size):
                ooms += 1
                size = c.next_batch_size(10_000)
                self.assertTrue(run_slice(c, size))
            state["cost"] *= 1.02
        self.assertLessEqual(ooms, 1)
        self.assertLess(c.current_batch_size, 32, "must have pre-shrunk as cost rose")

    def test_falling_cost_regrows_by_streaks(self):
        """Grow side: success streaks must expand fast as cost falls."""
        state, run_slice, c = self._make(initial_batch_size=8)
        state["cost"] = 300_000.0
        slices = 0
        for _ in range(40):
            c.observe_batch_cost(state["cost"])
            size = c.next_batch_size(10_000)
            if not run_slice(c, size):
                size = c.next_batch_size(10_000)
                run_slice(c, size)
            state["cost"] *= 0.95
            slices += 1
            if c.current_batch_size >= 32:
                break
        self.assertGreaterEqual(c.current_batch_size, 32, "streak growth must reach the cap")
        self.assertLessEqual(slices, 30, "growth must be fast (streak-driven, not damped)")

    def test_growth_gated_by_streak(self):
        state, run_slice, c = self._make(initial_batch_size=8, successes_before_growth=3)
        state["cost"] = 1_000.0
        c.observe_batch_cost(state["cost"])
        run_slice(c, 8)
        self.assertEqual(c.current_batch_size, 8, "no growth before the streak completes")
        run_slice(c, 8)
        self.assertEqual(c.current_batch_size, 8)
        run_slice(c, 8)
        self.assertGreaterEqual(c.current_batch_size, 12, "streak complete -> multiplicative growth")

    def test_growth_never_reprobes_known_oom_size(self):
        state, run_slice, c = self._make(initial_batch_size=32)
        state["cost"] = 120_000.0  # batch 32 infeasible here
        c.observe_batch_cost(state["cost"])
        self.assertFalse(run_slice(c, 32))  # OOM -> upper bound recorded
        self.assertEqual(c.oom_upper_bound, 32)
        # Streak growth must be clamped below the known failing size.
        for _ in range(6):
            size = c.next_batch_size(10_000)
            self.assertLess(size, 32)
            if not run_slice(c, size):
                size = c.next_batch_size(10_000)
                run_slice(c, size)

    def test_validation(self):
        with self.assertRaises(ValueError):
            HybridController(
                initial_batch_size=8, peak_memory_probe=lambda: 0,
                total_bytes=100, successes_before_growth=0,
            )
        with self.assertRaises(ValueError):
            HybridController(
                initial_batch_size=8, peak_memory_probe=lambda: 0,
                total_bytes=100, growth_factor=1.0,
            )


if __name__ == "__main__":
    unittest.main()
