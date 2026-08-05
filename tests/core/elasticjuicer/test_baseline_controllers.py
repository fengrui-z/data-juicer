"""Unit tests for baseline batch-size policies (drift experiment arms)."""

import unittest

from data_juicer.core.elasticjuicer.batch_controller import MinimumBatchSizeOOM
from data_juicer.core.elasticjuicer.baseline_controllers import (
    BisectController,
    FixedBatchController,
    HalvingController,
    MemoryProportionalController,
    make_baseline_controller,
)


class FixedControllerTest(unittest.TestCase):
    def test_constant_size_and_oom_propagates(self):
        c = FixedBatchController(initial_batch_size=16, max_batch_size=32)
        self.assertEqual(c.next_batch_size(1000), 16)
        c.observe_success(16)
        self.assertEqual(c.next_batch_size(1000), 16)
        with self.assertRaises(MinimumBatchSizeOOM):
            c.observe_oom(16)


class HalvingControllerTest(unittest.TestCase):
    def test_oom_halves_and_success_streak_grows(self):
        c = HalvingController(initial_batch_size=32, max_batch_size=32, grow_after_successes=2)
        c.observe_oom(32)
        self.assertEqual(c.current_batch_size, 16)
        c.observe_success(16)
        self.assertEqual(c.current_batch_size, 16)  # streak not complete yet
        c.observe_success(16)
        # x2 growth after streak, clamped just below the known OOM size (32)
        self.assertGreaterEqual(c.current_batch_size, 31)
        self.assertLessEqual(c.current_batch_size, 32)

    def test_floor_at_min(self):
        c = HalvingController(initial_batch_size=2, min_batch_size=1, max_batch_size=32)
        c.observe_oom(2)
        self.assertEqual(c.current_batch_size, 1)
        with self.assertRaises(MinimumBatchSizeOOM):
            c.observe_oom(1)


class BisectControllerTest(unittest.TestCase):
    def test_converges_to_boundary(self):
        c = BisectController(initial_batch_size=32, max_batch_size=32)
        # Boundary at 20: sizes above OOM, at/below succeed.
        for _ in range(12):
            size = c.next_batch_size(1000)
            if size >= 21:
                c.observe_oom(size)
            else:
                c.observe_success(size)
        self.assertGreaterEqual(c.success_lower_bound, 16)
        self.assertLessEqual(c.oom_upper_bound, 24)
        self.assertTrue(16 <= c.current_batch_size <= 20)


class ProportionalControllerTest(unittest.TestCase):
    def test_scales_by_headroom(self):
        peaks = iter([50_000_000])  # 50MB peak, 100MB total, target 0.85 -> ~x1.7
        c = MemoryProportionalController(
            initial_batch_size=10,
            max_batch_size=32,
            peak_memory_probe=lambda: next(peaks, 0),
            total_bytes=100_000_000,
            max_growth_factor=1.5,
        )
        c.observe_success(10)
        self.assertEqual(c.current_batch_size, 15)  # capped by max_growth_factor

    def test_oom_halves(self):
        c = MemoryProportionalController(
            initial_batch_size=16, max_batch_size=32, total_bytes=100
        )
        c.observe_oom(16)
        self.assertEqual(c.current_batch_size, 8)


class FactoryAndSurfaceTest(unittest.TestCase):
    def test_factory_and_state_surface(self):
        for policy in ("fixed", "halving", "bisect"):
            c = make_baseline_controller(policy, initial_batch_size=8, max_batch_size=32)
            state = c.state
            self.assertEqual(state.current_batch_size, 8)
            self.assertIsNone(state.oom_upper_bound)
        with self.assertRaises(ValueError):
            make_baseline_controller("proportional", initial_batch_size=8)
        with self.assertRaises(ValueError):
            make_baseline_controller("unknown", initial_batch_size=8)

    def test_seed_bounds_adopted(self):
        c = HalvingController(initial_batch_size=32, max_batch_size=32)
        seeded = c.seed_bounds(safe_batch_size=12, oom_upper_bound=16)
        self.assertEqual(seeded, 12)
        self.assertEqual(c.oom_upper_bound, 16)
        self.assertEqual(c.success_lower_bound, 12)
        self.assertLessEqual(c.next_batch_size(1000), 15)


if __name__ == "__main__":
    unittest.main()
