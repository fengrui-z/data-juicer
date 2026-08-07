"""Unit tests for the cost-aware controller (phase-3 elastic upgrade)."""

import unittest

from data_juicer.core.elasticjuicer.cost_aware_controller import CostAwareController


def make_world(total_bytes=100_000_000, per_unit_bytes=30.0, overhead=5_000_000):
    """Synthetic memory world: peak = overhead + per_unit * batch * cost."""

    state = {"peak": 0, "cost": 1.0}

    def probe():
        peak = state["peak"]
        state["peak"] = 0
        return peak

    def run_slice(controller, batch_size):
        """Emulate the harness: execute slice, return True on success.

        Peak accounting mirrors torch: max since last reset (a failed attempt
        leaves its partial-allocation peak in the window until consumed).
        """
        peak = overhead + per_unit_bytes * batch_size * state["cost"]
        state["peak"] = max(state["peak"], peak)
        budget = total_bytes * 0.85
        if peak > budget:
            controller.observe_oom(batch_size)
            return False
        controller.observe_success(batch_size)
        return True

    return state, probe, run_slice


class CostAwareControllerTest(unittest.TestCase):
    def test_requires_probe_and_budget(self):
        with self.assertRaises(ValueError):
            CostAwareController(initial_batch_size=8, total_bytes=100)
        with self.assertRaises(ValueError):
            CostAwareController(initial_batch_size=8, peak_memory_probe=lambda: 1)

    def test_rising_cost_preempts_oom(self):
        """Cost rising 2% per slice: the model must shrink BEFORE hitting OOM."""
        state, probe, run_slice = make_world()
        c = CostAwareController(
            initial_batch_size=32,
            max_batch_size=32,
            peak_memory_probe=probe,
            total_bytes=100_000_000,
            min_model_samples=3,
        )
        ooms = 0
        last_oom_step = None
        state["cost"] = 80_000.0  # start inside the feasible region
        for step in range(60):
            c.observe_batch_cost(state["cost"])  # harness hook per block
            size = c.next_batch_size(10_000)
            if not run_slice(c, size):
                ooms += 1
                last_oom_step = step
                # retry the same slice like the harness does
                size = c.next_batch_size(10_000)
                self.assertTrue(run_slice(c, size), "retried slice must succeed")
            state["cost"] *= 1.02
        # At most one cold-start OOM while the trend EMA converges; never a
        # chasing pattern (previous behavior OOMed every step).
        self.assertLessEqual(ooms, 1, "cost-aware controller must preempt the OOM wall")
        if ooms:
            self.assertLess(last_oom_step, 6, "cold-start OOM must happen early")
        self.assertLess(c.current_batch_size, 32, "must have pre-shrunk as cost rose")

    def test_falling_cost_regrows(self):
        state, probe, run_slice = make_world()
        c = CostAwareController(
            initial_batch_size=8,
            max_batch_size=32,
            peak_memory_probe=probe,
            total_bytes=100_000_000,
            min_model_samples=3,
        )
        state["cost"] = 300_000.0
        for _ in range(40):
            c.observe_batch_cost(state["cost"])
            size = c.next_batch_size(10_000)
            if not run_slice(c, size):
                size = c.next_batch_size(10_000)
                run_slice(c, size)
            state["cost"] *= 0.97
        self.assertGreater(c.current_batch_size, 8, "must regrow as cost falls")

    def test_cold_start_additive_growth(self):
        """Before the model has enough samples, grow additively (no wild jumps)."""
        state, probe, run_slice = make_world()
        c = CostAwareController(
            initial_batch_size=8,
            max_batch_size=32,
            peak_memory_probe=probe,
            total_bytes=100_000_000,
            min_model_samples=3,
        )
        state["cost"] = 1_000.0
        c.observe_batch_cost(state["cost"])
        run_slice(c, 8)
        self.assertLessEqual(c.current_batch_size, 10, "cold-start growth must be gradual")

    def test_oom_falls_back_to_halving_without_model(self):
        probe_state = {"peak": 0}
        c = CostAwareController(
            initial_batch_size=32,
            max_batch_size=32,
            peak_memory_probe=lambda: probe_state["peak"],
            total_bytes=100_000_000,
        )
        c.observe_oom(32)  # no samples yet -> halving fallback
        self.assertEqual(c.current_batch_size, 16)

    def test_cost_model_export_seed_roundtrip(self):
        """Phase-3c follow-up: the fitted model must survive a restart."""
        state, probe, run_slice = make_world()
        teacher = CostAwareController(
            initial_batch_size=32,
            max_batch_size=32,
            peak_memory_probe=probe,
            total_bytes=100_000_000,
            min_model_samples=3,
        )
        self.assertIsNone(teacher.export_cost_model(), "no export before enough evidence")
        state["cost"] = 80_000.0
        for _ in range(6):
            teacher.observe_batch_cost(state["cost"])
            run_slice(teacher, teacher.next_batch_size(10_000))
        snapshot = teacher.export_cost_model()
        self.assertIsNotNone(snapshot)

        student = CostAwareController(
            initial_batch_size=32,
            max_batch_size=32,
            peak_memory_probe=probe,
            total_bytes=100_000_000,
            min_model_samples=3,
        )
        self.assertTrue(student.seed_cost_model(snapshot))
        teacher_pred = teacher._predicted_batch_size()
        student_pred = student._predicted_batch_size()
        self.assertIsNotNone(student_pred)
        self.assertEqual(teacher_pred, student_pred, "seeded student must predict like the teacher")

    def test_seed_cost_model_rejects_malformed(self):
        c = CostAwareController(
            initial_batch_size=8,
            max_batch_size=32,
            peak_memory_probe=lambda: 0,
            total_bytes=100_000_000,
        )
        for bad in (None, {}, {"samples": "x"}, {"samples": -1}, {"samples": 0.5}):
            self.assertFalse(c.seed_cost_model(bad), f"must reject {bad!r}")

    def test_cost_model_profile_persistence_roundtrip(self):
        """Snapshot survives StageProfile merge + atomic save/load."""
        import tempfile

        from data_juicer.core.elasticjuicer.control_service import (
            ControlService,
            StageProfile,
            load_stage_profiles,
            save_stage_profiles,
        )

        snapshot = {
            "samples": 5.0,
            "sum_x": 1.0e9,
            "sum_y": 4.0e10,
            "sum_xx": 1.0e18,
            "sum_xy": 4.0e19,
            "cost_trend": 1.02,
            "last_block_cost": 80000.0,
            "fit_decay": 0.94,
        }
        profile = StageProfile(
            job_id="job",
            stage_id="stage-1",
            op_name="op",
            safe_batch_size=8,
            oom_upper_bound=None,
            observed_at_ms=1,
            cost_model=snapshot,
        )
        merged = ControlService._merge_stage_profile(None, profile)
        self.assertEqual(merged.cost_model, snapshot)
        richer = StageProfile(
            job_id="job",
            stage_id="stage-1",
            op_name="op",
            safe_batch_size=8,
            oom_upper_bound=None,
            observed_at_ms=2,
            cost_model={**snapshot, "samples": 9.0},
        )
        merged = ControlService._merge_stage_profile(merged, richer)
        self.assertEqual(merged.cost_model["samples"], 9.0, "keep the most-evidenced model")

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/profiles.json"
            save_stage_profiles(path, [merged], "fingerprint-abc")
            restored = load_stage_profiles(path, "fingerprint-abc")
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].cost_model["samples"], 9.0)

    def test_oom_window_does_not_contaminate_model(self):
        """A failed attempt's partial-allocation peak must not poison the fit.

        Without the OOM-window reset, the retry's success sample carries the
        failed attempt's inflated peak, biasing b up and collapsing the
        prediction to the minimum batch size.
        """
        state, probe, run_slice = make_world()
        c = CostAwareController(
            initial_batch_size=32,
            max_batch_size=32,
            peak_memory_probe=probe,
            total_bytes=100_000_000,
            min_model_samples=3,
        )
        state["cost"] = 80_000.0
        # Warm the model inside the feasible region.
        for _ in range(4):
            c.observe_batch_cost(state["cost"])
            self.assertTrue(run_slice(c, c.next_batch_size(10_000)))
        # Push cost up until one OOM happens, then retry like the harness.
        ooms = 0
        for _ in range(20):
            state["cost"] *= 1.06
            c.observe_batch_cost(state["cost"])
            size = c.next_batch_size(10_000)
            if not run_slice(c, size):
                ooms += 1
                size = c.next_batch_size(10_000)
                self.assertTrue(run_slice(c, size))
                recovered = c.current_batch_size
        self.assertGreater(ooms, 0)
        self.assertGreaterEqual(
            recovered, 8, "model must not collapse to the minimum after an OOM"
        )
        coefficients = c._model_coefficients()
        self.assertIsNotNone(coefficients, "model must stay usable after an OOM")
        self.assertGreater(coefficients[1], 0)


if __name__ == "__main__":
    unittest.main()
