"""Unit tests for preflight probing, throughput-aware planning and the
partition-boundary replan controller.  Everything here is GPU-free."""

import json
import os
import tempfile
import unittest

from data_juicer.core.elasticjuicer.config_contract import (
    effective_elastic_juicer_config,
    validate_elastic_juicer_config,
)
from data_juicer.core.elasticjuicer.gpu_probe import (
    PROBE_RESULTS_SCHEMA_VERSION,
    ProbeParams,
    StageProbeResult,
    StageProbeSpec,
    _column_batches,
    _cost_stats,
    _cycled_batch,
    atomic_write_json,
    load_probe_results,
    probe_fingerprint,
    resolve_probe_cache,
    save_probe_results,
)
from data_juicer.core.elasticjuicer.parallelism_controller import (
    PartitionReplanController,
    ReplanConfig,
    StageObservation,
    aggregate_stage_observations,
)
from data_juicer.core.elasticjuicer.parallelism_planner import (
    ClusterSpec,
    PlanInfeasibleError,
    StageDemand,
    plan_parallelism,
)


class _FakeOp:
    _batched_op = True

    def __init__(self, scale=1):
        self.scale = scale

    def process_batched(self, samples):
        return {"sample_id": list(samples["sample_id"])}

    def use_ray_actor(self):
        return True


def _spec(name="fake_op", scale=1, batch_size=8):
    return StageProbeSpec(
        stage_key=f"stage:{name}:{scale}",
        op_name=name,
        op_class=_FakeOp,
        init_args=(),
        init_kwargs={"scale": scale},
        batch_size=batch_size,
    )


def _result(stage_key="stage:fake_op:1", fingerprint="fp", peak=1_000_000, rps=100.0, batch_size=8):
    return StageProbeResult(
        stage_key=stage_key,
        op_name="fake_op",
        fingerprint=fingerprint,
        gpu_name="FakeGPU",
        resident_bytes=100_000,
        effective_peak_bytes=peak,
        peak_bytes_with_headroom=int(peak * 1.1),
        throughput_rows_per_sec=rps,
        probed_batch_size=batch_size,
        oom_upper_bound=None,
        cost_mean=4.0,
        cost_p50=4.0,
        cost_p95=9.0,
        cost_samples=64,
        probe_duration_sec=1.0,
        rows_probed=32,
    )


class ProbeFingerprintAndCacheTest(unittest.TestCase):
    def test_fingerprint_changes_with_op_config(self):
        params = ProbeParams()
        base = probe_fingerprint("op", (), {"a": 1}, "GPU", params)
        self.assertEqual(base, probe_fingerprint("op", (), {"a": 1}, "GPU", params))
        self.assertNotEqual(base, probe_fingerprint("op", (), {"a": 2}, "GPU", params))
        self.assertNotEqual(base, probe_fingerprint("op", (), {"a": 1}, "GPU2", params))
        self.assertNotEqual(base, probe_fingerprint("op2", (), {"a": 1}, "GPU", params))
        other = ProbeParams(steady_batches=5)
        self.assertNotEqual(base, probe_fingerprint("op", (), {"a": 1}, "GPU", other))

    def test_probe_params_validation(self):
        with self.assertRaises(ValueError):
            ProbeParams(steady_batches=0).validate()
        with self.assertRaises(ValueError):
            ProbeParams(batch_cap_multiplier=0).validate()
        with self.assertRaises(ValueError):
            ProbeParams(memory_headroom=2.0).validate()

    def test_cache_hit_requires_matching_fingerprint(self):
        params = ProbeParams()
        spec = _spec()
        fingerprint = probe_fingerprint(spec.op_name, spec.init_args, spec.init_kwargs, "FakeGPU", params)
        hit = _result(stage_key=spec.stage_key, fingerprint=fingerprint)
        cached = {"schema_version": PROBE_RESULTS_SCHEMA_VERSION, "stages": {spec.stage_key: hit.to_dict()}}
        hits, pending = resolve_probe_cache([spec], params, cached)
        self.assertIn(spec.stage_key, hits)
        self.assertTrue(hits[spec.stage_key].cache_hit)
        self.assertEqual(pending, [])

        stale = _result(stage_key=spec.stage_key, fingerprint="other")
        cached["stages"][spec.stage_key] = stale.to_dict()
        hits, pending = resolve_probe_cache([spec], params, cached)
        self.assertEqual(hits, {})
        self.assertEqual([s.stage_key for s in pending], [spec.stage_key])

    def test_save_and_load_roundtrip_with_corruption_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gpu_probe_results.json")
            save_probe_results(path, {"k": _result(stage_key="k")})
            loaded = load_probe_results(path)
            self.assertEqual(loaded["schema_version"], PROBE_RESULTS_SCHEMA_VERSION)
            self.assertIn("k", loaded["stages"])
            with open(path, "w") as handle:
                handle.write("{not json")
            self.assertIsNone(load_probe_results(path))

    def test_atomic_write_leaves_no_tmp_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            atomic_write_json(path, {"a": 1})
            self.assertEqual(sorted(os.listdir(tmp)), ["out.json"])
            with open(path) as handle:
                self.assertEqual(json.load(handle), {"a": 1})

    def test_batch_helpers(self):
        columns = {"x": list(range(10))}
        batches = _column_batches(columns, 4)
        self.assertEqual([len(b["x"]) for b in batches], [4, 4, 2])
        cycled = _cycled_batch(columns, 25)
        self.assertEqual(len(cycled["x"]), 25)
        self.assertEqual(cycled["x"][:10], list(range(10)))
        mean, p50, p95, count = _cost_stats([1, 2, 3, None, 4])
        self.assertEqual(count, 4)
        self.assertAlmostEqual(mean, 2.5)


class PlannerTest(unittest.TestCase):
    def _cluster(self, num_devices=2, mem_gib=8, cpus=16.0):
        return ClusterSpec(
            total_cpus=cpus,
            num_devices=num_devices,
            device_memory_bytes=mem_gib * 1024**3,
            device_memory_budget_fraction=1.0,
        )

    def _demand(self, key, rps=100.0, peak_mib=1024, batch_size=8, batches=100, num_gpus=0.25, num_cpus=1.0):
        return StageDemand(
            stage_key=key,
            op_name=key,
            probed_throughput_rps=rps,
            effective_peak_bytes=peak_mib * 1024**2,
            batch_size=batch_size,
            available_batches=batches,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
        )

    def test_minimum_plan_and_bottleneck_fill(self):
        cluster = self._cluster()
        slow = self._demand("slow", rps=10.0)
        fast = self._demand("fast", rps=500.0)
        plan = plan_parallelism([fast, slow], cluster, max_replicas_per_device=4)
        self.assertGreater(plan.stages["slow"].planned_actors, plan.stages["fast"].planned_actors)
        total = sum(stage.planned_actors for stage in plan.stages.values())
        self.assertGreaterEqual(total, 2)

    def test_memory_constraint_caps_actors(self):
        cluster = self._cluster(num_devices=1, mem_gib=2)
        demand = self._demand("big", peak_mib=1024)  # 1 GiB peak; 2 GiB device
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=8)
        self.assertEqual(plan.stages["big"].planned_actors, 2)

    def test_replica_limit_caps_actors(self):
        cluster = self._cluster(mem_gib=64)
        demand = self._demand("tiny", peak_mib=16)
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=2)
        self.assertEqual(plan.stages["tiny"].planned_actors, 4)  # 2 devices x 2

    def test_cpu_budget_caps_actors(self):
        cluster = self._cluster(mem_gib=64, cpus=3.0)
        demand = self._demand("cpu", peak_mib=16, num_cpus=1.0)
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=16)
        self.assertEqual(plan.stages["cpu"].planned_actors, 3)

    def test_available_batches_cap_actors(self):
        cluster = self._cluster(mem_gib=64)
        demand = self._demand("fewbatches", peak_mib=16, batches=2)
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=16)
        self.assertEqual(plan.stages["fewbatches"].planned_actors, 2)

    def test_minimum_plan_infeasible_fails_fast(self):
        cluster = self._cluster(mem_gib=1)
        demand = self._demand("toobig", peak_mib=2048)
        with self.assertRaises(PlanInfeasibleError):
            plan_parallelism([demand], cluster)

    def test_explicit_stage_is_fixed(self):
        cluster = self._cluster(mem_gib=64)
        demand = self._demand("pinned", peak_mib=16)
        demand.explicit = True
        demand.explicit_actors = 3
        other = self._demand("auto", rps=1.0, peak_mib=16)
        plan = plan_parallelism([demand, other], cluster, max_replicas_per_device=8)
        self.assertEqual(plan.stages["pinned"].planned_actors, 3)
        self.assertTrue(plan.stages["pinned"].fixed)

    def test_batch_hard_cap_honors_oom_bound(self):
        cluster = self._cluster(mem_gib=64)
        demand = self._demand("oombound", peak_mib=16, batch_size=8)
        demand.oom_upper_bound = 16
        plan = plan_parallelism([demand], cluster)
        self.assertLessEqual(plan.stages["oombound"].batch_hard_cap, 15)

    def test_planning_is_deterministic(self):
        cluster = self._cluster(mem_gib=8)
        demands = [self._demand(f"s{i}", rps=100.0 - i) for i in range(4)]
        first = plan_parallelism(list(demands), cluster, max_replicas_per_device=2)
        second = plan_parallelism(list(demands), cluster, max_replicas_per_device=2)
        self.assertEqual(first.to_dict()["stages"], second.to_dict()["stages"])


def _event(stage="stage:a", partition=0, batch_size=8, peak_mb=100.0, throughput=50.0, ok=True, error=None):
    class _Snap:
        pass

    class _Cuda:
        pass

    cuda = _Cuda()
    cuda.peak_allocated_mb = peak_mb
    snapshot = _Snap()
    snapshot.batch_size = batch_size
    snapshot.throughput = throughput
    snapshot.cuda = cuda
    snapshot.succeeded = ok
    snapshot.error_type = error

    class _Event:
        pass

    event = _Event()
    event.stage_id = stage
    event.op_name = stage.split(":")[-1]
    event.partition_id = partition
    event.snapshot = snapshot
    return event


class ObservationAggregationTest(unittest.TestCase):
    def test_partition_filter_and_oom_counting(self):
        events = [
            _event(partition=0, peak_mb=100, throughput=50),
            _event(partition=0, peak_mb=300, throughput=60),
            _event(partition=1, peak_mb=999, throughput=99),
            _event(partition=0, ok=False, error="OutOfMemoryError", throughput=0),
        ]
        observations = aggregate_stage_observations(events, partition_id=0)
        self.assertIn("stage:a", observations)
        observation = observations["stage:a"]
        self.assertEqual(observation.oom_batches, 1)
        self.assertEqual(observation.succeeded_batches, 2)
        # median of [50, 60]
        self.assertEqual(observation.rows_per_sec, 60.0)
        # median bytes/row of [12.5MiB/8, 37.5MiB/8]
        self.assertIsNotNone(observation.bytes_per_row)

    def test_empty_when_no_matching_partition(self):
        events = [_event(partition=5)]
        self.assertEqual(aggregate_stage_observations(events, partition_id=0), {})


class ReplanControllerTest(unittest.TestCase):
    def _controller(self, *, events_path=None, drift_threshold=1.3, epsilon=0.03, cooldown=1):
        cluster = ClusterSpec(total_cpus=16, num_devices=1, device_memory_bytes=8 * 1024**3)
        demand = StageDemand(
            stage_key="stage:a",
            op_name="op_a",
            probed_throughput_rps=100.0,
            effective_peak_bytes=1 * 1024**3,
            batch_size=8,
            available_batches=100,
            num_gpus=0.25,
            num_cpus=1.0,
        )
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=4)
        # force a deterministic 3-actor starting point for the rules below
        plan.stages["stage:a"].planned_actors = 3
        plan.stages["stage:a"].device_ids = [0, 0, 0]
        controller = PartitionReplanController(
            plan,
            {"stage:a": demand},
            cluster,
            ReplanConfig(drift_threshold=drift_threshold, marginal_gain_epsilon=epsilon, cooldown_partitions=cooldown),
            baseline_bytes_per_row={"stage:a": 100.0 * 1024**2 / 8},
            events_path=events_path,
        )
        return controller, plan

    def _obs(self, bytes_per_row, rps=90.0, oom=0):
        observation = StageObservation(stage_key="stage:a", succeeded_batches=10, oom_batches=oom, sample_count=10)
        observation.rows_per_sec = rps
        observation.bytes_per_row = bytes_per_row
        return {"stage:a": observation}

    def test_drift_up_triggers_scale_down_and_budget_redistribution(self):
        controller, plan = self._controller()
        baseline_per_row = 100.0 * 1024**2 / 8
        before_cap = plan.stages["stage:a"].batch_hard_cap
        changes = controller.on_partition_complete(0, self._obs(baseline_per_row * 2.0, oom=1), goodput_rows_per_sec=100)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].action, "scale_down")
        self.assertEqual(changes[0].actors_after, 2)
        self.assertGreaterEqual(plan.stages["stage:a"].batch_hard_cap, before_cap)

    def test_no_change_without_drift(self):
        controller, plan = self._controller()
        baseline_per_row = 100.0 * 1024**2 / 8
        changes = controller.on_partition_complete(0, self._obs(baseline_per_row * 1.1), goodput_rows_per_sec=100)
        self.assertEqual(changes, [])
        self.assertEqual(plan.stages["stage:a"].planned_actors, 3)

    def test_scale_up_then_revert_on_low_marginal_gain(self):
        controller, plan = self._controller(epsilon=0.05)
        plan.stages["stage:a"].planned_actors = 1
        plan.stages["stage:a"].device_ids = [0]
        observation = self._obs(bytes_per_row=None, rps=10.0)  # far below 60% of probed 100
        changes = controller.on_partition_complete(0, observation, goodput_rows_per_sec=10.0)
        self.assertEqual([c.action for c in changes], ["scale_up"])
        self.assertEqual(plan.stages["stage:a"].planned_actors, 2)
        # next partition: goodput barely moved -> revert and freeze
        changes = controller.on_partition_complete(1, observation, goodput_rows_per_sec=10.2)
        self.assertEqual([c.action for c in changes], ["revert_scale_up"])
        self.assertEqual(plan.stages["stage:a"].planned_actors, 1)
        # frozen: no further changes even under identical pressure
        changes = controller.on_partition_complete(3, observation, goodput_rows_per_sec=10.0)
        self.assertEqual(changes, [])

    def test_scale_up_kept_when_gain_meets_epsilon(self):
        controller, plan = self._controller(epsilon=0.05)
        plan.stages["stage:a"].planned_actors = 1
        plan.stages["stage:a"].device_ids = [0]
        observation = self._obs(bytes_per_row=None, rps=10.0)
        controller.on_partition_complete(0, observation, goodput_rows_per_sec=10.0)
        changes = controller.on_partition_complete(2, observation, goodput_rows_per_sec=12.0)
        self.assertEqual(changes, [])  # gain 20% >= 5% -> kept, no event
        self.assertEqual(plan.stages["stage:a"].planned_actors, 2)

    def test_cooldown_blocks_immediate_second_change(self):
        controller, plan = self._controller(cooldown=2)
        baseline_per_row = 100.0 * 1024**2 / 8
        controller.on_partition_complete(0, self._obs(baseline_per_row * 2.0, oom=1), goodput_rows_per_sec=100)
        self.assertEqual(plan.stages["stage:a"].planned_actors, 2)
        changes = controller.on_partition_complete(1, self._obs(baseline_per_row * 2.0, oom=1), goodput_rows_per_sec=100)
        self.assertEqual(changes, [])  # still inside cooldown window
        self.assertEqual(plan.stages["stage:a"].planned_actors, 2)

    def test_plan_events_are_appended(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plan_events.jsonl")
            controller, _plan = self._controller(events_path=path)
            baseline_per_row = 100.0 * 1024**2 / 8
            controller.on_partition_complete(0, self._obs(baseline_per_row * 2.0, oom=1), goodput_rows_per_sec=100)
            with open(path) as handle:
                lines = [json.loads(line) for line in handle]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["action"], "scale_down")
            self.assertEqual(lines[0]["partition_id"], 0)


class AutoParallelismContractTest(unittest.TestCase):
    def test_sub_options_require_main_flag(self):
        with self.assertRaises(ValueError):
            validate_elastic_juicer_config({"elastic_juicer_probe_enabled": False})
        with self.assertRaises(ValueError):
            validate_elastic_juicer_config({"elastic_juicer_replan_enabled": False})
        # main flag on: defaults validate cleanly
        validate_elastic_juicer_config({"elastic_juicer_auto_parallelism": True})

    def test_job_id_required_when_asked(self):
        with self.assertRaises(ValueError):
            validate_elastic_juicer_config({"elastic_juicer_auto_parallelism": True}, require_job_id=True)
        validate_elastic_juicer_config(
            {"elastic_juicer_auto_parallelism": True, "job_id": "j1"}, require_job_id=True
        )

    def test_invalid_numeric_options_rejected(self):
        with self.assertRaises(ValueError):
            validate_elastic_juicer_config(
                {"elastic_juicer_auto_parallelism": True, "elastic_juicer_drift_threshold": 0.9}
            )
        with self.assertRaises(ValueError):
            validate_elastic_juicer_config(
                {"elastic_juicer_auto_parallelism": True, "elastic_juicer_max_gpu_workers_per_device": 0}
            )

    def test_effective_snapshot_exposes_auto_parallelism(self):
        snapshot = effective_elastic_juicer_config({"elastic_juicer_auto_parallelism": True})
        self.assertTrue(snapshot["auto_parallelism"])
        self.assertTrue(snapshot["probe_enabled"])
        self.assertTrue(snapshot["replan_enabled"])


if __name__ == "__main__":
    unittest.main()
