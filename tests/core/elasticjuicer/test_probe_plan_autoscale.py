"""Unit tests for preflight probing, throughput-aware planning and the
partition-boundary replan controller.  Everything here is GPU-free."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.core.elasticjuicer.config_contract import (
    effective_elastic_juicer_config,
    validate_elastic_juicer_config,
)
from data_juicer.core.elasticjuicer.gpu_probe import (
    PROBE_RESULTS_FILENAME,
    PROBE_RESULTS_SCHEMA_VERSION,
    ProbeParams,
    StageProbeResult,
    StageProbeSpec,
    _column_batches,
    _cost_stats,
    _cycled_batch,
    atomic_write_json,
    dataset_cache_usable,
    dataset_snapshot_fingerprint,
    load_probe_results,
    probe_fingerprint,
    probe_gpu_stages,
    probe_stage_on_worker,
    resolve_probe_cache,
    sample_columns_digest,
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
    RuntimePlan,
    StageDemand,
    StagePlan,
    _batch_hard_cap,
    plan_parallelism,
    plan_envelope_fingerprint,
)
from data_juicer.core.elasticjuicer.performance_window import PerformanceWindow


class _StationaryFixtureController(PartitionReplanController):
    """These legacy allocation tests explicitly assume stationary input/resources.

    New comparability tests use the production class directly, including absent
    evidence. Do not restore an implicit adjacent-partition fallback in production.
    """

    def on_partition_complete(self, partition_id, observations, goodput_rows_per_sec):
        for observation in observations.values():
            observation.performance_window = PerformanceWindow(
                partition_id, plan_envelope_fingerprint(self.plan), {"resource": "fixed"},
                {"fixture_cost": 1.0}, goodput_rows_per_sec, 10, 80, partition_id * 10 + 1, partition_id * 10 + 9, [0.1],
            )
        return super().on_partition_complete(partition_id, observations, goodput_rows_per_sec)


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

    def test_fractional_gpu_budget_caps_per_actor_share(self):
        # Ray packs fractional-GPU actors up to 1/num_gpus per device no
        # matter where the plan lists them, so the honest per-actor budget
        # is the fractional share of the device budget -- never the whole
        # device.  (The W2' smoke crashed on a whole-device 86 GiB budget
        # while four co-located actors each really owned ~21.5 GiB.)
        cluster = self._cluster(num_devices=8, mem_gib=8, cpus=64.0)
        demand = self._demand("quarter", peak_mib=16, batch_size=8, num_gpus=0.25)
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=4)
        stage = plan.stages["quarter"]
        self.assertGreaterEqual(stage.planned_actors, 1)
        entitlement = int(cluster.device_memory_budget_bytes * 0.25)
        self.assertLessEqual(stage.per_actor_vram_budget_bytes, entitlement)

    def test_full_gpu_actor_keeps_whole_device_budget(self):
        # num_gpus=1.0 actors keep the whole-device share: the fractional
        # entitlement must not change classic full-GPU planning.
        cluster = self._cluster(num_devices=2, mem_gib=8)
        demand = self._demand("whole", peak_mib=16, batch_size=8, num_gpus=1.0)
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=1)
        stage = plan.stages["whole"]
        self.assertEqual(
            stage.per_actor_vram_budget_bytes, cluster.device_memory_budget_bytes
        )

    def test_budget_never_drops_below_probed_peak(self):
        cluster = self._cluster(num_devices=2, mem_gib=8)
        # probed peak 4 GiB exceeds the 2 GiB fractional entitlement: the
        # stage must never be told it owns less than what it measured.
        demand = self._demand("heavy", peak_mib=4096, batch_size=8, num_gpus=0.25)
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=4)
        stage = plan.stages["heavy"]
        self.assertEqual(stage.per_actor_vram_budget_bytes, 4 * 1024**3)

    def test_planning_gates_record_fractional_entitlement(self):
        # The plan is auditable evidence: it must state the entitlement the
        # budget was derived from (contract 5.1 plan outputs).
        cluster = self._cluster(num_devices=2, mem_gib=8)
        demand = self._demand("audited", peak_mib=16, batch_size=8, num_gpus=0.25)
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=4)
        gates = plan.planning_gates["per_stage"]["audited"]
        self.assertEqual(
            gates["per_actor_entitlement_bytes"], int(cluster.device_memory_budget_bytes * 0.25)
        )

    def test_plan_envelope_fingerprint_binds_resource_semantics(self):
        # Contract C3's execution-plan dimension: every plan output that
        # shapes profile learning (actors, placement, budget, hard cap,
        # replica limit, cluster) must change the fingerprint.
        from data_juicer.core.elasticjuicer.parallelism_planner import plan_envelope_fingerprint

        cluster = self._cluster(num_devices=2, mem_gib=8)
        demand = self._demand("fp", peak_mib=16, batch_size=8, num_gpus=0.25)
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=4)
        base = plan_envelope_fingerprint(plan)
        # deterministic for an identical plan
        self.assertEqual(base, plan_envelope_fingerprint(plan))

        stage = plan.stages["fp"]
        for mutate, restore in [
            ("per_actor_vram_budget_bytes", stage.per_actor_vram_budget_bytes),
            ("batch_hard_cap", stage.batch_hard_cap),
            ("planned_actors", stage.planned_actors),
        ]:
            setattr(stage, mutate, getattr(stage, mutate) + 1)
            self.assertNotEqual(base, plan_envelope_fingerprint(plan), mutate)
            setattr(stage, mutate, restore)
        self.assertEqual(base, plan_envelope_fingerprint(plan))
        # a different cluster is a different envelope
        other = plan_parallelism(
            [self._demand("fp", peak_mib=16, batch_size=8, num_gpus=0.25)],
            self._cluster(num_devices=4, mem_gib=8),
            max_replicas_per_device=4,
        )
        self.assertNotEqual(base, plan_envelope_fingerprint(other))

    def test_planning_is_deterministic(self):
        cluster = self._cluster(mem_gib=8)
        demands = [self._demand(f"s{i}", rps=100.0 - i) for i in range(4)]
        first = plan_parallelism(list(demands), cluster, max_replicas_per_device=2)
        second = plan_parallelism(list(demands), cluster, max_replicas_per_device=2)
        self.assertEqual(first.to_dict()["stages"], second.to_dict()["stages"])

    def test_fixture_batches_do_not_over_provision_actors(self):
        # The 4096-row smoke fixture: 8 partitions x batch 16 -> 32 batches
        # per partition. The old planner provisioned one actor per batch
        # (32 actors vs the 8-actor static baseline, ~56% slower); the
        # economics gates must stop growth exactly at the baseline.
        cluster = self._cluster(num_devices=8, mem_gib=80, cpus=256.0)
        demand = self._demand("fixture", peak_mib=256, batch_size=16, batches=32)
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=4)
        self.assertEqual(plan.stages["fixture"].planned_actors, 8)

    def test_startup_amortization_floor_caps_actors(self):
        # batch_seconds = 8 rows / 8 rows-per-sec = 1s, so an actor owning
        # fewer than ceil(0.10 * 100s / 1s) = 10 batches cannot amortize its
        # 100s startup; ceil(32 / 10) -> at most 3 actors.
        cluster = self._cluster(num_devices=2, mem_gib=64, cpus=256.0)
        demand = self._demand("slowstart", rps=8.0, peak_mib=16, batch_size=8, batches=32)
        demand.actor_startup_seconds = 100.0
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=8)
        self.assertEqual(plan.stages["slowstart"].planned_actors, 3)

    def test_relaxed_gates_recover_one_actor_per_batch(self):
        # Opting out of both floors restores the old over-provisioning
        # behavior, proving the caps are gate-driven, not hardcoded.
        cluster = self._cluster(num_devices=8, mem_gib=80, cpus=256.0)
        demand = self._demand("relaxed", peak_mib=256, batch_size=16, batches=32)
        plan = plan_parallelism(
            [demand],
            cluster,
            max_replicas_per_device=4,
            min_batches_per_actor=1,
            min_marginal_makespan_gain=0.0,
        )
        self.assertEqual(plan.stages["relaxed"].planned_actors, 32)

    def test_planning_gates_are_recorded(self):
        cluster = self._cluster(mem_gib=64)
        demand = self._demand("gates", peak_mib=16, batches=32)
        demand.actor_startup_seconds = 2.0
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=4)
        gates = plan.planning_gates
        self.assertEqual(gates["min_batches_per_actor"], 4)
        self.assertEqual(gates["startup_amortization_ratio"], 0.10)
        self.assertEqual(gates["min_marginal_makespan_gain"], 0.10)
        stage = gates["per_stage"]["gates"]
        self.assertEqual(stage["actor_startup_seconds"], 2.0)
        self.assertGreaterEqual(stage["per_actor_batch_floor"], 1)
        # the persisted runtime plan exposes the gates for audits
        self.assertIn("planning_gates", plan.to_dict())


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
    def test_structured_oom_classification_matches_the_local_retry_path(self):
        event = _event(ok=False, error="RuntimeError")
        event.snapshot.oom = True
        observation = aggregate_stage_observations([event])["stage:a"]
        self.assertEqual(observation.oom_batches, 1)

    def test_resident_model_memory_does_not_turn_batch_changes_into_drift(self):
        costs = []
        for size in (16, 1):
            event = _event(batch_size=size, peak_mb=(8 + 0.1 * size) * 1024)
            observation = aggregate_stage_observations(
                [event], resident_bytes_by_stage={"stage:a": 8 * 1024**3}
            )["stage:a"]
            costs.append(observation.bytes_per_row)
        self.assertAlmostEqual(costs[0] / 1024**3, 0.1)
        self.assertAlmostEqual(costs[1] / 1024**3, 0.1)

    def test_missing_or_incompatible_resident_measurement_disables_memory_drift(self):
        for residents in ({}, {"stage:a": 200 * 1024**2}):
            with self.subTest(residents=residents):
                result = aggregate_stage_observations(
                    [_event(peak_mb=100)], resident_bytes_by_stage=residents
                )["stage:a"]
                self.assertIsNone(result.bytes_per_row)
                self.assertEqual(result.succeeded_batches, 1)

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



class ResidentMemoryModelTest(unittest.TestCase):
    def test_probe_samples_allocator_residency_after_lazy_warmup(self):
        cuda = MagicMock()
        cuda.get_device_name.return_value = "FakeGPU"
        cuda.mem_get_info.return_value = (100_000, 200_000)
        cuda.memory_allocated.return_value = 0
        cuda.max_memory_allocated.return_value = 9600

        class LazyModel(_FakeOp):
            def process_batched(self, samples):
                cuda.memory_allocated.return_value = 8000
                return super().process_batched(samples)

        with patch("torch.cuda", cuda):
            result = probe_stage_on_worker(
                LazyModel, (), {}, {"sample_id": list(range(16))}, 16,
                {"warmup_batches": 1, "steady_batches": 1, "batch_cap_multiplier": 1},
            )
        self.assertEqual(result["allocator_resident_bytes"], 8000)
        self.assertEqual(result["allocator_peak_bytes"], 9600)
        self.assertEqual(result["resident_bytes"], 0)  # Different diagnostic scope.

    def test_legacy_cache_is_rejected_instead_of_reusing_total_peak_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, PROBE_RESULTS_FILENAME)
            atomic_write_json(path, {"schema_version": 2, "stages": {"k": _result().to_dict()}})
            self.assertIsNone(load_probe_results(path))

    def test_budget_reserves_resident_memory_before_sizing_working_batch(self):
        demand = StageDemand(
            stage_key="s", op_name="model", probed_throughput_rps=100,
            effective_peak_bytes=9600, resident_bytes=8000,
            batch_size=16, available_batches=100,
        )
        self.assertEqual(demand.per_row_vram_bytes, 100)
        self.assertEqual(_batch_hard_cap(demand, 10000), 20)
        self.assertEqual(_batch_hard_cap(demand, 8100), 1)

    def test_zero_working_slope_does_not_authorize_unbounded_growth(self):
        demand = StageDemand(
            stage_key="s", op_name="model", probed_throughput_rps=100,
            effective_peak_bytes=8000, resident_bytes=8000,
            batch_size=16, available_batches=100,
        )
        self.assertEqual(_batch_hard_cap(demand, 10000), 16)

    def test_probe_preserves_headroom_on_resident_and_working_memory(self):
        result = _result(peak=9600, batch_size=16)
        result.allocator_resident_bytes = 8000
        result.allocator_peak_bytes = 9600
        self.assertEqual(result.resident_bytes_with_headroom, 8800)
        self.assertEqual(result.working_bytes_per_row, 100)
        self.assertAlmostEqual(result.memory_headroom_factor, 1.1)

    def test_replan_budget_pressure_includes_resident_memory(self):
        cluster = ClusterSpec(total_cpus=4, num_devices=1, device_memory_bytes=10000, device_memory_budget_fraction=1)
        demand = StageDemand(
            stage_key="s", op_name="model", probed_throughput_rps=100,
            effective_peak_bytes=9600, resident_bytes=8000,
            batch_size=16, available_batches=100,
        )
        stage = StagePlan(stage_key="s", op_name="model", planned_actors=1,
                          device_ids=[0], per_actor_vram_budget_bytes=10000, batch_hard_cap=20)
        controller = _StationaryFixtureController(
            RuntimePlan(stages={"s": stage}, cluster=cluster), {"s": demand}, cluster,
            ReplanConfig(), baseline_bytes_per_row={"s": 100},
        )
        observations = {"s": StageObservation(stage_key="s", bytes_per_row=200, sample_count=10)}
        changes = controller.on_partition_complete(0, observations, 10)
        self.assertEqual([change.action for change in changes], ["scale_down"])
        self.assertEqual(stage.batch_hard_cap, 10)


class ReplanGpuFeasibilityTest(unittest.TestCase):
    def _controller(self, allocations, *, devices=1, fraction_budget=1.0):
        cluster = ClusterSpec(
            total_cpus=64,
            num_devices=devices,
            device_memory_bytes=64 * 1024**3,
            gpu_fraction_budget=fraction_budget,
        )
        demands = {}
        stages = {}
        for key, fraction, placement in allocations:
            demands[key] = StageDemand(
                stage_key=key, op_name=key, probed_throughput_rps=100,
                effective_peak_bytes=1024**2, batch_size=8,
                available_batches=100, num_gpus=fraction,
            )
            stages[key] = StagePlan(
                stage_key=key, op_name=key, planned_actors=len(placement),
                device_ids=list(placement), batch_hard_cap=8,
            )
        plan = RuntimePlan(stages=stages, cluster=cluster, max_replicas_per_device=16)
        return _StationaryFixtureController(plan, demands, cluster, ReplanConfig())

    def test_full_gpu_plan_cannot_add_a_fifth_actor_on_four_devices(self):
        controller = self._controller([("s", 1.0, [0, 1, 2, 3])], devices=4)
        observations = {"s": StageObservation(stage_key="s", rows_per_sec=10, sample_count=10)}

        changes = controller.on_partition_complete(0, observations, 10)

        self.assertEqual(changes, [])
        self.assertEqual(controller.plan.stages["s"].planned_actors, 4)

    def test_gpu_fraction_is_accounted_across_heterogeneous_stages(self):
        controller = self._controller([("large", 0.75, [0]), ("small", 0.25, [0, 1])], devices=2)
        self.assertEqual(controller._device_candidates(controller.demands["small"]), [1])

    def test_configured_fraction_budget_and_floating_point_boundary(self):
        for used, budget, expected in [(4, 0.5, [0]), (5, 0.5, []), (9, 1.0, [0]), (10, 1.0, [])]:
            with self.subTest(used=used, budget=budget):
                controller = self._controller([("s", 0.1, [0] * used)], fraction_budget=budget)
                self.assertEqual(controller._device_candidates(controller.demands["s"]), expected)

    def test_scale_up_and_revert_release_the_gpu_reservation(self):
        controller = self._controller([("s", 0.5, [0])])
        observations = {"s": StageObservation(stage_key="s", rows_per_sec=10, sample_count=10)}
        controller.on_partition_complete(0, observations, 10)
        self.assertEqual(controller._device_candidates(controller.demands["s"]), [])

        controller.on_partition_complete(1, observations, 10)
        self.assertEqual(controller._device_candidates(controller.demands["s"]), [0])


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
        controller = _StationaryFixtureController(
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

    def test_recompute_budget_respects_fractional_entitlement(self):
        # Boundary replanning must redistribute budget against the actor's
        # fractional share too, or drift-time hard limits stay 4x too
        # loose and never bind (the W2' no-op redistribution).
        controller, plan = self._controller()
        stage_plan = plan.stages["stage:a"]
        demand = controller.demands["stage:a"]
        entitlement = int(controller.cluster.device_memory_budget_bytes * demand.num_gpus)
        controller._recompute_budget_and_cap("stage:a")
        self.assertLessEqual(stage_plan.per_actor_vram_budget_bytes, entitlement)
        self.assertGreaterEqual(stage_plan.batch_hard_cap, 1)

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

    def test_cooldown_does_not_block_safety_adjustment(self):
        controller, plan = self._controller(cooldown=2)
        baseline_per_row = 100.0 * 1024**2 / 8
        controller.on_partition_complete(0, self._obs(baseline_per_row * 2.0, oom=1), goodput_rows_per_sec=100)
        self.assertEqual(plan.stages["stage:a"].planned_actors, 2)
        changes = controller.on_partition_complete(1, self._obs(baseline_per_row * 2.0, oom=1), goodput_rows_per_sec=100)
        self.assertEqual([change.action for change in changes], ["scale_down"])
        self.assertEqual(plan.stages["stage:a"].planned_actors, 1)

    def test_failed_exploration_still_responds_to_later_drift(self):
        controller, plan = self._controller()
        slow = self._obs(bytes_per_row=None, rps=10)
        controller.on_partition_complete(0, slow, 10)
        controller.on_partition_complete(1, slow, 10)
        self.assertIn("stage:a", controller._frozen_stages)
        cap_before = plan.stages["stage:a"].batch_hard_cap

        changes = controller.on_partition_complete(
            2, self._obs(bytes_per_row=20 * 1024**3 / 8, oom=1), 5
        )

        self.assertEqual([change.action for change in changes], ["scale_down"])
        self.assertLess(plan.stages["stage:a"].batch_hard_cap, cap_before)
        self.assertIn("stage:a", controller._frozen_stages)

    def test_performance_exploration_still_obeys_cooldown(self):
        controller, _plan = self._wide_controller(device_ids=[0])
        controller.config = ReplanConfig(cooldown_partitions=3)
        slow = self._obs(bytes_per_row=None, rps=10)
        self.assertEqual(controller.on_partition_complete(0, slow, 10)[0].action, "scale_up")
        self.assertEqual(controller.on_partition_complete(1, slow, 12), [])
        self.assertEqual(controller.on_partition_complete(2, slow, 12), [])
        self.assertEqual(controller.on_partition_complete(4, slow, 12)[0].action, "scale_up")

    def test_safety_preempts_pending_performance_verdict(self):
        for goodput in (10, 20):
            with self.subTest(goodput=goodput):
                controller, plan = self._controller()
                controller.on_partition_complete(0, self._obs(None, rps=10), 10)
                changes = controller.on_partition_complete(
                    1, self._obs(bytes_per_row=20 * 1024**3 / 8, oom=1), goodput
                )
                self.assertEqual([change.action for change in changes], ["scale_down"])
                self.assertNotIn("stage:a", controller._pending_scale_up)
                self.assertLess(plan.stages["stage:a"].batch_hard_cap, 8)

    def test_safety_tightens_at_minimum_actor_count_and_preserves_explicit_count(self):
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                controller, plan = self._wide_controller(device_ids=[0])
                stage = plan.stages["stage:a"]
                stage.fixed = fixed
                before = stage.batch_hard_cap
                changes = controller.on_partition_complete(
                    0, self._obs(bytes_per_row=8 * 1024**3, oom=1), 5
                )
                self.assertEqual(stage.planned_actors, 1)
                self.assertEqual([change.action for change in changes], ["scale_down"])
                self.assertLess(stage.batch_hard_cap, before)

    def test_repeated_safety_observation_never_relaxes_an_earlier_cap(self):
        controller, plan = self._wide_controller(device_ids=[0])
        stage = plan.stages["stage:a"]
        controller.on_partition_complete(0, self._obs(8 * 1024**3, oom=1), 5)
        cap_before = stage.batch_hard_cap
        controller.on_partition_complete(1, self._obs(4 * 1024**3, oom=1), 5)
        self.assertLessEqual(stage.batch_hard_cap, cap_before)

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

    def _wide_controller(self, *, device_ids, events_path=None):
        num_devices = max(device_ids) + 1
        cluster = ClusterSpec(
            total_cpus=64.0,
            num_devices=num_devices,
            device_memory_bytes=80 * 1024**3,
            device_memory_budget_fraction=1.0,
        )
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
        plan = plan_parallelism([demand], cluster, max_replicas_per_device=8)
        plan.stages["stage:a"].planned_actors = len(device_ids)
        plan.stages["stage:a"].device_ids = list(device_ids)
        controller = _StationaryFixtureController(
            plan,
            {"stage:a": demand},
            cluster,
            ReplanConfig(),
            baseline_bytes_per_row={"stage:a": 100.0 * 1024**2 / 8},
            events_path=events_path,
        )
        # Re-derive the stored budget for the mutated layout so before/after
        # envelope comparisons reflect the actual placement.
        controller._recompute_budget_and_cap("stage:a")
        return controller, plan

    def test_drift_scale_down_is_device_level_envelope_step(self):
        # 8 devices x 3 replicas = 24 actors: one replica must come off
        # EVERY device (24 -> 16), never a cosmetic 24 -> 23 whose per-device
        # maxima and budget stay unchanged.
        device_ids = [device for device in range(8) for _ in range(3)]
        controller, plan = self._wide_controller(device_ids=device_ids)
        baseline_per_row = 100.0 * 1024**2 / 8
        before_budget = plan.stages["stage:a"].per_actor_vram_budget_bytes
        before_cap = plan.stages["stage:a"].batch_hard_cap
        changes = controller.on_partition_complete(
            0, self._obs(baseline_per_row * 2.0, oom=1), goodput_rows_per_sec=100
        )
        self.assertEqual(len(changes), 1)
        change = changes[0]
        self.assertEqual(change.action, "scale_down")
        self.assertEqual(change.actors_before, 24)
        self.assertEqual(change.actors_after, 16)
        self.assertEqual(change.stage_max_replicas_per_device_before, 3)
        self.assertEqual(change.stage_max_replicas_per_device_after, 2)
        self.assertEqual(change.device_ids_after.count(0), 2)
        self.assertTrue(change.effective)
        # Under fractional GPUs the budget is honestly capped at the
        # entitlement (Ray packs up to 1/num_gpus replicas per device no
        # matter where the plan lists them), so shedding replicas cannot
        # grow it beyond that share; the envelope still moved via the
        # per-device replica maxima and the actor count.
        entitlement = int(80 * 1024**3 * 0.25)
        self.assertLessEqual(change.per_actor_vram_budget_after, entitlement)
        self.assertGreaterEqual(change.per_actor_vram_budget_after, 1 * 1024**3)
        self.assertLessEqual(change.batch_hard_cap_after, entitlement)
        self.assertGreaterEqual(change.batch_hard_cap_after, 1)

    def test_drift_tightens_hard_cap_even_without_actor_shedding(self):
        # 8 actors, one per device: no actor-shedding step exists, but the
        # observed per-row VRAM (drift) must still tighten the batch hard
        # cap against the fractional entitlement -- otherwise the next
        # partition's co-located actors keep colliding at the probed-head
        # cap (the W2' floor OOMs: observed per-row 6.4 GiB, entitlement
        # 21.5 GiB, honest cap ~3, hard limit stayed 16).
        device_ids = list(range(8))
        controller, plan = self._wide_controller(device_ids=device_ids)
        stage_plan = plan.stages["stage:a"]
        cap_before = stage_plan.batch_hard_cap
        budget = stage_plan.per_actor_vram_budget_bytes
        observed_per_row = 4 * 1024**3
        changes = controller.on_partition_complete(
            3, self._obs(bytes_per_row=observed_per_row, oom=1), goodput_rows_per_sec=100
        )
        self.assertEqual(len(changes), 1)
        change = changes[0]
        self.assertTrue(change.effective)
        # the envelope moved, not the actor count
        self.assertEqual(change.actors_before, change.actors_after)
        expected_cap = int(budget // observed_per_row)
        self.assertEqual(change.batch_hard_cap_after, expected_cap)
        self.assertLess(change.batch_hard_cap_after, cap_before)
        self.assertEqual(stage_plan.batch_hard_cap, expected_cap)

    def test_drift_with_smaller_observed_per_row_never_loosens_cap(self):
        # Only drift-up may tighten: an observation below the probed
        # per-row cost must leave the cap alone (relaxation belongs to the
        # scale-up path, not to noise).
        device_ids = list(range(8))
        controller, plan = self._wide_controller(device_ids=device_ids)
        stage_plan = plan.stages["stage:a"]
        cap_before = stage_plan.batch_hard_cap
        small_observation = 25 * 1024 * 1024  # drifted (2x baseline) yet below the probed 128 MiB/row
        changes = controller.on_partition_complete(
            3, self._obs(bytes_per_row=small_observation, oom=1), goodput_rows_per_sec=100
        )
        self.assertEqual(len(changes), 1)
        self.assertFalse(changes[0].effective)
        self.assertEqual(stage_plan.batch_hard_cap, cap_before)

    def test_drift_scale_down_at_one_replica_per_device_is_noop(self):
        # 8 actors, one per device: shedding actors cannot change the
        # resource envelope, so the action is recorded as an honest no-op.
        device_ids = list(range(8))
        controller, plan = self._wide_controller(device_ids=device_ids)
        baseline_per_row = 100.0 * 1024**2 / 8
        changes = controller.on_partition_complete(
            0, self._obs(baseline_per_row * 2.0, oom=1), goodput_rows_per_sec=100
        )
        self.assertEqual(len(changes), 1)
        change = changes[0]
        self.assertEqual(change.action, "scale_down")
        self.assertFalse(change.effective)
        self.assertEqual(change.actors_before, 8)
        self.assertEqual(change.actors_after, 8)
        self.assertEqual(change.device_ids_after, device_ids)
        self.assertEqual(change.stage_max_replicas_per_device_after, 1)
        self.assertIn("no-op", change.reason)

    def test_revert_sheds_recorded_device_not_generic_targets(self):
        controller, plan = self._wide_controller(device_ids=[0, 1, 1])
        observation = self._obs(bytes_per_row=None, rps=10.0)
        changes = controller.on_partition_complete(0, observation, goodput_rows_per_sec=10.0)
        self.assertEqual([c.action for c in changes], ["scale_up"])
        self.assertEqual(sorted(plan.stages["stage:a"].device_ids), [0, 0, 1, 1])
        # Low marginal gain -> exact undo of the recorded device (0). A
        # generic device-level step would also shed a replica from device 1.
        changes = controller.on_partition_complete(1, observation, goodput_rows_per_sec=10.05)
        self.assertEqual([c.action for c in changes], ["revert_scale_up"])
        self.assertEqual(sorted(changes[0].device_ids_after), [0, 1, 1])
        self.assertEqual(plan.stages["stage:a"].planned_actors, 3)

    def test_plan_events_record_envelope_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plan_events.jsonl")
            controller, _plan = self._controller(events_path=path)
            baseline_per_row = 100.0 * 1024**2 / 8
            controller.on_partition_complete(
                0, self._obs(baseline_per_row * 2.0, oom=1), goodput_rows_per_sec=100
            )
            with open(path) as handle:
                lines = [json.loads(line) for line in handle]
            self.assertEqual(len(lines), 1)
            record = lines[0]
            for key in (
                "device_ids_before",
                "device_ids_after",
                "stage_max_replicas_per_device_before",
                "stage_max_replicas_per_device_after",
                "plan_max_colocated_replicas_before",
                "plan_max_colocated_replicas_after",
                "per_actor_vram_budget_before",
                "per_actor_vram_budget_after",
                "effective",
            ):
                self.assertIn(key, record)
            self.assertTrue(record["effective"])


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

    def test_planner_economics_options_validated(self):
        base = {"elastic_juicer_auto_parallelism": True}
        with self.assertRaises(ValueError):
            validate_elastic_juicer_config({**base, "elastic_juicer_min_batches_per_actor": 0})
        with self.assertRaises(ValueError):
            validate_elastic_juicer_config({**base, "elastic_juicer_startup_amortization_ratio": 1.5})
        with self.assertRaises(ValueError):
            validate_elastic_juicer_config({**base, "elastic_juicer_min_marginal_makespan_gain": 1.0})
        with self.assertRaises(ValueError):
            validate_elastic_juicer_config({**base, "elastic_juicer_actor_startup_seconds": -1.0})
        validate_elastic_juicer_config({**base, "elastic_juicer_actor_startup_seconds": 2.0})


class DatasetSnapshotBindingTest(unittest.TestCase):
    """The warm fast path must be bound to the exact input bytes."""

    def test_fingerprint_binds_file_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "uniform.jsonl")
            with open(path, "w") as handle:
                handle.write('{"a": 1}\n{"a": 2}\n')
            first = dataset_snapshot_fingerprint([path])
            self.assertIsNotNone(first)
            with open(path, "a") as handle:
                handle.write('{"a": 3}\n')
            self.assertNotEqual(dataset_snapshot_fingerprint([path]), first)
            # restoring the exact bytes restores the fingerprint
            with open(path, "w") as handle:
                handle.write('{"a": 1}\n{"a": 2}\n')
            self.assertEqual(dataset_snapshot_fingerprint([path]), first)

    def test_fingerprint_requires_resolvable_file_backing(self):
        self.assertIsNone(dataset_snapshot_fingerprint([]))
        self.assertIsNone(dataset_snapshot_fingerprint(None))
        self.assertIsNone(dataset_snapshot_fingerprint(["/nonexistent/path.jsonl"]))
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "a.jsonl")
            second = os.path.join(tmp, "b.jsonl")
            with open(first, "w") as handle:
                handle.write("x\n")
            with open(second, "w") as handle:
                handle.write("y\n")
            self.assertIsNotNone(dataset_snapshot_fingerprint([first, second]))
            # the shard set itself is part of the fingerprint
            self.assertNotEqual(
                dataset_snapshot_fingerprint([first]), dataset_snapshot_fingerprint([first, second])
            )

    def test_dataset_cache_usable_requires_intact_binding(self):
        good = {"dataset": {"fingerprint": "fp1", "total_rows": 4096, "sample_digest": "d1"}}
        self.assertEqual(dataset_cache_usable(good, "fp1"), 4096)
        # a changed input snapshot must reject the cache
        self.assertIsNone(dataset_cache_usable(good, "fp2"))
        self.assertIsNone(dataset_cache_usable(good, None))
        self.assertIsNone(dataset_cache_usable({"dataset": {}}, "fp1"))
        self.assertIsNone(dataset_cache_usable({}, "fp1"))
        self.assertIsNone(dataset_cache_usable(None, "fp1"))
        zero_rows = {"dataset": {"fingerprint": "fp1", "total_rows": 0, "sample_digest": "d1"}}
        self.assertIsNone(dataset_cache_usable(zero_rows, "fp1"))
        no_digest = {"dataset": {"fingerprint": "fp1", "total_rows": 8}}
        self.assertIsNone(dataset_cache_usable(no_digest, "fp1"))

    def test_save_and_load_roundtrip_carries_dataset_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, PROBE_RESULTS_FILENAME)
            stages = {"stage:fake_op:1": _result()}
            save_probe_results(
                path, stages, dataset={"fingerprint": "fp", "total_rows": 4096, "sample_digest": "d"}
            )
            payload = load_probe_results(path)
            self.assertEqual(payload["dataset"]["total_rows"], 4096)
            # a save without a binding drops it: later runs stay cold
            save_probe_results(path, stages)
            self.assertIsNone(load_probe_results(path).get("dataset"))

    def test_sample_columns_digest_is_stable_and_content_sensitive(self):
        columns = {"id": [1, 2, 3], "cost": [1.5, 2.5, 3.5]}
        self.assertEqual(sample_columns_digest(columns), sample_columns_digest(dict(columns)))
        changed = {"id": [1, 2, 4], "cost": [1.5, 2.5, 3.5]}
        self.assertNotEqual(sample_columns_digest(columns), sample_columns_digest(changed))


class _FakeProbeTask:
    def __init__(self, owner):
        self._owner = owner

    def options(self, **kwargs):
        return self

    def remote(self, *args, **kwargs):
        self._owner.probe_calls += 1
        return object()


class _FakeRayModule:
    """Minimal ray stand-in: every remote task resolves to one canned result."""

    def __init__(self, raw_result):
        self.raw_result = raw_result
        self.probe_calls = 0

    def remote(self, *decorator_args, **decorator_kwargs):
        def decorate(fn):
            return _FakeProbeTask(self)

        return decorate

    def get(self, future, timeout=None):
        return self.raw_result


class WarmCacheGatingTest(unittest.TestCase):
    """probe_gpu_stages must reject the cache when the dataset changes."""

    @staticmethod
    def _raw():
        return {
            "gpu_name": "FakeGPU",
            "resident_bytes": 1_000_000,
            "allocator_resident_bytes": 2_000_000,
            "allocator_peak_bytes": 8_000_000,
            "effective_peak_bytes": 8_000_000,
            "peak_bytes_with_headroom": 8_800_000,
            "throughput_rows_per_sec": 100.0,
            "probed_batch_size": 8,
            "oom_upper_bound": None,
            "probe_duration_sec": 0.5,
            "rows_probed": 24,
            "init_seconds": 0.5,
        }

    @staticmethod
    def _write_cache(tmp, fingerprint, total_rows=4096):
        spec = _spec()
        params = ProbeParams()
        result = _result(stage_key=spec.stage_key)
        result.fingerprint = probe_fingerprint(
            spec.op_name, spec.init_args, spec.init_kwargs, "FakeGPU", params
        )
        path = os.path.join(tmp, PROBE_RESULTS_FILENAME)
        save_probe_results(
            path,
            {spec.stage_key: result},
            dataset={"fingerprint": fingerprint, "total_rows": total_rows, "sample_digest": "d"},
        )
        return path, spec, params

    def test_compatible_hit_skips_gpu_probe_and_preserves_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "data.jsonl")
            with open(data, "w") as handle:
                handle.write('{"a": 1}\n')
            fingerprint = dataset_snapshot_fingerprint([data])
            path, spec, params = self._write_cache(tmp, fingerprint)
            ray_fake = _FakeRayModule(self._raw())
            results = probe_gpu_stages(
                [spec],
                {"a": [1]},
                params,
                path,
                ray_module=ray_fake,
                dataset_fingerprint=fingerprint,
            )
            self.assertEqual(ray_fake.probe_calls, 0)
            self.assertTrue(results[spec.stage_key].cache_hit)
            payload = load_probe_results(path)
            self.assertEqual(payload["dataset"]["fingerprint"], fingerprint)
            self.assertEqual(payload["dataset"]["total_rows"], 4096)

    def test_dataset_change_rejects_cache_and_reprobes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "data.jsonl")
            with open(data, "w") as handle:
                handle.write('{"a": 1}\n')
            fingerprint = dataset_snapshot_fingerprint([data])
            path, spec, params = self._write_cache(tmp, fingerprint)
            with open(data, "a") as handle:
                handle.write('{"a": 2}\n')
            changed = dataset_snapshot_fingerprint([data])
            ray_fake = _FakeRayModule(self._raw())
            results = probe_gpu_stages(
                [spec],
                {"a": [1]},
                params,
                path,
                ray_module=ray_fake,
                dataset_fingerprint=changed,
                dataset_stats={"total_rows": 2, "sample_digest": "d2"},
            )
            self.assertEqual(ray_fake.probe_calls, 1)
            self.assertFalse(results[spec.stage_key].cache_hit)
            self.assertEqual(results[spec.stage_key].allocator_resident_bytes, 2_000_000)
            self.assertEqual(results[spec.stage_key].working_bytes_per_row, 750_000)
            payload = load_probe_results(path)
            self.assertEqual(payload["dataset"]["fingerprint"], changed)
            self.assertEqual(payload["dataset"]["total_rows"], 2)


class _StubRayData:
    def __init__(self, rows):
        self._rows = rows
        self.take_calls = 0
        self.count_calls = 0

    def take(self, n):
        self.take_calls += 1
        return self._rows[:n]

    def count(self):
        self.count_calls += 1
        return len(self._rows)


class WarmFastPathExecutorTest(unittest.TestCase):
    """_elastic_juicer_autoscale_setup must skip the sample read and the
    full row count on a fully compatible cache hit, and go cold otherwise."""

    def _op(self):
        op = _FakeOp()
        op._name = "fake_op"
        op._init_args = ()
        op._init_kwargs = {}
        op.num_gpus = 0.25
        op.num_cpus = 1.0
        op.batch_size = 8
        op.is_batched_op = lambda: True
        return op

    def _executor(self, work_dir, data_path):
        from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor

        executor = object.__new__(PartitionedRayExecutor)
        executor.cfg = {
            "elastic_juicer_auto_parallelism": True,
            "elastic_juicer_probe_enabled": True,
            "gpu_probe_sample_rows": 8,
            "dataset_path": data_path,
        }
        executor.work_dir = work_dir
        executor.num_partitions = 4
        executor._explicit_actor_op_ids = set()
        executor._ej_build_cluster_spec = lambda: ClusterSpec(
            total_cpus=16.0,
            num_devices=2,
            device_memory_bytes=8 * 1024**3,
            device_memory_budget_fraction=1.0,
        )
        return executor

    def _write_executor_cache(self, work_dir, data_path, stage_key, oom_upper_bound=None):
        params = ProbeParams(sample_rows=8)
        result = _result(stage_key=stage_key)
        result.fingerprint = probe_fingerprint("fake_op", (), {}, "FakeGPU", params)
        result.oom_upper_bound = oom_upper_bound
        fingerprint = dataset_snapshot_fingerprint([data_path])
        save_probe_results(
            os.path.join(work_dir, PROBE_RESULTS_FILENAME),
            {stage_key: result},
            dataset={"fingerprint": fingerprint, "total_rows": 4096, "sample_digest": "d"},
        )
        return fingerprint

    def test_warm_hit_skips_sample_read_and_full_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "uniform.jsonl")
            with open(data, "w") as handle:
                handle.write('{"a": 1}\n')
            stage_key = "stage-000000:fake_op"
            self._write_executor_cache(tmp, data, stage_key, oom_upper_bound=12)
            executor = self._executor(tmp, data)
            stub = _StubRayData([{"a": i} for i in range(16)])
            op = self._op()

            executor._elastic_juicer_autoscale_setup(
                type("Dataset", (), {"data": stub})(), [op]
            )

            self.assertEqual(stub.take_calls, 0)
            self.assertEqual(stub.count_calls, 0)
            self.assertIsNotNone(executor._ej_autoscale_state)
            # the cached probed OOM bound seeds the actor batch cap
            self.assertGreaterEqual(op.num_proc, 1)
            self.assertLessEqual(op._elastic_juicer_batch_hard_cap, 11)
            # the warm hit is re-persisted so file consumers see cache_hit=True
            persisted = load_probe_results(os.path.join(tmp, PROBE_RESULTS_FILENAME))
            self.assertTrue(persisted["stages"][stage_key]["cache_hit"])
            self.assertEqual(persisted["dataset"]["total_rows"], 4096)

    def test_data_change_falls_back_to_cold_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "uniform.jsonl")
            with open(data, "w") as handle:
                handle.write('{"a": 1}\n')
            stage_key = "stage-000000:fake_op"
            self._write_executor_cache(tmp, data, stage_key)
            with open(data, "a") as handle:
                handle.write('{"a": 2}\n')
            executor = self._executor(tmp, data)
            stub = _StubRayData([{"a": i} for i in range(16)])

            import data_juicer.core.elasticjuicer.gpu_probe as gpu_probe_module

            calls = []
            original = gpu_probe_module.probe_gpu_stages

            def fake_probe(specs, sample_columns, params, results_path, **kwargs):
                calls.append(kwargs.get("dataset_fingerprint"))
                return {spec.stage_key: _result(stage_key=spec.stage_key) for spec in specs}

            gpu_probe_module.probe_gpu_stages = fake_probe
            try:
                executor._elastic_juicer_autoscale_setup(
                    type("Dataset", (), {"data": stub})(), [self._op()]
                )
            finally:
                gpu_probe_module.probe_gpu_stages = original

            self.assertEqual(len(calls), 1)
            self.assertEqual(stub.take_calls, 1)
            self.assertEqual(stub.count_calls, 1)
            self.assertIsNotNone(executor._ej_autoscale_state)


if __name__ == "__main__":
    unittest.main()
