import json
from dataclasses import replace

import pytest

from data_juicer.core.elasticjuicer.execution_audit import DurableExecutionSink
from .test_async_metrics_sink import _event


def _sink(tmp_path, max_events=1):
    sink = DurableExecutionSink("job-a", "run-a", str(tmp_path), max_events=max_events)
    sink.expect_execution(
        {
            "execution_id": "exec-a",
            "stage_id": "stage-a",
            "plan_version": "plan-a",
            "planned_actors": 1,
            "applied_cap": 4,
            "partition_id": 0,
        }
    )
    sink.register_actor(
        {
            "execution_id": "exec-a",
            "actor_incarnation_id": "incarnation-a",
            "actor_id": "actor-a",
            "node_id": "node-a",
            "gpu_uuids": ["GPU-a"],
            "plan_version": "plan-a",
            "applied_cap": 4,
        }
    )
    return sink


def _sample(sequence):
    return replace(_event("job-a", sequence), run_id="run-a")


def test_final_watermark_and_durable_samples_survive_ring_eviction(tmp_path):
    sink = _sink(tmp_path)
    sink.begin_call("incarnation-a", 1, 8)
    sink.record_batch([_sample(1), _sample(2)])
    sink.end_call("incarnation-a", 1, 2, {"succeeded_batches": 2, "failed_batches": 0, "oom_batches": 0}, True)
    assert sink.snapshot()["dropped_events"] == 1
    assert sink.seal_run(True)["complete"] is True
    metrics = [json.loads(line) for line in (tmp_path / "elastic_juicer_metrics_events.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in metrics] == [1, 2]
    audit = json.loads((tmp_path / "elastic_juicer_execution_audit.json").read_text())
    assert audit["executions"]["exec-a"]["applied"] is True
    assert audit["actors"]["incarnation-a"]["final_sequence"] == 2
    assert audit["executions"]["exec-a"]["working_actors"] == 1
    assert audit["executions"]["exec-a"]["completed_input_rows"] == 8
    with pytest.raises(ValueError, match="sealed"):
        sink.begin_call("incarnation-a", 2, 1)


def test_partition_wall_is_durable_idempotent_and_finite(tmp_path):
    sink = _sink(tmp_path)
    assert sink.record_partition(0, 8, 2.5) == 0
    assert sink.record_partition(0, 8, 2.5) == 0
    with pytest.raises(ValueError, match="changed"):
        sink.record_partition(0, 8, 3.0)
    with pytest.raises(ValueError, match="finite"):
        sink.record_partition(1, 8, float("nan"))
    recovered = DurableExecutionSink("job-a", "run-a", str(tmp_path))
    assert recovered.partitions == {"0": {"partition_id": 0, "input_rows": 8, "wall_seconds": 2.5}}


def test_unfinished_call_and_missing_actor_cannot_be_certified(tmp_path):
    sink = _sink(tmp_path)
    sink.begin_call("incarnation-a", 1, 4)
    sink.record_batch([_sample(1)])
    assert sink.seal_run(True)["complete"] is False


def test_enforced_placement_cannot_seal_an_undersized_roster(tmp_path):
    sink = DurableExecutionSink("job-a", "run-a", str(tmp_path))
    sink.expect_execution({"execution_id": "exec", "stage_id": "stage-a", "plan_version": "plan-a",
                           "planned_actors": 2, "applied_cap": 4,
                           "placement": {"device_ids": [0, 0], "topology": [
                               {"device_id": 0, "node_id": "node-a", "gpu_uuid": "GPU-a"}]}})
    sink.register_actor({"execution_id": "exec", "actor_incarnation_id": "inc-a", "actor_id": "actor-a",
                         "plan_version": "plan-a", "applied_cap": 4, "placement_slot": 0,
                         "placement_verified": True, "node_id": "node-a", "gpu_uuids": ["GPU-a"]})
    assert sink.seal_run(True)["complete"] is False


def test_missing_final_event_is_rejected(tmp_path):
    sink = _sink(tmp_path)
    sink.begin_call("incarnation-a", 1, 4)
    with pytest.raises(ValueError, match="watermark"):
        sink.end_call("incarnation-a", 1, 1, {"succeeded_batches": 1, "failed_batches": 0, "oom_batches": 0}, True)


def test_expected_execution_without_any_actor_cannot_be_certified(tmp_path):
    sink = _sink(tmp_path)
    sink.expect_execution(
        {
            "execution_id": "missing",
            "stage_id": "stage-b",
            "plan_version": "plan-b",
            "planned_actors": 1,
            "applied_cap": 4,
            "partition_id": 0,
        }
    )
    assert sink.seal_run(True)["complete"] is False


def test_valid_json_journal_corruption_is_detected(tmp_path):
    _sink(tmp_path)
    path = tmp_path / "execution_journal.jsonl"
    path.write_text(path.read_text().replace('"plan-a"', '"plan-b"'))
    with pytest.raises(ValueError, match="checksum"):
        DurableExecutionSink("job-a", "run-a", str(tmp_path))


def test_duplicate_delivery_and_sink_replay_are_idempotent(tmp_path):
    sink = _sink(tmp_path)
    sink.begin_call("incarnation-a", 1, 4)
    sink.record_batch([_sample(1)])
    resumed = DurableExecutionSink("job-a", "run-a", str(tmp_path), max_events=1)
    resumed.record_batch([_sample(1)])
    resumed.end_call("incarnation-a", 1, 1, {"succeeded_batches": 1, "failed_batches": 0, "oom_batches": 0}, True)
    assert resumed.seal_run(True)["complete"] is True
    assert resumed.snapshot()["received_events"] == 1


def test_actor_must_confirm_the_expected_version_and_cap(tmp_path):
    sink = _sink(tmp_path)
    with pytest.raises(ValueError, match="envelope"):
        sink.register_actor(
            {
                "execution_id": "exec-a",
                "actor_incarnation_id": "bad",
                "actor_id": "actor-b",
                "node_id": "node-a",
                "gpu_uuids": [],
                "plan_version": "wrong",
                "applied_cap": 99,
            }
        )


def test_run_identity_and_sequence_gap_fail_closed(tmp_path):
    sink = _sink(tmp_path)
    sink.begin_call("incarnation-a", 1, 4)
    with pytest.raises(ValueError, match="run"):
        sink.record_batch([replace(_sample(1), run_id="old-run")])
    with pytest.raises(ValueError, match="sequence"):
        sink.record_batch([_sample(2)])


def test_failed_fsync_never_acknowledges_and_poisoned_sink_fails_closed(tmp_path, monkeypatch):
    sink = _sink(tmp_path)
    sink.begin_call("incarnation-a", 1, 4)
    monkeypatch.setattr(
        "data_juicer.core.elasticjuicer.execution_audit.os.fsync", lambda fd: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises(OSError):
        sink.record_batch([_sample(1)])
    assert sink.snapshot()["received_events"] == 0
    with pytest.raises(RuntimeError, match="journal"):
        sink.record_batch([_sample(1)])


def test_reporter_retries_lost_ack_without_duplicate_events(tmp_path):
    from types import SimpleNamespace
    from data_juicer.core.elasticjuicer.acknowledged_metrics import AcknowledgedMetricsReporter
    from .test_async_metrics_sink import _snapshot

    sink = _sink(tmp_path)
    handle = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=getattr(sink, name))
            for name in ("register_actor", "begin_call", "record_batch", "end_call")
        }
    )
    identity = {**sink.actors["incarnation-a"]["identity"], "stage_id": "stage-a", "op_name": "mapper"}
    # Use a second incarnation; identity is immutable after registration.
    identity["actor_incarnation_id"] = "incarnation-b"
    reporter = AcknowledgedMetricsReporter(
        handle, "job-a", "run-a", identity, lambda: None, max_buffered=1, ack_fn=lambda value: value
    )
    reporter.begin_call(4)
    lost = [False]

    def ack(value):
        if not lost[0]:
            lost[0] = True
            raise TimeoutError("lost response, not lost write")
        return value

    reporter.ack_fn = ack
    reporter.report(_snapshot())
    reporter.end_call(True)
    assert sink.snapshot()["received_events"] == 1
    assert reporter.snapshot()["pending_events"] == 0


def test_reporter_failure_cannot_be_hidden_by_advisory_snapshot_callback(tmp_path):
    from types import SimpleNamespace
    from data_juicer.core.elasticjuicer.acknowledged_metrics import AcknowledgedMetricsReporter, MetricsDeliveryError
    from .test_async_metrics_sink import _snapshot

    sink = _sink(tmp_path)
    handle = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=getattr(sink, name))
            for name in ("register_actor", "begin_call", "record_batch", "end_call")
        }
    )
    reporter = AcknowledgedMetricsReporter(
        handle,
        "job-a",
        "run-a",
        sink.actors["incarnation-a"]["identity"],
        lambda: None,
        max_buffered=1,
        ack_fn=lambda value: value,
        max_attempts=1,
    )
    reporter.begin_call(4)
    reporter.identity = {**reporter.identity, "stage_id": "stage-a", "op_name": "mapper"}
    reporter.ack_fn = lambda value: (_ for _ in ()).throw(TimeoutError("sink unavailable"))
    with pytest.raises(MetricsDeliveryError):
        reporter.report(_snapshot())
    with pytest.raises(MetricsDeliveryError):
        reporter.end_call(True)


def test_real_ray_sink_restart_replays_before_acknowledging_final_sequence(tmp_path):
    import ray
    from data_juicer.core.elasticjuicer.execution_audit import create_durable_execution_sink

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=2, include_dashboard=False)
    sink = create_durable_execution_sink("job-a", "run-a", str(tmp_path), max_events=1)
    try:
        expected = {
            "execution_id": "exec-a",
            "stage_id": "stage-a",
            "plan_version": "plan-a",
            "planned_actors": 1,
            "applied_cap": 4,
            "partition_id": 0,
        }
        ray.get(sink.expect_execution.remote(expected))
        ray.get(
            sink.register_actor.remote(
                {
                    "execution_id": "exec-a",
                    "actor_incarnation_id": "incarnation-a",
                    "actor_id": "actor-a",
                    "node_id": "node-a",
                    "gpu_uuids": [],
                    "plan_version": "plan-a",
                    "applied_cap": 4,
                }
            )
        )
        ray.get(sink.begin_call.remote("incarnation-a", 1, 4))
        ray.get(sink.record_batch.remote([_sample(1)]))
        ray.kill(sink, no_restart=False)
        ray.get(sink.record_batch.remote([_sample(1)]), timeout=30)
        ray.get(
            sink.end_call.remote(
                "incarnation-a", 1, 1, {"succeeded_batches": 1, "failed_batches": 0, "oom_batches": 0}, True
            )
        )
        assert ray.get(sink.seal_run.remote(True))["complete"] is True
        assert ray.get(sink.snapshot.remote())["received_events"] == 1
    finally:
        ray.kill(sink, no_restart=True)
        if started_here:
            ray.shutdown()
