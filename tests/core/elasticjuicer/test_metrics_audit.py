import json
from unittest.mock import patch

from data_juicer.core.elasticjuicer.metrics_audit import MetricsSidecarWriter


def _snapshot(*sequences, received=None, dropped=0, gaps=0):
    return {
        "job_id": "job", "received_events": received or len(sequences),
        "dropped_events": dropped, "sequence_gaps": gaps, "out_of_order_events": 0,
        "outcome_counts": {"oom_batches": 2},
        "events": [{"job_id": "job", "actor_incarnation_id": "inc", "sequence": i} for i in sequences],
    }


def _status(tmp_path):
    return json.loads((tmp_path / "elastic_juicer_metrics_integrity.json").read_text())


def test_evicted_events_are_reported_as_missing_not_complete(tmp_path):
    writer = MetricsSidecarWriter(str(tmp_path))
    assert writer.persist(_snapshot(3, 4, received=4, dropped=2)) == 2
    status = _status(tmp_path)
    assert status["persisted_events"] == 2
    assert status["known_loss"] is True
    assert status["complete"] is False
    assert status["outcome_counts"]["oom_batches"] == 2


def test_already_persisted_evictions_are_not_loss_or_duplicates(tmp_path):
    writer = MetricsSidecarWriter(str(tmp_path))
    assert writer.persist(_snapshot(1, 2)) == 2
    assert writer.persist(_snapshot(3, 4, received=4, dropped=2)) == 2
    assert writer.persist(_snapshot(3, 4, received=4, dropped=2)) == 0
    status = _status(tmp_path)
    assert status["known_loss"] is False
    assert status["observed_prefix_complete"] is True
    assert status["complete"] is False  # No producer end-of-stream acknowledgements yet.


def test_sequence_gaps_invalidate_even_fully_persisted_sink(tmp_path):
    writer = MetricsSidecarWriter(str(tmp_path))
    writer.persist(_snapshot(1, 3, gaps=1))
    assert _status(tmp_path)["known_loss"] is True


def test_failed_write_does_not_mark_events_written_and_can_retry(tmp_path):
    writer = MetricsSidecarWriter(str(tmp_path))
    with patch("data_juicer.core.elasticjuicer.metrics_audit.os.fsync", side_effect=OSError("disk error")):
        assert writer.persist(_snapshot(1, 2)) == 0
    assert writer.persist(_snapshot(1, 2)) == 2
    assert _status(tmp_path)["write_failures"] >= 1
    assert _status(tmp_path)["complete"] is False


def test_empty_snapshot_still_writes_integrity_and_unknown_producer_tail(tmp_path):
    writer = MetricsSidecarWriter(str(tmp_path))
    assert writer.persist(_snapshot()) == 0
    assert _status(tmp_path)["transport_status"] == "unverified"


def test_existing_events_are_reused_without_reappending(tmp_path):
    writer = MetricsSidecarWriter(str(tmp_path))
    writer.persist(_snapshot(1, 2))
    restarted = MetricsSidecarWriter(str(tmp_path))
    assert restarted.persist(_snapshot(1, 2)) == 0
    assert _status(tmp_path)["persisted_events"] == 2


def test_executor_persists_an_integrity_report_even_with_empty_buffer(tmp_path):
    from types import SimpleNamespace
    from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor

    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.work_dir = str(tmp_path)
    executor.cfg = {"_elastic_juicer_metrics_sink": SimpleNamespace(
        snapshot=SimpleNamespace(remote=lambda: _snapshot())
    )}
    with patch("data_juicer.core.executor.ray_executor_partitioned.ray", SimpleNamespace(
        get=lambda value, **kw: value
    )):
        assert executor._ej_persist_metrics_sidecar() == 0
    assert _status(tmp_path)["complete"] is False
