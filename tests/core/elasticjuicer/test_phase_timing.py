from types import SimpleNamespace

import pytest

from data_juicer.core.elasticjuicer.phase_timing import PhaseTiming, phase, timed
from .test_execution_audit import _sink, _sample


def test_spans_keep_nesting_and_failed_work():
    ticks = iter([10, 11, 12, 14, 17, 18])
    timer = PhaseTiming(clock=lambda: next(ticks))
    with timer.span("outer"):
        with pytest.raises(ValueError), timer.span("inner"):
            raise ValueError("failed work is still measured")
    assert timer.snapshot() == {
        "schema_version": 1,
        "clock": "driver_perf_counter",
        "wall_seconds": 8,
        "spans": [{"phase": "inner", "start": 2, "end": 4}, {"phase": "outer", "start": 1, "end": 7}],
    }


def test_uninstrumented_callers_are_compatible():
    owner = SimpleNamespace()
    with phase(owner, "none"):
        assert timed("method")(lambda self: 42)(owner) == 42


def test_diagnostics_do_not_change_ack_or_replay_contract(tmp_path):
    sink = _sink(tmp_path)
    with pytest.raises(ValueError, match="durable"):
        sink.record_actor_timing("incarnation-a", {"call_id": 1})
    sink.begin_call("incarnation-a", 1, 4)
    sink.record_batch([_sample(1)])
    sink.end_call("incarnation-a", 1, 1, {"succeeded_batches": 1, "failed_batches": 0, "oom_batches": 0}, True)
    journal_before = (tmp_path / "execution_journal.jsonl").read_bytes()
    sink.record_actor_timing("incarnation-a", {"call_id": 1})
    assert (tmp_path / "execution_journal.jsonl").read_bytes() == journal_before
    assert sink.seal_run(True)["complete"] is True
    recovered = type(sink)("job-a", "run-a", str(tmp_path))
    assert recovered.actor_timings == {}
    assert recovered.export_audit()["complete"] is True
