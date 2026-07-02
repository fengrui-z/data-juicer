import json

import pytest

from data_juicer.core.elasticjuicer.profiler.execution_observer import ExecutionObserver


class FakeClock:
    def __call__(self):
        return 123.0


def monitor_result():
    return {
        "resource_analysis": {
            "CPU util.": {"max": 0.75},
            "Used mem.": {"max": 2048},
            "GPU used mem.": {"max": 512},
            "GPU util.": {"max": 0.5},
        }
    }


def test_observer_normalizes_existing_monitor_metrics(tmp_path):
    observer = ExecutionObserver(str(tmp_path), clock=FakeClock())

    observation = observer.record_stage(
        stage_name="mapper",
        configured_batch_size=16,
        input_rows=100,
        output_rows=90,
        duration_sec=2,
        monitor_result=monitor_result(),
    )

    assert observation.throughput == 50
    assert observation.cpu_peak_percent == 75
    assert observation.memory_peak_mb == 2048
    assert observation.gpu_memory_peak_mb == 512
    assert observation.gpu_peak_percent == 50
    assert observation.timestamp == 123


def test_observer_persists_jsonl(tmp_path):
    observer = ExecutionObserver(str(tmp_path), clock=FakeClock())
    observer.record_stage(
        stage_name="filter",
        configured_batch_size=8,
        input_rows=10,
        output_rows=4,
        duration_sec=1,
    )

    rows = [json.loads(line) for line in (tmp_path / "observations.jsonl").read_text().splitlines()]

    assert len(rows) == 1
    assert rows[0]["stage_name"] == "filter"
    assert rows[0]["output_rows"] == 4


def test_observer_can_run_without_persistence():
    observer = ExecutionObserver()
    observer.record_stage(
        stage_name="op",
        configured_batch_size=1,
        input_rows=1,
        output_rows=1,
        duration_sec=0,
    )

    assert len(observer.observations) == 1
    assert observer.output_path is None


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ({"max": [0.1, 0.8, 0.3]}, 80),
        ({"max": []}, None),
        ({}, None),
    ],
)
def test_observer_handles_monitor_metric_shapes(metric, expected):
    result = {"resource_analysis": {"CPU util.": metric}}
    observer = ExecutionObserver()

    observation = observer.record_stage(
        stage_name="op",
        configured_batch_size=1,
        input_rows=1,
        output_rows=1,
        duration_sec=1,
        monitor_result=result,
    )

    assert observation.cpu_peak_percent == expected
