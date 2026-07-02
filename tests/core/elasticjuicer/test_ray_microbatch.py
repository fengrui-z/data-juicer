from types import SimpleNamespace

import pyarrow

from data_juicer.core.elasticjuicer.runtime.ray_microbatch import (
    RayAdaptiveActor,
    RayMetricsCollector,
    build_ray_actor_kwargs,
)


class StableScheduler:
    def __init__(self, initial_batch_size, min_batch_size, max_batch_size):
        self.controller = SimpleNamespace(
            current_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
        )

    def report_oom(self, batch_size, memory_mb):
        self.controller.current_batch_size = max(
            self.controller.min_batch_size,
            batch_size // 2,
        )

    def update(self, actual_memory_used):
        pass


class FakeSink:
    def __init__(self):
        self.payloads = []

    def record(self, payload):
        self.payloads.append(payload)


class ThresholdOperator:
    def __init__(self, threshold=2):
        self.threshold = threshold

    def process(self, table):
        if table.num_rows > self.threshold:
            raise RuntimeError("CUDA out of memory")
        return table


def test_ray_actor_microbatches_arrow_and_reports_metrics():
    sink = FakeSink()
    actor = RayAdaptiveActor(
        ThresholdOperator,
        operator_kwargs={"threshold": 2},
        stage_name="gpu_mapper",
        initial_batch_size=8,
        min_batch_size=1,
        max_batch_size=8,
        metrics_sink=sink,
        scheduler_factory=StableScheduler,
    )
    table = pyarrow.table({"value": list(range(7))})

    result = actor(table)

    assert result.to_pydict() == table.to_pydict()
    assert sink.payloads == [
        {
            "stage_name": "gpu_mapper",
            "outer_rows": 7,
            "successful_slices": 7,
            "oom_retries": 2,
            "current_batch_size": 1,
            "succeeded": True,
            "error_type": None,
        }
    ]


def test_ray_actor_reports_non_oom_failure():
    sink = FakeSink()

    class BrokenOperator:
        def process(self, table):
            raise RuntimeError("bad schema")

    actor = RayAdaptiveActor(
        BrokenOperator,
        metrics_sink=sink,
        scheduler_factory=StableScheduler,
    )

    try:
        actor(pyarrow.table({"value": [1]}))
    except RuntimeError:
        pass

    assert sink.payloads[0]["succeeded"] is False
    assert sink.payloads[0]["error_type"] == "RuntimeError"
    assert sink.payloads[0]["oom_retries"] == 0


def test_metrics_collector_aggregates_actor_feedback():
    collector = RayMetricsCollector()
    collector.record(
        {
            "stage_name": "op",
            "outer_rows": 10,
            "successful_slices": 3,
            "oom_retries": 1,
            "current_batch_size": 4,
            "succeeded": True,
        }
    )
    collector.record(
        {
            "stage_name": "op",
            "outer_rows": 5,
            "successful_slices": 0,
            "oom_retries": 0,
            "current_batch_size": 4,
            "succeeded": False,
        }
    )

    assert collector.snapshot()["op"] == {
        "calls": 2,
        "outer_rows": 15,
        "successful_slices": 3,
        "oom_retries": 1,
        "failures": 1,
        "current_batch_size": 4,
    }


def test_actor_kwargs_preserve_original_constructor():
    class ConstructorOperator:
        _init_args = ("model",)
        _init_kwargs = {"device": "cuda"}
        _name = "captioner"
        batch_size = 32

    operator = ConstructorOperator()

    kwargs = build_ray_actor_kwargs(
        operator,
        "process",
        min_batch_size=2,
        max_batch_size=16,
    )

    assert kwargs["operator_args"] == ("model",)
    assert kwargs["operator_kwargs"] == {"device": "cuda"}
    assert kwargs["initial_batch_size"] == 16
    assert kwargs["stage_name"] == "captioner"
