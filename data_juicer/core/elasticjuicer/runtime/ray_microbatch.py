"""Ray Actor wrapper and metrics feedback for adaptive micro-batching."""

from collections import defaultdict
from typing import Any, Dict, Optional

from .local_microbatch import AdaptiveMicrobatchExecutor, _batch_length
from ..scheduler.micro_scheduler import MicroScheduler


class RayMetricsCollector:
    """Small Ray-actor-friendly aggregation target."""

    def __init__(self):
        self._metrics = defaultdict(
            lambda: {
                "calls": 0,
                "outer_rows": 0,
                "successful_slices": 0,
                "oom_retries": 0,
                "failures": 0,
                "current_batch_size": 0,
            }
        )

    def record(self, payload: Dict[str, Any]):
        stage = self._metrics[payload["stage_name"]]
        stage["calls"] += 1
        stage["outer_rows"] += payload["outer_rows"]
        stage["successful_slices"] += payload["successful_slices"]
        stage["oom_retries"] += payload["oom_retries"]
        stage["failures"] += 0 if payload["succeeded"] else 1
        stage["current_batch_size"] = payload["current_batch_size"]

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return {stage: dict(metrics) for stage, metrics in self._metrics.items()}


def create_ray_metrics_collector():
    """Create a collector actor lazily after Ray has initialized."""

    import ray

    return ray.remote(RayMetricsCollector).remote()


def build_ray_actor_kwargs(
    operator,
    method_name: str,
    *,
    min_batch_size: int,
    max_batch_size: int,
    metrics_sink=None,
) -> Dict[str, Any]:
    """Build serializable constructor arguments for a wrapped operator."""

    return {
        "operator_class": operator.__class__,
        "operator_args": getattr(operator, "_init_args", None),
        "operator_kwargs": getattr(operator, "_init_kwargs", None),
        "method_name": method_name,
        "stage_name": getattr(operator, "_name", operator.__class__.__name__),
        "initial_batch_size": min(
            max_batch_size,
            max(min_batch_size, int(getattr(operator, "batch_size", 1))),
        ),
        "min_batch_size": min_batch_size,
        "max_batch_size": max_batch_size,
        "metrics_sink": metrics_sink,
    }


class RayAdaptiveActor:
    """Construct the original OP inside a Ray actor and micro-batch its calls."""

    def __init__(
        self,
        operator_class,
        operator_args=None,
        operator_kwargs=None,
        method_name: str = "process",
        stage_name: Optional[str] = None,
        initial_batch_size: int = 1,
        min_batch_size: int = 1,
        max_batch_size: int = 1000,
        metrics_sink=None,
        scheduler_factory=MicroScheduler,
    ):
        self.operator = operator_class(
            *(operator_args or ()),
            **(operator_kwargs or {}),
        )
        self.stage_name = stage_name or operator_class.__name__
        self.metrics_sink = metrics_sink
        scheduler = scheduler_factory(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )
        self.executor = AdaptiveMicrobatchExecutor(
            getattr(self.operator, method_name),
            scheduler,
        )

    def __call__(self, batch):
        outer_rows = _batch_length(batch)
        previous_slices = self.executor.successful_slices
        previous_ooms = self.executor.oom_retries
        try:
            result = self.executor(batch)
        except BaseException as error:
            self._record(
                outer_rows,
                previous_slices,
                previous_ooms,
                succeeded=False,
                error_type=error.__class__.__name__,
            )
            raise

        self._record(
            outer_rows,
            previous_slices,
            previous_ooms,
            succeeded=True,
        )
        return result

    def _record(
        self,
        outer_rows: int,
        previous_slices: int,
        previous_ooms: int,
        *,
        succeeded: bool,
        error_type: Optional[str] = None,
    ):
        if self.metrics_sink is None:
            return
        payload = {
            "stage_name": self.stage_name,
            "outer_rows": outer_rows,
            "successful_slices": self.executor.successful_slices - previous_slices,
            "oom_retries": self.executor.oom_retries - previous_ooms,
            "current_batch_size": self.executor.scheduler.controller.current_batch_size,
            "succeeded": succeeded,
            "error_type": error_type,
        }
        record = self.metrics_sink.record
        remote = getattr(record, "remote", None)
        if callable(remote):
            remote(payload)
        else:
            record(payload)
