"""Driver-side bridge from Ray actor counters to ElasticJuicer Tower/Captain."""

import threading
import time
from typing import Dict, Optional

from loguru import logger

from ..contracts import StageMetrics


class MetricsBridge:
    """Poll RayMetricsCollector and feed delta metrics into ElasticJuicer.

    Ray actors report cumulative counters.  The bridge keeps the previous
    snapshot and converts each polling interval into deltas so Tower receives
    fresh per-window throughput instead of repeatedly counting the same work.
    """

    def __init__(
        self,
        elastic_juicer,
        metrics_collector,
        poll_interval: float = 1.0,
        clock=time.time,
    ):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._elastic_juicer = elastic_juicer
        self._metrics_collector = metrics_collector
        self._poll_interval = poll_interval
        self._clock = clock
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_snapshot: Dict[str, Dict] = {}
        self._last_poll_time: Optional[float] = None
        self._cycle_count = 0
        self._errors = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._last_poll_time = self._clock()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ElasticJuicer-MetricsBridge",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"ElasticJuicer MetricsBridge started (poll_interval={self._poll_interval}s)")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval * 3)
            self._thread = None
        logger.info(f"ElasticJuicer MetricsBridge stopped after {self._cycle_count} cycles")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.bridge_cycle()
            except Exception as exc:
                self._errors += 1
                logger.warning(f"ElasticJuicer MetricsBridge cycle failed: {exc}")
            self._stop_event.wait(self._poll_interval)

    def bridge_cycle(self) -> None:
        import ray

        snapshot = ray.get(self._metrics_collector.snapshot.remote())
        self._cycle_count += 1
        now = self._clock()
        previous_poll_time = self._last_poll_time or now
        elapsed = max(1e-6, now - previous_poll_time)
        self._last_poll_time = now

        for stage_name, totals in snapshot.items():
            previous = self._last_snapshot.get(stage_name, {})
            delta_calls = self._counter_delta(totals, previous, "calls")
            delta_rows = self._counter_delta(totals, previous, "outer_rows")
            delta_ooms = self._counter_delta(totals, previous, "oom_retries")
            delta_failures = self._counter_delta(totals, previous, "failures")
            if delta_calls == 0 and delta_rows == 0 and delta_ooms == 0 and delta_failures == 0:
                continue

            captain = self._elastic_juicer.captains.get(stage_name)
            if captain is None:
                continue

            current_batch_size = int(totals.get("current_batch_size", 0) or 0)
            if captain.micro_scheduler is not None and current_batch_size > 0:
                controller = captain.micro_scheduler.controller
                controller.current_batch_size = max(
                    controller.min_batch_size,
                    min(controller.max_batch_size, current_batch_size),
                )

            captain.samples_processed += delta_rows
            captain.metrics = StageMetrics(
                stage_name=stage_name,
                queue_depth=0,
                current_parallelism=captain.metrics.current_parallelism,
                throughput=delta_rows / elapsed,
                avg_latency_ms=captain.metrics.avg_latency_ms,
                cpu_utilization=captain.metrics.cpu_utilization,
                memory_utilization=captain.metrics.memory_utilization,
                gpu_utilization=captain.metrics.gpu_utilization,
                oom_count=int(totals.get("oom_retries", 0) or 0),
                last_update=now,
            )
            self._elastic_juicer.update_stage_metrics(stage_name, captain.metrics)

        self._last_snapshot = {stage: dict(metrics) for stage, metrics in snapshot.items()}

    @staticmethod
    def _counter_delta(current: Dict, previous: Dict, key: str) -> int:
        return max(0, int(current.get(key, 0) or 0) - int(previous.get(key, 0) or 0))

    def get_stats(self) -> Dict:
        return {
            "is_running": self.is_running,
            "cycle_count": self._cycle_count,
            "errors": self._errors,
            "poll_interval": self._poll_interval,
        }
