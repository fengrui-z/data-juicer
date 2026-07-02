"""Observe-only integration for existing Data-Juicer execution boundaries."""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts import Clock, StageExecutionObservation


class ExecutionObserver:
    """Collect operator-stage observations without changing execution policy."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        clock: Clock = time.time,
    ):
        self._clock = clock
        self._lock = threading.Lock()
        self._observations: List[StageExecutionObservation] = []
        self.output_path: Optional[Path] = None

        if output_dir:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.output_path = directory / "observations.jsonl"

    @property
    def observations(self) -> List[StageExecutionObservation]:
        """Return a snapshot of observations collected so far."""

        with self._lock:
            return list(self._observations)

    def record_stage(
        self,
        *,
        stage_name: str,
        configured_batch_size: int,
        input_rows: int,
        output_rows: int,
        duration_sec: float,
        monitor_result: Optional[Dict[str, Any]] = None,
    ) -> StageExecutionObservation:
        """Translate an existing Monitor result into the shared contract."""

        analysis = self._get_analysis(monitor_result)
        duration_ms = max(0.0, duration_sec * 1000)
        throughput = input_rows / duration_sec if duration_sec > 0 else 0.0

        observation = StageExecutionObservation(
            stage_name=stage_name,
            configured_batch_size=max(1, configured_batch_size),
            input_rows=input_rows,
            output_rows=output_rows,
            duration_ms=duration_ms,
            throughput=throughput,
            cpu_peak_percent=self._percent_metric(analysis, "CPU util."),
            memory_peak_mb=self._max_metric(analysis, "Used mem."),
            gpu_memory_peak_mb=self._max_metric(analysis, "GPU used mem."),
            gpu_peak_percent=self._percent_metric(analysis, "GPU util."),
            timestamp=self._clock(),
        )

        with self._lock:
            self._observations.append(observation)
            if self.output_path is not None:
                with self.output_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(observation.to_dict(), sort_keys=True) + "\n")

        return observation

    @staticmethod
    def _get_analysis(monitor_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not monitor_result:
            return {}
        return monitor_result.get("resource_analysis", {})

    @staticmethod
    def _max_metric(analysis: Dict[str, Any], name: str) -> Optional[float]:
        metric = analysis.get(name)
        if not metric or metric.get("max") is None:
            return None
        values = metric["max"]
        if isinstance(values, list):
            return float(max(values)) if values else None
        return float(values)

    @classmethod
    def _percent_metric(cls, analysis: Dict[str, Any], name: str) -> Optional[float]:
        value = cls._max_metric(analysis, name)
        if value is None:
            return None
        # Existing Monitor reports utilization as a ratio.
        return min(100.0, max(0.0, value * 100))
