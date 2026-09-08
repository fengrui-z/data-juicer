"""Durable sink-prefix audit; deliberately not an end-to-end delivery claim."""

import json
import os

from loguru import logger

from .async_metrics_sink import METRICS_EVENTS_FILENAME, serialize_metrics_event
from .gpu_probe import atomic_write_json

METRICS_INTEGRITY_FILENAME = "elastic_juicer_metrics_integrity.json"


class MetricsSidecarWriter:
    """Persist samples with retryable acknowledgement and explicit loss state.

    Counters describe only events received by the sink. Until producers seal
    their final sequences, a lost tail or an actor that never reported cannot
    be ruled out. Consumers must not treat a complete prefix as a complete run.
    """

    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.path = os.path.join(work_dir, METRICS_EVENTS_FILENAME)
        self._written = set()
        self.write_failures = 0
        self.invalid_events = 0
        try:
            with open(self.path) as handle:
                for line in handle:
                    if line.strip():
                        try:
                            self._written.add(self._key(json.loads(line)))
                        except (ValueError, TypeError, AttributeError):
                            self.invalid_events += 1
        except FileNotFoundError:
            pass

    @staticmethod
    def _key(event):
        key = (event.get("job_id"), event.get("actor_incarnation_id"), event.get("sequence"))
        if (not all(isinstance(value, str) and value for value in key[:2])
                or isinstance(key[2], bool) or not isinstance(key[2], int) or key[2] < 1):
            raise ValueError("metrics event lacks a valid stream identity")
        return key

    def persist(self, snapshot):
        os.makedirs(self.work_dir, exist_ok=True)
        fresh = {}
        for event in snapshot.get("events", []):
            try:
                payload = event if isinstance(event, dict) else serialize_metrics_event(event)
                key = self._key(payload)
                if key not in self._written:
                    fresh[key] = json.dumps(payload, sort_keys=True)
            except (TypeError, ValueError, AttributeError):
                self.invalid_events += 1
        persisted = 0
        if fresh:
            try:
                with open(self.path, "a") as handle:
                    for line in fresh.values():
                        handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                # Never acknowledge before successful persistence. A partial
                # failed append may repeat on retry; readers deduplicate IDs.
                self._written.update(fresh)
                persisted = len(fresh)
            except OSError as error:
                self.write_failures += 1
                logger.warning(f"ElasticJuicer metrics sidecar: write failed ({error})")
        job_id = snapshot.get("job_id")
        received = snapshot.get("received_events")
        written = sum(key[0] == job_id for key in self._written)
        counts_match = isinstance(received, int) and received == written
        gaps = snapshot.get("sequence_gaps")
        reordered = snapshot.get("out_of_order_events")
        known_loss = (
            (isinstance(received, int) and received > written)
            or bool(gaps) or bool(reordered) or bool(self.write_failures) or bool(self.invalid_events)
        )
        status = {
            "schema_version": 1, "job_id": job_id,
            "received_events": received, "persisted_events": written,
            "buffer_evictions": snapshot.get("dropped_events"),
            "sequence_gaps": gaps, "out_of_order_events": reordered,
            "outcome_counts": snapshot.get("outcome_counts", {}),
            "write_failures": self.write_failures, "invalid_events": self.invalid_events,
            "known_loss": known_loss,
            "observed_prefix_complete": counts_match and gaps == 0 and reordered == 0 and not known_loss,
            "transport_status": "unverified", "complete": False,
            "limitations": ["producer final sequences and expected actor roster are not acknowledged"],
        }
        try:
            atomic_write_json(os.path.join(self.work_dir, METRICS_INTEGRITY_FILENAME), status)
        except OSError as error:
            self.write_failures += 1
            logger.warning(f"ElasticJuicer metrics integrity: write failed ({error})")
        return persisted
