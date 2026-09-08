"""Bounded, retryable delivery before actor results become visible to Ray."""

import os
import time

from .async_metrics_sink import ActorMetricsEvent
from .oom import snapshot_is_oom


class MetricsDeliveryError(RuntimeError):
    pass


def runtime_identity():
    import ray

    if not ray.is_initialized():
        raise MetricsDeliveryError("execution audit requires a real Ray worker")
    context = ray.get_runtime_context()
    identity = {
        "node_id": context.get_node_id(),
        "ray_actor_id": context.get_actor_id(),
        "ray_accelerator_ids": context.get_accelerator_ids(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_uuids": [],
        "pid": os.getpid(),
    }
    if identity["ray_accelerator_ids"].get("GPU"):
        import torch

        identity["gpu_uuids"] = [
            str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())
        ]
    return identity


class AcknowledgedMetricsReporter:
    def __init__(
        self,
        sink,
        job_id,
        run_id,
        identity,
        control_state_provider,
        max_buffered=64,
        ack_fn=None,
        timeout_seconds=15,
        max_attempts=3,
    ):
        if max_buffered < 1 or max_attempts < 1:
            raise ValueError("metrics buffer and retry budget must be positive")
        self.sink = sink
        self.job_id = job_id
        self.run_id = run_id
        self.identity = identity
        self.control_state_provider = control_state_provider
        self.max_buffered = max_buffered
        self.ack_fn = ack_fn
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.sequence = 0
        self.call_id = 0
        self.buffer = []
        self.poisoned = False
        self.counts = {"succeeded_batches": 0, "failed_batches": 0, "oom_batches": 0}
        self._rpc("register_actor", identity)

    def _rpc(self, method, *args):
        last_error = None
        for attempt in range(self.max_attempts):
            try:
                reference = getattr(self.sink, method).remote(*args)
                if self.ack_fn is not None:
                    return self.ack_fn(reference)
                import ray

                return ray.get(reference, timeout=self.timeout_seconds)
            except Exception as error:
                last_error = error
                if attempt + 1 < self.max_attempts:
                    time.sleep(0.05 * (attempt + 1))
        raise MetricsDeliveryError(f"{method} acknowledgement failed: {last_error}") from last_error

    def begin_call(self, input_rows):
        if self.poisoned:
            raise MetricsDeliveryError("prior metrics delivery failed")
        self.call_id += 1
        self._rpc("begin_call", self.identity["actor_incarnation_id"], self.call_id, input_rows)

    def report(self, snapshot):
        if self.poisoned:
            raise MetricsDeliveryError("prior metrics delivery failed")
        self.sequence += 1
        self.buffer.append(
            ActorMetricsEvent(
                job_id=self.job_id,
                run_id=self.run_id,
                actor_id=self.identity["actor_id"],
                actor_incarnation_id=self.identity["actor_incarnation_id"],
                stage_id=self.identity["stage_id"],
                op_name=self.identity["op_name"],
                sequence=self.sequence,
                observed_at_ms=int(snapshot.timestamp * 1000),
                emitted_at_ms=time.time_ns() // 1_000_000,
                source=snapshot.source,
                snapshot=snapshot,
                control=self.control_state_provider(),
                partition_id=self.identity.get("partition_id"),
            )
        )
        self.counts["succeeded_batches" if snapshot.succeeded else "failed_batches"] += 1
        self.counts["oom_batches"] += int(snapshot_is_oom(snapshot))
        if len(self.buffer) >= self.max_buffered:
            self.flush()
        return True

    def flush(self):
        if self.poisoned:
            raise MetricsDeliveryError("prior metrics delivery failed")
        if self.buffer:
            try:
                self._rpc("record_batch", self.buffer)
            except MetricsDeliveryError:
                self.poisoned = True  # The mapper's advisory callback may catch us.
                raise
            self.buffer.clear()

    def end_call(self, succeeded):
        self.flush()
        return self._rpc(
            "end_call", self.identity["actor_incarnation_id"], self.call_id, self.sequence, self.counts, succeeded
        )

    def snapshot(self):
        return {
            "submitted_events": self.sequence,
            "dropped_events": 0,
            "pending_events": len(self.buffer),
            "max_in_flight": self.max_buffered,
            "last_sequence": self.sequence,
            "delivery_mode": "acknowledged",
            "poisoned": self.poisoned,
        }
