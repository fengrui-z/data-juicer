"""Acknowledged batch delivery and actor execution evidence, scoped to a run.

The append-only journal is authoritative. Every ACK follows fsync. A worker
must ACK its call watermark before returning data; driver sealing is allowed
only after the dependent dataset has materialized. Sealing fences new work.
"""

import json
import hashlib
import math
import os
import tempfile

from .actor_resource_sampler import ActorResourceSnapshot, CudaDeviceMemory
from .async_metrics_sink import ActorControlMetrics, ActorMetricsEvent, AsyncMetricsSink, serialize_metrics_event
from .gpu_probe import atomic_write_json
from .oom import snapshot_is_oom


def _decode_event(value):
    value = dict(value)
    snapshot = dict(value.pop("snapshot"))
    if snapshot.get("cuda") is not None:
        snapshot["cuda"] = CudaDeviceMemory(**snapshot["cuda"])
    control = value.pop("control", None)
    return ActorMetricsEvent(
        snapshot=ActorResourceSnapshot(**snapshot),
        control=None if control is None else ActorControlMetrics(**control),
        **value,
    )


class DurableExecutionSink(AsyncMetricsSink):
    def __init__(self, job_id, run_id, work_dir, max_events=2048):
        super().__init__(job_id, max_events=max_events)
        self.run_id = run_id
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.journal = os.path.join(work_dir, "execution_journal.jsonl")
        self.executions = {}
        self.actors = {}
        self.partitions = {}
        self.sealed = False
        self.run_succeeded = False
        self._journal_failed = False
        self._journal_hash = ""
        if os.path.exists(self.journal):
            with open(self.journal) as handle:
                for line in handle:
                    record = json.loads(line)  # Torn journals fail closed.
                    if record["run_id"] != run_id or record["job_id"] != job_id:
                        raise ValueError("journal run identity mismatch")
                    checksum = record.pop("sha256")
                    if checksum != self._record_hash(record):
                        raise ValueError("journal checksum mismatch")
                    self._apply(record["kind"], record["payload"])
                    self._journal_hash = checksum

    def _record_hash(self, record):
        return hashlib.sha256((self._journal_hash + json.dumps(record, sort_keys=True)).encode()).hexdigest()

    def _append(self, kind, payload):
        if self._journal_failed:
            raise RuntimeError("journal write previously failed; restart/replay is required")
        record = {"run_id": self.run_id, "job_id": self.job_id, "kind": kind, "payload": payload}
        record["sha256"] = self._record_hash(record)
        try:
            with open(self.journal, "a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            self._journal_failed = True
            raise
        self._apply(kind, payload)
        self._journal_hash = record["sha256"]

    def _open(self):
        if self._journal_failed:
            raise RuntimeError("journal write previously failed; restart/replay is required")
        if self.sealed:
            raise ValueError("run is sealed; new work is fenced")

    def expect_execution(self, expectation):
        self._open()
        key = expectation["execution_id"]
        if key in self.executions:
            if self.executions[key] != expectation:
                raise ValueError("execution expectation changed")
            return key
        self._append("expect", dict(expectation))
        return key

    def register_actor(self, identity):
        self._open()
        expected = self.executions[identity["execution_id"]]
        if any(identity.get(key) != expected.get(key) for key in ("plan_version", "applied_cap")):
            raise ValueError("actor did not apply the expected envelope")
        if not identity.get("node_id") or not identity.get("actor_id"):
            raise ValueError("actor runtime identity is required")
        key = identity["actor_incarnation_id"]
        if key in self.actors:
            if self.actors[key]["identity"] != identity:
                raise ValueError("actor identity changed")
            return key
        placement = expected.get("placement")
        if placement:
            slot = identity.get("placement_slot", -1)
            if not isinstance(slot, int) or not 0 <= slot < len(placement["device_ids"]):
                raise ValueError("actor placement slot is absent or invalid")
            device_id = placement["device_ids"][slot]
            device = next(item for item in placement["topology"] if item["device_id"] == device_id)
            if identity["node_id"] != device["node_id"] or identity.get("gpu_uuids") != [device["gpu_uuid"]]:
                raise ValueError("actor actual GPU does not match the placement envelope")
            if any(value["identity"]["execution_id"] == identity["execution_id"]
                   and value["identity"].get("placement_slot") == slot for value in self.actors.values()):
                raise ValueError("placement slot already acknowledged; replacement requires a new epoch")
        self._append("register", dict(identity))
        return key

    def begin_call(self, incarnation, call_id, input_rows):
        self._open()
        actor = self.actors[incarnation]
        if actor["active_call"] == call_id:
            return call_id
        if actor["active_call"] is not None or call_id != actor["last_call"] + 1:
            raise ValueError("call sequence is not contiguous")
        self._append("begin", {"incarnation": incarnation, "call_id": call_id, "input_rows": input_rows})
        return call_id

    def record(self, event):
        return self.record_batch([event])

    def record_batch(self, events):
        self._open()
        pending = []
        latest = dict(self._last_sequence)
        for event in events:
            if event.run_id != self.run_id or event.job_id != self.job_id:
                raise ValueError("metrics run identity mismatch")
            actor = self.actors[event.actor_incarnation_id]
            expected = self.executions[actor["identity"]["execution_id"]]
            if (
                event.stage_id != expected["stage_id"]
                or event.actor_id != actor["identity"]["actor_id"]
                or event.snapshot.batch_size > actor["identity"]["applied_cap"]
            ):
                raise ValueError("metrics do not match the acknowledged execution envelope")
            previous = latest.get(event.actor_incarnation_id, 0)
            if event.sequence <= previous:
                continue  # Retransmission after a lost ACK.
            if actor["active_call"] is None or event.sequence != previous + 1:
                raise ValueError("metrics sequence is not contiguous within an active call")
            latest[event.actor_incarnation_id] = event.sequence
            pending.append(serialize_metrics_event(event))
        if pending:
            self._append("metrics", pending)
        return latest

    def end_call(self, incarnation, call_id, final_sequence, counts, succeeded):
        self._open()
        actor = self.actors[incarnation]
        if actor["last_call"] == call_id and actor["active_call"] is None:
            if actor["final_sequence"] != final_sequence or actor["counts"] != counts:
                raise ValueError("retry changed the final watermark")
            return final_sequence
        if (
            actor["active_call"] != call_id
            or self._last_sequence.get(incarnation, 0) != final_sequence
            or actor["counts"] != counts
        ):
            raise ValueError("final watermark or cumulative outcome counts do not match receipt")
        if actor["input_rows"] > 0 and final_sequence <= actor["final_sequence"]:
            raise ValueError("nonempty call has no measured events")
        self._append(
            "end",
            {
                "incarnation": incarnation,
                "call_id": call_id,
                "final_sequence": final_sequence,
                "succeeded": bool(succeeded),
            },
        )
        return final_sequence

    def _apply(self, kind, payload):
        if kind == "expect":
            self.executions[payload["execution_id"]] = payload
        elif kind == "register":
            self.actors[payload["actor_incarnation_id"]] = {
                "identity": payload,
                "active_call": None,
                "last_call": 0,
                "final_sequence": 0,
                "input_rows": 0,
                "failed_calls": 0,
                "completed_input_rows": 0,
                "counts": {"succeeded_batches": 0, "failed_batches": 0, "oom_batches": 0},
            }
        elif kind == "begin":
            actor = self.actors[payload["incarnation"]]
            actor["active_call"] = payload["call_id"]
            actor["input_rows"] = payload["input_rows"]
        elif kind == "metrics":
            for value in payload:
                event = _decode_event(value)
                if event.sequence <= self._last_sequence.get(event.actor_incarnation_id, 0):
                    continue
                super().record(event)
                counts = self.actors[event.actor_incarnation_id]["counts"]
                counts["succeeded_batches" if event.snapshot.succeeded else "failed_batches"] += 1
                counts["oom_batches"] += int(snapshot_is_oom(event.snapshot))
        elif kind == "end":
            actor = self.actors[payload["incarnation"]]
            actor["active_call"] = None
            actor["last_call"] = payload["call_id"]
            actor["final_sequence"] = payload["final_sequence"]
            actor["failed_calls"] += int(not payload["succeeded"])
            if payload["succeeded"]:
                actor["completed_input_rows"] += actor["input_rows"]
        elif kind == "partition":
            self.partitions[str(payload["partition_id"])] = payload
        elif kind == "seal":
            self.sealed = True
            self.run_succeeded = payload["succeeded"]
        else:
            raise ValueError(f"unknown journal record {kind!r}")

    def record_partition(self, partition_id, input_rows, wall_seconds):
        self._open()
        if input_rows < 0 or not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError("partition measurement requires nonnegative rows and positive finite wall time")
        value = {"partition_id": partition_id, "input_rows": input_rows, "wall_seconds": wall_seconds}
        previous = self.partitions.get(str(partition_id))
        if previous is not None:
            if previous != value:
                raise ValueError("partition measurement changed")
            return partition_id
        self._append("partition", value)
        return partition_id

    def seal_run(self, succeeded):
        if not self.sealed:
            self._append("seal", {"succeeded": bool(succeeded)})
        return self.export_audit()

    def export_audit(self):
        """Publish derived artifacts atomically; never export just the ring."""
        fd, temporary = tempfile.mkstemp(prefix=".metrics_", dir=self.work_dir)
        seen = set()
        try:
            with os.fdopen(fd, "w") as target:
                if os.path.exists(self.journal):
                    with open(self.journal) as source:
                        for line in source:
                            record = json.loads(line)
                            if record["kind"] == "metrics":
                                for event in record["payload"]:
                                    key = (event["actor_incarnation_id"], event["sequence"])
                                    if key not in seen:
                                        target.write(json.dumps(event, sort_keys=True) + "\n")
                                        seen.add(key)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, os.path.join(self.work_dir, "elastic_juicer_metrics_events.jsonl"))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        executions = {}
        for key, expected in self.executions.items():
            actors = [value for value in self.actors.values() if value["identity"]["execution_id"] == key]
            executions[key] = {
                **expected,
                "registered_actors": len(actors),
                "working_actors": sum(value["last_call"] > 0 for value in actors),
                "completed_input_rows": sum(value["completed_input_rows"] for value in actors),
                "applied": len(actors) == expected.get("planned_actors") and bool(actors),
                "actors": [value["identity"] for value in actors],
                "placement_verified": bool(expected.get("placement")) and bool(actors)
                    and len(actors) == expected.get("planned_actors")
                    and all(value["identity"].get("placement_verified") is True for value in actors),
            }
        roster_present = all(value["registered_actors"] > 0 for value in executions.values())
        watermarks = all(
            actor["active_call"] is None
            and actor["failed_calls"] == 0
            and actor["final_sequence"] == self._last_sequence.get(key, 0)
            for key, actor in self.actors.items()
        )
        prefix_complete = len(seen) == self._received_events and not self._journal_failed
        complete = self.sealed and self.run_succeeded and roster_present and watermarks and prefix_complete
        status = {
            "schema_version": 2,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "received_events": self._received_events,
            "persisted_events": len(seen),
            "buffer_evictions": self._dropped_events,
            "sequence_gaps": self._sequence_gaps,
            "out_of_order_events": self._out_of_order_events,
            "outcome_counts": dict(self._outcome_counts),
            "known_loss": not prefix_complete or not watermarks,
            "observed_prefix_complete": prefix_complete,
            "complete": complete,
            "transport_status": "complete" if complete else "incomplete" if self.sealed else "open",
            "sealed": self.sealed,
            "roster_present": roster_present,
            "watermarks_confirmed": watermarks,
        }
        atomic_write_json(
            os.path.join(self.work_dir, "elastic_juicer_execution_audit.json"),
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "job_id": self.job_id,
                "executions": executions,
                "actors": self.actors,
                "partitions": self.partitions,
                "sealed": self.sealed,
            },
        )
        atomic_write_json(os.path.join(self.work_dir, "elastic_juicer_metrics_integrity.json"), status)
        return status


def create_durable_execution_sink(job_id, run_id, work_dir, max_events=2048):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    remote = ray.remote(num_cpus=0, max_restarts=2, max_task_retries=2)(DurableExecutionSink)
    return remote.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)
    ).remote(job_id, run_id, work_dir, max_events=max_events)
