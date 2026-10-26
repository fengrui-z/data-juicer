"""Opt-in Python API for a bounded, resident DJ fork/join execution path.

Ray must already be initialized by the caller. This executor owns only its own
actors, never the cluster. It does not claim checkpoint recovery, live resizing,
automatic placement optimization, or exactly-once external side effects.
"""

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import ray

from data_juicer.core.elasticjuicer.dag_plan import (
    DAGContractError,
    DAGPayload,
    ResidentDAGPlan,
    payload_bytes,
    validate_columns,
)
from data_juicer.core.elasticjuicer.dag_worker import ResidentDAGWorker


@dataclass(frozen=True)
class DAGBatchResult:
    unit: int
    payload_ref: object
    input_ids: tuple
    accepted_ids: tuple
    rejected_ids: tuple

    def columns(self):
        """Materialize only when the consumer needs the final output payload."""
        return ray.get(self.payload_ref).columns


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


class ResidentDAGExecutor:
    """Reuse model instances across units, groups and successive input streams.

    ``adaptive`` changes only the local microbatch policy. Logical DAG, physical
    groups, worker placement and data transport remain identical in both arms.
    Operators explicitly declare immutable inputs and replay-safe invocation via
    ``elastic_juicer_dag_safe=True``. Failures fence/retire this executor's pool.
    """

    def __init__(self, plan: ResidentDAGPlan, *, adaptive=True, artifact_dir=None, runtime_env=None):
        if not ray.is_initialized():
            raise RuntimeError("initialize Ray explicitly before constructing ResidentDAGExecutor")
        self.plan, self.compiled = plan, plan.compile()
        self.adaptive = bool(adaptive)
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=False)
        self.actors, self.idle = {}, {}
        self.closed, self.running = False, False
        self._pause = threading.Event()
        self.last_summary = None
        started = time.perf_counter()
        required = {
            "CPU": sum(p.workers * p.num_cpus for p in plan.pools),
            "GPU": sum(p.workers * p.num_gpus for p in plan.pools),
        }
        total = ray.cluster_resources()
        if any(n > total.get(k, 0) + 1e-9 for k, n in required.items()):
            raise DAGContractError(f"pool exceeds cluster resources: requested {required}, total {total}")
        # ray.kill releases resources asynchronously. Allow a short bounded
        # handoff between owned pools, without changing other jobs' resources.
        while True:
            available = ray.available_resources()
            if all(n <= available.get(k, 0) + 1e-9 for k, n in required.items()):
                break
            if time.perf_counter() - started >= min(5, plan.startup_timeout_seconds):
                raise DAGContractError(f"insufficient free resources: requested {required}, available {available}")
            time.sleep(0.05)
        try:
            remote = ray.remote(max_restarts=0, max_task_retries=0)(ResidentDAGWorker)
            pending = {}
            for pool in plan.pools:
                self.idle[pool.name] = []
                for slot in range(pool.workers):
                    key = (pool.name, slot)
                    options = dict(num_cpus=pool.num_cpus, num_gpus=pool.num_gpus)
                    if runtime_env is not None:
                        options["runtime_env"] = runtime_env
                    actor = remote.options(**options).remote(plan, pool.name, slot, self.adaptive)
                    self.actors[key] = actor
                    pending[actor.ready.remote()] = key
            self.worker_identities = []
            while pending:
                done, _ = ray.wait(list(pending), timeout=1)
                for ref in done:
                    self.worker_identities.append(ray.get(ref))
                    self.idle[pending.pop(ref)[0]].append(self._key_for_identity(self.worker_identities[-1]))
                if time.perf_counter() - started > plan.startup_timeout_seconds:
                    raise TimeoutError("resident pool startup timed out")
            devices = [w["device"]["uuid"] for w in self.worker_identities if w["device"]]
            if len(devices) != len(set(devices)):
                raise DAGContractError("whole-GPU workers unexpectedly share a physical GPU")
            self.startup_seconds = time.perf_counter() - started
            if self.artifact_dir:
                _atomic_json(
                    self.artifact_dir / "pool.json",
                    {
                        "plan_fingerprint": plan.fingerprint(),
                        "adaptive": self.adaptive,
                        "startup_seconds": self.startup_seconds,
                        "workers": self.worker_identities,
                        "integration": "DJ load_ops + AdaptiveBatchController/OOMSafeAdaptiveMapper + resident DAG",
                        "control_service": False,
                        "automatic_topology": False,
                        "live_resizing": False,
                    },
                )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _key_for_identity(identity):
        return identity["pool"], identity["slot"]

    def request_pause(self):
        """Stop admitting input; drain admitted units. Close releases the GPUs."""
        self._pause.set()

    def iter_batches(self, batches):
        """Yield completed units; payloads travel between actors as ObjectRefs.

        Output order across units is unspecified; IDs/order within a unit are
        preserved. Caller-retained output refs are outside the executor's byte
        budget. Consume/release each result for bounded streaming memory.
        """
        if self.closed or self.running:
            raise RuntimeError("executor is closed or already executing")
        self.running = True
        self._pause.clear()
        run_id = uuid4().hex
        directory = self.artifact_dir / run_id if self.artifact_dir else None
        if directory:
            directory.mkdir()
        stream = (directory / "events.jsonl").open("x") if directory else None
        started = time.perf_counter()
        source = iter(batches)
        active, pending, seen = {}, {}, set()
        next_unit, admitted_rows, accepted_rows, rejected_rows, completed_units, peak_active = 0, 0, 0, 0, 0, 0
        exhausted, completed = False, False
        status, failure = "running", None
        try:
            while not exhausted or active:
                while not exhausted and not self._pause.is_set() and len(active) < self.compiled["capacity"]:
                    try:
                        columns = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    ids = tuple(validate_columns(columns, self.plan.id_column))
                    if seen.intersection(ids):
                        raise DAGContractError("duplicate input IDs across work units")
                    if payload_bytes(columns) > self.plan.max_batch_bytes:
                        raise DAGContractError("input exceeded max_batch_bytes")
                    seen.update(ids)
                    admitted_rows += len(ids)
                    active[next_unit] = {
                        "ids": ids,
                        "refs": {"$input": ray.put(DAGPayload(columns))},
                        "submitted": set(),
                        "done": set(),
                    }
                    next_unit += 1
                    peak_active = max(peak_active, len(active))
                if not active:
                    break
                # Downstream-first dispatch avoids filling a shared GPU pool
                # exclusively with upstream work while joins retain buffers.
                for group_name in reversed(self.compiled["group_order"]):
                    group = self.compiled["groups"][group_name]
                    for unit, state in active.items():
                        if not self.idle[group.pool]:
                            break
                        if (
                            group_name in state["submitted"]
                            or not self.compiled["dependencies"][group_name] <= state["done"]
                        ):
                            continue
                        parents = sorted(
                            {p for n in group.nodes for p in self.compiled["nodes"][n].parents if p not in group.nodes}
                        )
                        if any(not self.compiled["nodes"][n].parents for n in group.nodes):
                            parents.append("$input")
                        key = self.idle[group.pool].pop()
                        refs = (
                            self.actors[key]
                            .process.options(num_returns=len(group.nodes) + 1)
                            .remote(group_name, parents, *[state["refs"][p] for p in parents])
                        )
                        pending[refs[-1]] = dict(
                            unit=unit, group=group_name, key=key, refs=refs[:-1], submitted=time.perf_counter()
                        )
                        state["submitted"].add(group_name)
                if not pending:
                    raise DAGContractError("DAG scheduler made no progress")
                done, _ = ray.wait(list(pending), num_returns=1, timeout=0.2)
                for ref in done:
                    task = pending.pop(ref)
                    metadata = ray.get(ref)  # Small metadata only, never pixels.
                    state, group = active[task["unit"]], self.compiled["groups"][task["group"]]
                    self.idle[group.pool].append(task["key"])
                    state["done"].add(group.name)
                    state["refs"].update(zip(group.nodes, task["refs"]))
                    if stream:
                        for event in metadata:
                            stream.write(
                                json.dumps({"run_id": run_id, "unit": task["unit"], "group": group.name, **event})
                                + "\n"
                            )
                        stream.flush()
                    if self.plan.output in group.nodes:
                        terminal = next(e for e in metadata if e["node"] == self.plan.output)
                        actual = tuple(terminal["output_ids"])
                        if not set(actual) <= set(state["ids"]):
                            raise DAGContractError("terminal output changed source IDs")
                        rejected = tuple(i for i in state["ids"] if i not in set(actual))
                        accepted_rows += len(actual)
                        rejected_rows += len(rejected)
                        completed_units += 1
                        result = DAGBatchResult(
                            task["unit"], state["refs"][self.plan.output], state["ids"], actual, rejected
                        )
                        del active[task["unit"]]
                        yield result
                if any(time.perf_counter() - t["submitted"] > self.plan.task_timeout_seconds for t in pending.values()):
                    raise TimeoutError("DAG task timed out")
            if admitted_rows != accepted_rows + rejected_rows:
                raise DAGContractError("terminal ID conservation failed")
            status = "paused" if self._pause.is_set() and not exhausted else "complete"
            completed = True
        except BaseException as error:
            status, failure = "failed", {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            if stream:
                stream.close()
            self.running = False
            elapsed = time.perf_counter() - started
            self.last_summary = {
                "run_id": run_id,
                "status": status,
                "failure": failure,
                "plan_fingerprint": self.plan.fingerprint(),
                "adaptive": self.adaptive,
                "input_rows": admitted_rows,
                "accepted_rows": accepted_rows,
                "rejected_rows": rejected_rows,
                "completed_units": completed_units,
                "seconds_including_consumer": elapsed,
                "peak_inflight_units": peak_active,
                "reserved_payload_bytes": peak_active * self.compiled["unit_reservation_bytes"],
                "pool_startup_seconds": self.startup_seconds,
            }
            if not completed:
                self.close()  # Includes consumer cancellation and partial iteration.
            if directory:
                _atomic_json(directory / "summary.json", self.last_summary)

    def close(self):
        if self.closed:
            return
        self.closed = True
        actors, self.actors = self.actors, {}
        for actor in actors.values():
            try:
                ray.kill(actor, no_restart=True)
            except ray.exceptions.RayError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
