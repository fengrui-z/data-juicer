"""Experimental fixed-placement worker, independent of Ray Data pool internals.

Not wired into the public partition executor. The caller must serially bind,
execute, and finish each work unit. Only the operator/process is retained;
controllers, quotas, call/metric sequences and audit execution bindings are
fresh. No resizing, crash recovery or concurrent partitions are supported.
"""

from uuid import uuid4

from .ray_adaptive_mapper import RayAdaptiveMapperActor


class ResidentMapperWorker:
    def __init__(self, operator_class, **actor_options):
        if "operator_instance" in actor_options or "worker_lifetime" in actor_options:
            raise ValueError("operator lifetime is owned by the resident worker")
        if "execution_audit" in actor_options or "partition_id" in actor_options:
            raise ValueError("execution binding must be supplied by bind()")
        self.operator_class = operator_class
        self.options = dict(actor_options)
        self.physical_worker_id = uuid4().hex
        self.binding_index = -1
        self.binding = None
        self.actor = None
        self.finished = True
        self.failed = False
        self.calls = 0

    def bind(self, execution):
        if self.failed or not self.finished:
            raise RuntimeError("previous execution is not successfully finished")
        if not execution.get("placement") or not execution.get("run_id"):
            raise ValueError("resident worker requires an audited placement binding")
        partition = execution.get("partition_id")
        if isinstance(partition, bool) or not isinstance(partition, int) or partition < 0:
            raise ValueError("partition_id must be a non-negative integer")
        if self.binding:
            if self.actor.controller.hard_limit != self.binding["applied_cap"]:
                # Resetting a local controller must not silently undo an
                # actual safety quota merely because plan metadata is stale.
                raise ValueError("actual quota changed; fixed-envelope resident rebind is not supported")
            immutable = ("run_id", "stage_id", "plan_version", "planned_actors", "applied_cap", "placement")
            if any(execution.get(key) != self.binding.get(key) for key in immutable):
                raise ValueError("resident prototype requires an unchanged stage/plan/resource envelope")
            if partition != self.binding["partition_id"] + 1 or execution["execution_id"] == self.binding["execution_id"]:
                raise ValueError("resident executions must be distinct contiguous serial partitions")
        self.finished = False
        self.calls = 0
        self.binding_index += 1
        try:
            self.actor = RayAdaptiveMapperActor(
                self.operator_class, **self.options, execution_audit=execution, partition_id=partition,
                operator_instance=None if self.actor is None else self.actor.operator,
                worker_lifetime={"physical_worker_id": self.physical_worker_id,
                                 "binding_index": self.binding_index,
                                 "controller_reset": True},
            )
        except BaseException:
            self.failed = True  # Never silently replace a failed incarnation.
            raise
        self.binding = dict(execution)
        return dict(self.actor.metrics_reporter.identity)

    def __call__(self, batch):
        if self.finished or self.failed or self.actor is None:
            raise RuntimeError("worker has no active healthy execution")
        try:
            output = self.actor(batch)  # Includes the durable final ACK.
            self.calls += 1
            return output
        except BaseException:
            self.failed = True
            raise

    def finish_partition(self):
        if self.finished or self.failed or self.calls < 1:
            raise RuntimeError("cannot finish an empty, failed or already finished execution")
        try:
            self.actor.close()
            control = getattr(self.actor, "_control_service_handle", None)
            if control is not None:
                import ray
                # close() is best-effort for ordinary retiring Ray Data
                # actors; a resident rebind needs a confirmed release.
                ray.get(control.deregister.remote(self.actor.job_id, self.actor.actor_id,
                                                  self.actor.actor_incarnation_id), timeout=15)
            reference = getattr(self.actor, "_timing_ref", None)
            if reference is not None:
                import ray
                ray.get(reference, timeout=15)
        except BaseException:
            self.failed = True
            raise
        self.finished = True
        return {"physical_worker_id": self.physical_worker_id, "binding_index": self.binding_index,
                "calls": self.calls, "metrics": self.actor.get_metrics_state()}
