"""Opt-in public-executor fixed GPU pool; identical fresh/resident data paths.

Only one wave of input/output batches is resident on the driver. Materialized
outputs live in Ray's object store (and may spill), not a driver-side row list.
No retries, concurrent partitions, resizing or checkpoint recovery are implied.
"""

from .phase_timing import phase


def validate_managed_operators(ops):
    from data_juicer.ops.base_op import Mapper

    if len(ops) != 1:
        raise ValueError("managed workers require exactly one mapper")
    op = ops[0]
    if not (
        isinstance(op, Mapper)
        and op.is_batched_op()
        and op.use_ray_actor()
        and float(op.num_gpus or 0) > 0
        and getattr(op, "elastic_juicer_row_preserving", False) is True
    ):
        raise ValueError("managed workers require an explicitly row-preserving batched GPU actor mapper")
    if getattr(op, "skip_op_error", True):
        raise ValueError("managed workers require skip_op_error=false")


class ManagedMapperPool:
    def __init__(self, mode, timing=None):
        if mode not in ("managed_fresh", "resident"):
            raise ValueError("unknown managed worker mode")
        self.mode = mode
        self._ej_timing = timing
        self.workers = []
        self.closed = False
        self.options = None

    def map(self, dataset, *, constructor, selector, batch_size, num_cpus, num_gpus):
        import pyarrow as pa
        import ray
        from .resident_worker import ResidentMapperWorker

        if self.closed:
            raise RuntimeError("managed pool is closed or failed")
        execution = constructor["execution_audit"]
        options = {key: value for key, value in constructor.items() if key not in ("execution_audit", "partition_id")}
        try:
            if not execution or selector is None:
                raise ValueError("managed workers require audited placement")
            count = execution["planned_actors"]
            rows = dataset.count()
            if rows < count:
                raise ValueError("partition must contain at least one row per planned worker")
            # Small tail partitions still execute on every planned worker.
            dispatch_size = min(batch_size, max(1, rows // count))
            if self.options is not None and options != self.options:
                raise ValueError("managed worker constructor options changed across partitions")
            self.options = options
            with phase(self, "managed_start_bind", partition_id=execution["partition_id"]):
                if not self.workers:
                    remote = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus, max_restarts=0, max_task_retries=0)(
                        ResidentMapperWorker
                    )
                    # Append as each creation succeeds, so partial creation is cleaned up.
                    for _ in range(count):
                        self.workers.append(remote.options(**selector()).remote(**options))
                if len(self.workers) != count:
                    raise ValueError("managed worker count changed")
                ray.get([worker.bind.remote(execution) for worker in self.workers], timeout=300)
            output_refs, pending = [], []

            def drain():
                outputs = ray.get([ref for ref, _ in pending], timeout=300)
                for output, (_, expected_rows) in zip(outputs, pending):
                    table = output if isinstance(output, pa.Table) else pa.Table.from_pydict(output)
                    if len(table) != expected_rows:
                        raise ValueError("managed mapper violated row-preserving contract")
                    output_refs.append(ray.put(table))
                pending.clear()

            with phase(self, "managed_dispatch", partition_id=execution["partition_id"]):
                for index, batch in enumerate(
                    dataset.iter_batches(batch_size=dispatch_size, batch_format="pyarrow", prefetch_batches=0)
                ):
                    worker = self.workers[index % count]
                    pending.append((worker.__call__.remote(batch), len(batch)))
                    if len(pending) == count:
                        drain()
                if pending:
                    drain()
            with phase(self, "managed_finish_ack", partition_id=execution["partition_id"]):
                # Includes confirmed lease release and final actor timing delivery.
                ray.get([worker.finish_partition.remote() for worker in self.workers], timeout=300)
            if self.mode == "managed_fresh":
                self._retire()
            return ray.data.from_arrow_refs(output_refs)
        except BaseException:
            self.close()
            raise

    def _retire(self):
        import ray

        with phase(self, "managed_retire"):
            workers, self.workers = self.workers, []
            errors = []
            for worker in workers:
                try:
                    ray.kill(worker, no_restart=True)
                except Exception as error:
                    errors.append(error)
            if errors:
                raise RuntimeError("failed to retire managed worker(s)") from errors[0]

    def close(self):
        self.closed = True
        self._retire()
