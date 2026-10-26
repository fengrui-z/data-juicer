# Resident DAG runtime: experimental v0.1

Base revision: `22e54c2`. Development branch: `codex/ej-dag-resident-pool`.
The previous 20k/100k benchmark remains a separate frozen experiment.

## Delivered scope

`data_juicer.core.executor.resident_dag_executor.ResidentDAGExecutor` is an
explicit Python API. The existing `ray` and `ray_partitioned` executors and
their defaults are unchanged. This is not yet an `executor_type` CLI option.

A logical DAG specifies DJ mapper recipes, parents, input/output columns,
row-preservation or subset filtering, and optional image-version join keys.
Execution groups specify sequential or independent parallel execution. Worker
pools specify CPU resources and zero or one whole GPU per actor. Several groups
can use the same pool: its operators remain resident across groups, work units,
and successive calls to `iter_batches`.

With `adaptive=True`, each logical operator owns an existing
`AdaptiveBatchController` and `OOMSafeAdaptiveMapper`. With `adaptive=False`,
the same retry harness uses `FixedBatchController`; an OOM propagates. Neither
mode changes the graph, grouping, pool sizes, or transport. Inner operators
retain distinct microbatch caps rather than taking the minimum cap of a group.

This path does not yet connect Captain, shared profile seeding, acknowledged
execution audit, automatic topology search, or live resizing. Its JSON event
log is diagnostic evidence, not a durable exactly-once execution journal.
Per-model CUDA allocator peaks are not reported: concurrent branches share an
allocator, and resetting its peak per branch would produce misleading data.

## Operator contract

Operators must be registered, batched DJ `Mapper`s and explicitly set
`elastic_juicer_dag_safe = True`. This declares immutable inputs and replay-safe
execution, including after a partially executed OOM slice. External side
effects require their own idempotency; this runtime provides no transaction
guarantee for them. Error skipping and per-operator runtime environments are
rejected. Use compatible dependencies in a job-level Ray runtime environment.

Inputs and outputs are dictionaries of list/tuple columns with unique string or
integer IDs. An operator receives only its declared reads and the ID column.
It returns IDs and its declared writes; additional returned columns are ignored.
Input containers are isolated and NumPy image views are read-only. Operators
must not mutate other referenced payload objects. Output order is normalized
to input order. A preserving node must return exactly the same IDs; a subset
node may drop IDs but cannot add or duplicate them.

Joins match exact ID sets and optional version columns. Different branch
filtering does not silently become an inner join. Conflicting writes from
unrelated branches fail. Sequentially newer column versions supersede their
ancestors. Contraction into execution groups must remain acyclic. Parallel
groups cannot contain dependency edges or overlapping writes.

## Example

```python
import ray
from data_juicer.core.elasticjuicer.dag_plan import (
    DAGNode, ExecutionGroup, WorkerPool, ResidentDAGPlan,
)
from data_juicer.core.executor.resident_dag_executor import ResidentDAGExecutor

# Register/import your own pure, batched DJ operators on every worker.
plan = ResidentDAGPlan(
    nodes=(
        DAGNode("left", {"left_mapper": {}}, reads=("image",), writes=("score",), batch_size=32),
        DAGNode("right", {"right_mapper": {}}, reads=("image",), writes=("embedding",), batch_size=64),
        DAGNode("join", {"pack_mapper": {}}, parents=("left", "right"),
                reads=("score", "embedding"), writes=("record",)),
    ),
    groups=(ExecutionGroup("features", ("left", "right"), "gpu", parallel=True),
            ExecutionGroup("export", ("join",), "cpu")),
    pools=(WorkerPool("gpu", workers=8, num_cpus=4, num_gpus=1),
           WorkerPool("cpu", workers=2)),
    output="join", output_columns=("record",),
    operator_modules=("my_project.operators",),
)
ray.init(address="local", num_cpus=40, num_gpus=8)
try:
    with ResidentDAGExecutor(plan, adaptive=True, artifact_dir="run-001") as executor:
        for result in executor.iter_batches(input_column_batches):
            write_idempotently(result.columns(), result.rejected_ids)
finally:
    ray.shutdown()
```

Use the same plan with only `adaptive=False/True` changed for a controlled
comparison. Reuse an executor for successive streams to measure warm execution;
operator instances and their local OOM knowledge persist. A new physical plan
requires a new executor. GPU UUIDs and per-operator instance IDs are recorded.
Model initialization may be lazy inside an operator, so constructor time alone
must not be described as complete model warmup.

## Bounded execution, failure and pause

The scheduler sends object references to actors and reads only metadata. It
prioritizes ready downstream groups. Each admitted unit reserves
`(number_of_nodes + 1) * max_batch_bytes`; both admission count and byte budget
limit in-flight work. Inputs and each node output are checked against the
payload estimate. This is not a bound on temporary decode/model allocations or
physical VRAM. Operators must enforce their own temporary-memory limits.

Consume/release results promptly: references retained by the caller are outside
the executor budget. The final `output_columns` projection can discard pixels
before returning results. Global output order is unspecified; per-unit order
and IDs are preserved. Input ID validation retains a job-local ID set.

Failures, timeouts and closed partial iterators fence/retire all owned actors;
they never stop another Ray cluster or retry a whole task implicitly. Only the
local OOM harness retries a replay-safe slice. Always use the context manager.

`request_pause()` stops input admission and drains admitted units. The caller
can continue with the remaining input iterator and the same executor, or close
the pool to release GPUs. This is an in-process drain operation, not durable
checkpoint/resume across process failure. Completed outputs need caller-owned
commits before they can be considered durable.

## Versioned milestones

1. v0.1: explicit DAG/group contracts, fixed resident pools, exact joins, actual
   branch overlap, identical transport with fixed/adaptive microbatch controls.
2. v0.2: measured batching/transport improvements and representative real-model
   validation, preserving the v0.1 contract and recording paired evidence.
3. Next: compare a small set of fused/dedicated candidate plans, reuse compatible
   probes, then add queue-driven boundary replanning and durable drain/resume.

Every stage is committed separately. Performance claims require identical code,
models, input slice, logical DAG and budgets, output correctness, cold/warm time
separation, and repeated paired measurements. A GPU smoke test is not a speedup
claim. The interrupted 100k matrix must not restart implicitly during development.
