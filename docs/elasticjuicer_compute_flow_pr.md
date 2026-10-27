# Proposed PR: Add queue-aware compute-flow scheduling to resident DAG pools

Resident DAG pools already reuse models across execution groups, but their
downstream-first dispatch policy has no explicit demand-based concurrency plan.
This change adds an opt-in `ComputeFlowConfig` to `ResidentDAGExecutor`: a driver
controller assigns idle actor leases using ready/in-flight rows, measured service
cost and queue age, while each operator retains its own adaptive batch state.
Busy tasks finish normally; targets never preempt actors or change physical Ray
resource reservations. The existing dispatch policy remains the default.

The controller includes rate smoothing, periodic replanning, integer hysteresis,
idle borrowing and starvation protection. A static slot-ownership mode supports
controlled experiments. Plan/lease events record actual actor transitions and
link them to pool-specific epochs. `DAGNode.initial_batch_size` optionally starts
the existing local controller below its cap; global control never overwrites
local success/OOM learning.

This is a stacked change on Resident DAG v0.2.0 (`77885fb`). A PR against DJ
main requires the resident executor and its ElasticJuicer dependencies first.
There is no new default CLI executor, actor resizing, model eviction, automatic
placement search, durable checkpoint or exactly-once external commit guarantee.
All models assigned to a pool must fit together on each GPU actor.

Validation: 115 focused controller/real-Ray/mapper regressions passed. Six real
image arms ran on eight H20 GPUs, each processing the same 512 unique images
twice, with CPU decode, QualiCLIP/DINOv2 GPU branches, an exact join and full
artifact export. Cross-run exported output checks passed at fixed `1e-4`
absolute tolerance (maximum score difference `1.55e-6`, embedding difference
zero), with no physical OOMs and eight GPUs used per arm. Both dynamic modes
changed groups on all eight actors without reloading operator instances or
overlapping leases. The first attempt hit an underestimated decoded-payload
budget; the harness now derives the bound before creating GPU pools, and the
entire primary matrix was rerun.

The short experiment establishes integration and mechanism correctness, not a
general performance win. Warm dual-layer execution took 15.36 s versus 17.51 s
for static allocation with adaptive batching, 14.02 s for the existing shared
pool and 14.40 s for colocated parallel groups. Repeated larger comparisons are
required before recommending the new policy as the default. See
[the validation report](elasticjuicer_compute_flow_validation.md) for timing
definitions, experiment limits, provenance and reproducible commands.
