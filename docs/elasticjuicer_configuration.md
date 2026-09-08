# ElasticJuicer configuration contract

Status: experimental product surface; `ej` is the only stable controller policy.

ElasticJuicer keeps correctness and default behavior conservative while the
research policies accumulate repeated production-like evidence.  Enabling an
experimental policy is an explicit two-step action:

```yaml
elastic_juicer_adaptive_batching: true
elastic_juicer_experimental_policies: true
elastic_juicer_controller_policy: safebo
elastic_juicer_cost_column: img_pixels
```

## Public options

| Option | Default | Contract |
| --- | --- | --- |
| `elastic_juicer_adaptive_batching` | `false` | Master actor-local adaptive batching flag. |
| `elastic_juicer_controller_policy` | `ej` | `ej` is stable. Baseline, cost-aware, hybrid, SafeBO, and bandit policies are experimental. |
| `elastic_juicer_experimental_policies` | `false` | Required before selecting any policy other than `ej`; it does not enable adaptive batching by itself. |
| `elastic_juicer_cost_column` | `null` | Optional pre-operation numeric cost feature. Null invokes the cost-aware policies' defined no-signal fallback. |
| `elastic_juicer_profile_seed` | `false` | Advisory stage-profile seed. Failure or rejection degrades to an unseeded start. |
| `elastic_juicer_profiles_path` | `null` | Optional absolute local path stable across driver job IDs. The partitioned executor otherwise selects a path below its work directory. |
| `elastic_juicer_profile_ttl_ms` | `1800000` | Maximum profile age accepted by the control service. |

`elastic_juicer_pipeline_fingerprint`, detected resource class, Ray handles,
job IDs, and actor incarnation IDs are internal values.  They are not public
configuration knobs.

## Validation and failure behavior

- Policy names are enumerated and invalid names fail during configuration
  parsing, before Ray starts.
- Experimental policies require the explicit experimental opt-in.
- Policy, cost, seeding, and profile-path options require adaptive batching.
- An explicitly configured cost column is only valid for cost-aware policies.
  A missing configured column fails at the first concrete batch; leaving the
  option null is the supported no-signal mode.
- An explicit profiles path must be an absolute local path.  Persisted profile
  files are written with `fsync` plus `os.replace`; incompatible schema,
  pipeline fingerprint, resource class, stale data, or malformed entries are
  rejected and processing continues without a seed.
- CUDA-dependent experimental policies fail with a policy-specific message if
  their Ray actor has no CUDA device.

Every non-config-only job logs a compact `ElasticJuicer effective
configuration` record containing the enabled state, policy and stability tier,
cost signal, seed state, profiles path, and profile TTL.

## Default-change gates

`elastic_juicer_execution_audit=true` opts the public partitioned executor
into acknowledged delivery. Each run has an independent UUID and artifact
directory, addressed by `elastic_juicer_metrics_run.json`. Actors acknowledge
their applied cap, plan fingerprint, node/Ray identity and visible GPU UUIDs.
Before returning each outer batch they durably acknowledge every micro-batch
sample, the final sequence and cumulative outcome counts. The driver seals
the run only after materialization and export; sealing fences new calls.
No actor destructor or best-effort final asynchronous message is required.

The driver-node sink uses a checksummed append-only journal, bounded packets
and idempotent replay after lost acknowledgements or sink restart. A delivery
failure prevents the actor result from being accepted. Torn/corrupt journals
fail closed. The exported sidecar includes the journal's complete sample
history, not just the bounded Captain sampling ring. Audit failure leaves the
run incomplete. The opt-in adds blocking durability costs and does not by
itself enforce GPU placement or certify an actor-resize performance gain.

A successful, validated same-slice retry after a minimum-batch OOM records
local recovery evidence. With the existing reprobe budget enabled, a fresh
success window permits a larger-batch probe without requiring Captain.
Static bounds and quota caps still apply. A new OOM cancels that permission;
persistent minimum-batch failures still propagate after bounded retries.
The default policy and feature flags are unchanged.

Partition-boundary replanning checks safety before performance exploration.
Freezing an unprofitable scale-up or entering its cooldown never suppresses
drift-triggered batch-cap tightening. Explicit actor counts remain fixed, but
their batch cap can still tighten; a safety action never relaxes a previously
learned cap. GPU fractions are accounted across all planned stages on each
device. These are plan-feasibility constraints, not proof of runtime placement
or physical VRAM isolation.

The memory model reserves resident memory before dividing the remaining
budget by per-row working memory. Probe schema v3 records CUDA allocator
residency after warmup and the steady-state allocator peak; online drift
uses `(allocator_peak - allocator_resident) / batch_size` in the same scope.
Whole-device free-memory deltas remain conservative planning diagnostics,
not interchangeable drift baselines. Missing or incompatible residency
measurements disable that stage's memory-drift signal, not local OOM safety.
Old probe caches are rejected and must be reprobed. This linear model does
not yet predict input-feature-dependent costs, fragmentation, or changes
in model residency; it is not a physical memory-isolation guarantee.

Metrics snapshots carry the same OOM classification used by local retries.
The sink retains received success/failure/OOM counters and sequence-gap
counts independently of its bounded sample buffer. The driver writes
`elastic_juicer_metrics_integrity.json` alongside the event sidecar, including
received versus persisted counts, known loss and write failures. Events are
acknowledged only after `fsync`; consumers deduplicate by job, incarnation
and sequence because a failed append may be retried. These counters cover
sink receipt, not all producer attempts. Without an expected actor roster
and final-sequence acknowledgements, end-to-end completeness remains
unverified (`complete=false`), even when the observed prefix is intact.
This audit exposes loss; it does not yet provide a reliable transport.

Defaults change only through a separately reviewed decision record:

- `profile_seed=true`: at least 10 paired slow-start/high-cost recovery runs,
  100% output correctness, median resume-wall improvement of at least 5%, and
  no material regression in negative-control workloads.
- `controller_policy=safebo`: at least 10 randomized paired runs in each core
  workload class, zero unhandled OOMs, 100% correctness, a 95% confidence upper
  bound no worse than 3% versus `ej`, and a material probe reduction.

Until those gates pass, `ej` and `profile_seed=false` remain the product
defaults.
