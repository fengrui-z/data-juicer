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

Defaults change only through a separately reviewed decision record:

- `profile_seed=true`: at least 10 paired slow-start/high-cost recovery runs,
  100% output correctness, median resume-wall improvement of at least 5%, and
  no material regression in negative-control workloads.
- `controller_policy=safebo`: at least 10 randomized paired runs in each core
  workload class, zero unhandled OOMs, 100% correctness, a 95% confidence upper
  bound no worse than 3% versus `ej`, and a material probe reduction.

Until those gates pass, `ej` and `profile_seed=false` remain the product
defaults.
