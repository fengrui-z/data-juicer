"""Public configuration contract for the ElasticJuicer runtime.

This module deliberately has no Ray or Torch imports.  Configuration files,
CLI parsing, embedders, and the runtime can therefore share the same early
validation without starting a Ray job or touching a CUDA context.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

STABLE_CONTROLLER_POLICIES = frozenset({"ej"})
EXPERIMENTAL_CONTROLLER_POLICIES = frozenset(
    {
        "bandit",
        "bisect",
        "costaware",
        "fixed",
        "halving",
        "hybrid",
        "proportional",
        "safebo",
    }
)
CONTROLLER_POLICIES = STABLE_CONTROLLER_POLICIES | EXPERIMENTAL_CONTROLLER_POLICIES
COST_SIGNAL_POLICIES = frozenset({"bandit", "costaware", "hybrid", "safebo"})
CUDA_CONTROLLER_POLICIES = frozenset({"bandit", "costaware", "hybrid", "proportional", "safebo"})


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _optional_nonempty_string(value: Any, name: str) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string or null")
    return value.strip()


def controller_policy_stability(policy: str) -> str:
    """Return the user-facing stability tier for one validated policy."""

    if policy in STABLE_CONTROLLER_POLICIES:
        return "stable"
    if policy in EXPERIMENTAL_CONTROLLER_POLICIES:
        return "experimental"
    raise ValueError(f"unknown ElasticJuicer controller policy: {policy!r}")


def _validate_auto_parallelism_options(cfg: Any) -> None:
    """Validate the probe/plan/replan option block for auto-parallelism."""

    from .gpu_probe import ProbeParams
    from .parallelism_controller import ReplanConfig

    auto_parallelism = bool(_get(cfg, "elastic_juicer_auto_parallelism", False))
    probe_enabled = bool(_get(cfg, "elastic_juicer_probe_enabled", True))
    replan_enabled = bool(_get(cfg, "elastic_juicer_replan_enabled", True))

    if not auto_parallelism:
        active_options = []
        if not probe_enabled:
            active_options.append("elastic_juicer_probe_enabled=false")
        if not replan_enabled:
            active_options.append("elastic_juicer_replan_enabled=false")
        if active_options:
            raise ValueError(
                f"{', '.join(active_options)} is only meaningful with elastic_juicer_auto_parallelism=true"
            )
        return

    ProbeParams(
        warmup_batches=int(_get(cfg, "gpu_probe_warmup_batches", 1)),
        steady_batches=int(_get(cfg, "gpu_probe_steady_batches", 3)),
        sample_rows=int(_get(cfg, "gpu_probe_sample_rows", 256)),
        batch_cap_multiplier=int(_get(cfg, "gpu_probe_batch_cap_multiplier", 8)),
        memory_headroom=float(_get(cfg, "gpu_probe_memory_headroom", 0.10)),
        timeout_seconds=_get(cfg, "gpu_probe_timeout_seconds", None),
    ).validate()
    ReplanConfig(
        drift_threshold=float(_get(cfg, "elastic_juicer_drift_threshold", 1.3)),
        marginal_gain_epsilon=float(_get(cfg, "elastic_juicer_marginal_gain_epsilon", 0.03)),
        cooldown_partitions=int(_get(cfg, "elastic_juicer_parallelism_cooldown_partitions", 1)),
    ).validate()

    min_batches_per_actor = int(_get(cfg, "elastic_juicer_min_batches_per_actor", 4))
    if min_batches_per_actor < 1:
        raise ValueError("elastic_juicer_min_batches_per_actor must be >= 1")
    startup_amortization_ratio = float(_get(cfg, "elastic_juicer_startup_amortization_ratio", 0.10))
    if not 0.0 <= startup_amortization_ratio <= 1.0:
        raise ValueError("elastic_juicer_startup_amortization_ratio must be in [0, 1]")
    min_marginal_makespan_gain = float(_get(cfg, "elastic_juicer_min_marginal_makespan_gain", 0.10))
    if not 0.0 <= min_marginal_makespan_gain < 1.0:
        raise ValueError("elastic_juicer_min_marginal_makespan_gain must be in [0, 1)")
    actor_startup_seconds = _get(cfg, "elastic_juicer_actor_startup_seconds", None)
    if actor_startup_seconds is not None and float(actor_startup_seconds) < 0:
        raise ValueError("elastic_juicer_actor_startup_seconds must be non-negative or null")

    max_workers = int(_get(cfg, "elastic_juicer_max_gpu_workers_per_device", 4))
    if max_workers < 1:
        raise ValueError("elastic_juicer_max_gpu_workers_per_device must be >= 1")
    budget_fraction = float(_get(cfg, "elastic_juicer_device_memory_budget_fraction", 0.90))
    if not 0.0 < budget_fraction <= 1.0:
        raise ValueError("elastic_juicer_device_memory_budget_fraction must be in (0, 1]")


def validate_elastic_juicer_config(cfg: Any, *, require_job_id: bool = False) -> None:
    """Fail fast on invalid public ElasticJuicer configuration.

    The validation is intentionally limited to configuration semantics.  A
    dataset-column lookup and CUDA availability remain runtime checks because
    they depend on the concrete Ray actor and input batch.
    """

    enabled = bool(_get(cfg, "elastic_juicer_adaptive_batching", False))
    if _get(cfg, "elastic_juicer_enforce_gpu_placement", False) and not (
        _get(cfg, "elastic_juicer_execution_audit", False) and _get(cfg, "elastic_juicer_auto_parallelism", False)
    ):
        raise ValueError("elastic_juicer_enforce_gpu_placement requires execution audit and auto-parallelism")
    policy = str(_get(cfg, "elastic_juicer_controller_policy", "ej") or "ej").strip().lower()
    experimental_enabled = bool(_get(cfg, "elastic_juicer_experimental_policies", False))
    profile_seed = bool(_get(cfg, "elastic_juicer_profile_seed", False))
    cost_column = _optional_nonempty_string(
        _get(cfg, "elastic_juicer_cost_column", None),
        "elastic_juicer_cost_column",
    )
    profiles_path = _optional_nonempty_string(
        _get(cfg, "elastic_juicer_profiles_path", None),
        "elastic_juicer_profiles_path",
    )

    if policy not in CONTROLLER_POLICIES:
        allowed = ", ".join(sorted(CONTROLLER_POLICIES))
        raise ValueError(f"unknown elastic_juicer_controller_policy {policy!r}; choose one of: {allowed}")
    if policy in EXPERIMENTAL_CONTROLLER_POLICIES and not experimental_enabled:
        raise ValueError(
            f"elastic_juicer_controller_policy={policy!r} is experimental; "
            "set elastic_juicer_experimental_policies=true to opt in"
        )

    if not enabled:
        active_options = []
        if _get(cfg, "elastic_juicer_execution_audit", False):
            active_options.append("elastic_juicer_execution_audit")
        if policy != "ej":
            active_options.append("elastic_juicer_controller_policy")
        if cost_column is not None:
            active_options.append("elastic_juicer_cost_column")
        if profile_seed:
            active_options.append("elastic_juicer_profile_seed")
        if profiles_path is not None:
            active_options.append("elastic_juicer_profiles_path")
        if active_options:
            raise ValueError(f"{', '.join(active_options)} requires elastic_juicer_adaptive_batching=true")

    if cost_column is not None and policy not in COST_SIGNAL_POLICIES:
        supported = ", ".join(sorted(COST_SIGNAL_POLICIES))
        raise ValueError(
            "elastic_juicer_cost_column is only consumed by cost-aware policies "
            f"({supported}); current policy is {policy!r}"
        )

    if profiles_path is not None:
        if profiles_path.startswith(("s3://", "hdfs://")):
            raise ValueError("elastic_juicer_profiles_path currently supports local filesystems only")
        if not os.path.isabs(profiles_path):
            raise ValueError("elastic_juicer_profiles_path must be absolute so it remains stable across job restarts")

    if require_job_id and enabled and not _get(cfg, "job_id", None):
        raise ValueError("elastic_juicer_adaptive_batching requires a resolved job_id")

    _validate_auto_parallelism_options(cfg)
    if require_job_id and bool(_get(cfg, "elastic_juicer_auto_parallelism", False)) and not _get(cfg, "job_id", None):
        raise ValueError("elastic_juicer_auto_parallelism requires a resolved job_id")


def effective_elastic_juicer_config(cfg: Any) -> Dict[str, Any]:
    """Return a compact, safe-to-log snapshot of the effective public contract."""

    validate_elastic_juicer_config(cfg)
    policy = str(_get(cfg, "elastic_juicer_controller_policy", "ej") or "ej").strip().lower()
    cost_column = _optional_nonempty_string(
        _get(cfg, "elastic_juicer_cost_column", None),
        "elastic_juicer_cost_column",
    )
    profiles_path = _optional_nonempty_string(
        _get(cfg, "elastic_juicer_profiles_path", None),
        "elastic_juicer_profiles_path",
    )
    return {
        "enabled": bool(_get(cfg, "elastic_juicer_adaptive_batching", False)),
        "controller_policy": policy,
        "controller_policy_stability": controller_policy_stability(policy),
        "experimental_policies_enabled": bool(_get(cfg, "elastic_juicer_experimental_policies", False)),
        "cost_signal": "none" if cost_column is None else cost_column,
        "profile_seed": bool(_get(cfg, "elastic_juicer_profile_seed", False)),
        "profiles_path": profiles_path,
        "profile_ttl_ms": int(_get(cfg, "elastic_juicer_profile_ttl_ms", 1_800_000)),
        "auto_parallelism": bool(_get(cfg, "elastic_juicer_auto_parallelism", False)),
        "probe_enabled": bool(_get(cfg, "elastic_juicer_probe_enabled", True)),
        "replan_enabled": bool(_get(cfg, "elastic_juicer_replan_enabled", True)),
        "max_gpu_workers_per_device": int(_get(cfg, "elastic_juicer_max_gpu_workers_per_device", 4)),
        "min_batches_per_actor": int(_get(cfg, "elastic_juicer_min_batches_per_actor", 4)),
        "startup_amortization_ratio": float(_get(cfg, "elastic_juicer_startup_amortization_ratio", 0.10)),
        "min_marginal_makespan_gain": float(_get(cfg, "elastic_juicer_min_marginal_makespan_gain", 0.10)),
        "actor_startup_seconds": (
            None
            if _get(cfg, "elastic_juicer_actor_startup_seconds", None) is None
            else float(_get(cfg, "elastic_juicer_actor_startup_seconds"))
        ),
    }


def normalize_elastic_juicer_config(cfg: Any) -> Mapping[str, Any]:
    """Validate and normalize policy/string fields on mutable config objects."""

    validate_elastic_juicer_config(cfg)
    policy = str(_get(cfg, "elastic_juicer_controller_policy", "ej") or "ej").strip().lower()
    cost_column = _optional_nonempty_string(
        _get(cfg, "elastic_juicer_cost_column", None),
        "elastic_juicer_cost_column",
    )
    profiles_path = _optional_nonempty_string(
        _get(cfg, "elastic_juicer_profiles_path", None),
        "elastic_juicer_profiles_path",
    )
    values = {
        "elastic_juicer_controller_policy": policy,
        "elastic_juicer_cost_column": cost_column,
        "elastic_juicer_profiles_path": profiles_path,
    }
    if cfg is not None:
        for key, value in values.items():
            try:
                cfg[key] = value
            except (TypeError, AttributeError):
                setattr(cfg, key, value)
    return values
