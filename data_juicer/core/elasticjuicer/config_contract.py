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


def validate_elastic_juicer_config(cfg: Any, *, require_job_id: bool = False) -> None:
    """Fail fast on invalid public ElasticJuicer configuration.

    The validation is intentionally limited to configuration semantics.  A
    dataset-column lookup and CUDA availability remain runtime checks because
    they depend on the concrete Ray actor and input batch.
    """

    enabled = bool(_get(cfg, "elastic_juicer_adaptive_batching", False))
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
