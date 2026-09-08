"""Conservative input/resource keys for advisory profile inheritance.

Only homogeneous, explicitly declared input costs are eligible. Unknown or
mixed input never reads/writes a contextual profile. This is intentionally
not a similarity model: the operator owns the meaning of its cost/resource
declarations and must include every relevant budget in the latter.
"""

import hashlib
import json

from .adaptive_mapper import _batch_length
from .performance_window import input_cost_counts


def validate_context(value):
    if not isinstance(value, str) or len(value) > 8192:
        raise ValueError("profile context must be a bounded canonical JSON string")
    if not value:  # Legacy namespace, never a fallback for contextual reads.
        return value
    data = json.loads(value)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported profile context schema")
    if not data.get("cost") or not data.get("resources"):
        raise ValueError("profile context requires cost and resource evidence")
    if json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) != value:
        raise ValueError("profile context must be canonical and finite")
    return value


def context_id(value):
    return hashlib.sha256(value.encode()).hexdigest() if value else None


def make_context(batch, operator, controller, execution_audit=None, operator_args=()):
    """Return (canonical key, reason); do not infer budgets from free memory."""
    costs = input_cost_counts(batch, getattr(operator, "elastic_juicer_cost_columns", ()))
    if not costs or sum(costs.values()) != _batch_length(batch):
        return "", "missing_input_cost"
    if len(costs) != 1:
        return "", "mixed_input_cost"
    provider = getattr(operator, "elastic_juicer_profile_resource_context", None)
    if not callable(provider):
        return "", "missing_resource_contract"
    try:
        resources = provider()
        if not isinstance(resources, dict) or not resources:
            return "", "missing_resource_contract"
        value = json.dumps({
            "schema_version": 1, "cost": json.loads(next(iter(costs))),
            "resources": resources,
            # Constructor arguments may contain paths or credentials. Their
            # identity matters to matching, not their plaintext in journals.
            "operator_args_sha256": hashlib.sha256(json.dumps(
                operator_args, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode()).hexdigest(),
            "limits": [controller.min_batch_size, controller.max_batch_size, controller.hard_limit],
            "placement": (execution_audit or {}).get("placement"),
        }, sort_keys=True, separators=(",", ":"), allow_nan=False)
        validate_context(value)
        return value, "eligible"
    except Exception:  # Advisory operator declaration must not brick execution.
        return "", "invalid_resource_contract"
