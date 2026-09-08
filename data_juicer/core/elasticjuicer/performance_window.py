"""Fail-closed, same-condition evidence for a partition scale-up trial.

Cost columns are declared by the operator, not inferred from its latency.
Steady goodput uses a common, single-node monotonic interval after EVERY
actor's first successful batch. It is not adjacent-partition wall goodput.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
import math


def distribution(counts):
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())} if total else {}


def distance(left, right):
    return sum(abs(left.get(key, 0) - right.get(key, 0)) for key in left.keys() | right.keys()) / 2


def input_cost_counts(batch, columns):
    if not columns:
        return None
    try:
        data = batch.to_pydict() if hasattr(batch, "to_pydict") else batch
        counts = Counter()
        for row in zip(*(data[column] for column in columns)):
            values = tuple(float(value) for value in row)
            if not all(math.isfinite(value) and value >= 0 for value in values):
                return None
            counts[json.dumps(list(zip(columns, values)), separators=(",", ":"))] += 1
            if len(counts) > 256:
                return None  # Bounded evidence; no silent histogram truncation.
        return dict(counts)
    except (KeyError, TypeError, ValueError):
        return None


@dataclass
class PerformanceWindow:
    partition_id: int
    plan_version: str
    resource_state: dict
    cost_distribution: dict
    steady_goodput: float
    steady_batches: int
    steady_rows: int
    start: float
    end: float
    startup_seconds: list = field(default_factory=list)
    issues: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def invalid_reasons(self):
        reasons = list(self.issues)
        if not self.cost_distribution:
            reasons.append("missing_input_cost")
        elif (not all(math.isfinite(value) and value >= 0 for value in self.cost_distribution.values())
              or not math.isclose(sum(self.cost_distribution.values()), 1.0, abs_tol=1e-8)):
            reasons.append("invalid_cost_distribution")
        if not self.resource_state or not self.plan_version:
            reasons.append("missing_resource_or_plan")
        if not self.startup_seconds or any(not math.isfinite(value) or value < 0 for value in self.startup_seconds):
            reasons.append("missing_startup_measurement")
        if (
            self.steady_batches < 4
            or self.steady_rows < 16
            or self.end <= self.start
            or not all(math.isfinite(value) for value in (self.start, self.end, self.steady_goodput))
            or self.steady_goodput <= 0
        ):
            reasons.append("insufficient_stable_samples")
        return sorted(set(reasons))


def comparison_reasons(before, after, expected_plan):
    if after is None:
        return ["missing_window"]
    reasons = before.invalid_reasons() + after.invalid_reasons()
    if after.partition_id <= before.partition_id or after.start < before.end:
        reasons.append("stale_or_overlapping_window")
    if after.plan_version != expected_plan:
        reasons.append("unexpected_plan_version")
    if before.resource_state != after.resource_state:
        reasons.append("resource_change")
    if distance(before.cost_distribution, after.cost_distribution) > 0.05:
        reasons.append("input_cost_drift")
    return sorted(set(reasons))


def same_conditions(before, after):
    return bool(
        after
        and not after.invalid_reasons()
        and before.resource_state == after.resource_state
        and distance(before.cost_distribution, after.cost_distribution) <= 0.05
    )


def build_window(events, partition_id, actors, timings, resource_state):
    """events are complete ordered stage events, not a lossy metrics ring."""
    issues = []
    by_actor = defaultdict(list)
    costs = Counter()
    versions, nodes, quota = set(), set(), set()
    for event in events:
        snapshot = event["snapshot"]
        context = event.get("comparison_context") or {}
        identity = actors.get(event["actor_incarnation_id"], {}).get("identity", {})
        versions.add(identity.get("plan_version"))
        nodes.add(identity.get("node_id"))
        control = event.get("control") or {}
        quota.add((control.get("quota_revision"), control.get("hard_limit")))
        if not snapshot["succeeded"]:
            issues.append("oom_or_failed_batch")
            continue
        histogram = context.get("cost_counts")
        if not histogram or sum(histogram.values()) != snapshot["batch_size"]:
            issues.append("missing_input_cost")
            continue
        start, end = context.get("start"), context.get("end")
        if start is None or end is None or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            issues.append("missing_operator_interval")
            continue
        costs.update(histogram)
        by_actor[event["actor_incarnation_id"]].append((start, end, snapshot["batch_size"], histogram))
    expected = {key for key, value in actors.items() if value["identity"].get("partition_id") == partition_id}
    for key in expected:
        sequence = sorted(event["sequence"] for event in events if event["actor_incarnation_id"] == key)
        if sequence != list(range(1, actors[key]["final_sequence"] + 1)):
            issues.append("incomplete_metrics_window")
    if set(by_actor) != expected or not expected:
        issues.append("incomplete_actor_coverage")
    if len(nodes) != 1 or None in nodes:
        issues.append("unsupported_cross_node_clock")
    if len(versions) != 1 or None in versions or len(quota) != 1:
        issues.append("plan_or_quota_changed_within_window")
    ordered = [sorted(values) for values in by_actor.values()]
    start = max((values[0][1] for values in ordered), default=0)
    end = min((values[-1][1] for values in ordered), default=0)
    steady = [value for values in ordered for value in values[1:] if value[0] >= start and value[1] <= end]
    if any(sum(value[0] >= start and value[1] <= end for value in values[1:]) < 2 for values in ordered):
        issues.append("insufficient_per_actor_stable_samples")
    # Reject a cost transition even when the partition-wide mean stays fixed.
    chronological = sorted((value for values in ordered for value in values), key=lambda value: value[0])
    halves = [Counter(), Counter()]
    for i, value in enumerate(chronological):
        halves[int(i >= len(chronological) // 2)].update(value[3])
    if distance(*(distribution(half) for half in halves)) > 0.05:
        issues.append("input_cost_drift_within_window")
    startup = []
    for key in expected:
        timing = timings.get(key, {})
        if timing.get("call_id") != actors[key]["last_call"]:
            issues.append("missing_startup_measurement")
        else:
            startup.append(timing["init_to_first_call_seconds"])
    rows = sum(value[2] for value in steady)
    steady_costs = Counter()
    for value in steady:
        steady_costs.update(value[3])
    if distance(distribution(costs), distribution(steady_costs)) > 0.05:
        issues.append("steady_window_cost_selection")
    return PerformanceWindow(
        partition_id,
        next(iter(versions)) if len(versions) == 1 else "",
        dict(resource_state, nodes=sorted(str(node) for node in nodes), quota=sorted(quota, key=str)),
        distribution(costs),
        rows / (end - start) if end > start else 0,
        len(steady),
        rows,
        start,
        end,
        startup,
        sorted(set(issues)),
    )
