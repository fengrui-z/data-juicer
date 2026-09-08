from dataclasses import replace

import pytest

from data_juicer.core.elasticjuicer.parallelism_controller import PartitionReplanController, StageObservation
from data_juicer.core.elasticjuicer.parallelism_planner import plan_envelope_fingerprint
from data_juicer.core.elasticjuicer.performance_window import (
    PerformanceWindow,
    build_window,
    input_cost_counts,
)
from . import test_probe_plan_autoscale as fixtures


def controller(tmp_path):
    fixture, plan = fixtures.ReplanControllerTest()._controller()
    return PartitionReplanController(
        plan,
        fixture.demands,
        fixture.cluster,
        fixture.config,
        events_path=str(tmp_path / "plan_events.jsonl"),
        baseline_bytes_per_row=fixture.baseline_bytes_per_row,
    )


def observe(control, partition, goodput=10, cost="a", **changes):
    window = PerformanceWindow(
        partition,
        plan_envelope_fingerprint(control.plan),
        {"resource": "fixed"},
        {cost: 1.0},
        goodput,
        10,
        80,
        partition * 10 + 1,
        partition * 10 + 9,
        [0.1],
    )
    window = replace(window, **changes)
    return {"stage:a": StageObservation("stage:a", 10, 0, 10, None, 10, window)}


@pytest.mark.parametrize("gain,verdict,actors", [(12, "effective", 4), (10, "ineffective", 3)])
def test_same_cost_trial_verdict_uses_steady_window_not_partition_wall(tmp_path, gain, verdict, actors):
    control = controller(tmp_path)
    assert control.on_partition_complete(0, observe(control, 0), 100)[0].action == "scale_up"
    control.on_partition_complete(1, observe(control, 1, gain), 1)  # misleading partition wall
    assert control.performance_trials[-1]["verdict"] == verdict
    assert control.plan.stages["stage:a"].planned_actors == actors
    assert bool(control._frozen_stages) == (verdict == "ineffective")


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"cost": "b"}, "input_cost_drift"),
        ({"resource_state": {"resource": "changed"}}, "resource_change"),
        ({"steady_batches": 1}, "insufficient_stable_samples"),
        ({"issues": ["oom_or_failed_batch"]}, "oom_or_failed_batch"),
        ({"plan_version": "unexpected"}, "unexpected_plan_version"),
        ({"start": 1}, "stale_or_overlapping_window"),
        ({"partition_id": 0}, "wrong_partition_window"),
        ({"startup_seconds": []}, "missing_startup_measurement"),
        ({"startup_seconds": [float("nan")]}, "missing_startup_measurement"),
    ],
)
def test_incomparable_trial_rolls_back_without_freezing_and_reestablishes_baseline(tmp_path, change, reason):
    control = controller(tmp_path)
    control.on_partition_complete(0, observe(control, 0), 10)
    control.on_partition_complete(1, observe(control, 1, 1, **change), 1)
    record = control.performance_trials[-1]
    assert record["verdict"] == "inconclusive" and reason in record["reasons"]
    assert record["marginal_gain"] is None
    assert not control._frozen_stages
    assert control.plan.stages["stage:a"].planned_actors == 3
    assert control.on_partition_complete(2, observe(control, 2, cost="b"), 10)[0].action == "scale_up"
    control.on_partition_complete(3, observe(control, 3, 12, cost="b"), 12)
    assert control.performance_trials[-1]["verdict"] == "effective"


def test_freeze_is_condition_scoped_and_safety_survives(tmp_path):
    control = controller(tmp_path)
    control.on_partition_complete(0, observe(control, 0), 10)
    control.on_partition_complete(1, observe(control, 1), 10)
    assert control.on_partition_complete(2, observe(control, 2), 10) == []
    assert control.on_partition_complete(3, observe(control, 3, cost="b"), 10)[0].action == "scale_up"
    unsafe = observe(control, 4, cost="b")
    unsafe["stage:a"].oom_batches = 1
    unsafe["stage:a"].bytes_per_row = 20 * 1024**3
    assert control.on_partition_complete(4, unsafe, 1)[0].action == "scale_down"
    assert control.performance_trials[-1]["verdict"] == "inconclusive"


def test_missing_input_context_does_not_authorize_performance_exploration(tmp_path):
    control = controller(tmp_path)
    observation = {"stage:a": StageObservation("stage:a", sample_count=10, rows_per_sec=10)}
    assert control.on_partition_complete(0, observation, 10) == []


def test_declared_joint_cost_distribution_is_bounded_and_missing_is_unknown():
    assert input_cost_counts({"a": [1, 2], "b": [2, 1]}, ["a", "b"]) == {
        '[["a",1.0],["b",2.0]]': 1,
        '[["a",2.0],["b",1.0]]': 1,
    }
    for data, columns in [({}, []), ({}, ["a"]), ({"a": [float("nan")]}, ["a"]), ({"a": list(range(257))}, ["a"])]:
        assert input_cost_counts(data, columns) is None


def test_common_steady_window_excludes_each_actor_startup_and_rejects_lost_prefix():
    actors, timings, events = {}, {}, []
    for inc in ("a", "b"):
        actors[inc] = {
            "identity": {"partition_id": 0, "node_id": "node", "plan_version": "plan"},
            "last_call": 1,
            "final_sequence": 5,
        }
        timings[inc] = {"call_id": 1, "init_to_first_call_seconds": 0.2}
        for seq in range(1, 6):
            events.append(
                {
                    "actor_incarnation_id": inc,
                    "sequence": seq,
                    "snapshot": {"succeeded": True, "batch_size": 8},
                    "comparison_context": {"cost_counts": {"a": 8}, "start": seq, "end": seq + 0.5},
                }
            )
    window = build_window(events, 0, actors, timings, {"gpu": "fixed"})
    assert not window.invalid_reasons()
    assert window.start == 1.5 and window.steady_batches == 8
    assert window.steady_goodput == 64 / 4
    lost = build_window(events[1:], 0, actors, timings, {"gpu": "fixed"})
    assert "incomplete_metrics_window" in lost.issues
