"""Freeze the observed 64-batch admission boundary, not a recommendation.

Automatic capacity and explicit capacity are different contracts. Changing
the greedy search/economic policy requires new evidence, not a relaxed test.
"""

from data_juicer.core.elasticjuicer.parallelism_planner import (
    ClusterSpec,
    StageDemand,
    _marginal_makespan_gain,
    _rounds,
    plan_parallelism,
)


def demand(**updates):
    return StageDemand(
        stage_key="model", op_name="model", probed_throughput_rps=24,
        effective_peak_bytes=2**30, batch_size=8, available_batches=64,
        actor_startup_seconds=0, **updates,
    )


def test_automatic_integer_round_gate_stops_at_six_not_eight():
    cluster = ClusterSpec(total_cpus=36, num_devices=8, device_memory_bytes=96 * 2**30)
    stage = plan_parallelism([demand()], cluster).stages["model"]
    assert _rounds(64, 6) == 11 and _rounds(64, 7) == 10 and _rounds(64, 8) == 8
    assert _marginal_makespan_gain(64, 6) == 1 / 11 < .1
    assert stage.planned_actors == 6 and stage.fixed is False
    assert stage.device_ids == list(range(6))


def test_explicit_eight_reserves_capacity_without_changing_economic_gate():
    cluster = ClusterSpec(total_cpus=36, num_devices=8, device_memory_bytes=96 * 2**30)
    plan = plan_parallelism([demand(explicit=True, explicit_actors=8)], cluster)
    stage = plan.stages["model"]
    assert stage.planned_actors == 8 and stage.fixed is True
    assert stage.device_ids == list(range(8))
    assert plan.planning_gates["min_marginal_makespan_gain"] == .1
