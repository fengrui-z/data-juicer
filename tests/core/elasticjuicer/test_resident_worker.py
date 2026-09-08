from types import SimpleNamespace

import pytest

from data_juicer.core.elasticjuicer import resident_worker as module


class Operator:
    pass


class FakeActor:
    def __init__(self, operator_class, operator_instance=None, **options):
        self.operator = operator_instance or operator_class()
        self.controller = SimpleNamespace(hard_limit=options["execution_audit"]["applied_cap"])
        self.metrics_reporter = SimpleNamespace(identity=options)
        self.closed = False

    def __call__(self, batch):
        if batch == "fail":
            raise RuntimeError("ACK failed")
        return batch

    def close(self):
        self.closed = True

    def get_metrics_state(self):
        return {"last_sequence": 1}


def execution(partition=0, **kwargs):
    return dict(run_id="r", partition_id=partition, execution_id=f"e{partition}",
                stage_id="s", plan_version="p", planned_actors=4, applied_cap=8,
                placement={"device_ids": [0, 0, 1, 1]}, **kwargs)


def worker(monkeypatch):
    monkeypatch.setattr(module, "RayAdaptiveMapperActor", FakeActor)
    return module.ResidentMapperWorker(Operator)


def test_operator_reused_but_controller_and_execution_binding_reset(monkeypatch):
    value = worker(monkeypatch)
    first = value.bind(execution())
    original = value.actor
    assert value([1]) == [1]
    value.finish_partition()
    second = value.bind(execution(1))
    assert value.actor.operator is original.operator
    assert value.actor.controller is not original.controller
    assert original.closed
    assert first["worker_lifetime"]["physical_worker_id"] == second["worker_lifetime"]["physical_worker_id"]
    assert second["worker_lifetime"]["binding_index"] == 1


def test_ack_release_failure_fences_rebind(monkeypatch):
    import ray
    value = worker(monkeypatch)
    value.bind(execution())
    value([1])
    value.actor._control_service_handle = SimpleNamespace(deregister=SimpleNamespace(remote=lambda *args: None))
    value.actor.job_id = value.actor.actor_id = value.actor.actor_incarnation_id = "id"
    monkeypatch.setattr(ray, "get", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("release")))
    with pytest.raises(TimeoutError):
        value.finish_partition()
    with pytest.raises(RuntimeError):
        value.bind(execution(1))


def test_real_wrapper_retains_instance_without_constructor(monkeypatch):
    from data_juicer.core.elasticjuicer.ray_adaptive_mapper import RayAdaptiveMapperActor
    from tests.core.elasticjuicer.test_ray_adaptive_mapper_actor import NoopSampler
    constructions = []
    class Counting:
        def __init__(self):
            constructions.append(self)
    first = RayAdaptiveMapperActor(Counting, sampler_factory=NoopSampler)
    second = RayAdaptiveMapperActor(Counting, sampler_factory=NoopSampler, operator_instance=first.operator)
    assert constructions == [first.operator]
    assert first.operator is second.operator
    assert first.controller is not second.controller
    assert second._operator_init_seconds == 0


def test_no_rebind_before_finish_or_after_failure(monkeypatch):
    value = worker(monkeypatch)
    value.bind(execution())
    with pytest.raises(RuntimeError):
        value.bind(execution(1))
    with pytest.raises(RuntimeError):
        value.finish_partition()
    with pytest.raises(RuntimeError, match="ACK failed"):
        value("fail")
    with pytest.raises(RuntimeError):
        value.bind(execution(1))


def test_actual_quota_tightening_cannot_be_reset_by_stale_plan_metadata(monkeypatch):
    value = worker(monkeypatch)
    value.bind(execution())
    value([1])
    value.actor.controller.hard_limit = 4
    value.finish_partition()
    with pytest.raises(ValueError, match="actual quota changed"):
        value.bind(execution(1))


@pytest.mark.parametrize("change", [dict(plan_version="new"), dict(applied_cap=4), dict(planned_actors=2),
                                  dict(run_id="other"), dict(partition_id=3), dict(execution_id="e0"),
                                  dict(placement={"device_ids": [0, 1, 1, 1]})])
def test_reject_unvalidated_envelope_or_partition_change(monkeypatch, change):
    value = worker(monkeypatch)
    value.bind(execution())
    value([1])
    value.finish_partition()
    with pytest.raises(ValueError):
        value.bind({**execution(1), **change})
