from dataclasses import replace
import json
from types import SimpleNamespace

import pyarrow as pa
import pytest

from data_juicer.core.elasticjuicer.control_service import ControlService, StageProfile, load_stage_profiles, save_stage_profiles
from data_juicer.core.elasticjuicer.profile_context import context_id, make_context, validate_context
from data_juicer.core.elasticjuicer.ray_adaptive_mapper import RayAdaptiveMapperActor
from data_juicer.core.elasticjuicer.config_contract import validate_elastic_juicer_config
from tests.core.elasticjuicer.test_ray_adaptive_mapper_actor import NoopSampler, RemoteMethod


class ContextMapper:
    elastic_juicer_cost_columns = ("cost",)

    def __init__(self):
        self.budget = 24

    def elastic_juicer_profile_resource_context(self):
        return {"budget": self.budget}

    def process(self, batch):
        if sum(batch["cost"].to_pylist()) > self.budget:
            raise RuntimeError("CUDA out of memory")
        return batch


def batch(cost, n=32):
    return pa.table({"cost": [cost] * n})


def actor(monkeypatch, service=None):
    import ray
    monkeypatch.setattr(ray, "get", lambda ref, **kwargs: ref)
    value = RayAdaptiveMapperActor(ContextMapper, initial_batch_size=8, max_batch_size=8,
                                  sampler_factory=NoopSampler, job_id="job", resource_class="cpu",
                                  profile_seed_enabled=True, profile_context_enabled=True)
    service = service or ControlService("job")
    value._profile_control_service = SimpleNamespace(
        get_stage_profile=RemoteMethod(service.get_stage_profile),
        report_stage_profile=RemoteMethod(service.report_stage_profile),
    )
    return value, service


def test_cost_recurrence_resets_and_restores_only_matching_evidence(monkeypatch):
    value, service = actor(monkeypatch)
    assert value._profile_seed_evidence["status"] == "deferred_until_input"
    value(batch(1))
    light = value._profile_context
    assert value.controller.state.oom_events == 0
    value(batch(8))
    heavy = value._profile_context
    assert light != heavy
    assert value._profile_seed_evidence["status"] == "no_profile"
    assert value.controller.state.oom_events == 2
    value(batch(1))
    assert value._profile_seed_evidence["status"] == "applied"
    assert value._profile_seed_evidence["initial_batch_size"] == 8
    assert value.controller.state.oom_events == 0
    value(batch(8))
    assert value._profile_seed_evidence["status"] == "applied"
    assert value._profile_seed_evidence["initial_batch_size"] <= 3
    assert value.controller.state.oom_events == 0
    assert len(service.snapshot()["stage_profiles"]) == 2


def test_mixed_input_neither_inherits_nor_poisons_prior(monkeypatch):
    value, service = actor(monkeypatch)
    value(batch(1))
    prior = service.snapshot()["stage_profiles"]
    value(pa.table({"cost": [1, 8] * 16}))
    assert value._profile_seed_evidence["status"] == "mixed_input_cost"
    assert service.snapshot()["stage_profiles"] == prior
    value(batch(1))
    assert value._profile_seed_evidence["initial_batch_size"] == 8


def test_resource_and_limit_changes_cannot_inherit(monkeypatch):
    value, _ = actor(monkeypatch)
    value(batch(1))
    previous = value._profile_context
    value.operator.budget = 16
    value(batch(1))
    assert value._profile_context != previous
    assert value._profile_seed_evidence["status"] == "no_profile"
    value.controller.set_hard_limit(4)
    value(batch(1))
    assert value._profile_seed_evidence["status"] == "no_profile"
    assert value.controller.hard_limit == 4


def test_legacy_namespace_persistence_ttl_and_no_fabricated_safe_size(monkeypatch, tmp_path):
    value, service = actor(monkeypatch)
    context, _ = make_context(batch(1), value.operator, value.controller)
    base = StageProfile("job", "stage-0", "op", 8, None, 100, profile_context=context)
    service.report_stage_profile(base)
    assert service.get_stage_profile("job", "stage-0", now_ms=100) is None
    merged = service.report_stage_profile(replace(base, safe_batch_size=2, oom_upper_bound=4))
    assert merged.safe_batch_size == 2  # NOT an unobserved size 3.
    path = str(tmp_path / "profiles.json")
    save_stage_profiles(path, [merged], "pipeline", "plan")
    assert json.loads(open(path).read())["schema_version"] == 2  # Downgrade fence.
    assert load_stage_profiles(path, "pipeline", "plan") == [merged]
    assert service.get_stage_profile("job", "stage-0", now_ms=100, profile_context=context) == merged
    assert service.get_stage_profile("job", "stage-0", now_ms=2_000_000, profile_context=context) is None
    assert context_id(context)


def test_expired_context_not_rejuvenated_by_new_write(monkeypatch):
    value, service = actor(monkeypatch)
    context, _ = make_context(batch(1), value.operator, value.controller)
    base = StageProfile("job", "stage-0", "op", 2, 4, 100, profile_context=context)
    service.report_stage_profile(base)
    fresh = replace(base, safe_batch_size=8, oom_upper_bound=None, observed_at_ms=2_000_000)
    assert service.report_stage_profile(fresh) == fresh


@pytest.mark.parametrize("value", ["{}", "[]", '{"schema_version":2}', "x" * 8193, 3])
def test_bad_context_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        validate_context(value)


def test_missing_cost_and_resource_contract(monkeypatch):
    value, _ = actor(monkeypatch)
    assert make_context(pa.table({"other": [1]}), value.operator, value.controller)[1] == "missing_input_cost"
    value.operator.elastic_juicer_profile_resource_context = None
    assert make_context(batch(1), value.operator, value.controller)[1] == "missing_resource_contract"


def test_context_requires_opt_in_seed_and_ej():
    with pytest.raises(ValueError, match="profile_context"):
        validate_elastic_juicer_config({"elastic_juicer_profile_context": True})


def test_positional_operator_configuration_partitions_context(monkeypatch):
    value, _ = actor(monkeypatch)
    a, _ = make_context(batch(1), value.operator, value.controller, operator_args=("model-a",))
    b, _ = make_context(batch(1), value.operator, value.controller, operator_args=("model-b",))
    assert a != b
    assert make_context(batch(1), value.operator, value.controller, operator_args=(object(),))[0] == ""


def test_mid_call_quota_tightens_and_invalidates_profile_epoch(monkeypatch):
    from data_juicer.core.elasticjuicer.quota import QuotaEnvelope, current_time_ms
    value, service = actor(monkeypatch)
    value(batch(1))
    before = service.snapshot()["stage_profiles"]
    epoch = value._profile_context_epoch
    now = current_time_ms()
    quota = QuotaEnvelope("job", value.actor_id, value.actor_incarnation_id, revision=1,
                          issued_at_ms=now, expires_at_ms=now + 10000, max_batch_size=4, reason="test")
    value._apply_quota(quota)
    assert value.controller.hard_limit == 4
    assert value._profile_seed_evidence["status"] == "quota_changed"
    assert value._profile_context_epoch == epoch + 1
    value._report_stage_profile()
    assert service.snapshot()["stage_profiles"] == before
    value(batch(1))
    assert value._profile_seed_evidence["status"] == "no_profile"


def test_bad_service_context_response_is_not_seeded(monkeypatch):
    value, _ = actor(monkeypatch)
    wrong = StageProfile("job", "stage-0", "op", 2, 4, 100)
    value._profile_control_service.get_stage_profile = RemoteMethod(lambda *args: wrong)
    value(batch(1))
    assert value._profile_seed_evidence["status"] == "context_mismatch"
    assert value.controller.oom_upper_bound is None


def test_untrusted_advisory_bounds_do_not_partially_seed(monkeypatch):
    value, _ = actor(monkeypatch)
    context, _ = make_context(batch(1), value.operator, value.controller)
    invalid = StageProfile("job", value.stage_id, value.op_name, True, 4, 100,
                           op_fingerprint=value.op_fingerprint, resource_class=value.resource_class,
                           profile_context=context)
    value._profile_control_service.get_stage_profile = RemoteMethod(lambda *args: invalid)
    value(batch(1))
    assert value._profile_seed_evidence["status"] == "invalid_profile"
    assert value.controller.oom_upper_bound is None


def test_resource_provider_error_remains_advisory(monkeypatch):
    value, _ = actor(monkeypatch)
    value.operator.elastic_juicer_profile_resource_context = lambda: (_ for _ in ()).throw(KeyError("budget"))
    result = value(batch(1))
    assert result.num_rows == 32
    assert value._profile_seed_evidence["status"] == "invalid_resource_contract"
