import pytest

from data_juicer.core.elasticjuicer.gpu_placement import PlacementSelector, verify_worker_placement


def _topology():
    return [
        {"device_id": 0, "node_id": "node", "gpu_uuid": "GPU-a"},
        {"device_id": 1, "node_id": "node", "gpu_uuid": "GPU-b"},
    ]


def test_selector_binds_each_slot_and_preserves_operator_environment():
    env = {"env_vars": {"CUSTOM_OP": "preserved"}}
    selector = PlacementSelector(None, [1, 0, 1], _topology(), env)
    arguments = [selector() for _ in range(3)]
    assert [item["scheduling_strategy"].placement_group_bundle_index for item in arguments] == [1, 0, 1]
    assert arguments[0]["runtime_env"]["env_vars"]["CUSTOM_OP"] == "preserved"
    assert "ELASTIC_JUICER_GPU_UUID" not in env["env_vars"]
    assert selector()["runtime_env"]["env_vars"]["ELASTIC_JUICER_PLACEMENT_SLOT"] == "0"


def test_wrong_actual_device_fails_before_work(monkeypatch):
    monkeypatch.setenv("ELASTIC_JUICER_PLACEMENT_SLOT", "0")
    placement = {"device_ids": [1], "topology": _topology()}
    with pytest.raises(RuntimeError, match="actual node/GPU"):
        verify_worker_placement({"node_id": "node", "gpu_uuids": ["GPU-a"]}, placement)
    assert verify_worker_placement({"node_id": "node", "gpu_uuids": ["GPU-b"]}, placement)["placement_verified"]
