import os
from pathlib import Path

import pytest

from data_juicer.config.config import build_base_parser, init_configs
from data_juicer.core.elasticjuicer.config_contract import (
    CONTROLLER_POLICIES,
    controller_policy_stability,
    effective_elastic_juicer_config,
    normalize_elastic_juicer_config,
    validate_elastic_juicer_config,
)


def _enabled(**overrides):
    cfg = {
        "elastic_juicer_adaptive_batching": True,
        "elastic_juicer_controller_policy": "ej",
        "elastic_juicer_experimental_policies": False,
        "job_id": "job-test",
    }
    cfg.update(overrides)
    return cfg


def test_public_parser_exposes_the_contract_defaults():
    cfg = build_base_parser().parse_args(["--auto"])

    assert cfg.elastic_juicer_controller_policy == "ej"
    assert cfg.elastic_juicer_experimental_policies is False
    assert cfg.elastic_juicer_cost_column is None
    assert cfg.elastic_juicer_profiles_path is None


def test_parser_rejects_unknown_policy_before_runtime():
    parser = build_base_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--auto",
                "--elastic_juicer_controller_policy",
                "not-a-policy",
            ]
        )


def test_init_configs_applies_contract_to_cli_overrides():
    config_path = Path(__file__).parents[2] / "config" / "demo_4_test.yaml"

    cfg = init_configs(
        [
            "--config",
            str(config_path),
            "--elastic_juicer_adaptive_batching",
            "true",
            "--elastic_juicer_experimental_policies",
            "true",
            "--elastic_juicer_controller_policy",
            "safebo",
            "--elastic_juicer_cost_column",
            "img_pixels",
        ],
        load_configs_only=True,
    )

    assert cfg.elastic_juicer_controller_policy == "safebo"
    assert cfg.elastic_juicer_cost_column == "img_pixels"


def test_init_configs_rejects_experimental_policy_without_opt_in():
    config_path = Path(__file__).parents[2] / "config" / "demo_4_test.yaml"

    with pytest.raises(ValueError, match="experimental"):
        init_configs(
            [
                "--config",
                str(config_path),
                "--elastic_juicer_adaptive_batching",
                "true",
                "--elastic_juicer_controller_policy",
                "safebo",
            ],
            load_configs_only=True,
        )


def test_policy_inventory_and_stability_are_explicit():
    assert "ej" in CONTROLLER_POLICIES
    assert controller_policy_stability("ej") == "stable"
    assert controller_policy_stability("safebo") == "experimental"

    with pytest.raises(ValueError, match="unknown"):
        controller_policy_stability("not-a-policy")


@pytest.mark.parametrize("policy", sorted(CONTROLLER_POLICIES - {"ej"}))
def test_every_non_ej_policy_requires_explicit_experimental_opt_in(policy):
    with pytest.raises(ValueError, match="experimental"):
        validate_elastic_juicer_config(_enabled(elastic_juicer_controller_policy=policy))

    validate_elastic_juicer_config(
        _enabled(
            elastic_juicer_controller_policy=policy,
            elastic_juicer_experimental_policies=True,
        )
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("elastic_juicer_controller_policy", "safebo"),
        ("elastic_juicer_cost_column", "img_pixels"),
        ("elastic_juicer_profile_seed", True),
        ("elastic_juicer_profiles_path", "/tmp/ej-profiles.json"),
    ],
)
def test_active_options_require_adaptive_batching(option, value):
    cfg = {
        option: value,
        "elastic_juicer_experimental_policies": True,
    }

    with pytest.raises(ValueError, match="adaptive_batching"):
        validate_elastic_juicer_config(cfg)


def test_cost_column_is_only_valid_for_cost_aware_policies():
    with pytest.raises(ValueError, match="only consumed"):
        validate_elastic_juicer_config(_enabled(elastic_juicer_cost_column="img_pixels"))

    validate_elastic_juicer_config(
        _enabled(
            elastic_juicer_controller_policy="safebo",
            elastic_juicer_experimental_policies=True,
            elastic_juicer_cost_column="img_pixels",
        )
    )


def test_cost_aware_policy_has_a_supported_no_signal_mode():
    validate_elastic_juicer_config(
        _enabled(
            elastic_juicer_controller_policy="hybrid",
            elastic_juicer_experimental_policies=True,
            elastic_juicer_cost_column=None,
        )
    )


def test_explicit_profiles_path_is_absolute_and_local():
    with pytest.raises(ValueError, match="must be absolute"):
        validate_elastic_juicer_config(_enabled(elastic_juicer_profiles_path="relative/profiles.json"))
    with pytest.raises(ValueError, match="local filesystems"):
        validate_elastic_juicer_config(_enabled(elastic_juicer_profiles_path="s3://bucket/profiles.json"))

    validate_elastic_juicer_config(_enabled(elastic_juicer_profiles_path=os.path.abspath("profiles.json")))


def test_runtime_validation_requires_resolved_job_id():
    with pytest.raises(ValueError, match="resolved job_id"):
        validate_elastic_juicer_config(
            {"elastic_juicer_adaptive_batching": True},
            require_job_id=True,
        )


def test_normalization_and_effective_config_are_deterministic():
    cfg = _enabled(
        elastic_juicer_controller_policy="  SAFEBO ",
        elastic_juicer_experimental_policies=True,
        elastic_juicer_cost_column=" img_pixels ",
        elastic_juicer_profiles_path=" /tmp/ej-profiles.json ",
        elastic_juicer_profile_seed=True,
    )

    normalize_elastic_juicer_config(cfg)
    effective = effective_elastic_juicer_config(cfg)

    assert cfg["elastic_juicer_controller_policy"] == "safebo"
    assert cfg["elastic_juicer_cost_column"] == "img_pixels"
    assert cfg["elastic_juicer_profiles_path"] == "/tmp/ej-profiles.json"
    assert effective == {
        "enabled": True,
        "controller_policy": "safebo",
        "controller_policy_stability": "experimental",
        "experimental_policies_enabled": True,
        "cost_signal": "img_pixels",
        "profile_seed": True,
        "profile_context": False,
        "profiles_path": "/tmp/ej-profiles.json",
        "profile_ttl_ms": 1_800_000,
        "auto_parallelism": False,
        "probe_enabled": True,
        "replan_enabled": True,
        "max_gpu_workers_per_device": 4,
        "min_batches_per_actor": 4,
        "startup_amortization_ratio": 0.10,
        "min_marginal_makespan_gain": 0.10,
        "actor_startup_seconds": None,
    }
