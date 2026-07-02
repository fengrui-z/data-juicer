from types import SimpleNamespace

from data_juicer.core.executor.default_executor import (
    _create_elastic_juicer_observer,
)


def test_observer_factory_is_disabled_by_default(tmp_path):
    observer = _create_elastic_juicer_observer(SimpleNamespace(), str(tmp_path))

    assert observer is None
    assert not (tmp_path / "elastic_juicer").exists()


def test_observer_factory_uses_work_dir_default(tmp_path):
    cfg = SimpleNamespace(
        elastic_juicer_mode="observe",
        elastic_juicer_profile_dir=None,
    )

    observer = _create_elastic_juicer_observer(cfg, str(tmp_path))

    assert observer.output_path == tmp_path / "elastic_juicer" / "observations.jsonl"


def test_observer_factory_honors_explicit_profile_dir(tmp_path):
    profile_dir = tmp_path / "profiles"
    cfg = SimpleNamespace(
        elastic_juicer_mode="observe",
        elastic_juicer_profile_dir=str(profile_dir),
    )

    observer = _create_elastic_juicer_observer(cfg, str(tmp_path / "work"))

    assert observer.output_path == profile_dir / "observations.jsonl"
