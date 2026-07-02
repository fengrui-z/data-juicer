from types import SimpleNamespace

import pytest

from data_juicer.core.elasticjuicer.mode import ElasticJuicerMode, resolve_mode
from data_juicer.core.executor.default_executor import _resolve_elastic_juicer_mode


@pytest.mark.parametrize("mode", list(ElasticJuicerMode))
def test_explicit_modes_round_trip(mode):
    assert resolve_mode(mode.value) is mode


def test_legacy_adaptive_batch_size_maps_to_apply():
    assert resolve_mode("off", legacy_adaptive_batch_size=True) is ElasticJuicerMode.APPLY


def test_legacy_flag_does_not_downgrade_dynamic_mode():
    assert resolve_mode("dynamic", legacy_adaptive_batch_size=True) is ElasticJuicerMode.DYNAMIC


def test_conflicting_legacy_and_explicit_mode_is_rejected():
    with pytest.raises(ValueError, match="conflicts"):
        resolve_mode("observe", legacy_adaptive_batch_size=True)


def test_executor_mode_defaults_to_off():
    assert _resolve_elastic_juicer_mode(SimpleNamespace()) is ElasticJuicerMode.OFF
