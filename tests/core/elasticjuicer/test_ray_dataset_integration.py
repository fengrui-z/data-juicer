from types import SimpleNamespace

from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.elasticjuicer.mode import ElasticJuicerMode
from data_juicer.core.elasticjuicer.runtime.ray_microbatch import RayAdaptiveActor


class FakeOperator:
    _name = "gpu_op"
    _init_args = ()
    _init_kwargs = {}
    batch_size = 4


class FakeCfg(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_ray_dataset_builds_adaptive_actor_spec_when_dynamic():
    dataset = RayDataset.__new__(RayDataset)
    dataset._elastic_juicer_mode = ElasticJuicerMode.DYNAMIC
    dataset._elastic_juicer_min_batch_size = 1
    dataset._elastic_juicer_max_batch_size = 32
    dataset._elastic_juicer_metrics_collector = object()

    actor_class, kwargs = dataset._ray_adaptive_actor_spec(
        FakeOperator(),
        "process",
    )

    assert actor_class is RayAdaptiveActor
    assert kwargs["initial_batch_size"] == 4
    assert kwargs["max_batch_size"] == 32
    assert kwargs["metrics_sink"] is dataset._elastic_juicer_metrics_collector


def test_ray_dataset_keeps_existing_actor_when_not_dynamic():
    dataset = RayDataset.__new__(RayDataset)
    dataset._elastic_juicer_mode = ElasticJuicerMode.APPLY

    assert dataset._ray_adaptive_actor_spec(FakeOperator(), "process") is None


def test_ray_dataset_reuses_executor_metrics_collector():
    collector = object()
    cfg = FakeCfg(
        auto_op_parallelism=None,
        elastic_juicer_mode="dynamic",
        adaptive_batch_size=False,
        elastic_juicer_min_batch_size=1,
        elastic_juicer_max_batch_size=32,
        _ej_metrics_collector_ref=collector,
    )

    dataset = RayDataset(object(), cfg=cfg)

    assert dataset._elastic_juicer_metrics_collector is collector
