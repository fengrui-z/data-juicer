from types import SimpleNamespace

from data_juicer.core.elasticjuicer.mode import ElasticJuicerMode
from data_juicer.core.elasticjuicer.runtime.local_microbatch import (
    LocalMicrobatchRuntime,
)
from data_juicer.core.executor.default_executor import (
    _install_local_microbatch_runtime,
)
from data_juicer.ops.base_op import Mapper


class BatchedMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples, *args, **kwargs):
        return samples


class FakeScheduler:
    def __init__(self, initial_batch_size, min_batch_size, max_batch_size):
        self.controller = SimpleNamespace(
            current_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
        )

    def report_oom(self, batch_size, memory_mb):
        self.controller.current_batch_size = max(
            self.controller.min_batch_size,
            batch_size // 2,
        )

    def update(self, actual_memory_used):
        pass


def test_runtime_installs_on_batched_mapper():
    mapper = BatchedMapper(batch_size=4, auto_op_parallelism=False, num_proc=1)
    runtime = LocalMicrobatchRuntime(
        min_batch_size=1,
        max_batch_size=16,
        scheduler_factory=FakeScheduler,
    )

    installed = runtime.install([mapper])

    assert len(installed) == 1
    assert mapper.batch_size == 16
    assert mapper.process({"text": ["a", "b"]}) == {"text": ["a", "b"]}


def test_executor_helper_only_enables_dynamic_mode():
    cfg = SimpleNamespace(
        elastic_juicer_min_batch_size=1,
        elastic_juicer_max_batch_size=16,
    )

    assert _install_local_microbatch_runtime(cfg, [], ElasticJuicerMode.APPLY) is None
    assert isinstance(
        _install_local_microbatch_runtime(cfg, [], ElasticJuicerMode.DYNAMIC),
        LocalMicrobatchRuntime,
    )
