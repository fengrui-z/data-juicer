from unittest.mock import patch

from data_juicer.core.data import NestedDataset
from data_juicer.core.elasticjuicer.profiler.execution_observer import ExecutionObserver


class IdentityOp:
    _name = "identity"
    _op_cfg = {"identity": {}}
    batch_size = 4

    @staticmethod
    def use_cuda():
        return False

    @staticmethod
    def is_batched_op():
        return True

    @staticmethod
    def run(dataset, **kwargs):
        return dataset


class FakeDataset:
    def __len__(self):
        return 3


def test_nested_dataset_reuses_monitor_for_observe_mode(tmp_path):
    dataset = FakeDataset()
    observer = ExecutionObserver(str(tmp_path))
    monitor_payload = {
        "time": 0.1,
        "sampling interval": 0.1,
        "resource": [],
        "resource_analysis": {
            "CPU util.": {"max": 0.25},
            "Used mem.": {"max": 100},
        },
    }

    with patch(
        "data_juicer.core.data.dj_dataset.Monitor.monitor_func",
        return_value=(dataset, monitor_payload),
    ) as monitor:
        result = NestedDataset.process(
            dataset,
            [IdentityOp()],
            open_monitor=False,
            execution_observer=observer,
        )

    assert result is dataset
    assert monitor.call_count == 1
    assert len(observer.observations) == 1
    assert observer.observations[0].input_rows == 3
    assert observer.observations[0].output_rows == 3
    assert observer.observations[0].configured_batch_size == 4
