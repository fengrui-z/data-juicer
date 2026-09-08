from types import SimpleNamespace
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
import ray

from data_juicer.core.elasticjuicer.config_contract import validate_elastic_juicer_config
from data_juicer.core.elasticjuicer.managed_mapper_pool import ManagedMapperPool, validate_managed_operators
from tests.core.elasticjuicer.test_ray_dataset_adaptive_batching import BatchedActorMapper, _dataset


def config(**updates):
    return dict(
        executor_type="ray_partitioned",
        elastic_juicer_adaptive_batching=True,
        elastic_juicer_execution_audit=True,
        elastic_juicer_auto_parallelism=True,
        elastic_juicer_enforce_gpu_placement=True,
        elastic_juicer_replan_enabled=False,
        elastic_juicer_worker_mode="resident",
        checkpoint={"enabled": False},
        partition={"mode": "manual", "max_concurrent_partitions": 1},
        **updates,
    )


def test_contract_and_operator_opt_in():
    validate_elastic_juicer_config(config())
    validate_elastic_juicer_config({})
    op = BatchedActorMapper(num_gpus=1, skip_op_error=False)
    with pytest.raises(ValueError, match="row-preserving"):
        validate_managed_operators([op])
    op.elastic_juicer_row_preserving = True
    validate_managed_operators([op])
    with pytest.raises(ValueError, match="exactly one"):
        validate_managed_operators([op, op])
    op.skip_op_error = True
    with pytest.raises(ValueError, match="skip_op_error"):
        validate_managed_operators([op])


@pytest.mark.parametrize(
    "change",
    [
        {"elastic_juicer_worker_mode": "bad"},
        {"executor_type": "ray"},
        {"elastic_juicer_adaptive_batching": False},
        {"elastic_juicer_enforce_gpu_placement": False},
        {"elastic_juicer_replan_enabled": True},
        {"checkpoint": {"enabled": True}},
        {"partition": {"mode": "auto", "max_concurrent_partitions": 1}},
        {"partition": {"mode": "manual", "max_concurrent_partitions": 2}},
        {"op_fusion": True},
        {"use_checkpoint": True},
        {"_user_provided_job_id": True},
    ],
)
def test_unsupported_modes_fail_fast(change):
    with pytest.raises(ValueError):
        validate_elastic_juicer_config({**config(), **change})


def fake_runtime(monkeypatch, bad_output=False, fail_finish=False):
    made, killed = [], []

    class Remote:
        def options(self, **kwargs):
            return self

        def remote(self, **kwargs):
            worker = SimpleNamespace(bind=MagicMock(), __call__=MagicMock(), finish_partition=MagicMock())
            worker.__call__.remote.side_effect = lambda batch: batch.slice(0, 0) if bad_output else batch
            if fail_finish:
                worker.finish_partition.remote.side_effect = RuntimeError("ACK failure")
            made.append(worker)
            return worker

    monkeypatch.setattr(ray, "remote", lambda **kwargs: lambda cls: Remote())
    monkeypatch.setattr(ray, "get", lambda refs, **kwargs: refs)
    monkeypatch.setattr(ray, "put", lambda table: table)
    monkeypatch.setattr(ray, "kill", lambda worker, **kwargs: killed.append(worker))
    monkeypatch.setattr(ray.data, "from_arrow_refs", pa.concat_tables)
    return made, killed


class Input:
    def __init__(self, rows):
        self.table = pa.table({"id": list(range(rows))})

    def count(self):
        return len(self.table)

    def iter_batches(self, batch_size, **kwargs):
        for i in range(0, len(self.table), batch_size):
            yield self.table.slice(i, batch_size)


def execute(pool, rows=37, partition=0):
    return pool.map(
        Input(rows),
        constructor={
            "operator_class": BatchedActorMapper,
            "initial_batch_size": 8,
            "execution_audit": {"planned_actors": 4, "partition_id": partition},
            "partition_id": partition,
        },
        selector=lambda: {},
        batch_size=8,
        num_cpus=1,
        num_gpus=0.5,
    )


@pytest.mark.parametrize("mode,expected", [("resident", 4), ("managed_fresh", 8)])
def test_same_path_order_tail_all_workers_and_lifetime(monkeypatch, mode, expected):
    made, killed = fake_runtime(monkeypatch)
    pool = ManagedMapperPool(mode)
    assert execute(pool).to_pydict() == {"id": list(range(37))}
    assert execute(pool, rows=5, partition=1).to_pydict() == {"id": list(range(5))}
    assert len(made) == expected
    assert all(worker.__call__.remote.call_count > 0 for worker in made)
    pool.close()
    pool.close()
    assert len(killed) == expected


@pytest.mark.parametrize("bad_output,fail_finish", [(True, False), (False, True)])
def test_failure_retires_all_workers_and_fences_pool(monkeypatch, bad_output, fail_finish):
    made, killed = fake_runtime(monkeypatch, bad_output, fail_finish)
    pool = ManagedMapperPool("resident")
    with pytest.raises((ValueError, RuntimeError)):
        execute(pool)
    assert made == killed
    with pytest.raises(RuntimeError, match="closed"):
        execute(pool)


@pytest.mark.parametrize("rows", [0, 3])
def test_too_small_partition_fails_before_worker_creation(monkeypatch, rows):
    made, killed = fake_runtime(monkeypatch)
    with pytest.raises(ValueError, match="one row"):
        execute(ManagedMapperPool("resident"), rows=rows)
    assert not made and not killed


def test_dataset_managed_failure_never_runtime_env_fallback():
    dataset = _dataset(True)
    dataset._auto_proc = False
    dataset._elastic_juicer_managed_pool = object()
    dataset._run_single_op = MagicMock(side_effect=RuntimeError("ACK failure"))
    op = BatchedActorMapper(runtime_env={"env_vars": {"A": "1"}})
    with pytest.raises(RuntimeError, match="ACK failure"):
        dataset.process([op])
    assert dataset._run_single_op.call_count == 1


def test_ray_dataset_routes_managed_pool_with_same_adaptive_options(monkeypatch):
    dataset = _dataset(True)
    dataset._elastic_juicer_managed_pool = MagicMock()
    dataset._elastic_juicer_cfg = {
        "elastic_juicer_execution_audit": True,
        "_elastic_juicer_audit_context": {"run_id": "r"},
    }
    dataset._elastic_juicer_partition_id = 0
    op = BatchedActorMapper(num_proc=4, num_gpus=0.5, batch_size=8)
    monkeypatch.setattr(ray, "get", lambda ref, **kwargs: ref)
    original = dataset.data
    dataset._run_single_op(op, cached_columns={"id"}, stage_id="s")
    call = dataset._elastic_juicer_managed_pool.map.call_args
    assert call.args == (original,)
    assert call.kwargs["constructor"]["operator_class"] is type(op)
    assert call.kwargs["constructor"]["execution_audit"]["planned_actors"] == 4
    assert call.kwargs["constructor"]["partition_id"] == 0
    original.map_batches.assert_not_called()


def test_executor_cleanup_precedes_seal_and_placement_release(monkeypatch, tmp_path):
    from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor

    events = []
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.tmp_dir = str(tmp_path / "temporary")
    executor._ej_begin_execution_audit = lambda: None
    executor._run_impl = lambda *args: "result"
    executor._ej_managed_pool = SimpleNamespace(close=lambda: events.append("workers"))
    executor._ej_finalize_execution_audit = lambda ok: events.append(("seal", ok))
    executor._ej_gpu_placement_lease = SimpleNamespace(close=lambda: events.append("placement"))
    assert executor._run_with_audit_cleanup() == "result"
    assert events == ["workers", ("seal", True), "placement"]
    events.clear()
    executor._run_impl = MagicMock(side_effect=RuntimeError("execution failed"))
    with pytest.raises(RuntimeError):
        executor._run_with_audit_cleanup()
    assert events == ["workers", ("seal", False), "placement"]
