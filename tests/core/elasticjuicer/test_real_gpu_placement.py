"""Explicit opt-in real-GPU tests; never silently substitute a CPU fake."""

import os

import pytest
from data_juicer.ops.base_op import OPERATORS, Mapper

pytestmark = pytest.mark.skipif(
    os.environ.get("ELASTIC_JUICER_RUN_GPU_TESTS") != "1",
    reason="set ELASTIC_JUICER_RUN_GPU_TESTS=1 on an idle GPU host",
)


class PlacementGpuActor:
    def ready(self, placement):
        import torch
        from data_juicer.core.elasticjuicer.acknowledged_metrics import runtime_identity
        from data_juicer.core.elasticjuicer.gpu_placement import verify_worker_placement

        identity = runtime_identity()
        verified = verify_worker_placement(identity, placement)
        value = (torch.ones(1024, device="cuda") * 2).sum().item()
        torch.cuda.synchronize()
        return {**identity, **verified, "gpu_result": value}


@OPERATORS.register_module("ej_placement_gpu_test_mapper")
class PlacementGpuMapper(Mapper):
    _batched_op = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = None

    def process_batched(self, samples):
        import torch

        if self.weights is None:
            self.weights = torch.ones((128, 128), device="cuda")
        result = self.weights @ self.weights
        checksum = result[0, 0].item()
        torch.cuda.synchronize()
        return {**samples, "checksum": [checksum] * len(samples["text"])}

    def use_ray_actor(self):
        return True


def test_real_fractional_actors_follow_uuid_mapping_through_resize():
    import ray
    from data_juicer.core.elasticjuicer.gpu_placement import GpuPlacementLease, PlacementSelector

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=8, num_gpus=2, include_dashboard=False)
    lease = None
    actors = []
    try:
        lease = GpuPlacementLease(2, cpus_per_device=2)
        remote = ray.remote(num_cpus=0.5, num_gpus=0.5)(PlacementGpuActor)
        for device_ids in ([0, 1], [0, 0, 1, 1], [1]):
            selector = PlacementSelector(lease.group, device_ids, lease.topology)
            placement = {"device_ids": device_ids, "topology": lease.topology}
            actors = [remote.options(**selector()).remote() for _ in device_ids]
            results = ray.get([actor.ready.remote(placement) for actor in actors], timeout=60)
            assert [value["device_id"] for value in results] == device_ids
            assert all(value["gpu_result"] == 2048 for value in results)
            assert all(value["placement_verified"] for value in results)
            for actor in actors:
                ray.kill(actor, no_restart=True)
            actors = []
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        if lease is not None:
            lease.close()
        if started_here:
            ray.shutdown()


def test_public_partitioned_run_enforces_gpu_placement_and_final_delivery(tmp_path):
    import json
    import ray
    import yaml
    from data_juicer.config import init_configs
    from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=12, num_gpus=2, include_dashboard=False)
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps({"text": f"row-{i}"}) + "\n" for i in range(128)))
    config = {
        "project_name": "placement-gpu",
        "executor_type": "ray_partitioned",
        "dataset_path": str(source),
        "export_path": str(tmp_path / "output.jsonl"),
        "work_dir": str(tmp_path / "work"),
        "auto_op_parallelism": False,
        "partition": {"mode": "manual", "num_of_partitions": 2, "max_concurrent_partitions": 1},
        "checkpoint": {"enabled": False},
        "elastic_juicer_adaptive_batching": True,
        "elastic_juicer_execution_audit": True,
        "elastic_juicer_auto_parallelism": True,
        "elastic_juicer_enforce_gpu_placement": True,
        "elastic_juicer_replan_enabled": False,
        "elastic_juicer_max_gpu_workers_per_device": 2,
        "gpu_probe_batch_cap_multiplier": 1,
        "gpu_probe_warmup_batches": 1,
        "gpu_probe_steady_batches": 1,
        "process": [{"ej_placement_gpu_test_mapper": {"batch_size": 8, "num_proc": 2, "num_gpus": 0.5, "num_cpus": 1}}],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    cfg = init_configs(["--config", str(path)])
    cfg.work_dir = str(tmp_path / "work")
    executor = PartitionedRayExecutor(cfg)
    try:
        executor.run(skip_return=True)
        context = json.loads((tmp_path / "work" / "elastic_juicer_metrics_run.json").read_text())
        from pathlib import Path

        directory = Path(context["directory"])
        assert json.loads((directory / "elastic_juicer_metrics_integrity.json").read_text())["complete"]
        audit = json.loads((directory / "elastic_juicer_execution_audit.json").read_text())
        assert len(audit["executions"]) == 2
        assert all(item["applied"] and item["placement_verified"] for item in audit["executions"].values())
        output = tmp_path / "output.jsonl"
        files = [output] if output.is_file() else [p for p in output.rglob("*") if p.is_file()]
        rows = [json.loads(line) for file in files for line in file.read_text().splitlines()]
        assert sorted(row["text"] for row in rows) == sorted(f"row-{i}" for i in range(128))
        assert all(row["checksum"] == 128 for row in rows)
    finally:
        for key in ("_elastic_juicer_metrics_sink", "_elastic_juicer_control_service"):
            if cfg.get(key) is not None:
                ray.kill(cfg.get(key), no_restart=True)
        if started_here:
            ray.shutdown()
