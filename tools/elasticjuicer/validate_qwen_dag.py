"""Focused 8-GPU integration/correctness check, not a performance benchmark."""

import argparse
import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--reference-suite",
    type=Path,
    required=True,
    help="Frozen Qwen suite with source/configs/inputs and validated full-fused outputs",
)
parser.add_argument("--output", type=Path, required=True, help="New artifact directory; never overwrite a validation")
parser.add_argument("--rows", type=int, default=128)
args = parser.parse_args()
if args.rows < 1:
    parser.error("--rows must be positive")
ROOT = args.output.resolve()
ROOT.mkdir(parents=True, exist_ok=False)
PRODUCT = Path(__file__).resolve().parents[2]
FROZEN = args.reference_suite.resolve()
ADAPTER_DIR = Path(__file__).resolve().parent
paths = [str(PRODUCT), str(ADAPTER_DIR), str(FROZEN / "source")]
sys.path[:0] = paths
sys.modules["cudf"] = None
os.environ.update(
    PYTHONPATH=os.pathsep.join(paths),
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True",
    TOKENIZERS_PARALLELISM="false",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
)

import numpy as np
import ray
import yaml
from data_juicer.core.elasticjuicer.dag_plan import DAGNode, ExecutionGroup, WorkerPool, ResidentDAGPlan
from data_juicer.core.executor.resident_dag_executor import ResidentDAGExecutor
from qwen_dag_ops import NORMAL, CROP, MODEL


def make_plan(cfg):
    policy = cfg["policy"]
    nodes = [
        DAGNode(
            "normalize",
            {"ej_verify_qwen_normalize": {"policy": policy}},
            reads=("image_path", "image_sha256", "text"),
            writes=NORMAL,
            batch_size=16,
            row_policy="subset",
        )
    ]
    for stage in ("ocr", "quality"):
        nodes.append(
            DAGNode(
                stage,
                {"ej_verify_qwen_model": {"stage": stage, "model_config": cfg}},
                parents=("normalize",),
                reads=("image_path", "pixels", "canonical_hash"),
                writes=MODEL[stage],
                batch_size=cfg["batch_caps"][stage],
            )
        )
    nodes.append(
        DAGNode(
            "crop",
            {"ej_verify_qwen_crop": {"policy": policy}},
            parents=("ocr", "quality"),
            reads=("image_sha256", "pixels", "canonical_hash", "ocr_polygons", "ocr_confidences", "quality_score"),
            writes=CROP,
            batch_size=16,
            row_policy="subset",
            join_keys=("canonical_hash",),
        )
    )
    for stage in ("embedding", "aesthetic"):
        nodes.append(
            DAGNode(
                stage,
                {"ej_verify_qwen_model": {"stage": stage, "model_config": cfg}},
                parents=("crop",),
                reads=("image_path", "pixels", "crop_hash"),
                writes=MODEL[stage],
                batch_size=cfg["batch_caps"][stage],
            )
        )
    nodes.append(
        DAGNode(
            "pack",
            {"ej_verify_qwen_pack": {}},
            parents=("embedding", "aesthetic"),
            reads=("jpeg", "aesthetic_score", "dinov2_embedding"),
            writes=("jpeg_sha256",),
            batch_size=16,
            join_keys=("crop_hash",),
        )
    )
    return ResidentDAGPlan(
        tuple(nodes),
        (
            ExecutionGroup("normalize", ("normalize",), "decode"),
            ExecutionGroup("screen", ("ocr", "quality"), "gpu", parallel=True),
            ExecutionGroup("crop", ("crop",), "post"),
            ExecutionGroup("features", ("embedding", "aesthetic"), "gpu", parallel=True),
            ExecutionGroup("pack", ("pack",), "post"),
        ),
        (WorkerPool("decode", 8), WorkerPool("post", 4), WorkerPool("gpu", 8, 4, 1)),
        "pack",
        operator_modules=("qwen_dag_ops",),
        max_inflight_units=16,
        max_batch_bytes=128 * 1024**2,
        object_budget_bytes=16 * 1024**3,
        output_columns=(
            "image_sha256",
            "canonical_hash",
            "original_width",
            "original_height",
            "crop_hash",
            "crop_rect",
            "text_fraction",
            "export_pixel_hash",
            "jpeg_sha256",
            "quality_score",
            "dinov2_embedding",
            "aesthetic_score",
        ),
    )


def compare(records, reference):
    assert set(records) == set(reference), (set(records) - set(reference), set(reference) - set(records))
    errors = {"aesthetic_max_abs": 0.0, "quality_max_abs": 0.0, "embedding_max_abs": 0.0}
    for key, actual in records.items():
        expected = reference[key]
        for col, value in actual.items():
            if col in ("aesthetic_score", "quality_score", "dinov2_embedding"):
                a, b = np.asarray(value), np.asarray(expected[col])
                delta = float(np.max(np.abs(a - b)))
                field = {
                    "aesthetic_score": "aesthetic_max_abs",
                    "quality_score": "quality_max_abs",
                    "dinov2_embedding": "embedding_max_abs",
                }[col]
                errors[field] = max(errors[field], delta)
                assert np.allclose(a, b, atol=1e-4, rtol=0), (key, col, delta)
            else:
                assert value == expected[col], (key, col)
    return errors


def main():
    if (ROOT / "validation.json").exists():
        raise RuntimeError("Validation results already exist")
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    assert len(smi.splitlines()) == 8 and all(int(v) < 100 for v in smi.splitlines()), "GPUs are occupied"
    cfg = yaml.safe_load((FROZEN / "configs/n20000.yaml").read_text())
    cfg["product_root"] = str(PRODUCT)
    source = [json.loads(l) for l in (FROZEN / "inputs/qwen20k.jsonl").read_text().splitlines()[: args.rows]]
    ids = {r["id"] for r in source}
    reference = {}
    for file in (FROZEN / "cells/n20000-dj-full-fused/run/units").glob("*.json"):
        if file.name.endswith(".commit.json"):
            continue
        value = json.loads(file.read_text())
        reference.update({r["id"]: r for r in value["accepted"] if r["id"] in ids})
    plan = make_plan(cfg)
    product_commit = subprocess.check_output(["git", "-C", str(PRODUCT), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "purpose": "8-GPU real-model functional validation; no speedup claim",
        "product_commit": product_commit,
        "plan_fingerprint": plan.fingerprint(),
        "rows_per_pass": len(source),
        "unique_images": len(source),
        "expected_accepted": len(reference),
        "tolerance": 1e-4,
        "arms": [],
        "adapter_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ADAPTER_DIR / "qwen_dag_ops.py", Path(__file__)]
        },
    }
    (ROOT / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    ray.init(
        address="local",
        num_cpus=64,
        num_gpus=8,
        include_dashboard=False,
        object_store_memory=16 * 1024**3,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join(paths)}},
    )
    try:
        for adaptive in (False, True):
            arm = "adaptive" if adaptive else "fixed"
            print("START", arm, flush=True)
            with ResidentDAGExecutor(plan, adaptive=adaptive, artifact_dir=ROOT / arm) as executor:
                assert len({w["device"]["uuid"] for w in executor.worker_identities if w["device"]}) == 8
                for repeat in range(2):
                    inputs = (
                        {
                            k: [row[k] for row in source[i : i + 16]]
                            for k in ("id", "image_path", "image_sha256", "text")
                        }
                        for i in range(0, len(source), 16)
                    )
                    records, rejected = {}, set()
                    for batch in executor.iter_batches(inputs):
                        columns = batch.columns()
                        for i, key in enumerate(columns["id"]):
                            records[key] = {k: v[i] for k, v in columns.items()}
                        rejected.update(batch.rejected_ids)
                    errors = compare(records, reference)
                    assert rejected == ids - set(reference)
                    result["arms"].append(
                        {"arm": arm, "pass": repeat, "summary": executor.last_summary, "comparison": errors}
                    )
                    (ROOT / f"{arm}-pass{repeat}.json").write_text(json.dumps(records) + "\n")
                    print("PASS", arm, repeat, executor.last_summary["seconds_including_consumer"], errors, flush=True)
    finally:
        ray.shutdown()
    result["execution_checks"] = {}
    for arm in ("fixed", "adaptive"):
        identity = json.loads((ROOT / arm / "pool.json").read_text())
        workers = {w["worker_id"]: w for w in identity["workers"]}
        events = [
            json.loads(line) for path in (ROOT / arm).glob("*/events.jsonl") for line in path.read_text().splitlines()
        ]
        used = {e["worker_id"] for e in events if workers[e["worker_id"]]["device"]}
        assert len(used) == 8, (arm, "GPU slots executed", len(used))
        units, instances = {}, {}
        for event in events:
            units.setdefault((event["run_id"], event["unit"]), {})[event["node"]] = event
            instances.setdefault((event["node"], event["worker_id"]), set()).add(event["operator_instance"])
        assert all(len(ids) == 1 for ids in instances.values()), "operator instance was replaced"
        overlap = {}
        for left, right in [("ocr", "quality"), ("embedding", "aesthetic")]:
            overlap[left + "+" + right] = sum(
                min(v[left]["ended"], v[right]["ended"]) > max(v[left]["started"], v[right]["started"])
                for v in units.values()
                if v[left]["input_rows"] and v[right]["input_rows"]
            )
            assert overlap[left + "+" + right] > 0
        result["execution_checks"][arm] = {
            "gpu_workers_executed": len(used),
            "instances_persisted": True,
            "branch_overlap_units": overlap,
            "materialized_gpu_column_bytes": sum(
                e["payload_bytes"] for e in events if e["node"] in ("ocr", "quality", "embedding", "aesthetic")
            ),
            "emitted_gpu_column_bytes": sum(
                e["emitted_column_bytes"] for e in events if e["node"] in ("ocr", "quality", "embedding", "aesthetic")
            ),
        }
        assert all(
            e["emitted_column_bytes"] < e["payload_bytes"]
            for e in events
            if e["node"] in ("ocr", "quality", "embedding", "aesthetic") and e["input_rows"]
        )
    result["status"] = "passed"
    (ROOT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("VALIDATED", ROOT / "validation.json", flush=True)


if __name__ == "__main__":
    main()
