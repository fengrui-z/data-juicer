"""Local verification adapters: unchanged frozen Qwen transforms and real models."""

import hashlib
import sys

sys.modules["cudf"] = None
from data_juicer.ops.base_op import Mapper, OPERATORS
from ejdag.core import normalize_rows, crop_rows, validate_features
from ejdag.models import RealBackend


def rows(batch):
    return [{k: v[i] for k, v in batch.items()} for i in range(len(batch["id"]))]


def columns(records, keys):
    return {k: [r[k] for r in records] for k in ("id", *keys)}


NORMAL = ("pixels", "canonical_hash", "original_width", "original_height")
CROP = ("pixels", "jpeg", "crop_hash", "crop_rect", "text_fraction", "export_pixel_hash")
MODEL = {
    "ocr": ("ocr_polygons", "ocr_confidences"),
    "quality": ("quality_score",),
    "embedding": ("dinov2_embedding",),
    "aesthetic": ("aesthetic_score",),
}


@OPERATORS.register_module("ej_verify_qwen_normalize")
class Normalize(Mapper):
    _batched_op = True
    elastic_juicer_dag_safe = True

    def __init__(self, policy, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy

    def process_batched(self, batch):
        accepted, _ = normalize_rows(rows(batch), self.policy)
        return columns(accepted, NORMAL)


@OPERATORS.register_module("ej_verify_qwen_crop")
class Crop(Mapper):
    _batched_op = True
    elastic_juicer_dag_safe = True

    def __init__(self, policy, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy

    def process_batched(self, batch):
        accepted, _ = crop_rows(rows(batch), self.policy)
        return columns(accepted, CROP)


@OPERATORS.register_module("ej_verify_qwen_model")
class Model(Mapper):
    _batched_op = True
    elastic_juicer_dag_safe = True

    def __init__(self, stage, model_config, **kwargs):
        super().__init__(**kwargs)
        self.stage = stage
        self.backend = RealBackend(stage, model_config)

    def process_batched(self, batch):
        return columns(self.backend.infer(rows(batch)), MODEL[self.stage])


@OPERATORS.register_module("ej_verify_qwen_pack")
class Pack(Mapper):
    _batched_op = True
    elastic_juicer_dag_safe = True

    def process_batched(self, batch):
        for row in rows(batch):
            validate_features(row)
        return {"id": list(batch["id"]), "jpeg_sha256": [hashlib.sha256(j).hexdigest() for j in batch["jpeg"]]}
