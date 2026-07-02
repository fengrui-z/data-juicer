import pytest
import numpy as np

from data_juicer.core.elasticjuicer.predictor.memory_predictor import (
    MemoryPredictor,
    PredictionResult,
)
from data_juicer.core.elasticjuicer.predictor.feature_extractor import SampleFeatures


def _make_features(batch_size=1, text_length=100, estimated_size_mb=0.001):
    return SampleFeatures(
        batch_size=batch_size,
        text_length=text_length,
        num_tokens=text_length // 5,
        estimated_size_mb=estimated_size_mb,
    )


def test_predict_returns_none_before_min_samples():
    pred = MemoryPredictor(op_name="test_op", min_samples_for_prediction=5)
    features = _make_features()
    assert pred.predict(features) is None


def test_predict_works_after_enough_observations():
    pred = MemoryPredictor(op_name="test_op", min_samples_for_prediction=3)
    for i in range(5):
        f = _make_features(text_length=100 + i * 50)
        pred.observe(f, actual_memory_mb=50.0 + i * 10)

    features = _make_features(text_length=200)
    result = pred.predict(features)
    assert result is not None
    assert isinstance(result, PredictionResult)
    assert result.predicted_memory_mb >= 0


def test_confidence_interval_is_ordered():
    pred = MemoryPredictor(op_name="test_op", min_samples_for_prediction=3)
    for i in range(10):
        f = _make_features(text_length=100 + i * 20)
        pred.observe(f, actual_memory_mb=100.0 + i * 5)

    result = pred.predict(_make_features(text_length=200))
    assert result is not None
    assert result.confidence_lower <= result.predicted_memory_mb
    assert result.predicted_memory_mb <= result.confidence_upper


def test_recommend_batch_size_returns_positive():
    pred = MemoryPredictor(op_name="test_op", min_samples_for_prediction=3)
    for i in range(10):
        f = _make_features(text_length=100)
        pred.observe(f, actual_memory_mb=10.0)

    features = _make_features(text_length=100)
    bs = pred.recommend_batch_size(features, available_memory_mb=5000)
    assert bs >= 1


def test_export_import_roundtrip_preserves_predictions():
    pred = MemoryPredictor(op_name="test_op", min_samples_for_prediction=3)
    for i in range(10):
        f = _make_features(text_length=100 + i * 10)
        pred.observe(f, actual_memory_mb=50.0 + i * 2)

    exported = pred.export_model()

    pred2 = MemoryPredictor(op_name="other")
    pred2.import_model(exported)

    assert pred2.op_name == "test_op"
    assert pred2.weights is not None
    assert len(pred2.feature_history) == len(pred.feature_history)
    assert len(pred2.memory_history) == len(pred.memory_history)

    features = _make_features(text_length=150)
    r1 = pred.predict(features)
    r2 = pred2.predict(features)
    assert r1 is not None
    assert r2 is not None
    assert abs(r1.predicted_memory_mb - r2.predicted_memory_mb) < 1e-6


def test_import_with_empty_history():
    pred = MemoryPredictor(op_name="test_op")
    exported = pred.export_model()

    pred2 = MemoryPredictor(op_name="other")
    pred2.import_model(exported)
    assert len(pred2.feature_history) == 0
    assert pred2.predict(_make_features()) is None


def test_monotonic_synthetic_data():
    pred = MemoryPredictor(op_name="test_op", min_samples_for_prediction=5)
    for text_len in [100, 200, 400, 800, 1600]:
        f = _make_features(text_length=text_len)
        pred.observe(f, actual_memory_mb=text_len * 0.1)

    small = pred.predict(_make_features(text_length=100))
    large = pred.predict(_make_features(text_length=1600))
    assert small is not None
    assert large is not None
    assert large.predicted_memory_mb > small.predicted_memory_mb


def test_get_safe_prediction_uses_upper_bound():
    result = PredictionResult(
        predicted_memory_mb=100,
        confidence_lower=80,
        confidence_upper=120,
    )
    safe = result.get_safe_prediction(safety_margin=0.9)
    assert safe == 120 / 0.9


def test_model_stats_reflect_training():
    pred = MemoryPredictor(op_name="test_op", min_samples_for_prediction=3)
    for i in range(10):
        pred.observe(_make_features(text_length=100 + i * 10), actual_memory_mb=50.0 + i)

    stats = pred.get_model_stats()
    assert stats["model_trained"] is True
    assert stats["total_updates"] == 10
    assert stats["samples_in_window"] == 10
    assert "avg_prediction_error_mb" in stats
