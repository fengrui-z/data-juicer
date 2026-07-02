import json
import pytest
import tempfile
from pathlib import Path

from data_juicer.core.elasticjuicer.profiler.profiling_store import (
    ProfilingStore,
    ResourceThroughputCurve,
)
from data_juicer.core.elasticjuicer.profiler.resource_monitor import (
    OpExecutionStats,
    ResourceSnapshot,
)


def _make_snapshots(n=10, base_batch=10, base_mem=100):
    return [
        ResourceSnapshot(
            timestamp=float(i),
            batch_size=base_batch + i,
            cpu_percent=20.0,
            memory_mb=base_mem + i * 10,
            latency_ms=50.0 + i,
            throughput=(base_batch + i) / (50.0 + i) * 1000,
        )
        for i in range(n)
    ]


def test_create_store_in_temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        assert Path(tmpdir).exists()


def test_update_and_retrieve_execution_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="filter_a")
        for snap in _make_snapshots(5):
            stats.update(snap)

        store.update_execution_stats("filter_a", stats)
        retrieved = store.get_execution_stats("filter_a")
        assert retrieved is not None
        assert retrieved.total_batches == 5


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="op_x")
        for snap in _make_snapshots(6):
            stats.update(snap)
        store.update_execution_stats("op_x", stats)
        store.save_all()

        store2 = ProfilingStore(storage_dir=tmpdir)
        loaded = store2.get_execution_stats("op_x")
        assert loaded is not None
        assert loaded.total_batches == 6


def test_throughput_curve_fitted_with_enough_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="op_y")
        for snap in _make_snapshots(10):
            stats.update(snap)

        store.update_execution_stats("op_y", stats)
        curve = store.get_throughput_curve("op_y")
        assert curve is not None
        assert curve.n_samples > 0


def test_throughput_curve_not_fitted_with_too_few():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="op_z")
        for snap in _make_snapshots(2):
            stats.update(snap)

        store.update_execution_stats("op_z", stats)
        curve = store.get_throughput_curve("op_z")
        assert curve is None


def test_predict_memory_for_batch():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="op_m")
        for snap in _make_snapshots(10, base_batch=5, base_mem=50):
            stats.update(snap)
        store.update_execution_stats("op_m", stats)

        predicted = store.predict_memory_for_batch("op_m", batch_size=20)
        assert predicted is not None
        assert predicted > 0


def test_predict_memory_returns_none_for_unknown_op():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        assert store.predict_memory_for_batch("nonexistent", 10) is None


def test_get_safe_batch_size_conservative_for_unknown():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        bs = store.get_safe_batch_size("unknown_op", available_memory_mb=5000)
        assert bs == 1


def test_corrupted_stats_file_handled():
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_file = Path(tmpdir) / "execution_stats.pkl"
        stats_file.write_text("not a valid pickle")

        store = ProfilingStore(storage_dir=tmpdir)
        assert store.get_execution_stats("anything") is None


def test_resource_throughput_curve_predict():
    curve = ResourceThroughputCurve(
        op_name="test",
        coefficients={"batch_coef": 2.0, "memory_coef": 0.1, "intercept": 10.0},
        model_type="linear",
    )
    t = curve.predict_throughput(batch_size=10, memory_mb=100)
    assert t == 2.0 * 10 + 0.1 * 100 + 10.0


def test_resource_throughput_curve_power_model():
    curve = ResourceThroughputCurve(
        op_name="test",
        coefficients={"scale": 5.0, "power": 0.8},
        model_type="power",
    )
    t = curve.predict_throughput(batch_size=10, memory_mb=0)
    assert abs(t - 5.0 * (10 ** 0.8)) < 1e-6


def test_export_report(tmp_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="report_op")
        for snap in _make_snapshots(5):
            stats.update(snap)
        store.update_execution_stats("report_op", stats)

        report_path = str(tmp_path / "report.md")
        store.export_report(report_path)
        content = Path(report_path).read_text()
        assert "report_op" in content
        assert "ElasticJuicer" in content
