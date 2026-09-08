"""Preflight GPU probing for ElasticJuicer throughput-aware actor planning.

Disposable Ray tasks run representative batches of every eligible GPU
operator before the formal job starts and collect:

- resident device memory (model/state loaded on the device);
- effective per-batch peak device memory (torch allocator peak and
  whole-device free-memory delta, whichever is larger) plus headroom;
- warmup + steady-state throughput in rows/s;
- an OOM upper bound by probing upward from the configured batch size;
- cost-column distribution statistics as the drift baseline.

Results are written to ``work_dir/gpu_probe_results.json`` with a
provenance fingerprint per stage (operator config + GPU + probe params).
A stage whose fingerprint matches the cache skips probing entirely; any
fingerprint change invalidates that stage's cached result.

The file additionally binds the input snapshot: a ``dataset`` block
carries a content fingerprint of the source files plus the total row
count and the probe sample digest.  A run whose dataset fingerprint no
longer matches rejects the whole cache (fail-closed) so stale
sample/cost evidence is never reused, while a fully compatible hit may
skip re-reading the sample and recounting rows entirely (see
:func:`dataset_cache_usable`).

The module keeps all arithmetic pure (dataclass inputs, dict outputs) so
the planner and tests run without Ray or CUDA.  Only
:func:`probe_gpu_stages` touches Ray.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .oom import is_oom_error

PROBE_RESULTS_SCHEMA_VERSION = 3
PROBE_RESULTS_FILENAME = "gpu_probe_results.json"


@dataclass(frozen=True)
class ProbeParams:
    """Public probe knobs; every field participates in the cache fingerprint."""

    warmup_batches: int = 1
    steady_batches: int = 3
    sample_rows: int = 256
    batch_cap_multiplier: int = 8
    memory_headroom: float = 0.10
    timeout_seconds: Optional[float] = None

    def validate(self) -> None:
        if self.warmup_batches < 0:
            raise ValueError("gpu_probe_warmup_batches must be >= 0")
        if self.steady_batches < 1:
            raise ValueError("gpu_probe_steady_batches must be >= 1")
        if self.sample_rows < 1:
            raise ValueError("gpu_probe_sample_rows must be >= 1")
        if self.batch_cap_multiplier < 1:
            raise ValueError("gpu_probe_batch_cap_multiplier must be >= 1")
        if not 0.0 <= self.memory_headroom <= 1.0:
            raise ValueError("gpu_probe memory_headroom must be in [0, 1]")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("gpu_probe_timeout_seconds must be positive or null")


@dataclass(frozen=True)
class StageProbeSpec:
    """One operator scheduled for probing."""

    stage_key: str
    op_name: str
    op_class: Any
    init_args: Tuple[Any, ...]
    init_kwargs: Dict[str, Any]
    batch_size: int
    num_gpus: float = 1.0

    def __post_init__(self):
        if not self.stage_key:
            raise ValueError("stage_key must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass
class StageProbeResult:
    """Measured outcome of one stage probe."""

    stage_key: str
    op_name: str
    fingerprint: str
    gpu_name: str
    resident_bytes: int
    effective_peak_bytes: int
    peak_bytes_with_headroom: int
    throughput_rows_per_sec: float
    probed_batch_size: int
    oom_upper_bound: Optional[int]
    cost_mean: Optional[float]
    cost_p50: Optional[float]
    cost_p95: Optional[float]
    cost_samples: int
    probe_duration_sec: float
    rows_probed: int
    cache_hit: bool = False
    init_seconds: float = 0.0
    allocator_resident_bytes: Optional[int] = None
    allocator_peak_bytes: Optional[int] = None

    @property
    def memory_headroom_factor(self) -> float:
        return max(1.0, self.peak_bytes_with_headroom / max(1, self.effective_peak_bytes))

    @property
    def resident_bytes_with_headroom(self) -> int:
        # Use the same allocator scope as online CUDA peaks. Whole-device
        # free-memory deltas can include unrelated processes and caches.
        if self.working_bytes_per_row is None:
            return 0
        return int(self.allocator_resident_bytes * self.memory_headroom_factor)

    @property
    def working_bytes_per_row(self) -> Optional[float]:
        if (
            self.allocator_resident_bytes is None
            or self.allocator_peak_bytes is None
            or not 0 <= self.allocator_resident_bytes <= self.allocator_peak_bytes
            or self.probed_batch_size < 1
        ):
            return None
        return (self.allocator_peak_bytes - self.allocator_resident_bytes) / self.probed_batch_size

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def probe_fingerprint(
    op_name: str,
    init_args: Tuple[Any, ...],
    init_kwargs: Dict[str, Any],
    gpu_name: str,
    params: ProbeParams,
) -> str:
    """Deterministic cache key: op config + GPU identity + probe params."""

    payload = {
        "op_name": op_name,
        "init_args": repr(tuple(init_args)),
        "init_kwargs": repr(sorted((k, v) for k, v in dict(init_kwargs).items())),
        "gpu_name": gpu_name,
        "params": asdict(params),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """tmp + fsync + os.replace so a crash never leaves a torn file."""

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".probe_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_probe_results(path: str) -> Optional[Dict[str, Any]]:
    """Load a cached probe file; corrupt or wrong-schema files return None."""

    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != PROBE_RESULTS_SCHEMA_VERSION:
        return None
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        return None
    return payload


def dataset_snapshot_fingerprint(paths: Optional[Sequence[str]]) -> Optional[str]:
    """Content fingerprint of the file-backed input snapshot.

    Every source path contributes its path, size, and file bytes to a
    single sha256, so any data edit, shard reordering, or added/removed
    shard changes the fingerprint.  The cost is one sequential read per
    file -- the same bytes the cached full row count was derived from.
    Returns None when the input is not plain file-backed (missing path,
    empty glob): callers must then treat the dataset binding as unusable
    (fail-closed).
    """

    resolved: List[str] = []
    for raw in paths or ():
        if not isinstance(raw, str) or not raw:
            return None
        if any(char in raw for char in "*?["):
            matches = sorted(glob.glob(raw))
            if not matches:
                return None
            resolved.extend(matches)
        elif os.path.isfile(raw):
            resolved.append(raw)
        else:
            return None
    if not resolved:
        return None
    digest = hashlib.sha256()
    for path in sorted(resolved):
        digest.update(path.encode("utf-8"))
        digest.update(str(os.path.getsize(path)).encode("ascii"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sample_columns_digest(sample_columns: Dict[str, List[Any]]) -> str:
    """Stable digest of the probe sample that was actually measured."""

    payload = json.dumps(
        {key: list(values) for key, values in sample_columns.items()},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_cache_usable(cached: Optional[Dict[str, Any]], fingerprint: Optional[str]) -> Optional[int]:
    """Cached total row count when the dataset binding still matches.

    The binding is only usable when the stored fingerprint equals the
    current input snapshot's fingerprint and both the total row count and
    the sample digest are intact.  Anything else returns None so callers
    fall back to the cold path (fail-closed).
    """

    if not isinstance(cached, dict) or fingerprint is None:
        return None
    binding = cached.get("dataset")
    if not isinstance(binding, dict) or binding.get("fingerprint") != fingerprint:
        return None
    total_rows = binding.get("total_rows")
    if isinstance(total_rows, bool) or not isinstance(total_rows, int) or total_rows <= 0:
        return None
    sample_digest = binding.get("sample_digest")
    if not isinstance(sample_digest, str) or not sample_digest:
        return None
    return total_rows


def save_probe_results(
    path: str,
    stages: Dict[str, StageProbeResult],
    dataset: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "schema_version": PROBE_RESULTS_SCHEMA_VERSION,
        "generated_at_ms": int(time.time() * 1000),
        "stages": {key: result.to_dict() for key, result in stages.items()},
    }
    if dataset:
        payload["dataset"] = dict(dataset)
    atomic_write_json(path, payload)


def _column_batches(sample_columns: Dict[str, List[Any]], batch_size: int) -> List[Dict[str, List[Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not sample_columns:
        return []
    row_count = len(next(iter(sample_columns.values())))
    batches = []
    for start in range(0, row_count, batch_size):
        stop = min(start + batch_size, row_count)
        batches.append({key: list(values[start:stop]) for key, values in sample_columns.items()})
    return batches


def _cycled_batch(sample_columns: Dict[str, List[Any]], size: int) -> Dict[str, List[Any]]:
    """Build a batch of exactly ``size`` rows, cycling the sample rows.

    The probe is advisory: repeated sample rows are acceptable because only
    resource behaviour (memory/throughput/OOM boundary) is measured.
    """

    if not sample_columns:
        raise ValueError("sample_columns must not be empty")
    row_count = len(next(iter(sample_columns.values())))
    if row_count == 0:
        raise ValueError("sample_columns must contain at least one row")
    batch: Dict[str, List[Any]] = {}
    for key, values in sample_columns.items():
        repeats = size // row_count + 1
        batch[key] = (list(values) * repeats)[:size]
    return batch


def _cost_stats(values: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None, None, 0
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)

    def _quantile(q: float) -> float:
        index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
        return ordered[index]

    return mean, _quantile(0.50), _quantile(0.95), len(ordered)


def probe_stage_on_worker(
    op_class: Any,
    init_args: Tuple[Any, ...],
    init_kwargs: Dict[str, Any],
    sample_columns: Dict[str, List[Any]],
    batch_size: int,
    params_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Worker-side probe body.  Runs inside a disposable Ray task with GPUs.

    Returns a plain dict so Ray serialization stays on primitive types.
    """

    import torch

    params = ProbeParams(**params_dict)
    started = time.time()
    init_started = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    free_before, _total = torch.cuda.mem_get_info()
    min_free_observed = free_before

    operator = op_class(*init_args, **init_kwargs)
    torch.cuda.synchronize()
    # Actor startup proxy: CUDA context creation plus operator construction
    # (model load) is the dominant fixed cost of bringing one more actor up.
    init_seconds = time.perf_counter() - init_started
    free_after_init, _ = torch.cuda.mem_get_info()
    resident_bytes = max(0, free_before - free_after_init)

    def _run_batch(batch: Dict[str, List[Any]]) -> int:
        nonlocal min_free_observed
        result = operator.process_batched(batch)
        torch.cuda.synchronize()
        _free_now, _ = torch.cuda.mem_get_info()
        min_free_observed = min(min_free_observed, _free_now)
        if isinstance(result, dict):
            first = next(iter(result.values()), [])
            return len(first) if hasattr(first, "__len__") else len(batch[next(iter(batch))])
        return len(result) if result is not None else 0

    # Warmup at the configured batch size.
    warmup_batches = _column_batches(sample_columns, batch_size)
    if not warmup_batches:
        raise ValueError("probe received no sample rows")
    warm_index = 0
    for _ in range(params.warmup_batches):
        _run_batch(warmup_batches[warm_index % len(warmup_batches)])
        warm_index += 1

    # Steady-state throughput + peak memory at the configured batch size.
    # Warmup may lazily initialize the model, so measure residency here.
    allocator_resident_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    rows = 0
    steady_wall = 0.0
    for i in range(params.steady_batches):
        batch = warmup_batches[(warm_index + i) % len(warmup_batches)]
        batch_rows = len(batch[next(iter(batch))])
        batch_start = time.perf_counter()
        _run_batch(batch)
        steady_wall += time.perf_counter() - batch_start
        rows += batch_rows
    torch_peak_bytes = torch.cuda.max_memory_allocated()
    throughput = rows / steady_wall if steady_wall > 0 else 0.0

    free_delta_peak = max(0, free_before - min_free_observed)
    effective_peak_bytes = max(torch_peak_bytes, free_delta_peak)
    peak_with_headroom = int(effective_peak_bytes * (1.0 + params.memory_headroom))

    # OOM boundary: geometric climb from the configured batch size.  The
    # first classified OOM becomes the exclusive upper bound seed.
    oom_upper_bound: Optional[int] = None
    cap = batch_size * params.batch_cap_multiplier
    size = batch_size * 2
    torch.cuda.empty_cache()
    while size <= cap:
        try:
            _run_batch(_cycled_batch(sample_columns, size))
        except BaseException as error:  # noqa: BLE001 - probe must classify and stop
            if is_oom_error(error):
                oom_upper_bound = size
                torch.cuda.empty_cache()
                break
            raise
        size *= 2

    return {
        "gpu_name": gpu_name,
        "resident_bytes": int(resident_bytes),
        "allocator_resident_bytes": int(allocator_resident_bytes),
        "allocator_peak_bytes": int(torch_peak_bytes),
        "effective_peak_bytes": int(effective_peak_bytes),
        "peak_bytes_with_headroom": int(peak_with_headroom),
        "throughput_rows_per_sec": float(throughput),
        "probed_batch_size": int(batch_size),
        "oom_upper_bound": oom_upper_bound,
        "probe_duration_sec": float(time.time() - started),
        "rows_probed": int(rows),
        "init_seconds": float(init_seconds),
    }


def resolve_probe_cache(
    specs: List[StageProbeSpec],
    params: ProbeParams,
    cached: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, StageProbeResult], List[StageProbeSpec]]:
    """Split specs into cache hits and stages that still need probing."""

    hits: Dict[str, StageProbeResult] = {}
    pending: List[StageProbeSpec] = []
    cached_stages = (cached or {}).get("stages", {}) if cached else {}
    for spec in specs:
        entry = cached_stages.get(spec.stage_key)
        if isinstance(entry, dict) and entry.get("fingerprint") == _spec_fingerprint(spec, params, entry.get("gpu_name", "")):
            result = StageProbeResult(**{key: entry[key] for key in entry if key in StageProbeResult.__dataclass_fields__})
            result.cache_hit = True
            hits[spec.stage_key] = result
        else:
            pending.append(spec)
    return hits, pending


def _spec_fingerprint(spec: StageProbeSpec, params: ProbeParams, gpu_name: str) -> str:
    return probe_fingerprint(spec.op_name, spec.init_args, spec.init_kwargs, gpu_name, params)


def probe_gpu_stages(
    specs: List[StageProbeSpec],
    sample_columns: Dict[str, List[Any]],
    params: ProbeParams,
    results_path: str,
    cost_column: Optional[str] = None,
    ray_module: Any = None,
    dataset_fingerprint: Optional[str] = None,
    dataset_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, StageProbeResult]:
    """Probe every spec serially and persist merged results.

    Serial probing is the default because concurrent large-model loads can
    saturate shared storage; the cache makes repeat runs cheap.

    ``dataset_fingerprint`` gates the whole cache: a snapshot whose
    fingerprint no longer matches is rejected before stage resolution, so
    stale sample/cost evidence is re-measured instead of reused.  When
    ``dataset_stats`` is provided (total row count, sample digest) it is
    bound to the fingerprint in the same atomic write as the stage
    results; a missing count keeps the previous valid binding instead of
    caching a wrong one.
    """

    params.validate()
    for spec in specs:
        if spec.batch_size < 1:
            raise ValueError(f"stage {spec.stage_key} has an invalid batch_size")

    cached = load_probe_results(results_path)
    if cached is not None and dataset_cache_usable(cached, dataset_fingerprint) is None:
        cached = None
    results, pending = resolve_probe_cache(specs, params, cached)

    if pending:
        if ray_module is None:
            import ray

            ray_module = ray
        probe_task = ray_module.remote(num_cpus=1)(probe_stage_on_worker)

    for spec in pending:
        future = probe_task.options(num_gpus=max(spec.num_gpus, 0.0)).remote(
            spec.op_class,
            tuple(spec.init_args),
            dict(spec.init_kwargs),
            sample_columns,
            spec.batch_size,
            asdict(params),
        )
        raw = ray_module.get(future, timeout=params.timeout_seconds)
        fingerprint = probe_fingerprint(
            spec.op_name, spec.init_args, spec.init_kwargs, raw["gpu_name"], params
        )
        cost_values = []
        if cost_column and cost_column in sample_columns:
            cost_values = [v for v in sample_columns[cost_column] if isinstance(v, (int, float))]
        cost_mean, cost_p50, cost_p95, cost_samples = _cost_stats(cost_values)
        results[spec.stage_key] = StageProbeResult(
            stage_key=spec.stage_key,
            op_name=spec.op_name,
            fingerprint=fingerprint,
            gpu_name=raw["gpu_name"],
            resident_bytes=raw["resident_bytes"],
            allocator_resident_bytes=raw.get("allocator_resident_bytes"),
            allocator_peak_bytes=raw.get("allocator_peak_bytes"),
            effective_peak_bytes=raw["effective_peak_bytes"],
            peak_bytes_with_headroom=raw["peak_bytes_with_headroom"],
            throughput_rows_per_sec=raw["throughput_rows_per_sec"],
            probed_batch_size=raw["probed_batch_size"],
            oom_upper_bound=raw["oom_upper_bound"],
            cost_mean=cost_mean,
            cost_p50=cost_p50,
            cost_p95=cost_p95,
            cost_samples=cost_samples,
            probe_duration_sec=raw["probe_duration_sec"],
            rows_probed=raw["rows_probed"],
            cache_hit=False,
            init_seconds=float(raw.get("init_seconds", 0.0) or 0.0),
        )

    dataset_block: Optional[Dict[str, Any]] = None
    if dataset_stats is not None and dataset_fingerprint is not None:
        dataset_block = {"fingerprint": dataset_fingerprint, **dataset_stats}
    elif cached is not None:
        previous = cached.get("dataset")
        if isinstance(previous, dict):
            dataset_block = dict(previous)
    save_probe_results(results_path, results, dataset=dataset_block)
    return results
