"""Shared contracts for ElasticJuicer components.

The scheduler, profiler, predictor, and executor integrations exchange these
small value objects.  Keeping them in a dependency-free module prevents the
control plane from depending on a concrete runtime such as psutil or Ray.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from time import time
from typing import Optional, Protocol


class TopologyMode(Enum):
    """Placement preference produced by the macro scheduler."""

    CO_LOCATION = "co_location"
    DISTRIBUTED = "distributed"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class MemoryState:
    """Point-in-time host/device memory observation, measured in MB."""

    timestamp: float
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    gpu_total_mb: Optional[float] = None
    gpu_used_mb: Optional[float] = None
    gpu_available_mb: Optional[float] = None
    gpu_percent: Optional[float] = None

    def __post_init__(self):
        for name in ("total_memory_mb", "used_memory_mb", "available_memory_mb"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= self.memory_percent <= 100:
            raise ValueError("memory_percent must be in [0, 100]")
        if self.gpu_percent is not None and not 0 <= self.gpu_percent <= 100:
            raise ValueError("gpu_percent must be in [0, 100]")

    def get_available_memory(self, use_gpu: bool = False) -> float:
        """Return available device memory when requested and observable."""

        if use_gpu and self.gpu_available_mb is not None:
            return self.gpu_available_mb
        return self.available_memory_mb


class MemoryStateProvider(Protocol):
    """Injectable source of memory observations."""

    def __call__(self) -> MemoryState: ...


class Clock(Protocol):
    """Injectable wall clock used by controllers and monitors."""

    def __call__(self) -> float: ...


@dataclass(frozen=True)
class BatchObservation:
    """Measured outcome of one logical micro-batch."""

    stage_name: str
    batch_size: int
    latency_ms: float
    throughput: float
    memory_peak_mb: float
    memory_delta_mb: float = 0.0
    gpu_memory_peak_mb: Optional[float] = None
    succeeded: bool = True
    error_type: Optional[str] = None
    timestamp: float = field(default_factory=time)

    def __post_init__(self):
        if not self.stage_name:
            raise ValueError("stage_name must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        for name in ("latency_ms", "throughput", "memory_peak_mb"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.succeeded and self.error_type is not None:
            raise ValueError("successful observations cannot contain error_type")


@dataclass(frozen=True)
class BatchDecision:
    """Controller output for the next logical micro-batch."""

    batch_size: int
    reason: str
    confidence: float = 1.0

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not self.reason:
            raise ValueError("reason must not be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class StageExecutionObservation:
    """Observed outcome of one complete operator stage execution."""

    stage_name: str
    configured_batch_size: int
    input_rows: int
    output_rows: int
    duration_ms: float
    throughput: float
    cpu_peak_percent: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    gpu_memory_peak_mb: Optional[float] = None
    gpu_peak_percent: Optional[float] = None
    succeeded: bool = True
    error_type: Optional[str] = None
    timestamp: float = field(default_factory=time)

    def __post_init__(self):
        if not self.stage_name:
            raise ValueError("stage_name must not be empty")
        if self.configured_batch_size < 1:
            raise ValueError("configured_batch_size must be at least 1")
        for name in ("input_rows", "output_rows"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("duration_ms", "throughput"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("cpu_peak_percent", "gpu_peak_percent"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
        for name in ("memory_peak_mb", "gpu_memory_peak_mb"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.succeeded and self.error_type is not None:
            raise ValueError("successful observations cannot contain error_type")

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass
class StageMetrics:
    """Windowed performance metrics for one executable operator stage."""

    stage_name: str
    queue_depth: int = 0
    current_parallelism: int = 1
    throughput: float = 0.0
    avg_latency_ms: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    oom_count: int = 0
    last_update: float = field(default_factory=time)

    def __post_init__(self):
        if not self.stage_name:
            raise ValueError("stage_name must not be empty")
        if self.queue_depth < 0:
            raise ValueError("queue_depth must be non-negative")
        if self.current_parallelism < 1:
            raise ValueError("current_parallelism must be at least 1")
        for name in ("throughput", "avg_latency_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("cpu_utilization", "memory_utilization", "gpu_utilization"):
            if not 0 <= getattr(self, name) <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
        if self.oom_count < 0:
            raise ValueError("oom_count must be non-negative")


@dataclass
class ResourceQuota:
    """Resource allocation produced for one stage controller."""

    captain_id: str
    target_parallelism: int
    cpu_quota: float
    memory_quota_mb: float
    gpu_quota: float = 0.0
    target_throughput: float = 0.0
    topology_mode: TopologyMode = TopologyMode.ADAPTIVE

    def __post_init__(self):
        if not self.captain_id:
            raise ValueError("captain_id must not be empty")
        if self.target_parallelism < 1:
            raise ValueError("target_parallelism must be at least 1")
        for name in ("cpu_quota", "memory_quota_mb", "gpu_quota", "target_throughput"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class ClusterState:
    """Capacity snapshot consumed by the macro scheduler."""

    total_cpu_cores: int
    total_memory_mb: float
    total_gpu_count: int
    available_cpu_cores: float
    available_memory_mb: float
    available_gpus: float
    timestamp: float = field(default_factory=time)

    def __post_init__(self):
        for name in (
            "total_cpu_cores",
            "total_memory_mb",
            "total_gpu_count",
            "available_cpu_cores",
            "available_memory_mb",
            "available_gpus",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.available_cpu_cores > self.total_cpu_cores:
            raise ValueError("available_cpu_cores cannot exceed total_cpu_cores")
        if self.available_memory_mb > self.total_memory_mb:
            raise ValueError("available_memory_mb cannot exceed total_memory_mb")
        if self.available_gpus > self.total_gpu_count:
            raise ValueError("available_gpus cannot exceed total_gpu_count")
