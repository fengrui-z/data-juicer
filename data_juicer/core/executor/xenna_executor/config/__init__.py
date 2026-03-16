"""
Configuration System for Xenna Executor

This module provides configuration classes for the Xenna executor,
including streaming settings, resource configuration, and memory safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ExecutionMode(Enum):
    """Execution mode for Xenna pipeline."""
    STREAMING = "streaming"
    BATCH = "batch"
    SERVING = "serving"


class VerbosityLevel(Enum):
    """Verbosity level for logging."""
    NONE = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5


@dataclass
class StreamingConfig:
    """
    Configuration for streaming execution mode.

    These parameters control the streaming behavior, autoscaling,
    and backpressure mechanisms.
    """

    # Autoscaling settings
    autoscale_interval_s: float = 60.0
    """How often to run the stage auto-scaler (seconds)."""

    speed_estimation_window_s: float = 180.0
    """Window size for estimating processing speed."""

    min_data_points: int = 5
    """Minimum data points for speed estimation."""

    # Backpressure settings
    max_queued_multiplier: float = 1.0
    """
    Multiplier for max queued tasks calculation.
    max_queued = num_actors * slots_per_actor * max_queued_multiplier
    Higher values = more buffering = higher throughput but more memory.
    """

    max_queued_lower_bound: int = 8
    """
    Minimum max_queued to prevent starvation when stages scale down.
    """

    # Logging
    verbosity_level: VerbosityLevel = VerbosityLevel.INFO
    """Verbosity level for autoscaler logging."""

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "StreamingConfig":
        """Create from dictionary."""
        return cls(
            autoscale_interval_s=config.get("autoscale_interval_s", 60.0),
            speed_estimation_window_s=config.get("speed_estimation_window_s", 180.0),
            min_data_points=config.get("min_data_points", 5),
            max_queued_multiplier=config.get("max_queued_multiplier", 1.0),
            max_queued_lower_bound=config.get("max_queued_lower_bound", 8),
            verbosity_level=VerbosityLevel[
                config.get("verbosity_level", "INFO").upper()
            ],
        )


@dataclass
class ResourceConfig:
    """
    Configuration for resource management.

    Controls CPU/GPU allocation, worker sizing, and resource efficiency.
    """

    # CPU allocation
    cpu_allocation_percentage: float = 0.95
    """Percentage of CPU resources to allocate to pipeline (rest for Ray)."""

    # Default worker resources
    default_cpus_per_worker: float = 1.0
    """Default CPU cores per worker."""

    default_gpus_per_worker: float = 0.0
    """Default GPUs per worker."""

    # GPU settings
    enable_gpu_fractional: bool = True
    """Allow fractional GPU allocation (< 1.0 GPU per worker)."""

    clear_cuda_on_cpu_actors: bool = True
    """Clear CUDA_VISIBLE_DEVICES for CPU-only actors."""

    # Memory
    max_memory_per_worker_gb: Optional[float] = None
    """Maximum memory per worker in GB (None for auto)."""

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ResourceConfig":
        """Create from dictionary."""
        return cls(
            cpu_allocation_percentage=config.get("cpu_allocation_percentage", 0.95),
            default_cpus_per_worker=config.get("default_cpus_per_worker", 1.0),
            default_gpus_per_worker=config.get("default_gpus_per_worker", 0.0),
            enable_gpu_fractional=config.get("enable_gpu_fractional", True),
            clear_cuda_on_cpu_actors=config.get("clear_cuda_on_cpu_actors", True),
            max_memory_per_worker_gb=config.get("max_memory_per_worker_gb"),
        )


@dataclass
class MemorySafetyConfig:
    """
    Configuration for memory safety features.

    These settings control memory management, backpressure, and
    out-of-memory prevention.
    """

    # Backpressure
    enable_backpressure: bool = True
    """Enable backpressure to prevent memory overflow."""

    memory_check_interval_s: float = 1.0
    """Interval for memory usage checks."""

    high_memory_threshold: float = 0.85
    """Memory usage threshold (0-1) to trigger backpressure."""

    critical_memory_threshold: float = 0.95
    """Memory usage threshold to trigger emergency measures."""

    # Emergency actions
    emergency_action: str = "pause"
    """
    Action when critical memory threshold is reached.
    Options: 'pause', 'scale_down', 'checkpoint'
    """

    # Garbage collection
    enable_auto_gc: bool = True
    """Enable automatic garbage collection under memory pressure."""

    gc_interval_s: float = 30.0
    """Interval for garbage collection checks."""

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MemorySafetyConfig":
        """Create from dictionary."""
        return cls(
            enable_backpressure=config.get("enable_backpressure", True),
            memory_check_interval_s=config.get("memory_check_interval_s", 1.0),
            high_memory_threshold=config.get("high_memory_threshold", 0.85),
            critical_memory_threshold=config.get("critical_memory_threshold", 0.95),
            emergency_action=config.get("emergency_action", "pause"),
            enable_auto_gc=config.get("enable_auto_gc", True),
            gc_interval_s=config.get("gc_interval_s", 30.0),
        )


@dataclass
class FaultToleranceConfig:
    """
    Configuration for fault tolerance and recovery.
    """

    # Retry settings
    num_setup_attempts: int = 1
    """Number of attempts to run stage setup."""

    num_run_attempts: int = 1
    """Number of attempts to run process_data per task."""

    ignore_failures: bool = False
    """Ignore failures in process_data (continue processing)."""

    reset_workers_on_failure: bool = False
    """Reset workers when a failure occurs."""

    max_setup_failure_percentage: Optional[float] = None
    """Maximum percentage of setup failures before pipeline fails."""

    # Worker lifecycle
    worker_max_lifetime_m: int = 0
    """Maximum worker lifetime in minutes (0 = unlimited)."""

    worker_restart_interval_m: int = 1
    """Interval between worker restarts (minutes)."""

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FaultToleranceConfig":
        """Create from dictionary."""
        return cls(
            num_setup_attempts=config.get("num_setup_attempts", 1),
            num_run_attempts=config.get("num_run_attempts", 1),
            ignore_failures=config.get("ignore_failures", False),
            reset_workers_on_failure=config.get("reset_workers_on_failure", False),
            max_setup_failure_percentage=config.get("max_setup_failure_percentage"),
            worker_max_lifetime_m=config.get("worker_max_lifetime_m", 0),
            worker_restart_interval_m=config.get("worker_restart_interval_m", 1),
        )


@dataclass
class XennaConfig:
    """
    Main configuration class for Xenna executor.

    This class aggregates all configuration for streaming execution,
    resource management, memory safety, and fault tolerance.

    Example:
        >>> config = XennaConfig(
        ...     execution_mode=ExecutionMode.STREAMING,
        ...     streaming=StreamingConfig(
        ...         autoscale_interval_s=30.0,
        ...         max_queued_multiplier=2.0,
        ...     ),
        ...     resource=ResourceConfig(
        ...         cpu_allocation_percentage=0.9,
        ...     ),
        ... )
    """

    # Execution mode
    execution_mode: ExecutionMode = ExecutionMode.STREAMING
    """Pipeline execution mode."""

    # Streaming settings
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    """Streaming-specific configuration."""

    # Resource settings
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    """Resource management configuration."""

    # Memory safety
    memory_safety: MemorySafetyConfig = field(default_factory=MemorySafetyConfig)
    """Memory safety configuration."""

    # Fault tolerance
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)
    """Fault tolerance configuration."""

    # Pipeline settings
    slots_per_actor: int = 2
    """Number of concurrent tasks per actor."""

    logging_interval_s: float = 60.0
    """Interval for status logging."""

    over_provision_factor: Optional[float] = None
    """Over-provision factor for worker allocation."""

    # Debug options
    log_worker_allocation: bool = False
    """Log detailed worker allocation layout."""

    return_last_stage_outputs: bool = True
    """Return outputs from the last stage."""

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "XennaConfig":
        """Create configuration from dictionary."""
        # Parse execution mode
        mode_str = config.get("execution_mode", "streaming").upper()
        execution_mode = ExecutionMode[mode_str]

        # Parse sub-configurations
        streaming = StreamingConfig.from_dict(config.get("streaming", {}))
        resource = ResourceConfig.from_dict(config.get("resource", {}))
        memory_safety = MemorySafetyConfig.from_dict(config.get("memory_safety", {}))
        fault_tolerance = FaultToleranceConfig.from_dict(config.get("fault_tolerance", {}))

        return cls(
            execution_mode=execution_mode,
            streaming=streaming,
            resource=resource,
            memory_safety=memory_safety,
            fault_tolerance=fault_tolerance,
            slots_per_actor=config.get("slots_per_actor", 2),
            logging_interval_s=config.get("logging_interval_s", 60.0),
            over_provision_factor=config.get("over_provision_factor"),
            log_worker_allocation=config.get("log_worker_allocation", False),
            return_last_stage_outputs=config.get("return_last_stage_outputs", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_mode": self.execution_mode.value,
            "streaming": {
                "autoscale_interval_s": self.streaming.autoscale_interval_s,
                "speed_estimation_window_s": self.streaming.speed_estimation_window_s,
                "min_data_points": self.streaming.min_data_points,
                "max_queued_multiplier": self.streaming.max_queued_multiplier,
                "max_queued_lower_bound": self.streaming.max_queued_lower_bound,
                "verbosity_level": self.streaming.verbosity_level.name,
            },
            "resource": {
                "cpu_allocation_percentage": self.resource.cpu_allocation_percentage,
                "default_cpus_per_worker": self.resource.default_cpus_per_worker,
                "default_gpus_per_worker": self.resource.default_gpus_per_worker,
                "enable_gpu_fractional": self.resource.enable_gpu_fractional,
                "clear_cuda_on_cpu_actors": self.resource.clear_cuda_on_cpu_actors,
                "max_memory_per_worker_gb": self.resource.max_memory_per_worker_gb,
            },
            "memory_safety": {
                "enable_backpressure": self.memory_safety.enable_backpressure,
                "memory_check_interval_s": self.memory_safety.memory_check_interval_s,
                "high_memory_threshold": self.memory_safety.high_memory_threshold,
                "critical_memory_threshold": self.memory_safety.critical_memory_threshold,
                "emergency_action": self.memory_safety.emergency_action,
                "enable_auto_gc": self.memory_safety.enable_auto_gc,
                "gc_interval_s": self.memory_safety.gc_interval_s,
            },
            "fault_tolerance": {
                "num_setup_attempts": self.fault_tolerance.num_setup_attempts,
                "num_run_attempts": self.fault_tolerance.num_run_attempts,
                "ignore_failures": self.fault_tolerance.ignore_failures,
                "reset_workers_on_failure": self.fault_tolerance.reset_workers_on_failure,
                "max_setup_failure_percentage": self.fault_tolerance.max_setup_failure_percentage,
                "worker_max_lifetime_m": self.fault_tolerance.worker_max_lifetime_m,
                "worker_restart_interval_m": self.fault_tolerance.worker_restart_interval_m,
            },
            "slots_per_actor": self.slots_per_actor,
            "logging_interval_s": self.logging_interval_s,
            "over_provision_factor": self.over_provision_factor,
            "log_worker_allocation": self.log_worker_allocation,
            "return_last_stage_outputs": self.return_last_stage_outputs,
        }


# Convenience aliases for backward compatibility
# These map to fault tolerance settings
num_setup_attempts: int = 1
num_run_attempts: int = 1
ignore_failures: bool = False
reset_workers_on_failure: bool = False
worker_max_lifetime_m: int = 0
