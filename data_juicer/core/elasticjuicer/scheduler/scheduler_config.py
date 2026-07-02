"""
Scheduler Configuration

Centralized configuration for micro and macro schedulers.
"""

from dataclasses import dataclass
@dataclass
class SchedulerConfig:
    """Configuration for ElasticJuicer schedulers"""
    
    # Batch size control
    initial_batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 1000
    
    # Memory management
    target_memory_utilization: float = 0.85  # 85% utilization target
    safety_buffer_mb: float = 1000.0  # 1GB safety buffer
    use_gpu_memory: bool = False
    
    # PID tuning
    pid_kp: float = 0.5   # Proportional gain
    pid_ki: float = 0.05  # Integral gain
    pid_kd: float = 0.1   # Derivative gain
    
    # Auto-adjustment
    enable_auto_adjust: bool = True
    enable_prediction: bool = True
    
    # Predictor settings
    predictor_window_size: int = 100
    predictor_min_samples: int = 5
    predictor_confidence_level: float = 0.95
    
    # Safety settings
    max_batch_change_ratio: float = 0.5  # Max 50% change per adjustment
    oom_backoff_ratio: float = 0.5  # Reduce to 50% on OOM

    def __post_init__(self):
        if self.min_batch_size < 1:
            raise ValueError("min_batch_size must be at least 1")
        if not self.min_batch_size <= self.initial_batch_size <= self.max_batch_size:
            raise ValueError("initial_batch_size must be within batch size bounds")
        for name in ("target_memory_utilization", "predictor_confidence_level"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be in (0, 1)")
        for name in ("max_batch_change_ratio", "oom_backoff_ratio"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.safety_buffer_mb < 0:
            raise ValueError("safety_buffer_mb must be non-negative")
        if self.predictor_window_size < 1:
            raise ValueError("predictor_window_size must be at least 1")
        if not 1 <= self.predictor_min_samples <= self.predictor_window_size:
            raise ValueError("predictor_min_samples must be within predictor window")
        for name in ("pid_kp", "pid_ki", "pid_kd"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
    
    @classmethod
    def conservative(cls) -> 'SchedulerConfig':
        """Conservative configuration (prioritizes safety)"""
        return cls(
            target_memory_utilization=0.70,
            safety_buffer_mb=2000.0,
            max_batch_change_ratio=0.25,
        )
    
    @classmethod
    def aggressive(cls) -> 'SchedulerConfig':
        """Aggressive configuration (prioritizes throughput)"""
        return cls(
            target_memory_utilization=0.95,
            safety_buffer_mb=500.0,
            max_batch_change_ratio=0.75,
        )
    
    @classmethod
    def gpu(cls) -> 'SchedulerConfig':
        """GPU-optimized configuration"""
        return cls(
            use_gpu_memory=True,
            target_memory_utilization=0.90,
            safety_buffer_mb=1024.0,  # 1GB buffer for GPU
        )
