"""
Scheduler Configuration

Centralized configuration for micro and macro schedulers.
"""

from dataclasses import dataclass
from typing import Optional


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
