"""
Memory Safety Controller

Provides memory monitoring, backpressure control, and emergency
measures to prevent out-of-memory errors during streaming execution.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from loguru import logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, memory monitoring will be limited")


class MemoryState(Enum):
    """Memory pressure states."""
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryStats:
    """Memory statistics snapshot."""
    total_gb: float
    available_gb: float
    used_gb: float
    used_percent: float
    timestamp: float = field(default_factory=time.time)

    @property
    def available_percent(self) -> float:
        return 1.0 - self.used_percent


@dataclass
class BackpressureState:
    """State for backpressure control."""
    is_active: bool = False
    paused_stages: set = field(default_factory=set)
    last_adjustment: float = 0.0
    adjustment_count: int = 0


class MemorySafetyController:
    """
    Controls memory usage and applies backpressure to prevent OOM.

    Features:
    - Real-time memory monitoring
    - Automatic backpressure application
    - Emergency actions (pause, scale down, checkpoint)
    - Garbage collection triggers
    - Configurable thresholds

    Example:
        >>> controller = MemorySafetyController(
        ...     max_queued_multiplier=2.0,
        ...     max_queued_lower_bound=8,
        ...     high_memory_threshold=0.85,
        ... )
        >>> controller.start_monitoring()
        >>> if controller.should_apply_backpressure():
        ...     controller.apply_backpressure()
    """

    def __init__(
        self,
        max_queued_multiplier: float = 1.0,
        max_queued_lower_bound: int = 8,
        high_memory_threshold: float = 0.85,
        critical_memory_threshold: float = 0.95,
        check_interval_s: float = 1.0,
        enable_auto_gc: bool = True,
        gc_interval_s: float = 30.0,
    ):
        """
        Initialize memory safety controller.

        Args:
            max_queued_multiplier: Multiplier for max queued tasks
            max_queued_lower_bound: Minimum max queued tasks
            high_memory_threshold: Threshold to trigger backpressure
            critical_memory_threshold: Threshold for emergency action
            check_interval_s: Memory check interval
            enable_auto_gc: Enable automatic garbage collection
            gc_interval_s: GC interval in seconds
        """
        self.max_queued_multiplier = max_queued_multiplier
        self.max_queued_lower_bound = max_queued_lower_bound
        self.high_memory_threshold = high_memory_threshold
        self.critical_memory_threshold = critical_memory_threshold
        self.check_interval_s = check_interval_s
        self.enable_auto_gc = enable_auto_gc
        self.gc_interval_s = gc_interval_s

        # State
        self._state = MemoryState.NORMAL
        self._backpressure = BackpressureState()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_high_memory: Optional[Callable] = None
        self._on_critical_memory: Optional[Callable] = None

        # Stats tracking
        self._stats_history: list = []
        self._max_history_size = 1000
        self._last_gc_time = time.time()

    @property
    def state(self) -> MemoryState:
        """Current memory state."""
        return self._state

    @property
    def is_backpressure_active(self) -> bool:
        """Whether backpressure is currently active."""
        return self._backpressure.is_active

    def get_current_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            return MemoryStats(
                total_gb=memory.total / (1024**3),
                available_gb=memory.available / (1024**3),
                used_gb=memory.used / (1024**3),
                used_percent=memory.percent / 100.0,
            )
        else:
            # Fallback: estimate from process
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            used_mb = rusage.ru_maxrss / 1024  # Convert to MB
            return MemoryStats(
                total_gb=16.0,  # Assume 16GB
                available_gb=8.0,  # Estimate
                used_gb=used_mb / 1024,
                used_percent=0.5,  # Estimate
            )

    def calculate_max_queued(self, num_actors: int, slots_per_actor: int) -> int:
        """
        Calculate maximum queued tasks based on backpressure.

        Args:
            num_actors: Number of actors in the stage
            slots_per_actor: Slots per actor

        Returns:
            Maximum queued tasks
        """
        base = num_actors * slots_per_actor * self.max_queued_multiplier
        return max(int(base), self.max_queued_lower_bound)

    def should_apply_backpressure(self) -> bool:
        """Check if backpressure should be applied."""
        stats = self.get_current_memory_stats()

        if stats.used_percent >= self.critical_memory_threshold:
            self._state = MemoryState.CRITICAL
            return True
        elif stats.used_percent >= self.high_memory_threshold:
            self._state = MemoryState.HIGH
            return True

        self._state = MemoryState.NORMAL
        return False

    def apply_backpressure(self, stage_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply backpressure to control memory usage.

        Args:
            stage_id: Optional stage to pause

        Returns:
            Action taken
        """
        action = {
            "timestamp": time.time(),
            "memory_state": self._state.value,
            "action": None,
        }

        if self._state == MemoryState.CRITICAL:
            # Critical: take emergency action
            action["action"] = "emergency"

            # Force garbage collection
            if self.enable_auto_gc:
                collected = gc.collect()
                action["gc_collected"] = collected
                logger.warning(
                    f"Critical memory ({self.get_current_memory_stats().used_percent:.1%}), "
                    f"GC collected {collected} objects"
                )

            # Call critical callback
            if self._on_critical_memory:
                self._on_critical_memory()

        elif self._state == MemoryState.HIGH:
            # High: apply backpressure
            action["action"] = "backpressure"
            self._backpressure.is_active = True

            if stage_id:
                self._backpressure.paused_stages.add(stage_id)

            # Call high memory callback
            if self._on_high_memory:
                self._on_high_memory()

        self._backpressure.last_adjustment = time.time()
        self._backpressure.adjustment_count += 1

        return action

    def release_backpressure(self, stage_id: Optional[str] = None) -> bool:
        """
        Release backpressure when memory normalizes.

        Args:
            stage_id: Stage to resume

        Returns:
            Whether backpressure was released
        """
        if not self._backpressure.is_active:
            return False

        if stage_id:
            self._backpressure.paused_stages.discard(stage_id)

        # Only release if all stages resumed and memory is OK
        if not self._backpressure.paused_stages and self._state == MemoryState.NORMAL:
            self._backpressure.is_active = False
            logger.info("Backpressure released")
            return True

        return False

    def start_monitoring(self):
        """Start background memory monitoring thread."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                stats = self.get_current_memory_stats()
                self._stats_history.append(stats)

                # Trim history
                if len(self._stats_history) > self._max_history_size:
                    self._stats_history = self._stats_history[-self._max_history_size:]

                # Check for backpressure
                if self.should_apply_backpressure():
                    self.apply_backpressure()

                # Periodic GC
                if (
                    self.enable_auto_gc
                    and time.time() - self._last_gc_time > self.gc_interval_s
                ):
                    gc.collect()
                    self._last_gc_time = time.time()

            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")

            time.sleep(self.check_interval_s)

    def set_callbacks(
        self,
        on_high_memory: Optional[Callable] = None,
        on_critical_memory: Optional[Callable] = None,
    ):
        """
        Set callbacks for memory events.

        Args:
            on_high_memory: Callback for high memory
            on_critical_memory: Callback for critical memory
        """
        self._on_high_memory = on_high_memory
        self._on_critical_memory = on_critical_memory

    def get_memory_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Get memory usage trend.

        Args:
            window_size: Number of recent samples to analyze

        Returns:
            Trend information
        """
        if len(self._stats_history) < 2:
            return {"trend": "unknown", "rate": 0.0}

        recent = self._stats_history[-window_size:]
        if len(recent) < 2:
            return {"trend": "unknown", "rate": 0.0}

        # Calculate rate of change
        first = recent[0]
        last = recent[-1]
        time_diff = last.timestamp - first.timestamp

        if time_diff <= 0:
            return {"trend": "stable", "rate": 0.0}

        rate = (last.used_percent - first.used_percent) / time_diff

        if rate > 0.01:
            trend = "increasing"
        elif rate < -0.01:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "rate": rate,
            "current_percent": last.used_percent,
            "window_samples": len(recent),
        }

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of memory statistics."""
        if not self._stats_history:
            return {"status": "no_data"}

        recent = self._stats_history[-100:]

        return {
            "current_state": self._state.value,
            "backpressure_active": self._backpressure.is_active,
            "current_percent": recent[-1].used_percent if recent else 0,
            "min_percent": min(s.used_percent for s in recent),
            "max_percent": max(s.used_percent for s in recent),
            "avg_percent": sum(s.used_percent for s in recent) / len(recent),
            "total_gb": recent[-1].total_gb if recent else 0,
            "samples": len(self._stats_history),
        }
