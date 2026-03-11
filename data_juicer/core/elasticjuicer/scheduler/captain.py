"""
Captain: Local Per-Operator Scheduler for ElasticJuicer

Based on Autothrottle's bi-level architecture, Captain is the local scheduler
that manages a single operator stage under Tower's global constraints.

Key responsibilities:
1. Execute Micro-Scheduler (JABAS-style batch size control) within quota
2. Report metrics to Tower
3. Enforce resource quotas from Tower
4. Handle local OOM events and recovery
5. Coordinate with adjacent Captains in pipeline

References:
- Autothrottle (NSDI 2024): Bi-level control for SLO-targeted microservices
- Report Section 5.1: Tower/Captain architecture
"""

import time
from dataclasses import dataclass
from typing import Optional, Callable, List
from collections import deque
import psutil

from ..scheduler.micro_scheduler import MicroScheduler, BatchSizeController
from ..scheduler.tower import ResourceQuota, StageMetrics, TopologyMode
from ..profiler.resource_monitor import ResourceMonitor, ResourceSnapshot
from ..predictor.memory_predictor import MemoryPredictor
from ..predictor.feature_extractor import FeatureExtractor


@dataclass
class CaptainConfig:
    """Configuration for Captain scheduler"""
    stage_name: str
    initial_batch_size: int = 32
    report_interval_sec: float = 1.0      # How often to report to Tower
    quota_check_interval_sec: float = 0.5  # How often to check quota
    enable_micro_scheduler: bool = True    # Use JABAS-style control
    enable_prediction: bool = True         # Use memory prediction
    emergency_backoff_ratio: float = 0.5   # OOM backoff ratio


class Captain:
    """
    Local Per-Operator Scheduler (Captain from Autothrottle architecture)
    
    Captain manages a single operator stage, executing micro-scheduling decisions
    (batch size adjustment) within the global constraints set by Tower.
    
    Key design:
    - Tower sets "what to achieve" (target parallelism, resource quota, SLO)
    - Captain decides "how to achieve it" (batch size, local optimization)
    - Bi-level decoupling enables scalability and autonomy
    """
    
    def __init__(
        self,
        config: CaptainConfig,
        tower_callback: Optional[Callable[[StageMetrics], None]] = None
    ):
        """
        Initialize Captain local scheduler
        
        Args:
            config: Captain configuration
            tower_callback: Callback function to report metrics to Tower
        """
        self.config = config
        self.tower_callback = tower_callback
        
        # Micro-scheduler for batch size control
        if config.enable_micro_scheduler:
            self.micro_scheduler = MicroScheduler(
                initial_batch_size=config.initial_batch_size,
                max_batch_size=1024,
                min_batch_size=1
            )
        else:
            self.micro_scheduler = None
        
        # Resource monitoring
        self.monitor = ResourceMonitor()
        
        # Memory prediction
        if config.enable_prediction:
            self.predictor = MemoryPredictor(op_name=config.stage_name)
            self.feature_extractor = FeatureExtractor()
        else:
            self.predictor = None
            self.feature_extractor = None
        
        # Current resource quota from Tower
        self.quota: Optional[ResourceQuota] = None
        
        # Current stage metrics
        self.metrics = StageMetrics(stage_name=config.stage_name)
        
        # Queue simulation
        self.queue: deque = deque()
        
        # Timing
        self.last_report_time = time.time()
        self.last_quota_check_time = time.time()
        
        # Processing statistics
        self.samples_processed = 0
        self.total_latency_ms = 0.0
        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        
        # OOM tracking
        self.oom_events = 0
        self.last_oom_time = 0.0
    
    def set_quota(self, quota: ResourceQuota):
        """
        Receive resource quota from Tower
        
        Args:
            quota: Resource allocation from Tower
        """
        self.quota = quota
        
        # Update micro-scheduler constraints if quota changed
        if self.micro_scheduler and quota.memory_quota_mb > 0:
            # Adjust max batch size based on memory quota
            # Rough estimate: 100MB per sample for typical multimodal data
            estimated_max_batch = max(1, int(quota.memory_quota_mb / 100))
            self.micro_scheduler.controller.max_batch_size = min(
                self.micro_scheduler.controller.max_batch_size,
                estimated_max_batch
            )
    
    def enqueue_samples(self, samples: List):
        """
        Add samples to processing queue
        
        Args:
            samples: List of samples to process
        """
        for sample in samples:
            self.queue.append(sample)
        
        # Update queue depth metric
        self.metrics.queue_depth = len(self.queue)
    
    def process_batch(
        self, 
        operator_func: Callable,
        sample_batch: Optional[List] = None
    ) -> Optional[List]:
        """
        Process a batch using the operator, with Captain's orchestration
        
        This is the core execution loop that:
        1. Gets batch size from micro-scheduler
        2. Dequeues samples
        3. Monitors execution
        4. Updates predictor and scheduler
        5. Checks quota compliance
        
        Args:
            operator_func: The actual operator function to execute
            sample_batch: Optional pre-formed batch (if None, dequeue from queue)
            
        Returns:
            Processed results or None if queue empty
        """
        start_time = time.time()
        
        # Get current batch size recommendation
        if self.micro_scheduler:
            current_batch_size = self.micro_scheduler.controller.current_batch_size
        else:
            current_batch_size = self.config.initial_batch_size
        
        # Dequeue samples if not provided
        if sample_batch is None:
            if len(self.queue) == 0:
                return None
            
            actual_batch_size = min(current_batch_size, len(self.queue))
            sample_batch = [self.queue.popleft() for _ in range(actual_batch_size)]
        else:
            actual_batch_size = len(sample_batch)
        
        # Extract features for prediction (if enabled)
        predicted_memory_mb = None
        if self.predictor and self.feature_extractor and len(sample_batch) > 0:
            features = self.feature_extractor.extract_from_sample(sample_batch[0])
            features.batch_size = actual_batch_size  # Set batch size
            prediction = self.predictor.predict(features)
            if prediction:
                predicted_memory_mb = prediction.predicted_memory_mb
        
        # Monitor execution
        with self.monitor.measure_execution(
            self.config.stage_name, 
            actual_batch_size
        ):
            try:
                # Execute operator
                results = operator_func(sample_batch)
                
                # Record success
                self.samples_processed += actual_batch_size
                
            except MemoryError as e:
                # OOM event - get approximate snapshot
                snapshot_approx = ResourceSnapshot(
                    timestamp=time.time(),
                    batch_size=actual_batch_size,
                    cpu_percent=psutil.cpu_percent(),
                    memory_mb=psutil.virtual_memory().used / (1024 * 1024),
                    latency_ms=0
                )
                self._handle_oom(actual_batch_size, snapshot_approx)
                raise
        
        # Get recorded stats
        op_stats = self.monitor.get_stats(self.config.stage_name)
        if op_stats and op_stats.snapshots:
            snapshot = op_stats.snapshots[-1]  # Get latest snapshot
            # Update predictor with actual memory usage
            if self.predictor and self.feature_extractor and len(sample_batch) > 0:
                features = self.feature_extractor.extract_from_sample(sample_batch[0])
                self.predictor.observe(features, snapshot.memory_mb)
            
            # Update micro-scheduler
            if self.micro_scheduler:
                self.micro_scheduler.update(
                    actual_memory_used=snapshot.memory_mb,
                    sample_features=None  # Already updated predictor above
                )
            
            # Update metrics
            latency_ms = snapshot.latency_ms
            self.total_latency_ms += latency_ms
            self.latency_history.append(latency_ms)
            
            throughput = snapshot.throughput
            self.throughput_history.append(throughput)
            
            self.metrics.avg_latency_ms = (
                sum(self.latency_history) / len(self.latency_history)
                if self.latency_history else 0
            )
            self.metrics.throughput = (
                sum(self.throughput_history) / len(self.throughput_history)
                if self.throughput_history else 0
            )
            self.metrics.cpu_utilization = snapshot.cpu_percent
            self.metrics.memory_utilization = (
                (snapshot.memory_mb / self.quota.memory_quota_mb * 100)
                if self.quota and self.quota.memory_quota_mb > 0
                else 0
            )
            self.metrics.gpu_utilization = snapshot.gpu_utilization or 0
            self.metrics.queue_depth = len(self.queue)
            self.metrics.oom_count = self.oom_events
            self.metrics.current_parallelism = 1  # Single-actor for now
        
        # Check if should report to Tower
        current_time = time.time()
        if current_time - self.last_report_time >= self.config.report_interval_sec:
            self._report_to_tower()
            self.last_report_time = current_time
        
        # Check quota compliance
        if current_time - self.last_quota_check_time >= self.config.quota_check_interval_sec:
            self._check_quota_compliance()
            self.last_quota_check_time = current_time
        
        return results
    
    def _handle_oom(self, batch_size: int, snapshot: Optional[ResourceSnapshot]):
        """
        Handle OOM event with emergency backoff
        
        Args:
            batch_size: Batch size that caused OOM
            snapshot: Resource snapshot at OOM time
        """
        self.oom_events += 1
        self.last_oom_time = time.time()
        
        # Emergency backoff
        if self.micro_scheduler:
            new_batch_size = max(1, batch_size // 2)
            self.micro_scheduler.controller.current_batch_size = new_batch_size
            self.micro_scheduler.controller.max_batch_size = batch_size
        
        # Update metrics
        self.metrics.oom_count = self.oom_events
    
    def _report_to_tower(self):
        """Report current metrics to Tower"""
        if self.tower_callback:
            # Update timestamp
            self.metrics.last_update = time.time()
            
            # Send to Tower
            self.tower_callback(self.metrics)
    
    def _check_quota_compliance(self):
        """
        Check if current resource usage is within Tower's quota
        
        If exceeding quota, apply throttling
        """
        if not self.quota:
            return
        
        # Check memory quota
        current_memory_mb = psutil.virtual_memory().used / (1024 * 1024)
        if current_memory_mb > self.quota.memory_quota_mb:
            # Exceeding memory quota, reduce batch size
            if self.micro_scheduler:
                reduction_ratio = self.quota.memory_quota_mb / current_memory_mb
                new_batch_size = max(
                    1, 
                    int(self.micro_scheduler.controller.current_batch_size * reduction_ratio)
                )
                self.micro_scheduler.controller.current_batch_size = new_batch_size
    
    def _get_available_memory_mb(self) -> float:
        """Get available system memory in MB"""
        return psutil.virtual_memory().available / (1024 * 1024)
    
    def get_stats(self) -> dict:
        """Get Captain statistics"""
        return {
            'stage_name': self.config.stage_name,
            'samples_processed': self.samples_processed,
            'queue_depth': len(self.queue),
            'current_batch_size': (
                self.micro_scheduler.controller.current_batch_size 
                if self.micro_scheduler else self.config.initial_batch_size
            ),
            'avg_latency_ms': self.metrics.avg_latency_ms,
            'avg_throughput': self.metrics.throughput,
            'oom_events': self.oom_events,
            'quota': {
                'target_parallelism': self.quota.target_parallelism if self.quota else 1,
                'memory_quota_mb': self.quota.memory_quota_mb if self.quota else 0,
                'cpu_quota': self.quota.cpu_quota if self.quota else 0,
            } if self.quota else None
        }


class CaptainPool:
    """
    Manages multiple Captains in a pipeline
    
    Coordinates execution across multiple stages, ensuring data flows
    correctly and all Captains report to Tower.
    """
    
    def __init__(self, tower_callback: Optional[Callable[[StageMetrics], None]] = None):
        """
        Initialize Captain pool
        
        Args:
            tower_callback: Shared callback to Tower for all Captains
        """
        self.tower_callback = tower_callback
        self.captains: dict[str, Captain] = {}
    
    def add_captain(self, config: CaptainConfig) -> Captain:
        """
        Add a new Captain to the pool
        
        Args:
            config: Configuration for the Captain
            
        Returns:
            The created Captain instance
        """
        captain = Captain(config, self.tower_callback)
        self.captains[config.stage_name] = captain
        return captain
    
    def get_captain(self, stage_name: str) -> Optional[Captain]:
        """Get Captain by stage name"""
        return self.captains.get(stage_name)
    
    def set_quotas(self, quotas: dict[str, ResourceQuota]):
        """
        Distribute quotas from Tower to all Captains
        
        Args:
            quotas: Dict mapping captain_id to ResourceQuota
        """
        for captain_id, quota in quotas.items():
            # Extract stage name from captain_id
            stage_name = quota.captain_id.replace('captain_', '').rsplit('_', 1)[0]
            
            if stage_name in self.captains:
                self.captains[stage_name].set_quota(quota)
    
    def get_all_stats(self) -> dict[str, dict]:
        """Get statistics from all Captains"""
        return {
            name: captain.get_stats()
            for name, captain in self.captains.items()
        }
