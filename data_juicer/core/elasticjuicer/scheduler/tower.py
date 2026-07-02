"""
Tower: Global Macro-Scheduler for ElasticJuicer

Based on Autothrottle's bi-level architecture, Tower is the global resource allocator
that sets performance targets and resource quotas for local Captains.

Key responsibilities:
1. Monitor global queue depth and cluster resource utilization
2. Set target parallelism for each operator stage
3. Allocate resource budgets to Captains
4. Make topology decisions (co-location vs distributed)
5. Handle global SLA guarantees

References:
- Autothrottle (NSDI 2024): Bi-level control for SLO-targeted microservices
- Report Section 5.1: Tower/Captain architecture
"""

import time
from collections import deque
from typing import Dict, List, Optional

from ..contracts import ClusterState, ResourceQuota, StageMetrics, TopologyMode


class Tower:
    """
    Global Macro-Scheduler (Tower from Autothrottle architecture)
    
    Tower doesn't directly control individual actors' behavior. Instead, it:
    1. Monitors global system state (queue depths, resource utilization)
    2. Sets performance targets and resource quotas for Captains
    3. Makes high-level topology decisions
    4. Ensures cluster-wide SLA guarantees
    
    The bi-level design (Tower + Captain) solves the single-point bottleneck
    problem of centralized schedulers, enabling high-frequency local decisions
    under global constraints.
    """
    
    def __init__(
        self,
        cluster_state: ClusterState,
        target_queue_depth: int = 100,
        sla_latency_ms: float = 5000.0,
        update_interval_sec: float = 5.0,
        history_window: int = 20
    ):
        """
        Initialize Tower global scheduler
        
        Args:
            cluster_state: Initial cluster resource state
            target_queue_depth: Target queue depth to maintain
            sla_latency_ms: SLA latency target (max allowed latency)
            update_interval_sec: How often to recompute resource allocation
            history_window: Window size for tracking metrics history
        """
        self.cluster = cluster_state
        self.target_queue_depth = target_queue_depth
        self.sla_latency_ms = sla_latency_ms
        self.update_interval = update_interval_sec
        
        # Track all stages and their metrics
        self.stages: Dict[str, StageMetrics] = {}

        # Track resource quotas allocated to each Captain
        self.quotas: Dict[str, ResourceQuota] = {}

        # Map captain_id -> stage_name for reliable lookup
        self._captain_to_stage: Dict[str, str] = {}
        
        # Metrics history for trend analysis
        self.metrics_history: Dict[str, deque] = {}
        self.history_window = history_window
        
        # Last allocation time
        self.last_allocation_time = time.time()
        
        # SLA violation tracking
        self.sla_violations = 0
        self.total_requests = 0
    
    def register_stage(self, stage_name: str, initial_parallelism: int = 1) -> str:
        if stage_name in self.stages:
            existing_id = self._captain_id_for_stage(stage_name)
            if existing_id is not None:
                return existing_id

        captain_id = f"captain_{stage_name}_{id(self.stages) & 0xFFFF:04x}_{len(self.stages)}"

        self.stages[stage_name] = StageMetrics(
            stage_name=stage_name,
            current_parallelism=initial_parallelism
        )

        self.metrics_history[stage_name] = deque(maxlen=self.history_window)

        initial_quota = self._compute_initial_quota(captain_id, initial_parallelism)
        self.quotas[captain_id] = initial_quota
        self._captain_to_stage[captain_id] = stage_name

        return captain_id
    
    def update_stage_metrics(self, stage_name: str, metrics: StageMetrics):
        """
        Update metrics for a stage (called by Captain)
        
        Args:
            stage_name: Name of the stage
            metrics: Latest metrics from Captain
        """
        if stage_name not in self.stages:
            raise ValueError(f"Stage {stage_name} not registered")
        
        # Update current metrics
        self.stages[stage_name] = metrics
        
        # Add to history
        self.metrics_history[stage_name].append({
            'timestamp': metrics.last_update,
            'queue_depth': metrics.queue_depth,
            'throughput': metrics.throughput,
            'latency_ms': metrics.avg_latency_ms,
            'cpu_util': metrics.cpu_utilization,
            'memory_util': metrics.memory_utilization
        })
        
        # Track SLA violations
        self.total_requests += 1
        if metrics.avg_latency_ms > self.sla_latency_ms:
            self.sla_violations += 1
    
    def allocate_resources(self) -> Dict[str, ResourceQuota]:
        current_time = time.time()

        if current_time - self.last_allocation_time < self.update_interval:
            return self.quotas

        self.last_allocation_time = current_time

        bottlenecks = self._identify_bottlenecks()

        stage_parallelism: Dict[str, int] = {}
        for captain_id in self.quotas:
            stage_name = self._get_stage_from_captain(captain_id)
            if stage_name not in self.stages:
                continue
            stage_parallelism[stage_name] = self._compute_target_parallelism(
                self.stages[stage_name],
                is_bottleneck=(stage_name in bottlenecks),
            )

        total_parallelism = max(1, sum(stage_parallelism.values()))

        for captain_id, quota in self.quotas.items():
            stage_name = self._get_stage_from_captain(captain_id)
            if stage_name not in stage_parallelism:
                continue

            target_parallelism = stage_parallelism[stage_name]
            weight = target_parallelism / total_parallelism

            quota.target_parallelism = target_parallelism
            quota.cpu_quota = self.cluster.available_cpu_cores * weight
            quota.memory_quota_mb = self.cluster.available_memory_mb * weight
            quota.gpu_quota = self.cluster.available_gpus * weight
            quota.target_throughput = self._compute_target_throughput(
                self.stages[stage_name]
            )
            quota.topology_mode = self._decide_topology(
                stage_name, self.stages[stage_name]
            )

        return self.quotas
    
    def _identify_bottlenecks(self) -> List[str]:
        """
        Identify bottleneck stages based on queue depth and latency
        
        A stage is a bottleneck if:
        1. Queue depth > target_queue_depth
        2. Latency approaching SLA limit
        3. Throughput declining over time
        
        Returns:
            List of bottleneck stage names
        """
        bottlenecks = []
        
        for stage_name, metrics in self.stages.items():
            # Check queue depth
            queue_pressure = metrics.queue_depth > self.target_queue_depth
            
            # Check latency
            latency_pressure = metrics.avg_latency_ms > (self.sla_latency_ms * 0.8)
            
            # Check throughput trend
            throughput_declining = False
            if stage_name in self.metrics_history and len(self.metrics_history[stage_name]) >= 3:
                recent = list(self.metrics_history[stage_name])[-3:]
                throughputs = [m['throughput'] for m in recent]
                if len(throughputs) >= 2:
                    throughput_declining = throughputs[-1] < throughputs[0] * 0.9
            
            if queue_pressure or latency_pressure or throughput_declining:
                bottlenecks.append(stage_name)
        
        return bottlenecks
    
    def _compute_target_parallelism(
        self, 
        metrics: StageMetrics, 
        is_bottleneck: bool
    ) -> int:
        """
        Compute target parallelism for a stage
        
        Strategy:
        - If bottleneck: Increase parallelism to drain queue
        - If underutilized: Decrease parallelism to free resources
        - Consider resource availability
        
        Args:
            metrics: Current stage metrics
            is_bottleneck: Whether this stage is a bottleneck
            
        Returns:
            Target parallelism (number of actors)
        """
        current = metrics.current_parallelism
        
        if is_bottleneck:
            # Estimate needed parallelism to drain queue
            if metrics.throughput > 0:
                # Time to process queue at current throughput
                queue_drain_time = metrics.queue_depth / metrics.throughput
                
                # If drain time > SLA, scale up
                if queue_drain_time > (self.sla_latency_ms / 1000.0):
                    scale_factor = min(2.0, queue_drain_time / (self.sla_latency_ms / 1000.0))
                    target = int(current * scale_factor)
                else:
                    target = current + 1  # Conservative increase
            else:
                target = current + 1  # No throughput data, try increasing
        else:
            # Check if we can scale down (free resources)
            if metrics.queue_depth < self.target_queue_depth * 0.5 and current > 1:
                target = max(1, current - 1)
            else:
                target = current  # Keep current level
        
        # Clamp to available resources
        max_possible = self._estimate_max_parallelism()
        target = min(target, max_possible)
        
        return max(1, target)  # At least 1 actor
    
    def _compute_target_throughput(self, metrics: StageMetrics) -> float:
        """
        Compute target throughput to meet SLA
        
        Args:
            metrics: Current stage metrics
            
        Returns:
            Target throughput in samples/sec
        """
        # To meet SLA, we need throughput >= queue_depth / (SLA_time - current_latency)
        sla_time_sec = self.sla_latency_ms / 1000.0
        current_latency_sec = metrics.avg_latency_ms / 1000.0
        
        remaining_time = max(0.1, sla_time_sec - current_latency_sec)
        
        if metrics.queue_depth > 0:
            target = metrics.queue_depth / remaining_time
        else:
            target = metrics.throughput  # Maintain current
        
        return max(1.0, target)
    
    def _decide_topology(
        self, 
        stage_name: str, 
        metrics: StageMetrics
    ) -> TopologyMode:
        """
        Decide topology mode for operator placement
        
        Based on Report Section 5.4:
        - CO_LOCATION: High transfer cost, sufficient local resources
        - DISTRIBUTED: Different resource bottlenecks, ample bandwidth
        
        Args:
            stage_name: Name of the stage
            metrics: Current metrics
            
        Returns:
            Topology mode decision
        """
        # Check resource pressure
        high_cpu = metrics.cpu_utilization > 80
        high_memory = metrics.memory_utilization > 80
        high_gpu = metrics.gpu_utilization > 80
        
        # If single resource bottleneck, distribute to specialize
        bottleneck_count = sum([high_cpu, high_memory, high_gpu])
        
        if bottleneck_count >= 2:
            # Multiple bottlenecks on same node -> distribute
            return TopologyMode.DISTRIBUTED
        elif bottleneck_count == 0:
            # No pressure -> co-locate for efficiency
            return TopologyMode.CO_LOCATION
        else:
            # Single bottleneck -> adaptive
            return TopologyMode.ADAPTIVE
    
    def _estimate_max_parallelism(self) -> int:
        """
        Estimate maximum parallelism given available resources
        
        Returns:
            Maximum number of actors cluster can support
        """
        # Conservative estimate: assume each actor needs 1 CPU + 1GB memory
        cpu_limit = int(self.cluster.available_cpu_cores)
        memory_limit = int(self.cluster.available_memory_mb / 1024)  # 1GB per actor
        
        return max(1, min(cpu_limit, memory_limit))
    
    def _get_stage_from_captain(self, captain_id: str) -> str:
        return self._captain_to_stage.get(captain_id, captain_id)
    
    def _compute_initial_quota(
        self,
        captain_id: str,
        parallelism: int
    ) -> ResourceQuota:
        total_stages = max(1, len(self.stages))

        return ResourceQuota(
            captain_id=captain_id,
            target_parallelism=parallelism,
            cpu_quota=self.cluster.available_cpu_cores / total_stages,
            memory_quota_mb=self.cluster.available_memory_mb / total_stages,
            gpu_quota=self.cluster.available_gpus / total_stages,
            target_throughput=10.0,
            topology_mode=TopologyMode.ADAPTIVE
        )

    def _captain_id_for_stage(self, stage_name: str) -> Optional[str]:
        prefix = f"captain_{stage_name}_"
        for captain_id in self.quotas:
            if captain_id.startswith(prefix):
                return captain_id
        return None
    
    def get_sla_compliance_rate(self) -> float:
        """
        Calculate SLA compliance rate
        
        Returns:
            Percentage of requests meeting SLA (0-100)
        """
        if self.total_requests == 0:
            return 100.0
        
        return ((self.total_requests - self.sla_violations) / self.total_requests) * 100.0
    
    def get_global_stats(self) -> Dict:
        """Get global system statistics"""
        return {
            'total_stages': len(self.stages),
            'total_parallelism': sum(q.target_parallelism for q in self.quotas.values()),
            'sla_compliance_rate': self.get_sla_compliance_rate(),
            'total_requests': self.total_requests,
            'sla_violations': self.sla_violations,
            'cluster_cpu_util': (
                (self.cluster.total_cpu_cores - self.cluster.available_cpu_cores) /
                self.cluster.total_cpu_cores * 100
            ) if self.cluster.total_cpu_cores > 0 else 0,
            'cluster_memory_util': (
                (self.cluster.total_memory_mb - self.cluster.available_memory_mb) /
                self.cluster.total_memory_mb * 100
            ) if self.cluster.total_memory_mb > 0 else 0
        }
