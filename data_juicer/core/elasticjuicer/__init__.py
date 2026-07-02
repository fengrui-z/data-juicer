"""
ElasticJuicer: Adaptive Resource Scheduling for Data-Juicer

A system that provides dynamic resource management and OOM prevention for
multimodal data processing pipelines.
"""

__version__ = "0.1.0"

from .contracts import (
    AllocationPlan,
    BatchRecommendation,
    BatchDecision,
    BatchObservation,
    ClusterState,
    MemoryState,
    ResourceQuota,
    ResourceQuotaSnapshot,
    StageExecutionObservation,
    StageMetrics,
    TopologyMode,
    TowerMode,
)

from .mode import ElasticJuicerMode, resolve_mode
from .elastic_juicer import ElasticJuicer, detect_cluster_state

__all__ = [
    "AllocationPlan",
    "BatchRecommendation",
    "BatchDecision",
    "BatchObservation",
    "ClusterState",
    "MemoryState",
    "ResourceQuota",
    "ResourceQuotaSnapshot",
    "StageExecutionObservation",
    "StageMetrics",
    "TopologyMode",
    "TowerMode",
    "ElasticJuicerMode",
    "resolve_mode",
    "ElasticJuicer",
    "detect_cluster_state",
]
