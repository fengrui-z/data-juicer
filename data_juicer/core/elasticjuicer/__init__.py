"""
ElasticJuicer: Adaptive Resource Scheduling for Data-Juicer

A system that provides dynamic resource management and OOM prevention for
multimodal data processing pipelines.
"""

__version__ = "0.1.0"

from .contracts import (
    BatchRecommendation,
    BatchDecision,
    BatchObservation,
    ClusterState,
    MemoryState,
    ResourceQuota,
    StageExecutionObservation,
    StageMetrics,
    TopologyMode,
)

from .mode import ElasticJuicerMode, resolve_mode

__all__ = [
    "BatchRecommendation",
    "BatchDecision",
    "BatchObservation",
    "ClusterState",
    "MemoryState",
    "ResourceQuota",
    "StageExecutionObservation",
    "StageMetrics",
    "TopologyMode",
    "ElasticJuicerMode",
    "resolve_mode",
]
