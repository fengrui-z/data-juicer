"""
ElasticJuicer: Adaptive Resource Scheduling for Data-Juicer

A system that provides dynamic resource management and OOM prevention for
multimodal data processing pipelines.
"""

__version__ = "0.1.0"

from .contracts import (
    BatchDecision,
    BatchObservation,
    ClusterState,
    MemoryState,
    ResourceQuota,
    StageExecutionObservation,
    StageMetrics,
    TopologyMode,
)

__all__ = [
    "BatchDecision",
    "BatchObservation",
    "ClusterState",
    "MemoryState",
    "ResourceQuota",
    "StageExecutionObservation",
    "StageMetrics",
    "TopologyMode",
]
