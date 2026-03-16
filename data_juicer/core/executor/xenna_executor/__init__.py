"""
Data-Juicer Xenna Executor Integration

This package provides streaming processing, dynamic resource scheduling,
and memory safety control for Data-Juicer by integrating Cosmos-Xenna.

Key Features:
- True streaming processing with backpressure control
- Dynamic resource autoscaling (Rust-based)
- Memory safety with slot-based task management
- GPU isolation and fractional GPU allocation
- Multi-stage pipelining with concurrent execution
"""

from .executor import XennaExecutor
from .adapter import (
    OpToStageAdapter,
    FilterStageAdapter,
    MapperStageAdapter,
    DeduplicatorStageAdapter,
    create_stage_from_op,
)
from .config import XennaConfig, StreamingConfig, ResourceConfig
from .utils import ResourceMapper, MemorySafetyController, DataConverter

__version__ = "0.1.0"
__all__ = [
    "XennaExecutor",
    "OpToStageAdapter",
    "FilterStageAdapter",
    "MapperStageAdapter",
    "DeduplicatorStageAdapter",
    "create_stage_from_op",
    "XennaConfig",
    "StreamingConfig",
    "ResourceConfig",
    "ResourceMapper",
    "MemorySafetyController",
    "DataConverter",
]
