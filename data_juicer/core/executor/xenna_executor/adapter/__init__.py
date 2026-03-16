"""
Operator to Stage Adapters

This module provides adapters that convert Data-Juicer operators
to Cosmos-Xenna Stage interfaces, enabling streaming execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic

from loguru import logger

try:
    from cosmos_xenna.pipelines.v1 import (
        Stage,
        Resources,
        NodeInfo,
        WorkerMetadata,
    )
    XENNA_AVAILABLE = True
except ImportError:
    XENNA_AVAILABLE = False
    # Placeholder for development
    class Stage:
        pass
    class Resources:
        def __init__(self, cpus=0.0, gpus=0, is_spmd=False):
            self.cpus = cpus
            self.gpus = gpus
            self.is_spmd = is_spmd
    class NodeInfo:
        pass
    class WorkerMetadata:
        pass

T = TypeVar('T')
V = TypeVar('V')


class OpToStageAdapter(Stage, Generic[T, V], ABC):
    """
    Base adapter that wraps Data-Juicer operators as Xenna Stages.

    This adapter bridges the gap between Data-Juicer's pull-based
    batch processing and Xenna's push-based streaming execution.

    Key responsibilities:
    - Convert DJ operator semantics to Stage interface
    - Handle batch size adaptation
    - Map resource requirements
    - Manage state for stateful operators
    """

    def __init__(
        self,
        dj_op: Any,
        op_index: int = 0,
        xenna_config: Optional[Any] = None,
        memory_controller: Optional[Any] = None,
    ):
        """
        Initialize the adapter.

        Args:
            dj_op: Data-Juicer operator instance
            op_index: Index of this operator in the pipeline
            xenna_config: Xenna configuration
            memory_controller: Memory safety controller
        """
        self.dj_op = dj_op
        self.op_index = op_index
        self.xenna_config = xenna_config
        self.memory_controller = memory_controller

        # Extract batch size from operator
        self._batch_size = getattr(dj_op, "batch_size", 1000)

        # Track operator name for logging
        self._op_name = getattr(dj_op, "_name", type(dj_op).__name__)

        # State management
        self._is_setup = False
        self._node_info_cache: Dict[str, Any] = {}

    @property
    def stage_batch_size(self) -> int:
        """Return the batch size for Xenna streaming."""
        return self._batch_size

    @property
    def required_resources(self) -> Resources:
        """
        Map Data-Juicer resource requirements to Xenna Resources.

        Handles:
        - CPU-only operators
        - GPU operators (fractional and whole)
        - Distributed operators (SPMD)
        """
        # Get resource requirements from DJ operator
        num_cpus = getattr(self.dj_op, "num_cpus", None)
        num_gpus = getattr(self.dj_op, "num_gpus", None)
        is_cuda = getattr(self.dj_op, "accelerator", "cpu") == "cuda"

        # Default values
        if num_cpus is None:
            num_cpus = 1.0
        if num_gpus is None:
            num_gpus = 1.0 if is_cuda else 0.0

        # Check for SPMD (distributed inference)
        is_spmd = getattr(self.dj_op, "is_distributed", False)

        return Resources(
            cpus=float(num_cpus),
            gpus=float(num_gpus),
            is_spmd=is_spmd,
        )

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:
        """
        Node-level setup - runs once per node.

        This is where shared resources should be initialized:
        - Model loading (once per node, shared across workers)
        - S3 clients
        - Shared caches
        """
        node_id = getattr(node_info, "node_id", "unknown")
        logger.info(f"[{self._op_name}] Setting up on node: {node_id}")

        # Cache node info
        self._node_info_cache[node_id] = {
            "node_info": node_info,
            "worker_metadata": worker_metadata,
        }

        # Call operator's node setup if available
        if hasattr(self.dj_op, "setup_on_node"):
            self.dj_op.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: WorkerMetadata) -> None:
        """
        Worker-level setup - runs once per worker.

        This is for worker-specific initialization:
        - GPU selection
        - Worker-local caches
        - Model loading (if not shared at node level)
        """
        worker_id = getattr(worker_metadata, "worker_id", "unknown")
        logger.debug(f"[{self._op_name}] Setting up worker: {worker_id}")

        # Call operator's setup if available
        if hasattr(self.dj_op, "setup"):
            self.dj_op.setup()

        self._is_setup = True

    @abstractmethod
    def process_data(self, in_data: List[T]) -> Optional[List[V]]:
        """
        Process data - must be implemented by subclasses.

        Args:
            in_data: List of input samples

        Returns:
            List of processed samples, or None to filter all
        """
        pass

    def _list_to_batch(self, samples: List[Dict]) -> Dict[str, List]:
        """Convert list of dicts to dict of lists (batch format)."""
        if not samples:
            return {}
        keys = samples[0].keys()
        return {key: [s.get(key) for s in samples] for key in keys}

    def _batch_to_list(self, batch: Dict[str, List]) -> List[Dict]:
        """Convert dict of lists to list of dicts."""
        if not batch:
            return []
        keys = list(batch.keys())
        if not keys:
            return []
        num_samples = len(batch[keys[0]])
        return [{key: batch[key][i] for key in keys} for i in range(num_samples)]


class MapperStageAdapter(OpToStageAdapter):
    """
    Adapter for Data-Juicer Mapper operators.

    Mappers transform data samples without filtering.
    Supports both single-sample and batched processing.
    """

    def process_data(self, in_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Process samples using mapper semantics.

        Mappers always return the same number of samples
        (possibly with modified content).
        """
        if not in_data:
            return []

        try:
            # Check if operator is batched
            is_batched = self.dj_op.is_batched_op() if hasattr(self.dj_op, "is_batched_op") else False

            if is_batched:
                # Batch processing - convert list of dicts to dict of lists
                batch = self._list_to_batch(in_data)
                result = self.dj_op.process(batch)
                # Convert back to list of dicts
                return self._batch_to_list(result)
            else:
                # Single-sample processing
                results = []
                for sample in in_data:
                    processed = self.dj_op.process(sample)
                    if processed is not None:
                        results.append(processed)
                return results if results else None

        except Exception as e:
            logger.error(f"[{self._op_name}] Error processing data: {e}")
            if self.xenna_config and not getattr(self.xenna_config, "ignore_failures", False):
                raise
            return None


class FilterStageAdapter(OpToStageAdapter):
    """
    Adapter for Data-Juicer Filter operators.

    Filters remove samples that don't meet criteria.
    Implements two-phase processing: compute_stats -> filter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stats_cache: Dict[int, Dict] = {}

    def process_data(self, in_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Process samples using filter semantics.

        Filters return a subset of input samples.
        None return means all samples filtered.
        """
        if not in_data:
            return []

        try:
            results = []
            is_batched = self.dj_op.is_batched_op() if hasattr(self.dj_op, "is_batched_op") else False

            if is_batched:
                # Batch processing for filters
                batch = self._list_to_batch(in_data)
                # First compute stats
                stats_batch = self.dj_op.compute_stats(batch)
                # Then filter
                keep_mask = list(self.dj_op.process(stats_batch))
                for i, (sample, keep) in enumerate(zip(in_data, keep_mask)):
                    if keep:
                        results.append(sample)
            else:
                # Single-sample processing
                for sample in in_data:
                    sample_with_stats = self.dj_op.compute_stats(sample)
                    should_keep = self.dj_op.process(sample_with_stats)
                    if should_keep:
                        results.append(sample)

            return results if results else None

        except Exception as e:
            logger.error(f"[{self._op_name}] Error filtering data: {e}")
            if self.xenna_config and not getattr(self.xenna_config, "ignore_failures", False):
                raise
            return None


class DeduplicatorStageAdapter(OpToStageAdapter):
    """
    Adapter for Data-Juicer Deduplicator operators.

    Deduplicators require global state - this adapter manages
    the state across streaming batches and performs convergence
    at appropriate points.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hash_state: Dict[Any, int] = {}  # hash -> sample_index
        self._pending_samples: List[Dict] = []
        self._processed_count = 0

    def process_data(self, in_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Process samples using deduplication semantics.

        Deduplication in streaming mode:
        1. Compute hashes for samples
        2. Track seen hashes
        3. Filter duplicates (keep first occurrence)
        """
        if not in_data:
            return []

        try:
            results = []

            for sample in in_data:
                # Compute hash for this sample
                sample_with_hash = self.dj_op.compute_hash(sample)
                hash_value = sample_with_hash.get("hash")

                if hash_value is not None:
                    if hash_value not in self._hash_state:
                        # First occurrence - keep it
                        self._hash_state[hash_value] = self._processed_count
                        results.append(sample)
                    # else: duplicate, skip
                else:
                    # No hash, pass through
                    results.append(sample)

                self._processed_count += 1

            return results if results else None

        except Exception as e:
            logger.error(f"[{self._op_name}] Error deduplicating data: {e}")
            if self.xenna_config and not getattr(self.xenna_config, "ignore_failures", False):
                raise
            return None


class SelectorStageAdapter(OpToStageAdapter):
    """Adapter for Data-Juicer Selector operators."""

    def process_data(self, in_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Process samples using selector semantics.

        Selectors choose specific samples from the dataset.
        In streaming mode, this operates on the accumulated batch.
        """
        if not in_data:
            return []

        try:
            batch = self._list_to_batch(in_data)
            result = self.dj_op.process(batch)
            if hasattr(result, "to_list"):
                return result.to_list()
            return self._batch_to_list(result) if isinstance(result, dict) else list(result)

        except Exception as e:
            logger.error(f"[{self._op_name}] Error selecting data: {e}")
            if self.xenna_config and not getattr(self.xenna_config, "ignore_failures", False):
                raise
            return None


class GrouperStageAdapter(OpToStageAdapter):
    """Adapter for Data-Juicer Grouper operators."""

    def process_data(self, in_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Group samples according to grouper logic."""
        if not in_data:
            return []

        try:
            batch = self._list_to_batch(in_data)
            grouped = self.dj_op.process(batch)
            return grouped if grouped else None

        except Exception as e:
            logger.error(f"[{self._op_name}] Error grouping data: {e}")
            if self.xenna_config and not getattr(self.xenna_config, "ignore_failures", False):
                raise
            return None


class AggregatorStageAdapter(OpToStageAdapter):
    """Adapter for Data-Juicer Aggregator operators."""

    def process_data(self, in_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Aggregate samples according to aggregator logic."""
        if not in_data:
            return []

        try:
            batch = self._list_to_batch(in_data)
            aggregated = self.dj_op.process(batch)
            if aggregated is not None:
                return [aggregated]
            return None

        except Exception as e:
            logger.error(f"[{self._op_name}] Error aggregating data: {e}")
            if self.xenna_config and not getattr(self.xenna_config, "ignore_failures", False):
                raise
            return None


def create_stage_from_op(
    dj_op: Any,
    op_index: int = 0,
    xenna_config: Optional[Any] = None,
    memory_controller: Optional[Any] = None,
) -> Stage:
    """
    Factory function to create appropriate Stage adapter for a Data-Juicer operator.

    This function inspects the operator type and returns the matching adapter.

    Args:
        dj_op: Data-Juicer operator instance
        op_index: Index in the pipeline
        xenna_config: Xenna configuration
        memory_controller: Memory safety controller

    Returns:
        Appropriate Stage adapter instance
    """
    # Import operator base classes
    try:
        from data_juicer.ops import Mapper, Filter, Deduplicator, Selector, Grouper, Aggregator

        # Map operator types to adapters
        if isinstance(dj_op, Mapper):
            adapter_cls = MapperStageAdapter
        elif isinstance(dj_op, Filter):
            adapter_cls = FilterStageAdapter
        elif isinstance(dj_op, Deduplicator):
            adapter_cls = DeduplicatorStageAdapter
        elif isinstance(dj_op, Selector):
            adapter_cls = SelectorStageAdapter
        elif isinstance(dj_op, Grouper):
            adapter_cls = GrouperStageAdapter
        elif isinstance(dj_op, Aggregator):
            adapter_cls = AggregatorStageAdapter
        else:
            logger.warning(
                f"Unknown operator type {type(dj_op).__name__}, "
                "using MapperStageAdapter"
            )
            adapter_cls = MapperStageAdapter

    except ImportError:
        # Fallback: use class name heuristics
        op_type = type(dj_op).__name__.lower()

        if "filter" in op_type:
            adapter_cls = FilterStageAdapter
        elif "deduplicator" in op_type:
            adapter_cls = DeduplicatorStageAdapter
        elif "selector" in op_type:
            adapter_cls = SelectorStageAdapter
        elif "grouper" in op_type:
            adapter_cls = GrouperStageAdapter
        elif "aggregator" in op_type:
            adapter_cls = AggregatorStageAdapter
        else:
            adapter_cls = MapperStageAdapter

    return adapter_cls(
        dj_op=dj_op,
        op_index=op_index,
        xenna_config=xenna_config,
        memory_controller=memory_controller,
    )
