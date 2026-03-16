"""
Xenna Executor - Streaming executor for Data-Juicer using Cosmos-Xenna

This module provides the main XennaExecutor class that integrates
Cosmos-Xenna's streaming capabilities into Data-Juicer's executor framework.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.data import NestedDataset
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.executor.base import ExecutorBase
from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin
from data_juicer.core.exporter import Exporter
from data_juicer.ops import load_ops

from .adapter import create_stage_from_op
from .config import XennaConfig
from .utils import MemorySafetyController, ResourceMapper

# Optional: Cosmos-Xenna import
try:
    from cosmos_xenna.pipelines.v1 import (
        StageSpec,
        PipelineSpec,
        PipelineConfig,
        ExecutionMode,
        StreamingSpecificSpec,
        run_pipeline,
    )
    XENNA_AVAILABLE = True
except ImportError:
    XENNA_AVAILABLE = False
    StageSpec = None
    PipelineSpec = None
    PipelineConfig = None
    ExecutionMode = None
    StreamingSpecificSpec = None
    run_pipeline = None
    logger.warning(
        "Cosmos-Xenna not available. Install it from: "
        "https://github.com/NVIDIA/cosmos-xenna to use XennaExecutor."
    )


class XennaExecutor(ExecutorBase, DAGExecutionMixin, EventLoggingMixin):
    """
    Streaming executor using Cosmos-Xenna for Data-Juicer.

    This executor provides:
    - True streaming processing with push-based data flow
    - Dynamic resource autoscaling (Rust-based algorithm)
    - Memory safety with backpressure and slot control
    - GPU isolation and fractional GPU allocation
    - Multi-stage pipelining with concurrent execution

    Example:
        >>> from data_juicer.core.executor import ExecutorFactory
        >>>
        >>> # Via factory
        >>> executor = ExecutorFactory.create_executor('xenna')
        >>> executor.run()
        >>>
        >>> # Direct instantiation
        >>> from jsonargparse import Namespace
        >>> config = Namespace(
        ...     dataset_path='input.jsonl',
        ...     process=[{'whitespace_filter': {'min_ratio': 0.5}}],
        ...     export_path='output.jsonl',
        ... )
        >>> executor = XennaExecutor(config)
        >>> result = executor.run()
    """

    def __init__(self, cfg: Optional[Namespace] = None):
        """
        Initialize XennaExecutor.

        Args:
            cfg: Data-Juicer configuration (jsonargparse Namespace)
        """
        if not XENNA_AVAILABLE:
            raise ImportError(
                "Cosmos-Xenna is not installed. Install it from: "
                "https://github.com/NVIDIA/cosmos-xenna"
            )

        super().__init__(cfg)
        self.executor_type = "xenna"

        # Initialize mixins
        EventLoggingMixin.__init__(self, self.cfg)
        DAGExecutionMixin.__init__(self)

        # Working directory
        self.work_dir = getattr(self.cfg, "work_dir", "./work_dir")

        # Number of processes
        self.np = getattr(self.cfg, "np", None) or 1

        # Extract Xenna-specific configuration
        self.xenna_config = self._parse_xenna_config()

        # Initialize memory safety controller
        self.memory_controller = MemorySafetyController(
            max_queued_multiplier=self.xenna_config.streaming.max_queued_multiplier,
            max_queued_lower_bound=self.xenna_config.streaming.max_queued_lower_bound,
        )

        # Initialize resource mapper
        self.resource_mapper = ResourceMapper()

        # Dataset builder
        self.dataset_builder = DatasetBuilder(self.cfg, executor_type=self.executor_type)

        # Exporter setup
        self._setup_exporter()

        # Checkpoint manager (optional)
        self.ckpt_manager = None
        if getattr(self.cfg, "use_checkpoint", False):
            self._setup_checkpoint_manager()

        logger.info(f"XennaExecutor initialized with mode: {self.xenna_config.execution_mode}")

    def _parse_xenna_config(self) -> XennaConfig:
        """Parse Xenna-specific configuration from Data-Juicer config."""
        xenna_cfg = getattr(self.cfg, "xenna", None)
        if xenna_cfg is None:
            return XennaConfig()
        return XennaConfig.from_dict(dict(xenna_cfg) if hasattr(xenna_cfg, "items") else xenna_cfg)

    def _setup_exporter(self):
        """Setup the exporter for output data."""
        export_extra_args = (
            dict(self.cfg.export_extra_args)
            if hasattr(self.cfg, "export_extra_args")
            else {}
        )
        self.exporter = Exporter(
            self.cfg.export_path,
            self.cfg.export_type,
            self.cfg.export_shard_size,
            self.cfg.export_in_parallel,
            self.np,
            keep_stats_in_res_ds=getattr(self.cfg, "keep_stats_in_res_ds", False),
            keep_hashes_in_res_ds=getattr(self.cfg, "keep_hashes_in_res_ds", False),
            **export_extra_args,
        )

    def _setup_checkpoint_manager(self):
        """Setup checkpoint manager for fault tolerance."""
        from data_juicer.utils.ckpt_utils import CheckpointManager
        self.ckpt_dir = os.path.join(self.work_dir, "ckpt")
        self.ckpt_manager = CheckpointManager(
            self.ckpt_dir, self.cfg.process, self.np
        )
        if self.ckpt_manager.ckpt_available:
            logger.info("Found existing checkpoint, resuming from it.")
            self.cfg.process = self.ckpt_manager.get_left_process_list()

    def run(
        self,
        dataset: Optional[Any] = None,
        load_data_np: Optional[PositiveInt] = None,
        skip_export: bool = False,
        skip_return: bool = False,
    ) -> Optional[Any]:
        """
        Run the streaming data processing pipeline.

        Args:
            dataset: Optional pre-loaded dataset
            load_data_np: Number of workers for loading data
            skip_export: Whether to skip exporting results
            skip_return: Whether to skip returning the result

        Returns:
            Processed dataset (unless skip_return=True)
        """
        start_time = time.time()

        # 1. Load or use provided dataset
        dataset = self._load_dataset(dataset, load_data_np)
        logger.info(f"Dataset loaded with {len(dataset)} samples")

        # 2. Load operations
        ops = load_ops(self.cfg.process)
        logger.info(f"Loaded {len(ops)} operations")

        # 3. Initialize DAG execution planning
        self._initialize_dag_execution(self.cfg, ops=ops)

        # 4. Log job start
        job_config = self._build_job_config()
        self.log_job_start(job_config, len(ops))

        # 5. Convert Data-Juicer operators to Xenna stages
        stages = self._create_xenna_stages(ops)
        logger.info(f"Created {len(stages)} Xenna stages")

        # 6. Create Xenna pipeline specification
        pipeline_spec = self._create_pipeline_spec(dataset, stages)

        # 7. Run the streaming pipeline
        logger.info("Starting Xenna streaming pipeline execution...")
        result = self._run_streaming_pipeline(pipeline_spec, ops)

        # 8. Export results
        if not skip_export and result is not None:
            logger.info("Exporting processed dataset...")
            self.exporter.export(result)

        # 9. Log completion
        duration = time.time() - start_time
        self.log_job_complete(duration, self.cfg.export_path)
        logger.info(f"Pipeline completed in {duration:.2f} seconds")

        if not skip_return:
            return result

    def _load_dataset(self, dataset, load_data_np):
        """Load dataset from various sources."""
        if dataset is not None:
            return dataset

        if self.ckpt_manager and self.ckpt_manager.ckpt_available:
            logger.info("Loading dataset from checkpoint...")
            return self.ckpt_manager.load_ckpt()

        logger.info("Loading dataset from dataset builder...")
        if load_data_np is None:
            load_data_np = self.np
        load_kwargs = {"num_proc": load_data_np}
        if getattr(self.cfg, "load_dataset_kwargs", None):
            load_kwargs.update(dict(self.cfg.load_dataset_kwargs))
        return self.dataset_builder.load_dataset(**load_kwargs)

    def _build_job_config(self) -> Dict[str, Any]:
        """Build job configuration for logging."""
        config = {
            "work_dir": self.work_dir,
            "executor_type": self.executor_type,
            "execution_mode": self.xenna_config.execution_mode.name,
        }
        if hasattr(self.cfg, "dataset_path"):
            config["dataset_path"] = self.cfg.dataset_path
        if hasattr(self.cfg, "dataset"):
            config["dataset"] = str(self.cfg.dataset)
        return config

    def _create_xenna_stages(self, ops: List[Any]) -> List[Any]:
        """
        Convert Data-Juicer operators to Xenna StageSpecs.

        This is the core adaptation layer between DJ operators and Xenna stages.
        """
        stages = []
        for i, op in enumerate(ops):
            # Create stage adapter for each operator
            stage = create_stage_from_op(
                op,
                op_index=i,
                xenna_config=self.xenna_config,
                memory_controller=self.memory_controller,
            )

            # Create StageSpec with optional overrides
            stage_spec = StageSpec(
                stage=stage,
                num_workers=getattr(op, "num_workers", None),
                num_workers_per_node=getattr(op, "num_workers_per_node", None),
                slots_per_actor=self.xenna_config.slots_per_actor,
                over_provision_factor=self.xenna_config.over_provision_factor,
            )
            stages.append(stage_spec)

        return stages

    def _create_pipeline_spec(
        self,
        dataset: Any,
        stages: List[Any]
    ) -> Any:
        """Create Xenna PipelineSpec from dataset and stages."""
        # Convert dataset to list for Xenna input
        input_data = self._convert_dataset_to_list(dataset)

        # Create pipeline configuration
        pipeline_config = self._create_pipeline_config()

        return PipelineSpec(
            input_data=input_data,
            stages=stages,
            config=pipeline_config,
        )

    def _create_pipeline_config(self) -> Any:
        """Create Xenna PipelineConfig from XennaConfig."""
        # Map execution mode
        mode_map = {
            "STREAMING": ExecutionMode.STREAMING,
            "BATCH": ExecutionMode.BATCH,
            "SERVING": ExecutionMode.SERVING,
        }
        execution_mode = mode_map.get(
            self.xenna_config.execution_mode.name,
            ExecutionMode.STREAMING
        )

        # Create streaming-specific config if needed
        mode_specific = None
        if execution_mode == ExecutionMode.STREAMING:
            mode_specific = StreamingSpecificSpec(
                autoscale_interval_s=self.xenna_config.streaming.autoscale_interval_s,
                autoscale_speed_estimation_window_duration_s=(
                    self.xenna_config.streaming.speed_estimation_window_s
                ),
                autoscale_speed_estimation_min_data_points=(
                    self.xenna_config.streaming.min_data_points
                ),
                max_queued_multiplier=self.xenna_config.streaming.max_queued_multiplier,
                max_queued_lower_bound=self.xenna_config.streaming.max_queued_lower_bound,
            )

        return PipelineConfig(
            execution_mode=execution_mode,
            num_setup_attempts_python=self.xenna_config.fault_tolerance.num_setup_attempts,
            num_run_attempts_python=self.xenna_config.fault_tolerance.num_run_attempts,
            ignore_failures=self.xenna_config.fault_tolerance.ignore_failures,
            reset_workers_on_failure=self.xenna_config.fault_tolerance.reset_workers_on_failure,
            slots_per_actor=self.xenna_config.slots_per_actor,
            worker_max_lifetime_m=self.xenna_config.fault_tolerance.worker_max_lifetime_m,
            logging_interval_s=self.xenna_config.logging_interval_s,
            return_last_stage_outputs=True,
            mode_specific=mode_specific,
            cpu_allocation_percentage=self.xenna_config.resource.cpu_allocation_percentage,
        )

    def _convert_dataset_to_list(self, dataset: Any) -> List[Dict[str, Any]]:
        """Convert Data-Juicer dataset to list of samples for Xenna."""
        if isinstance(dataset, list):
            return dataset

        # Handle NestedDataset (HuggingFace Dataset wrapper)
        if hasattr(dataset, "to_list"):
            return dataset.to_list()

        # Handle HuggingFace Dataset
        if hasattr(dataset, "__iter__"):
            return [dict(sample) for sample in dataset]

        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    def _run_streaming_pipeline(
        self,
        pipeline_spec: Any,
        ops: List[Any]
    ) -> Optional[Any]:
        """
        Execute the Xenna streaming pipeline.

        This is where the actual streaming execution happens with:
        - Backpressure control
        - Dynamic autoscaling
        - Memory safety
        """
        try:
            # Execute pipeline using Xenna's run_pipeline
            results = run_pipeline(pipeline_spec)

            if results is None:
                logger.warning("Pipeline returned no results")
                return None

            # Convert results back to Data-Juicer format
            return self._convert_results_to_dataset(results, ops)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            if not self.xenna_config.fault_tolerance.ignore_failures:
                raise
            return None

    def _convert_results_to_dataset(
        self,
        results: List[Any],
        ops: List[Any]
    ) -> Any:
        """Convert Xenna results back to Data-Juicer dataset format."""
        # Filter out None results (filtered samples)
        filtered_results = [r for r in results if r is not None]

        if not filtered_results:
            logger.warning("All samples were filtered out")
            return NestedDataset.from_dict({})

        # Create NestedDataset from results
        return NestedDataset.from_list(filtered_results)

    def sample_data(
        self,
        dataset_to_sample=None,
        load_data_np=None,
        sample_ratio: float = 1.0,
        sample_algo: str = "uniform",
        **kwargs
    ):
        """
        Sample a subset from the given dataset.

        Args:
            dataset_to_sample: Dataset to sample from
            load_data_np: Number of workers for loading
            sample_ratio: Ratio of samples to keep
            sample_algo: Sampling algorithm

        Returns:
            Sampled dataset
        """
        from data_juicer.utils.sample import random_sample

        dataset = self._load_dataset(dataset_to_sample, load_data_np)

        if sample_algo == "uniform":
            return random_sample(dataset, sample_ratio)
        else:
            raise ValueError(f"Unsupported sample_algo: {sample_algo}")
