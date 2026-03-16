"""
Resource Mapper

Maps Data-Juicer resource specifications to Cosmos-Xenna resource format.
Handles CPU, GPU, memory, and distributed execution resource mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from loguru import logger

try:
    from cosmos_xenna.pipelines.v1 import Resources
    XENNA_AVAILABLE = True
except ImportError:
    XENNA_AVAILABLE = False

    @dataclass
    class Resources:
        cpus: float = 0.0
        gpus: float = 0.0
        is_spmd: bool = False


@dataclass
class ResourceSpec:
    """Internal resource specification."""
    cpus: float
    gpus: float
    memory_gb: Optional[float]
    is_spmd: bool
    num_workers: Optional[int] = None


class ResourceMapper:
    """
    Maps Data-Juicer operator resource requirements to Xenna Resources.

    This class handles:
    - CPU/GPU allocation mapping
    - Memory requirement translation
    - Distributed execution (SPMD) configuration
    - Fractional GPU allocation
    """

    # Default resource values
    DEFAULT_CPUS = 1.0
    DEFAULT_GPUS = 0.0
    DEFAULT_MEMORY_GB = 4.0

    def __init__(
        self,
        default_cpus: float = DEFAULT_CPUS,
        default_gpus: float = DEFAULT_GPUS,
        enable_fractional_gpu: bool = True,
    ):
        """
        Initialize resource mapper.

        Args:
            default_cpus: Default CPU cores per worker
            default_gpus: Default GPUs per worker
            enable_fractional_gpu: Allow fractional GPU allocation
        """
        self.default_cpus = default_cpus
        self.default_gpus = default_gpus
        self.enable_fractional_gpu = enable_fractional_gpu

        # Cache for mapped resources
        self._cache: Dict[str, ResourceSpec] = {}

    def map_operator_to_resources(
        self,
        op: Any,
        op_name: Optional[str] = None,
    ) -> Resources:
        """
        Map a Data-Juicer operator to Xenna Resources.

        Args:
            op: Data-Juicer operator instance
            op_name: Optional operator name for caching

        Returns:
            Xenna Resources object
        """
        # Check cache
        if op_name and op_name in self._cache:
            cached = self._cache[op_name]
            return Resources(
                cpus=cached.cpus,
                gpus=cached.gpus,
                is_spmd=cached.is_spmd,
            )

        # Extract resource requirements from operator
        spec = self._extract_resource_spec(op)

        # Cache the result
        if op_name:
            self._cache[op_name] = spec

        logger.debug(
            f"Mapped {op_name or type(op).__name__} to "
            f"Resources(cpus={spec.cpus}, gpus={spec.gpus}, is_spmd={spec.is_spmd})"
        )

        return Resources(
            cpus=spec.cpus,
            gpus=spec.gpus,
            is_spmd=spec.is_spmd,
        )

    def _extract_resource_spec(self, op: Any) -> ResourceSpec:
        """Extract resource specification from operator."""
        # Get CPU requirements
        cpus = self._get_cpu_requirement(op)

        # Get GPU requirements
        gpus = self._get_gpu_requirement(op)

        # Get memory requirements
        memory_gb = self._get_memory_requirement(op)

        # Check for SPMD/distributed execution
        is_spmd = self._check_spmd(op)

        # Get number of workers if specified
        num_workers = getattr(op, "num_workers", None)

        return ResourceSpec(
            cpus=cpus,
            gpus=gpus,
            memory_gb=memory_gb,
            is_spmd=is_spmd,
            num_workers=num_workers,
        )

    def _get_cpu_requirement(self, op: Any) -> float:
        """Extract CPU requirement from operator."""
        # Try various attribute names
        for attr in ["num_cpus", "cpu_required", "cpus"]:
            value = getattr(op, attr, None)
            if value is not None:
                return float(value)

        # Check if CUDA operator (needs at least 1 CPU alongside GPU)
        if getattr(op, "accelerator", "cpu") == "cuda":
            return 1.0

        return self.default_cpus

    def _get_gpu_requirement(self, op: Any) -> float:
        """Extract GPU requirement from operator."""
        # Try various attribute names
        for attr in ["num_gpus", "gpu_required", "gpus"]:
            value = getattr(op, attr, None)
            if value is not None:
                gpus = float(value)
                # Validate fractional GPU
                if not self.enable_fractional_gpu and 0 < gpus < 1:
                    logger.warning(
                        f"Fractional GPU allocation disabled, "
                        f"rounding {gpus} to 1"
                    )
                    gpus = 1.0
                return gpus

        # Check accelerator type
        if getattr(op, "accelerator", "cpu") == "cuda":
            return 1.0

        return self.default_gpus

    def _get_memory_requirement(self, op: Any) -> Optional[float]:
        """Extract memory requirement from operator."""
        for attr in ["memory", "mem_required"]:
            value = getattr(op, attr, None)
            if value is not None:
                # Handle string format (e.g., "4GB")
                if isinstance(value, str):
                    return self._parse_memory_string(value)
                return float(value)
        return None

    def _parse_memory_string(self, mem_str: str) -> float:
        """Parse memory string to GB."""
        mem_str = mem_str.upper().strip()

        multipliers = {
            "TB": 1024,
            "GB": 1,
            "MB": 1 / 1024,
            "KB": 1 / (1024 * 1024),
            "B": 1 / (1024 * 1024 * 1024),
        }

        for suffix, mult in multipliers.items():
            if mem_str.endswith(suffix):
                try:
                    value = float(mem_str[:-len(suffix)].strip())
                    return value * mult
                except ValueError:
                    pass

        # Try parsing as raw number
        try:
            return float(mem_str)
        except ValueError:
            logger.warning(f"Could not parse memory string: {mem_str}")
            return self.DEFAULT_MEMORY_GB

    def _check_spmd(self, op: Any) -> bool:
        """Check if operator requires SPMD execution."""
        # Check for explicit SPMD flag
        if getattr(op, "is_spmd", False):
            return True

        # Check for distributed execution
        if getattr(op, "is_distributed", False):
            return True

        # Check for multi-GPU without fractional
        gpus = self._get_gpu_requirement(op)
        if gpus >= 2 and not self.enable_fractional_gpu:
            return True

        return False

    def calculate_max_workers(
        self,
        total_cpus: float,
        total_gpus: float,
        resources_per_worker: Resources,
    ) -> int:
        """
        Calculate maximum number of workers that can be allocated.

        Args:
            total_cpus: Total available CPU cores
            total_gpus: Total available GPUs
            resources_per_worker: Resources required per worker

        Returns:
            Maximum number of workers
        """
        # CPU-limited
        cpu_workers = int(total_cpus / max(resources_per_worker.cpus, 0.1))

        # GPU-limited (if GPUs required)
        if resources_per_worker.gpus > 0:
            gpu_workers = int(total_gpus / max(resources_per_worker.gpus, 0.1))
            return min(cpu_workers, gpu_workers)

        return cpu_workers

    def get_resource_efficiency(
        self,
        allocated: Resources,
        requested: Resources,
    ) -> float:
        """
        Calculate resource allocation efficiency.

        Args:
            allocated: Actually allocated resources
            requested: Requested resources

        Returns:
            Efficiency ratio (0-1, higher is better)
        """
        cpu_eff = min(allocated.cpus / max(requested.cpus, 0.1), 1.0)
        gpu_eff = (
            min(allocated.gpus / max(requested.gpus, 0.1), 1.0)
            if requested.gpus > 0
            else 1.0
        )

        # Weight GPU efficiency higher if GPUs are involved
        if requested.gpus > 0:
            return 0.3 * cpu_eff + 0.7 * gpu_eff
        return cpu_eff

    def clear_cache(self):
        """Clear the resource mapping cache."""
        self._cache.clear()
