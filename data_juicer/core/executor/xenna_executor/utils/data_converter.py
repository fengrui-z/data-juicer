"""
Data Converter

Converts data between Data-Juicer and Cosmos-Xenna formats.
Handles dataset conversion, sample transformation, and result aggregation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from loguru import logger


class DataConverter:
    """
    Converts data between Data-Juicer and Xenna formats.

    Data-Juicer uses:
    - NestedDataset (HuggingFace Dataset wrapper)
    - Dict of lists for batched data
    - List of dicts for individual samples

    Xenna uses:
    - List of samples as input
    - List of samples as output
    - Samples can be any picklable type
    """

    @staticmethod
    def dataset_to_list(dataset: Any) -> List[Dict[str, Any]]:
        """
        Convert Data-Juicer dataset to list of samples.

        Args:
            dataset: Data-Juicer dataset (NestedDataset, HF Dataset, etc.)

        Returns:
            List of sample dictionaries
        """
        # Handle list directly
        if isinstance(dataset, list):
            return dataset

        # Handle NestedDataset with to_list
        if hasattr(dataset, "to_list"):
            return dataset.to_list()

        # Handle HuggingFace Dataset
        if hasattr(dataset, "__iter__") and hasattr(dataset, "__len__"):
            return [DataConverter._sample_to_dict(s) for s in dataset]

        # Handle dict of lists (batched format)
        if isinstance(dataset, dict):
            return DataConverter.batch_to_list(dataset)

        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    @staticmethod
    def list_to_dataset(samples: List[Dict[str, Any]], dataset_type: str = "nested") -> Any:
        """
        Convert list of samples to Data-Juicer dataset.

        Args:
            samples: List of sample dictionaries
            dataset_type: Type of dataset to create ("nested", "hf")

        Returns:
            Data-Juicer dataset
        """
        if not samples:
            logger.warning("Empty sample list, creating empty dataset")
            return DataConverter._create_empty_dataset(dataset_type)

        if dataset_type == "nested":
            try:
                from data_juicer.core.data import NestedDataset
                return NestedDataset.from_list(samples)
            except ImportError:
                logger.warning("NestedDataset not available, using HuggingFace Dataset")
                dataset_type = "hf"

        if dataset_type == "hf":
            try:
                from datasets import Dataset
                return Dataset.from_list(samples)
            except ImportError:
                raise ImportError("HuggingFace datasets not available")

        raise ValueError(f"Unknown dataset type: {dataset_type}")

    @staticmethod
    def batch_to_list(batch: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Convert batched data (dict of lists) to list of samples.

        Args:
            batch: Dict where each key maps to a list of values

        Returns:
            List of sample dictionaries
        """
        if not batch:
            return []

        keys = list(batch.keys())
        if not keys:
            return []

        # Get number of samples from first key
        num_samples = len(batch[keys[0]])

        # Validate all keys have same length
        for key in keys:
            if len(batch[key]) != num_samples:
                raise ValueError(
                    f"Inconsistent batch sizes: key '{key}' has "
                    f"{len(batch[key])} values, expected {num_samples}"
                )

        return [{key: batch[key][i] for key in keys} for i in range(num_samples)]

    @staticmethod
    def list_to_batch(samples: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Convert list of samples to batched format.

        Args:
            samples: List of sample dictionaries

        Returns:
            Dict where each key maps to a list of values
        """
        if not samples:
            return {}

        # Collect all keys
        keys = set()
        for sample in samples:
            keys.update(sample.keys())

        # Build batch
        batch = {key: [] for key in keys}
        for sample in samples:
            for key in keys:
                batch[key].append(sample.get(key))

        return batch

    @staticmethod
    def _sample_to_dict(sample: Any) -> Dict[str, Any]:
        """Convert a single sample to dictionary."""
        if isinstance(sample, dict):
            return sample

        # Handle PyArrow Table
        if hasattr(sample, "to_pydict"):
            return sample.to_pydict()

        # Handle LazyDict (from datasets library)
        try:
            from datasets.formatting.formatting import LazyDict
            if isinstance(sample, LazyDict):
                return dict(sample)
        except ImportError:
            pass

        # Handle object with __dict__
        if hasattr(sample, "__dict__"):
            return sample.__dict__

        raise ValueError(f"Cannot convert sample to dict: {type(sample)}")

    @staticmethod
    def _create_empty_dataset(dataset_type: str) -> Any:
        """Create an empty dataset."""
        if dataset_type == "nested":
            try:
                from data_juicer.core.data import NestedDataset
                from datasets import Dataset
                return NestedDataset(Dataset.from_dict({}))
            except ImportError:
                pass

        if dataset_type == "hf":
            try:
                from datasets import Dataset
                return Dataset.from_dict({})
            except ImportError:
                pass

        return []

    @staticmethod
    def merge_samples(
        samples: List[Dict[str, Any]],
        merge_strategy: str = "update",
    ) -> Dict[str, Any]:
        """
        Merge multiple samples into one.

        Args:
            samples: List of samples to merge
            merge_strategy: How to handle conflicts ("update", "keep_first", "concat")

        Returns:
            Merged sample
        """
        if not samples:
            return {}
        if len(samples) == 1:
            return samples[0].copy()

        result = {}

        for sample in samples:
            for key, value in sample.items():
                if key not in result:
                    result[key] = value
                elif merge_strategy == "update":
                    result[key] = value
                elif merge_strategy == "concat":
                    if isinstance(result[key], list):
                        if isinstance(value, list):
                            result[key].extend(value)
                        else:
                            result[key].append(value)
                    else:
                        result[key] = [result[key], value]
                # "keep_first": do nothing

        return result

    @staticmethod
    def split_sample(
        sample: Dict[str, Any],
        split_keys: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Split a sample into multiple samples based on list values.

        Args:
            sample: Sample to split
            split_keys: Keys whose values are lists to split on

        Returns:
            List of split samples
        """
        if not split_keys:
            return [sample]

        # Get split length from first split key
        split_length = None
        for key in split_keys:
            if key in sample and isinstance(sample[key], list):
                split_length = len(sample[key])
                break

        if split_length is None or split_length == 0:
            return [sample]

        # Create split samples
        results = []
        for i in range(split_length):
            split_sample = {}
            for key, value in sample.items():
                if key in split_keys and isinstance(value, list):
                    if i < len(value):
                        split_sample[key] = value[i]
                else:
                    split_sample[key] = value
            results.append(split_sample)

        return results

    @staticmethod
    def filter_none_samples(samples: List[Optional[Dict]]) -> List[Dict[str, Any]]:
        """
        Filter out None samples from a list.

        Args:
            samples: List that may contain None values

        Returns:
            List without None values
        """
        return [s for s in samples if s is not None]

    @staticmethod
    def validate_samples(
        samples: List[Dict[str, Any]],
        required_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate samples and return statistics.

        Args:
            samples: Samples to validate
            required_keys: Keys that must be present

        Returns:
            Validation results
        """
        results = {
            "total": len(samples),
            "valid": 0,
            "invalid": 0,
            "missing_keys": [],
            "errors": [],
        }

        if required_keys is None:
            required_keys = []

        for i, sample in enumerate(samples):
            if sample is None:
                results["invalid"] += 1
                results["errors"].append(f"Sample {i}: None value")
                continue

            if not isinstance(sample, dict):
                results["invalid"] += 1
                results["errors"].append(f"Sample {i}: Not a dict")
                continue

            # Check required keys
            missing = [k for k in required_keys if k not in sample]
            if missing:
                results["missing_keys"].extend(missing)
                results["invalid"] += 1
                continue

            results["valid"] += 1

        # Deduplicate missing keys
        results["missing_keys"] = list(set(results["missing_keys"]))

        return results
