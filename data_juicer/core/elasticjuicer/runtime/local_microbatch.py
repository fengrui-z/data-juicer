"""Adaptive, lossless micro-batching for the local executor."""

from collections.abc import Mapping
from typing import Any, Callable, Iterable, List, Optional

from ..scheduler.micro_scheduler import MicroScheduler
from ..scheduler.oom import is_oom_error


def _batch_length(batch) -> int:
    if hasattr(batch, "num_rows"):
        return int(batch.num_rows)
    if isinstance(batch, Mapping):
        for value in batch.values():
            try:
                return len(value)
            except TypeError:
                continue
        raise ValueError("batched mappings must contain a sized column")
    return len(batch)


def _slice_batch(batch, start: int, end: int, total: int):
    if hasattr(batch, "num_rows") and callable(getattr(batch, "slice", None)):
        return batch.slice(start, end - start)
    if isinstance(batch, Mapping):
        sliced = {}
        for key, value in batch.items():
            try:
                sized_column = not isinstance(value, (str, bytes)) and len(value) == total
                sliced[key] = value[start:end] if sized_column else value
            except TypeError:
                sliced[key] = value
        return sliced
    return batch[start:end]


def _merge_outputs(outputs: List[Any]):
    if not outputs:
        return []

    first = outputs[0]
    if hasattr(first, "schema") and hasattr(first, "num_rows"):
        import pyarrow

        return pyarrow.concat_tables(outputs)
    if isinstance(first, Mapping):
        merged = {}
        for output in outputs:
            for key, value in output.items():
                target = merged.setdefault(key, [])
                if isinstance(value, (str, bytes)):
                    target.append(value)
                else:
                    try:
                        target.extend(value)
                    except TypeError:
                        target.append(value)
        return merged
    if isinstance(first, tuple):
        return tuple(item for output in outputs for item in output)

    merged = []
    for output in outputs:
        if isinstance(output, list):
            merged.extend(output)
        else:
            try:
                merged.extend(list(output))
            except TypeError:
                merged.append(output)
    return merged


class AdaptiveMicrobatchExecutor:
    """Split one outer batch and retry the same slice after a classified OOM."""

    def __init__(
        self,
        function: Callable,
        scheduler: MicroScheduler,
        max_retries_per_slice: int = 16,
    ):
        if max_retries_per_slice < 1:
            raise ValueError("max_retries_per_slice must be at least 1")
        self.function = function
        self.scheduler = scheduler
        self.max_retries_per_slice = max_retries_per_slice
        self.oom_retries = 0
        self.successful_slices = 0

    def __call__(self, batch, *args, **kwargs):
        total = _batch_length(batch)
        if total == 0:
            return self.function(batch, *args, **kwargs)

        outputs = []
        offset = 0
        while offset < total:
            retries = 0
            while True:
                batch_size = min(
                    self.scheduler.controller.current_batch_size,
                    total - offset,
                )
                microbatch = _slice_batch(batch, offset, offset + batch_size, total)
                try:
                    output = self.function(microbatch, *args, **kwargs)
                    break
                except BaseException as error:
                    if not is_oom_error(error):
                        raise
                    if batch_size <= self.scheduler.controller.min_batch_size:
                        raise
                    retries += 1
                    self.oom_retries += 1
                    if retries > self.max_retries_per_slice:
                        raise
                    self.scheduler.report_oom(batch_size, memory_mb=0)

            outputs.append(output)
            offset += batch_size
            self.successful_slices += 1
            self.scheduler.update(actual_memory_used=0)

        return _merge_outputs(outputs)


class LocalMicrobatchRuntime:
    """Install micro-batching on eligible local Mapper and Filter instances."""

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 1000,
        scheduler_factory: Optional[Callable[..., MicroScheduler]] = None,
    ):
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be at least 1")
        if max_batch_size < min_batch_size:
            raise ValueError("max_batch_size must be >= min_batch_size")
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.scheduler_factory = scheduler_factory or MicroScheduler
        self.executors: List[AdaptiveMicrobatchExecutor] = []

    def install(self, operators: Iterable) -> List[AdaptiveMicrobatchExecutor]:
        from data_juicer.ops.base_op import Filter, Mapper

        installed = []
        for operator in operators:
            if not operator.is_batched_op():
                continue
            if isinstance(operator, Mapper):
                attribute = "process"
            elif isinstance(operator, Filter):
                attribute = "compute_stats"
            else:
                continue

            original = getattr(operator, attribute)
            initial = min(
                self.max_batch_size,
                max(self.min_batch_size, int(operator.batch_size)),
            )
            scheduler = self.scheduler_factory(
                initial_batch_size=initial,
                min_batch_size=self.min_batch_size,
                max_batch_size=self.max_batch_size,
            )
            executor = AdaptiveMicrobatchExecutor(original, scheduler)
            setattr(operator, attribute, executor)
            operator.batch_size = self.max_batch_size
            installed.append(executor)

        self.executors.extend(installed)
        return installed
