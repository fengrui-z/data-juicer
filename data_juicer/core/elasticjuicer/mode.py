"""ElasticJuicer mode resolution and legacy compatibility."""

from enum import Enum


class ElasticJuicerMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    RECOMMEND = "recommend"
    APPLY = "apply"


def resolve_mode(configured_mode="off", legacy_adaptive_batch_size=False):
    """Resolve one mode while preserving the legacy adaptive batch flag."""

    try:
        mode = ElasticJuicerMode(configured_mode or "off")
    except ValueError as error:
        supported = ", ".join(item.value for item in ElasticJuicerMode)
        raise ValueError(f"Unsupported elastic_juicer_mode {configured_mode!r}; expected one of {supported}") from error

    if not legacy_adaptive_batch_size:
        return mode
    if mode in (ElasticJuicerMode.OFF, ElasticJuicerMode.APPLY):
        return ElasticJuicerMode.APPLY
    raise ValueError(
        "adaptive_batch_size=True conflicts with elastic_juicer_mode="
        f"{mode.value!r}; use elastic_juicer_mode='apply'"
    )
