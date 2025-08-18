# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .tasks import (
    BaseModel,
    ClassificationModel,
    DetectionModel,
    SegmentationModel,
    attempt_load_one_weight,
    attempt_load_weights,
    guess_model_scale,
    guess_model_task,
    parse_model,
    torch_safe_load,
    yaml_model_load,
)

# Import custom MGA modules so they are available to parse_model via globals()
try:  # safe import in case MGA modules removed
    from mga_yolo.nn.modules.seg import MGAMaskHead  # noqa: F401
except Exception:  # pragma: no cover - non-fatal
    MGAMaskHead = None  # type: ignore

__all__ = (
    "attempt_load_one_weight",
    "attempt_load_weights",
    "parse_model",
    "yaml_model_load",
    "guess_model_task",
    "guess_model_scale",
    "torch_safe_load",
    "DetectionModel",
    "SegmentationModel",
    "ClassificationModel",
    "BaseModel",
    "MGAMaskHead",
)
