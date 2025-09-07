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
try:  # MGA custom module
    from mga_yolo.nn.modules.segmentation import MGAMaskHead  # noqa: F401
except Exception:
    MGAMaskHead = None  # type: ignore
    
try:  # Masked ECA attention
    from mga_yolo.nn.modules.masked_eca import MaskECA  # noqa: F401
except Exception:
    MaskECA = None  # type: ignore
    
try:  # Masked SPADE attention
    from mga_yolo.nn.modules.masked_spade import MaskSPADE  # noqa: F401
except Exception:
    MaskSPADE = None  # type: ignore

try:  # Masked CBAM attention
    from mga_yolo.nn.modules.masked_cbam import MaskCBAM  # noqa: F401
except Exception:
    MaskCBAM = None  # type: ignore

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
    # Our modules
    "MGAMaskHead",
    "MaskECA",
    "MaskSPADE",
    "MaskCBAM",
)
