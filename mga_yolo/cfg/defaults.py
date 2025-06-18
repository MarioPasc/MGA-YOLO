# mga_yolo/cfg/defaults.py
from __future__ import annotations

import dataclasses as _dc
import numbers
from pathlib import Path
from typing import Any, Mapping

import yaml


# ────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────
def _to_int_tuple(seq: Any) -> tuple[int, ...]:
    """Convert a scalar / sequence of str|int to a tuple[int, …]."""
    if isinstance(seq, numbers.Integral):
        return (int(seq),)
    if isinstance(seq, (str, bytes)):
        # split on comma or whitespace
        parts = [p for p in str(seq).replace(",", " ").split() if p]
        return tuple(int(p) for p in parts)
    return tuple(int(x) for x in seq)


# ────────────────────────────────────────────────────────────────
# Main dataclass
# ────────────────────────────────────────────────────────────────
@_dc.dataclass(slots=True)
class MGAConfig:
    # ── Model & data ────────────────────────────────────────────
    model_cfg: str | Path = "yolov8n.pt"
    data_yaml: str | Path = "data/stenosis.yaml"
    masks_dir: str | Path = "data/masks"

    # ── Training hyper-parameters ───────────────────────────────
    epochs: int = 100
    imgsz: int = 512
    batch: int = 4
    device: str = "cuda:0"

    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    save_period: int = 10
    iou: float = 0.5
    
    single_cls: bool = True  # single-class training

    # ── MGA-specific knobs ──────────────────────────────────────
    target_layers: tuple[int, ...] = (15, 18, 21)
    reduction_ratio: int = 16
    kernel_size: int = 7
    sam_cam_fusion: str = "add"          # {add|multiply}
    mga_pyramid_fusion: str = "add"      # {add|concat|mul}

    # ── Experiment bookkeeping ─────────────────────────────────
    project: str | Path = "runs/mga-yolo"
    name: str = "exp"
    visualize_features: bool = False

    # ── Augmentation dictionary (hsv_h, hsv_s, …) ──────────────
    augmentation_config: Mapping[str, float] = _dc.field(
        default_factory=lambda: {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
        }
    )

    # ─────────────────── YAML helpers ───────────────────────────
    @classmethod
    def load(cls, path: str | Path) -> "MGAConfig":
        """Load a YAML file and coerce types to match the dataclass."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        # handle target_layers (str list → tuple[int,…])
        if "target_layers" in raw:
            raw["target_layers"] = _to_int_tuple(raw["target_layers"])

        # merge augmentation sub-dict if present
        aug = raw.get("augmentation_config")
        if aug is not None:
            # inherit defaults, then override
            raw["augmentation_config"] = {
                **cls().augmentation_config,  # default template
                **aug,
            }

        return cls(**raw)

    def dump(self, dest: str | Path) -> None:
        """Serialize the config (including defaults) to YAML."""
        with open(dest, "w") as f:
            yaml.safe_dump(_dc.asdict(self), f, sort_keys=False)

    # convenience immutable copy
    def copy(self, **updates) -> "MGAConfig":
        """Return a new MGAConfig with selected fields overridden."""
        return _dc.replace(self, **updates)
