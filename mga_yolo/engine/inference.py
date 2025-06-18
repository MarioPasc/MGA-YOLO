"""
Simple inference wrapper that adds mask-guided attention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from mga_yolo.external.ultralytics import YOLO

from mga_yolo.cfg.defaults import MGAConfig
from mga_yolo.models.hooks import HookManager


class MaskGuidedInference:
    def __init__(self, cfg: MGAConfig, weight_path: str | Path):
        self.cfg = cfg
        self.model = YOLO(weight_path)
        self.hooks = HookManager(cfg)
        self.hooks.register(self.model)

    def __call__(self, images: Sequence[str | Path], **predict_kwargs):
        # supply image paths to HookManager then run predict
        self.hooks.set_batch_paths(images)
        return self.model.predict(list(images), **predict_kwargs)
