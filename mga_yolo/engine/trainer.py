"""
Thin training wrapper around Ultralytics YOLO.
"""
from __future__ import annotations

from mga_yolo.external.ultralytics.ultralytics import YOLO
from mga_yolo.external.ultralytics.ultralytics.engine.trainer import BaseTrainer

from mga_yolo.cfg.defaults import MGAConfig
from mga_yolo.models.hooks import HookManager

from typing import Sequence
from pathlib import Path

class MaskGuidedTrainer:
    # ───────────────────────── init unchanged ───────────────────────── #
    def __init__(self, cfg: MGAConfig):
        self.cfg = cfg
        self.yolo = YOLO(cfg.model_cfg)
        self.hooks = HookManager(cfg)
        self.hooks.register(self.yolo)

    # ───────────────────────── fixed train() ────────────────────────── #
    def train(self) -> YOLO:
        """
        Delegates everything to Ultralytics’ built-in trainer, but wires a
        one-argument callback that supplies image-paths to the HookManager.
        """
        def on_batch_end(trainer: BaseTrainer):
            """Ultralytics passes only the trainer object."""
            batch = getattr(trainer, "batch", None)

            # dataloader in Ultralytics ≥8 returns a dict with 'im_file'
            paths: Sequence[str | Path] = []
            if isinstance(batch, dict):
                paths = batch.get("im_file", [])
            elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                # older loaders: first entry is the image-tensor, second the path list
                paths = batch[1] if isinstance(batch[1], (list, tuple)) else []
            self.hooks.set_batch_paths(paths)

        # register *fixed-signature* callback
        self.yolo.add_callback("on_batch_end", on_batch_end)

        # launch training
        self.yolo.train(
            data=self.cfg.data_yaml,
            epochs=self.cfg.epochs,
            imgsz=self.cfg.imgsz,
            batch=self.cfg.batch,
            device=self.cfg.device,
            project=self.cfg.project,
            name=self.cfg.name,
        )
        return self.yolo
