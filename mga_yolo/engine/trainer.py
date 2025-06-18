"""
Thin training wrapper around Ultralytics YOLO.
"""
from __future__ import annotations

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

from ..cfg.defaults import MGAConfig
from ..models.hooks import HookManager


class MaskGuidedTrainer:
    def __init__(self, cfg: MGAConfig):
        self.cfg = cfg
        self.yolo = YOLO(cfg.model_cfg)
        self.hooks = HookManager(cfg)
        self.hooks.register(self.yolo)

    def train(self) -> YOLO:
        """
        Delegates nearly everything to Ultralytics’ built-in trainer,
        only intercepting `batch_end` to provide image paths to the
        HookManager.
        """
        trainer: BaseTrainer = self.yolo.trainer

        def on_batch_end(_, batch, batch_i):
            paths = batch.get("im_file", [])         # Ultralytics dataloader
            self.hooks.set_batch_paths(paths)

        # Register callback and run
        self.yolo.add_callback("on_batch_end", on_batch_end)
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
