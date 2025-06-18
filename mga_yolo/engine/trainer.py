"""
Mask-Guided-Attention trainer for Ultralytics-YOLO
=================================================

* Keeps the refactored, lightweight design (single callback; no custom
  DetectionTrainer subclass needed).
* Restores the rich logging, banner, mask-folder diagnostics and runtime
  statistics found in the original implementation.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Sequence

import yaml
from mga_yolo.cfg.defaults import MGAConfig
from mga_yolo.models.hooks import HookManager
from mga_yolo.external.ultralytics.ultralytics import YOLO
from mga_yolo.external.ultralytics.ultralytics.engine.trainer import BaseTrainer

# ───────────────────────── module-level logger ────────────────────────── #
LOGGER = logging.getLogger("mga_yolo.trainer")

# ╭───────────────────────────────────────────────────────────────────────╮
# │ MaskGuidedTrainer                                                    │
# ╰───────────────────────────────────────────────────────────────────────╯
class MaskGuidedTrainer:
    def __init__(self, cfg: MGAConfig) -> None:
        # ── pretty banner & logging ─────────────────────────────────────── #
        self._print_banner()

        self.cfg = cfg
        self.yolo = YOLO(cfg.model_cfg)

        # stats
        self._start_time = time.time()
        self._batch_count = 0
        self._mga_active = True

        # ── hook manager ───────────────────────────────────────────────── #
        self.hooks = HookManager(cfg)
        self.hooks.register(self.yolo)

        # log mask information
        self._log_mask_information()

        LOGGER.info("Mask-Guided trainer initialised.")

    # ────────────────────────────── helpers ───────────────────────────── #
    def _print_banner(self) -> None:
        banner = r"""
╔════════════════════════════════════════════════╗
║          Mask-Guided Attention – YOLO          ║
╚════════════════════════════════════════════════╝
"""
        LOGGER.info(banner)
        print(banner)

    def _log_mask_information(self) -> None:
        mdir = Path(self.cfg.masks_dir)
        try:
            files = list(mdir.iterdir())
            LOGGER.info("Mask folder: %s  (%d files)", mdir, len(files))
            if files:
                LOGGER.debug("Sample masks: %s", [f.name for f in files[:5]])
        except Exception as e:
            LOGGER.error("Cannot access mask folder %s: %s", mdir, e)

    def _log_mga_statistics(self) -> None:
        elapsed = time.time() - self._start_time
        LOGGER.info(
            "[MGA-STATS] batches=%d  runtime=%.1fs  target_layers=%s  active=%s",
            self._batch_count,
            elapsed,
            self.cfg.target_layers,
            self._mga_active,
        )

    # ────────────────────────────── train ─────────────────────────────── #
    def train(self) -> YOLO:
        """
        Launch Ultralytics training while feeding image paths to the HookManager
        and logging periodic MGA statistics.
        """

        def on_batch_end(trainer: BaseTrainer):  # Ultralytics passes only self
            batch = getattr(trainer, "batch", None)

            # dataloader returns a dict with 'im_file' (current versions)
            paths: Sequence[str | Path] = []
            if isinstance(batch, dict):
                paths = batch.get("im_file", [])
            elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                paths = batch[1] if isinstance(batch[1], (list, tuple)) else []

            self.hooks.set_batch_paths(paths)

            # stats
            self._batch_count += 1
            if self._batch_count % 250 == 0:
                self._log_mga_statistics()

        # register callback once
        self.yolo.add_callback("on_batch_end", on_batch_end)

        # log config summary
        LOGGER.info(
            "Starting training → epochs=%d img_size=%d batch=%d device=%s",
            self.cfg.epochs,
            self.cfg.imgsz,
            self.cfg.batch,
            self.cfg.device,
        )

        # run training
        self.yolo.train(
            data=self.cfg.data_yaml,
            epochs=self.cfg.epochs,
            imgsz=self.cfg.imgsz,
            batch=self.cfg.batch,
            device=self.cfg.device,
            project=self.cfg.project,
            name=self.cfg.name,
            iou=self.cfg.iou,
            **self.cfg.augmentation_config,
        )

        runtime = time.time() - self._start_time
        LOGGER.info("Training finished in %.1f s  (batches=%d)", runtime, self._batch_count)
        return self.yolo
