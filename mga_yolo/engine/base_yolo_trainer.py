"""
Base YOLO trainer for Ultralytics-YOLO
=======================================

A lightweight wrapper around Ultralytics YOLO that provides:
- Simple configuration interface
- Training progress logging
- Training statistics tracking
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from mga_yolo.external.ultralytics.ultralytics import YOLO
from mga_yolo.external.ultralytics.ultralytics.engine.trainer import BaseTrainer

# ───────────────────────── module-level logger ────────────────────────── #
LOGGER = logging.getLogger("mga_yolo.base_trainer")

# ╭───────────────────────────────────────────────────────────────────────╮
# │ BaseYOLO                                                              │
# ╰───────────────────────────────────────────────────────────────────────╯
class BaseYOLO:
    def __init__(self, model_cfg: str) -> None:
        """
        Initialize a base YOLO trainer without mask-guided attention.
        
        Args:
            model_cfg: Path to the YOLO model configuration file or pretrained weights
        """
        # ── pretty banner & logging ─────────────────────────────────────── #
        self._print_banner()
        
        self.model_cfg = model_cfg
        self.yolo = YOLO(model_cfg)
        
        # stats
        self._start_time = time.time()
        self._batch_count = 0
        
        LOGGER.info(f"Base YOLO trainer initialized with model: {model_cfg}")

    # ────────────────────────────── helpers ───────────────────────────── #
    def _print_banner(self) -> None:
        banner = r"""
╔════════════════════════════════════════════════╗
║                Base YOLO Trainer               ║
╚════════════════════════════════════════════════╝
"""
        LOGGER.info(banner)
        print(banner)

    def _log_statistics(self) -> None:
        elapsed = time.time() - self._start_time
        LOGGER.info(
            "[YOLO-STATS] batches=%d  runtime=%.1fs",
            self._batch_count,
            elapsed,
        )

    # ────────────────────────────── train ─────────────────────────────── #
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: str = '0',
        project: str = 'runs/train',
        name: str = 'exp',
        iou: float = 0.7,
        **kwargs
    ) -> YOLO:
        """
        Launch Ultralytics training with the given parameters.
        
        Args:
            data_yaml: Path to the dataset configuration file
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            device: Device to train on ('cpu', '0', '0,1,2,3', etc.)
            project: Project directory
            name: Experiment name
            iou: IoU threshold for NMS
            **kwargs: Additional arguments to pass to YOLO.train()
            
        Returns:
            Trained YOLO model
        """
        def on_batch_end(trainer: BaseTrainer):
            # Update batch counter for statistics
            self._batch_count += 1
            if self._batch_count % 250 == 0:
                self._log_statistics()

        # Register callback for statistics
        self.yolo.add_callback("on_batch_end", on_batch_end)
        
        # Log training configuration
        LOGGER.info(
            "Starting training → epochs=%d img_size=%d batch=%d device=%s",
            epochs,
            imgsz,
            batch,
            device,
        )
        
        # Run training
        self.yolo.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            iou=iou,
            **kwargs
        )
        
        runtime = time.time() - self._start_time
        LOGGER.info("Training finished in %.1f s  (batches=%d)", runtime, self._batch_count)
        return self.yolo