"""Custom MGA-YOLO dataloader utilities.

Provides a thin wrapper around Ultralytics' build_dataloader to allow future
insertion of mask-specific sampling strategies or curriculum learning.
"""
from __future__ import annotations
from typing import Any
from ultralytics.utils import LOGGER
from ultralytics.data.utils import build_dataloader as yolo_build_dataloader


def build_dataloader(dataset, batch_size: int, workers: int, shuffle: bool, rank: int, mode: str = "train", **kwargs) -> Any:
    """Build a dataloader with logging (placeholder for custom sampling logic)."""
    LOGGER.debug(
        f"[MGA] build_dataloader: bs={batch_size} workers={workers} shuffle={shuffle} rank={rank} mode={mode} size={len(dataset)}"
    )
    return yolo_build_dataloader(dataset, batch_size, workers, shuffle, rank)
