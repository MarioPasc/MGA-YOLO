"""
Utility for locating and loading binary vessel masks.
"""
from __future__ import annotations


import re
from pathlib import Path
from typing import Optional

from mga_yolo import LOGGER as log

import torch
from PIL import Image

from mga_yolo import LOGGER
import logging
LOGGER = logging.getLogger("mga_yolo.mask_io")

def find_mask_path(masks_dir: str | Path, img_basename: str) -> Optional[Path]:
    """
    Heuristically match `img_basename` (without extension) to a mask file.
    Ported & cleaned from user’s `mask.py`. 
    """
    masks_dir = Path(masks_dir)
    stem = Path(img_basename).stem

    # exact stem match
    for p in masks_dir.iterdir():
        if p.stem == stem:
            return p

    # prefix match (e.g. "_mask" suffix)
    for p in masks_dir.iterdir():
        if p.stem.startswith(stem) or p.stem == f"{stem}_mask":
            return p

    # numerical id match
    m = re.search(r"(\d+)$", stem)
    if m:
        num = m.group(1)
        for p in masks_dir.iterdir():
            if num in p.stem:
                return p

    log.warning("No mask found for %s", img_basename)
    return None


def load_mask(path: Path) -> torch.Tensor:
    """Load a **binary** mask as `float32` tensor in [0,1]."""
    from torchvision.transforms.functional import to_tensor
    mask = Image.open(path).convert("L")
    tensor_mask = (to_tensor(mask) > 0).float()
    LOGGER.info(f"Loaded mask from {path}, shape: {tensor_mask.shape}")
    return tensor_mask
