"""
HookManager — Mask-Guided CBAM for Ultralytics-YOLOv8
----------------------------------------------------
* Works with YOLO ≥ 8 (DetectionModel backend)
* Handles **per-image** masks inside a batch
* Falls back gracefully when a mask is missing or unreadable
* Optional one-shot visualisation (mask ↓, before, after) controlled by
  `cfg.visualize_features`
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt  # only used when visualise=True
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from mga_yolo.cfg.defaults import MGAConfig
from mga_yolo.nn.mga_cbam import MaskGuidedCBAM
from mga_yolo.utils.mask_io import find_mask_path  # exact+prefix; we add extra logic

LOGGER = logging.getLogger("mga_yolo.hooks")


# ────────────────────────────────────────────────────────────────────────────
# util helpers
# ────────────────────────────────────────────────────────────────────────────
def _extra_mask_lookup(masks_dir: Path, stem: str) -> Optional[Path]:
    """More aggressive lookup: numeric id match and *_mask suffix."""
    files = list(masks_dir.iterdir())
    # numeric id
    m = re.search(r"(\d+)$", stem)
    if m:
        num = m.group(1)
        for p in files:
            if num in p.stem:
                return p
    # *_mask suffix
    for p in files:
        if p.stem == f"{stem}_mask":
            return p
    return None


def _load_mask(path: Path, size: Tuple[int, int]) -> Optional[torch.Tensor]:
    try:
        img = Image.open(path).convert("L")
        img = T.Resize(size, interpolation=T.InterpolationMode.NEAREST)(img)
        return T.ToTensor()(img).unsqueeze(0)  # [1,1,H,W]
    except Exception:
        LOGGER.exception("Failed to load or resize mask %s", path)
        return None


def _visualise_once(
    save_dir: Path,
    tag: str,
    mask: torch.Tensor,
    before: torch.Tensor,
    after: torch.Tensor,
) -> None:
    """
    Save a single 3-panel PNG (mask ↓, before, after) once per `tag`.
    The tensors are assumed 2-D (H×W) on CPU.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{tag}.png"
    if out_path.exists():
        return

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(mask, cmap="gray")
    axs[0].set_title("Mask ↓")
    axs[1].imshow(before, cmap="viridis")
    axs[1].set_title("Feat before")
    axs[2].imshow(after, cmap="viridis")
    axs[2].set_title("Feat after")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved feature-map visualisation → %s", out_path)


# ────────────────────────────────────────────────────────────────────────────
# main class
# ────────────────────────────────────────────────────────────────────────────
class HookManager:
    """Attach mask-guided CBAM blocks to arbitrary backbone layers."""

    def __init__(
        self,
        cfg: MGAConfig,
        get_image_path_fn: Optional[Callable[[int], str]] = None,
    ) -> None:
        self.cfg = cfg
        self.get_image_path_fn = get_image_path_fn

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._module_cache: Dict[tuple[str, int], nn.Module] = {}
        self._batch_paths: List[Path] = []

        # one-shot visualisation book-keeping
        self._visualised: set[str] = set()

    # ─────────────────── public API ─────────────────── #
    def set_batch_paths(self, paths: Sequence[str | Path]) -> None:
        self._batch_paths = [Path(p) for p in paths]

    def register(self, model: nn.Module) -> None:
        self.clear()
        wanted = {f"model.{str(i)}" for i in self.cfg.target_layers}

        if not hasattr(model, "model"):
            raise AttributeError("Expected `model.model` ModuleList")

        for name, module in model.model.named_modules():

            if name in wanted:
                h = module.register_forward_hook(self._hook_fn(name))
                self._handles.append(h)
                LOGGER.info("Hook registered on layer %s", name)

    def clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._module_cache.clear()

    # context-manager sugar
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.clear()
        return False

    # ─────────────────── internal ─────────────────── #
    def _hook_fn(self, layer_name: str):
        def inner(_mod: nn.Module, _inp, output: torch.Tensor):
            B, C, H, W = output.shape
            out_chunks = []

            for idx in range(B):
                # --- resolve image path ------------------------------------ #
                img_path = (
                    self._batch_paths[idx]
                    if idx < len(self._batch_paths)
                    else Path(self.get_image_path_fn(idx))
                    if self.get_image_path_fn
                    else None
                )
                if img_path is None:
                    out_chunks.append(output[idx : idx + 1])
                    continue

                # --- resolve mask ------------------------------------------ #
                mpath = find_mask_path(self.cfg.masks_dir, img_path.stem)
                if mpath is None:
                    mpath = _extra_mask_lookup(Path(self.cfg.masks_dir), img_path.stem)
                if mpath is None:
                    out_chunks.append(output[idx : idx + 1])
                    continue

                mask = _load_mask(mpath, (H, W))
                if mask is None:
                    out_chunks.append(output[idx : idx + 1])
                    continue
                mask = mask.to(output.device)

                # --- CBAM (cached per layer+channels) ---------------------- #
                key = (layer_name, C)
                block = self._module_cache.get(key)
                if block is None:
                    block = MaskGuidedCBAM(
                        in_channels=C,
                        reduction=self.cfg.reduction_ratio,
                        fusion=self.cfg.mga_pyramid_fusion,
                        sam_cam_fusion=self.cfg.sam_cam_fusion,
                    ).to(output.device)
                    self._module_cache[key] = block

                feat_before = output[idx : idx + 1]
                feat_after = block(feat_before, mask.expand_as(feat_before))

                # optional visualisation — only first image seen for that tag
                if (
                    self.cfg.visualize_features
                    and (tag := f"{img_path.stem}_{layer_name}") not in self._visualised
                ):
                    _visualise_once(
                        save_dir=Path(self.cfg.project_dir) / "feature_vis",
                        tag=tag.replace(".", "-"),
                        mask=mask.squeeze().cpu(),
                        before=feat_before.mean(1).squeeze().cpu(),
                        after=feat_after.mean(1).squeeze().cpu(),
                    )
                    self._visualised.add(tag)

                out_chunks.append(feat_after)

            return torch.cat(out_chunks, dim=0)

        return inner

    # allow changing config on the fly
    def set_config(self, cfg: MGAConfig) -> None:
        self.cfg = cfg
        self._module_cache.clear()

    # repr for debugging
    def __repr__(self) -> str:
        return (
            f"HookManager(target_layers={self.cfg.target_layers}, "
            f"hooks={len(self._handles)})"
        )
