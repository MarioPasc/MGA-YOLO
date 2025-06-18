"""
HookManager — injects MaskGuidedCBAM *without touching the backbone*.
Works with Ultralytics YOLO ≥ 8 where `YOLO.model` is a DetectionModel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn

from mga_yolo.nn.mga_cbam import MaskGuidedCBAM
from mga_yolo.utils.mask_io import find_mask_path, load_mask
from mga_yolo.cfg.defaults import MGAConfig


class HookManager:
    """Attach mask-guided CBAM blocks to arbitrary backbone layers."""

    def __init__(self, cfg: MGAConfig) -> None:
        self.cfg = cfg
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._img_paths: List[Path] | None = None
        self._module_cache: dict[tuple[str, int], nn.Module] = {}

    # ───────────────────────── public API ───────────────────────── #
    def set_batch_paths(self, img_paths: Iterable[str | Path]) -> None:
        """Record the paths of the *current* batch (needed to fetch masks)."""
        self._img_paths = [Path(p) for p in img_paths]

    def register(self, model: nn.Module) -> None:
        """Register forward-hooks on the layers listed in cfg.target_layers."""
        self.clear()
        wanted = {str(i) for i in self.cfg.target_layers}

        if not hasattr(model, "model"):
            raise AttributeError("Expected a YOLO model with attribute `.model`")

        for name, module in model.model.named_modules():
            if name in wanted:
                h = module.register_forward_hook(self._hook_fn(layer_name=name))
                self._handles.append(h)

    def clear(self) -> None:
        """Remove all hooks (call at the end of training/inference)."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._module_cache.clear()

    # context-manager sugar
    def __enter__(self):  # with HookManager(cfg) as hm:
        return self

    def __exit__(self, *exc):
        self.clear()
        return False

    # ───────────────────────── internal ─────────────────────────── #
    def _hook_fn(self, layer_name: str):
        """Build the actual closure that runs at forward-time."""

        def inner(_module: nn.Module, _inp, output):
            feat = output if isinstance(output, torch.Tensor) else output[0]

            # ---------------- mask lookup ---------------- #
            if self._img_paths is None:
                return feat  # nothing we can do

            mask_path = find_mask_path(self.cfg.masks_dir, self._img_paths[0].stem)
            if mask_path is None:
                return feat

            mask = load_mask(mask_path).to(feat.device)
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0), size=feat.shape[-2:], mode="nearest"
            )

            # ---------------- CBAM block (cached) -------- #
            key = (layer_name, feat.shape[1])
            block = self._module_cache.get(key)
            if block is None:
                block = MaskGuidedCBAM(
                    in_channels=feat.shape[1],
                    reduction=self.cfg.reduction_ratio,
                    fusion=self.cfg.mga_pyramid_fusion,
                ).to(feat.device)
                self._module_cache[key] = block

            return block(feat, mask)

        return inner
