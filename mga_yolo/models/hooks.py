"""
HookManager — injects MaskGuidedCBAM *without touching the backbone*.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List

import torch.nn as nn

from mga_yolo.layers.mga_cbam import MaskGuidedCBAM
from mga_yolo.utils.mask_io import find_mask_path, load_mask
from ..cfg.defaults import MGAConfig


class HookManager:
    def __init__(self, cfg: MGAConfig):
        self.cfg = cfg
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._img_paths: List[Path] | None = None

    # ───────────────────────── public API ───────────────────────── #
    def set_batch_paths(self, img_paths: Iterable[str | Path]) -> None:
        self._img_paths = [Path(p) for p in img_paths]

    def register(self, model: nn.Module) -> None:
        for idx in self.cfg.target_layers:
            layer = model.model[idx]      # Ultralytics keeps backbone in .model
            self._handles.append(layer.register_forward_hook(self._hook_fn(idx)))

    def clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ───────────────────────── context manager ──────────────────── #
    def __enter__(self):  # → with HookManager(cfg) as hm:
        return self

    def __exit__(self, *exc):
        self.clear()
        return False

    # ───────────────────────── internal ─────────────────────────── #
    def _hook_fn(self, idx: int):
        mga = None  # lazy init inside closure to keep weights on correct device

        def inner(module: nn.Module, x, y):
            nonlocal mga
            feat = y if isinstance(y, torch.Tensor) else y[0]
            if mga is None:
                mga = MaskGuidedCBAM(
                    feat.shape[1],
                    reduction=self.cfg.reduction,
                    fusion=self.cfg.fusion_mode,
                ).to(feat.device)
            # pick corresponding mask for *first* image in batch
            if self._img_paths is None:
                raise RuntimeError("set_batch_paths must be called before forward pass")
            mask_path = find_mask_path(self.cfg.masks_dir, self._img_paths[0].stem)
            if mask_path is None:
                return feat
            mask = load_mask(mask_path).to(feat.device)
            return mga(feat, mask)

        return inner
