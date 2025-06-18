"""
Inference wrapper for MGA-YOLO.

If `save_feature_maps=True`, we swap the regular HookManager for
VisualizingHookManager, which saves three-panel figures (mask ↓,
feature-map before CBAM, feature-map after CBAM) for every target layer.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from mga_yolo.cfg.defaults import MGAConfig
from mga_yolo.models.hooks import HookManager
from mga_yolo.utils.mask_io import load_mask, find_mask_path
from mga_yolo.nn.mga_cbam import MaskGuidedCBAM
from mga_yolo import LOGGER

# --- Use the vendored fork -------------------------------------------------- #
from mga_yolo.external.ultralytics.ultralytics import YOLO
from mga_yolo.external.ultralytics.ultralytics.engine.trainer import BaseTrainer


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ 1. VisualizingHookManager                                                │
# ╰──────────────────────────────────────────────────────────────────────────╯
class VisualizingHookManager(HookManager):
    """Same core logic as HookManager but also stores (mask, before, after)."""

    def __init__(self, cfg: MGAConfig, save_dir: str | Path):
        super().__init__(cfg)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # one item per (img-stem, layer)
        self._vis_pool: list[dict] = []

    # ───────────────────────── internal override ─────────────────────────── #
    def _hook_fn(self, layer_name: str):
        """Capture feat-maps before/after CBAM and schedule a plot."""
        LOGGER.info(f"Hooking layer {layer_name} for visualisation.")
        def inner(_module, _inp, output):
            feat_before = output if isinstance(output, torch.Tensor) else output[0]

            if self._img_paths is None:  # safety
                return feat_before

            # --- fetch mask (first image in batch) -------------------------- #
            img_stem = Path(self._img_paths[0]).stem
            mpath = find_mask_path(self.cfg.masks_dir, img_stem)
            if mpath is None:
                return feat_before  # no mask → skip

            mask = load_mask(mpath).to(feat_before.device)
            mask_ds = F.interpolate(mask.unsqueeze(0), size=feat_before.shape[-2:], mode="nearest")  # [1,1,H,W]

            # --- run / cache CBAM ------------------------------------------ #
            key = (layer_name, feat_before.shape[1])
            block = self._module_cache.get(key)
            if block is None:
                block = MaskGuidedCBAM(
                    in_channels=feat_before.shape[1],
                    reduction=self.cfg.reduction_ratio,
                    fusion=self.cfg.mga_pyramid_fusion,
                ).to(feat_before.device)
                self._module_cache[key] = block

            feat_after = block(feat_before, mask_ds)

            # --- enqueue visualisation ------------------------------------- #
            with torch.no_grad():
                self._vis_pool.append(
                    dict(
                        stem=img_stem,
                        layer=layer_name,
                        mask=mask_ds.squeeze().cpu(),
                        before=feat_before.mean(1, keepdim=False).squeeze().cpu(),
                        after=feat_after.mean(1, keepdim=False).squeeze().cpu(),
                    )
                )

            return feat_after

        return inner

    # ──────────────────────── visual dump helper ─────────────────────────── #
    def dump_figures(self) -> None:
        """Write PNG figures and clear the pool."""
        LOGGER.info(f"Dumping {len(self._vis_pool)} visualizations to {self.save_dir}")
        if not self._vis_pool:
            return

        for item in self._vis_pool:
            stem = item["stem"]
            layer = item["layer"].replace(".", "-")
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].imshow(item["mask"], cmap="gray")
            axs[0].set_title("Mask ↓")
            axs[1].imshow(item["before"], cmap="viridis")
            axs[1].set_title("Feat before")
            axs[2].imshow(item["after"], cmap="viridis")
            axs[2].set_title("Feat after")

            for ax in axs:
                ax.axis("off")

            fig.suptitle(f"{stem} — layer {layer}")
            out_path = self.save_dir / f"{stem}_layer-{layer}.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        # clear to avoid duplicate plots on next call
        self._vis_pool.clear()


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ 2. MaskGuidedInference                                                   │
# ╰──────────────────────────────────────────────────────────────────────────╯
class MaskGuidedInference:
    def __init__(
        self,
        cfg: MGAConfig,
        weight_path: str | Path,
        save_feature_maps: bool = True,
        feature_dir: str | Path = "feature_maps",
    ):
        self.cfg = cfg
        self.model = YOLO(weight_path)

        # Choose the right hook manager
        if save_feature_maps:
            self.hooks: HookManager = VisualizingHookManager(cfg, feature_dir)
        else:
            self.hooks = HookManager(cfg)

        self.hooks.register(self.model)
        
        LOGGER.info(f"Using {self.hooks.__class__.__name__} for feature visualization.")
        
        # ensure image paths reach the hooks during predict
        def on_batch_end(trainer: BaseTrainer):
            batch = getattr(trainer, "batch", {})
            paths: Sequence[str | Path] = batch.get("im_file", [])
            self.hooks.set_batch_paths(paths)

        self.model.add_callback("on_batch_end", on_batch_end)

    # ──────────────────────────── call ───────────────────────────────────── #
    def __call__(self, images: Sequence[str | Path], **predict_kwargs):
        # let the hook know which images we're about to process
        self.hooks.set_batch_paths(images)

        # run inference
        results = self.model.predict(list(images), **predict_kwargs)

        # dump visualisations if requested
        if isinstance(self.hooks, VisualizingHookManager):
            self.hooks.dump_figures()

        return results
