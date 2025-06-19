"""
Inference wrapper for MGA-YOLO
─────────────────────────────
If `save_feature_maps=True`, we extend the generic `HookManager` with a
Visualising mix-in that stores mean feature maps **before** and **after**
Mask-Guided CBAM, plus the (down-sampled) mask.

PNG figures are written to `<feature_dir>/<img_stem>_layer-<idx>.png`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from mga_yolo.cfg.defaults import MGAConfig
from mga_yolo.models.hooks import HookManager
from mga_yolo.utils.mask_io import load_mask, find_mask_path
from mga_yolo.nn.mga_cbam import MaskGuidedCBAM  # ← new path
from mga_yolo import LOGGER
import logging
LOGGER = logging.getLogger("mga_yolo.inference")

# vendored fork
from mga_yolo.external.ultralytics.ultralytics import YOLO


# ╭────────────────────────────────────────────────────────────────────╮
# │ VisualisingHookManager                                            │
# ╰────────────────────────────────────────────────────────────────────╯
class VisualisingHookManager(HookManager):
    """Add visual-dump capability on top of the standard HookManager."""

    def __init__(self, cfg: MGAConfig, save_dir: str | Path):
        super().__init__(cfg)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._vis_pool: list[dict] = []

    # ───────────────── hook override ───────────────── #
    def _hook_fn(self, layer_name: str):
        def inner(_module, _inp, output):
            feat_before = output if isinstance(output, torch.Tensor) else output[0]
            if not self._batch_paths:           # no image info
                return feat_before

            img_stem = Path(self._batch_paths[0]).stem
            mpath = find_mask_path(self.cfg.masks_dir, img_stem)
            if mpath is None:
                return feat_before

            
            mask = load_mask(mpath).to(feat_before.device)
            LOGGER.debug(f"Original mask shape: {mask.shape}")
            mask_ds = F.interpolate(mask.unsqueeze(0),
                                size=feat_before.shape[-2:],
                                mode="nearest")
            LOGGER.debug(f"Downsampled mask shape: {mask_ds.shape}")

            key = (layer_name, feat_before.shape[1])
            block = self._module_cache.get(key)
            if block is None:
                LOGGER.debug(f"Before MGA-CBAM: in_channels={feat_before.shape[1]}")

                # This shit working now yay
                block = MaskGuidedCBAM(
                    in_channels=feat_before.shape[1],
                    reduction_ratio=self.cfg.reduction_ratio,
                    sam_cam_fusion=self.cfg.sam_cam_fusion,
                    mga_pyramid_fusion=self.cfg.mga_pyramid_fusion,
                ).to(feat_before.device)
                self._module_cache[key] = block
            LOGGER.debug(f"MGA-CBAM block created with reduction={self.cfg.reduction_ratio}")

            # Before and after block call
            LOGGER.debug(f"Calling block with feat shape {feat_before.shape} and mask shape {mask_ds.shape}")
            feat_after = block(feat_before, mask_ds)
            LOGGER.debug(f"After block call: feat_after shape {feat_after.shape}")
            
            # enqueue for plotting
            with torch.no_grad():
                self._vis_pool.append(
                    dict(
                        stem=img_stem,
                        layer=layer_name,
                        mask=mask_ds.squeeze().cpu(),
                        before=feat_before.mean(1).squeeze().cpu(),
                        after=feat_after.mean(1).squeeze().cpu(),
                    )
                )
            return feat_after

        return inner

    # ───────────── dump figures once per call ───────────── #
    def dump_figures(self) -> None:
        LOGGER.debug("Dumping %d visualisations → %s",
                    len(self._vis_pool), self.save_dir)
        for item in self._vis_pool:
            layer_tag = item["layer"].replace(".", "-")
            fname = f"{item['stem']}_layer-{layer_tag}.png"
            out_path = self.save_dir / fname

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(item["mask"], cmap="gray")
            axs[0].set_title("Mask ↓")
            axs[1].imshow(item["before"], cmap="viridis")
            axs[1].set_title("Feat before")
            axs[2].imshow(item["after"], cmap="viridis")
            axs[2].set_title("Feat after")
            for ax in axs:
                ax.axis("off")
            fig.suptitle(f"{item['stem']} — layer {layer_tag}")
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

        self._vis_pool.clear()


# ╭────────────────────────────────────────────────────────────────────╮
# │ MaskGuidedInference                                               │
# ╰────────────────────────────────────────────────────────────────────╯
class MaskGuidedInference:
    def __init__(
        self,
        cfg: MGAConfig,
        weight_path: str | Path,
        save_feature_maps: bool = False,
        feature_dir: str | Path = "feature_maps",
    ):
        self.cfg = cfg
        self.model = YOLO(weight_path)

        self.hooks: HookManager
        if save_feature_maps:
            LOGGER.debug("Using VisualisingHookManager for feature maps.")
            self.hooks = VisualisingHookManager(cfg, feature_dir)
        else:
            self.hooks = HookManager(cfg)

        self.hooks.register(self.model)

    # ─────────────────────────── call ─────────────────────────── #
    def __call__(self, images: Sequence[str | Path], **predict_kwargs):
        # supply paths so HookManager can fetch masks
        self.hooks.set_batch_paths(images)

        results = self.model.predict(list(images), **predict_kwargs)

        if isinstance(self.hooks, VisualisingHookManager):
            self.hooks.dump_figures()

        return results
