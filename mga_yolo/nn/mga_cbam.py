"""
Mask-Guided CBAM: multiplies features with a binary vessel mask
*before* the CBAM, then fuses the result with the original features.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mga_yolo.nn.cbam import CBAM

from mga_yolo import LOGGER
import logging
LOGGER = logging.getLogger("mga_yolo.nn.mga_cbam")

_FuseOut = Literal["add", "multiply", "concat"]


class MaskGuidedCBAM(nn.Module):
    """
    1. Mask the feature map  Fmasked = F ⊗ M
    2. Attention  F̃ = CBAM(Fmasked)
    3. Fuse with the original F using add / multiply / concat
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        sam_cam_fusion: Literal["sequential", "concat", "add"] = "sequential",
        mga_pyramid_fusion: _FuseOut = "add",
    ) -> None:
        super().__init__()
        self.cbam = CBAM(in_channels, reduction_ratio, sam_cam_fusion)
        self.fuse = mga_pyramid_fusion
        if self.fuse == "concat":
            self.post = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)

    # -------------------------------------------------------- #
    def forward(self, feat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            LOGGER.debug("No mask – applying plain CBAM.")
            att = self.cbam(feat)
            return self._fuse(feat, att)

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)              # B,1,H,W
        mask = F.interpolate(mask.float(), size=feat.shape[-2:], mode="nearest")
        mask = mask.expand_as(feat)               # B,C,H,W

        masked_feat = feat * mask
        att = self.cbam(masked_feat)
        return self._fuse(feat, att)

    # -------------------------------------------------------- #
    def _fuse(self, feat: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        if self.fuse == "add":
            return feat + att
        if self.fuse == "multiply":
            return feat * att
        if self.fuse == "concat":
            return self.post(torch.cat([feat, att], 1))
        raise RuntimeError("Unreachable")