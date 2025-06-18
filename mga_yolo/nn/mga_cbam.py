"""
Mask-Guided CBAM: multiplies features with a binary vessel mask
*before* the CBAM, then fuses the result with the original features.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from mga_yolo.nn.cbam import CBAM

_Fusion = Literal["add", "mul", "concat"]


class MaskGuidedCBAM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        fusion: _Fusion = "add",
    ):
        super().__init__()
        self.cbam = CBAM(in_channels, reduction)
        self.fusion: _Fusion = fusion
        if fusion == "concat":
            self.post = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: (B,     1, H, W)  or  (B, H, W)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = torch.nn.functional.interpolate(
            mask.float(), size=feat.shape[-2:], mode="nearest"
        )
        masked = feat * mask                       # B,C,H,W
        attend = self.cbam(masked)                 # B,C,H,W

        if self.fusion == "add":
            return feat + attend
        if self.fusion == "mul":
            return feat * attend
        if self.fusion == "concat":
            return self.post(torch.cat([feat, attend], dim=1))
        raise ValueError(f"Unknown fusion: {self.fusion}")
