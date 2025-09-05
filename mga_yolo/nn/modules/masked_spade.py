"""Mask-conditioned SPADE normalization block for feature refinement.

Implements a lightweight SPADE (Spatially-Adaptive Denormalization) module inspired by:
  Park et al., "Semantic Image Synthesis with Spatially-Adaptive Normalization", CVPR 2019.

Usage pattern in this project:
  - Module takes [feature, mask] as inputs, where feature: (B,C,H,W), mask: (B,K,H,W) or (B,1,H,W)
  - Produces per-channel, per-spatial gamma(m), beta(m) via a tiny conv branch on the mask.
  - Applies normalization on the feature (affine-free) then FiLM-style modulation:
        y = gamma(m) * norm(x) + beta(m)

Notes:
  - If mask spatial size != feature size, it is bilinearly resized to match feature.
  - If mask is missing, falls back to a plain normalization (no modulation).
  - Designed to be placed at P3/P4/P5 FPN levels; we tag a best-effort scale_name for logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MaskSPADE", "MaskSPADEConfig"]


@dataclass
class MaskSPADEConfig:
    channels: int
    hidden: int = 64
    mask_channels: int = 1  # if >1, treated as multi-class mask/features
    norm_type: str = "in"  # "in" (InstanceNorm2d) or "bn" (BatchNorm2d)
    use_sigmoid_mask: bool = True
    eps: float = 1e-6


class MaskSPADE(nn.Module):
    """SPADE block conditioned on a spatial mask.

    Inputs:
        - Either a single tensor x: (B,C,H,W), or a sequence [x, mask].
          mask may be (B,1,H,W), (B,K,H,W), or without channel dim (B,H,W).

    Returns:
        (B,C,H,W) refined feature.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 64,
        mask_channels: int = 1,
        norm_type: str = "in",
        use_sigmoid_mask: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.cfg = MaskSPADEConfig(
            channels=channels,
            hidden=hidden,
            mask_channels=mask_channels,
            norm_type=norm_type,
            use_sigmoid_mask=use_sigmoid_mask,
            eps=eps,
        )

        # Normalization without affine parameters; modulation provides affine.
        self.norm: nn.Module
        if norm_type.lower() == "bn":
            self.norm = nn.BatchNorm2d(channels, affine=False, eps=eps)
        else:
            self.norm = nn.InstanceNorm2d(channels, affine=False, eps=eps)

        # Mask conditioning branch: Conv -> Act -> split heads (gamma, beta)
        in_mc = max(1, mask_channels)
        self.shared = nn.Sequential(
            nn.Conv2d(in_mc, hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv_gamma = nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=True)
        self.conv_beta = nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=True)

        # Best-effort scale tag for logging
        self.scale_name: str = {256: "P3", 512: "P4", 1024: "P5"}.get(channels, f"C{channels}")

        self._initialize()

    def _initialize(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                # affine=False; running stats init handled by nn
                pass

    @staticmethod
    def _prep_mask(mask: torch.Tensor, target_hw: Tuple[int, int], use_sigmoid: bool) -> torch.Tensor:
        if mask.dim() == 3:  # (B,H,W)
            mask = mask.unsqueeze(1)
        b, k, h, w = mask.shape
        H, W = target_hw
        if (h, w) != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
        if use_sigmoid:
            mask = mask.sigmoid()
        return mask

    def forward(self, x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            assert len(x) == 2, "MaskSPADE expects [feature, mask] as inputs"
            feat, mask = x  # type: ignore[assignment]
        else:
            feat, mask = x, None  # type: ignore[assignment]
        assert isinstance(feat, torch.Tensor) and feat.dim() == 4, "feature must be (B,C,H,W)"
        b, c, H, W = feat.shape

        x_hat = self.norm(feat)

        if mask is None:
            # No modulation if mask missing
            return x_hat

        mask = self._prep_mask(mask, (H, W), self.cfg.use_sigmoid_mask)
        # match dtype to parameter dtype for AMP stability
        param_dtype = self.conv_gamma.weight.dtype
        h = self.shared(mask.to(param_dtype))
        gamma = self.conv_gamma(h)
        beta = self.conv_beta(h)

        # Ensure modulation tensors share feature dtype
        gamma = gamma.to(feat.dtype)
        beta = beta.to(feat.dtype)
        y = gamma * x_hat + beta
        return y

    def extra_repr(self) -> str:  # pragma: no cover
        c = self.cfg
        return (
            f"C={c.channels}, hidden={c.hidden}, maskC={c.mask_channels}, norm={c.norm_type}, "
            f"sigmoid_mask={c.use_sigmoid_mask}, scale='{self.scale_name}'"
        )
