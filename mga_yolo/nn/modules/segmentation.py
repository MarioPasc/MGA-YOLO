"""MGAMaskHead module.

A lightweight segmentation head used in MGA-YOLO to derive a coarse single-channel
(or multi-channel) mask directly from an FPN feature map without upsampling to the
input image resolution. Each feature level (e.g. P3, P4, P5) gets its own instance
of this module. The produced mask can guide subsequent attention (e.g. MGAAttention)
by highlighting salient spatial regions.

Design goals:
- Minimal computational overhead (few parameters / FLOPs)
- Stable gradients (uses Conv -> Norm -> Activation -> Conv pattern)
- Flexible number of output channels (default = 1 for binary/foreground masks)
- Shape preserving: output spatial dims == input spatial dims

Typical YAML usage example:
    - [16, 1, MGAMaskHead, [256, 64]]  # in_channels=256, hidden=64 -> out_channels=1 (default)

If you want multiple mask channels, pass out_channels > 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MGAMaskHead"]


@dataclass
class MGAMaskHeadConfig:
    """Configuration dataclass for MGAMaskHead.

    Attributes:
        in_channels: Number of channels in the input feature map.
        hidden_channels: Channels in the intermediate representation.
        out_channels: Number of output mask channels (default 1).
        norm: Normalization layer type ("bn", "ln", None).
        act: Activation function constructor (default nn.SiLU).
        dropout: Optional spatial dropout probability after hidden conv.
    """

    in_channels: int
    hidden_channels: int
    out_channels: int = 1
    norm: Optional[str] = "bn"
    act: type = nn.SiLU
    dropout: float = 0.0


class MGAMaskHead(nn.Module):
    """Lightweight coarse mask prediction head.

    Architecture (default):
        Conv1x1(in -> hidden) -> Norm -> Activation -> (Dropout) -> Conv3x3(hidden -> out)

    Notes:
        - No upsampling: outputs are coarse masks aligned with the feature map resolution.
        - Produces logits; apply sigmoid() outside if probabilities are needed.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        norm: Optional[str] = "bn",
        act: type = nn.SiLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cfg = MGAMaskHeadConfig(in_channels, hidden_channels, out_channels, norm, act, dropout)

        layers = []
        # 1x1 projection
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False))
        if norm == "bn":
            layers.append(nn.BatchNorm2d(hidden_channels))
        elif norm == "ln":  # channel-last LayerNorm adaptation
            layers.append(ChannelLastLayerNorm(hidden_channels))
        if act is not None:
            layers.append(act())
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.proj = nn.Sequential(*layers)
        # 3x3 conv to produce logits
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True)

        self._initialize()

    def _initialize(self) -> None:
        # Kaiming init for conv layers and zero bias for stability
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C_in, H, W) -> (B, C_out, H, W)
        feat = self.proj(x)
        mask_logits = self.head(feat)
        return mask_logits

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"in={c.in_channels}, hidden={c.hidden_channels}, out={c.out_channels}, "
            f"norm={c.norm}, act={c.act.__name__ if c.act else None}, dropout={c.dropout}"
        )


class ChannelLastLayerNorm(nn.Module):
    """LayerNorm operating on channel dimension for NCHW by permuting to NHWC temporarily."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.ln(x_perm)
        return x_norm.permute(0, 3, 1, 2)
