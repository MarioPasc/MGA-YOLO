"""
Lightweight CBAM module.

Reference: Woo et al., “CBAM: Convolutional Block Attention Module”, ECCV 2018
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Literal, Optional

class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        # Since the spatial dimensions are already reduced to 1×1
        # by the global pooling operations, a 1×1 convolution is
        # mathematically equivalent to a fully connected layer
        # in this context.
        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=mid,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid,
                out_channels=channels,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x + x * att # Skip connection with attention


class _SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = torch.cat((x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)), 1)
        att = torch.sigmoid(self.conv(att))
        return x + x * att # Skip connection with attention


class CBAM(nn.Module):
    """CBAM with optional fusion order."""
    def __init__(
        self,
        channels: int,
        r: int = 16,
        sam_cam_fusion: Literal["sequential", "concat", "add"] = "sequential",
    ):
        super().__init__()
        self.ca = _ChannelAttention(channels, r)
        self.sa = _SpatialAttention()
        self.fusion = sam_cam_fusion
        if self.fusion == "concat":
            self.post = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusion == "sequential":
            return self.sa(self.ca(x))
        cam = self.ca(x)
        sam = self.sa(x)
        if self.fusion == "add":
            return cam + sam
        if self.fusion == "concat":
            return self.post(torch.cat([cam, sam], 1))
        raise ValueError(f"Unknown fusion {self.fusion}")
