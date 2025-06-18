"""
Lightweight CBAM module.

Reference: Woo et al., “CBAM: Convolutional Block Attention Module”, ECCV 2018
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Flatten(2),                       # B, C, HW
            nn.AdaptiveAvgPool1d(1),             # B, C, 1
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.mlp(x).unsqueeze(-1).unsqueeze(-1)   # B, C, 1, 1
        return x * w.sigmoid()


class _SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg, mx = x.mean(1, keepdim=True), x.max(1, keepdim=True).values
        w = self.conv(torch.cat([avg, mx], dim=1)).sigmoid()   # B, 1, H, W
        return x * w


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.ca = _ChannelAttention(in_channels, reduction)
        self.sa = _SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))
