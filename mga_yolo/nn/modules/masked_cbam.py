import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Sequence

from mga_yolo.nn.modules.probmaskgater import ProbMaskGater

class MaskCBAM(nn.Module):
    """
    Mask-guided CBAM for feature refinement at FPN levels (P3/P4/P5).

    Components:
      • CAM_m: Masked channel attention (masked avg + masked max pooling).
      • SAM_m: Spatial attention that also ingests the mask as an input plane.
      • α-skip: out = x + α * (refined - x), α=softplus(β) > 0 (learnable).

    Args:
        channels:     input channels C
        r:            reduction ratio for CAM MLP
        spatial_k:    kernel size for SAM conv (odd)
        use_sigmoid_mask: apply sigmoid to mask before use
        tiny_mask_thr: fallback to global pooling if mean(mask) below this
        eps:          numerical stability epsilon

    Forward:
        x:    (B, C, H, W)
        mask: (B, 1, H, W) or (B, H, W). If None → vanilla CBAM behavior.

    Returns:
        (B, C, H, W) refined feature
    """
    def __init__(
        self,
        channels: int,
        r: int = 16,
        spatial_k: int = 7,
        use_sigmoid_mask: bool = True,
        tiny_mask_thr: float = 1e-4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert r > 0 and channels > 0
        self.C = channels
        self.r = r
        self.k = spatial_k if spatial_k % 2 == 1 else spatial_k + 1
        self.use_sigmoid_mask = use_sigmoid_mask
        self.tiny_thr = tiny_mask_thr
        self.eps = eps

        # ---- CAM (MLP shared for avg/max pooled descriptors) ----
        hidden = max(1, channels // r)
        self.cam_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
        )

        # ---- SAM (takes [max(x), avg(x), mask] as 3 channels) ----
        self.sam_conv = nn.Conv2d(3, 1, kernel_size=self.k, padding=self.k // 2, bias=False)

        # ---- α-skip (learnable positive strength) ----
        self.beta = nn.Parameter(torch.zeros((), dtype=torch.float32))  # α=softplus(β)>0
        
        # ---- Mask Gating Mode ----
        MGA_PROB_MODE = os.getenv("MGA_PROB_MODE", False)
        if MGA_PROB_MODE:
            MGA_PROB_APPROACH = os.getenv("MGA_PROB_APPROACH", "gumbel")
            # Curate MGA_PROB_APPROACH to be a literal from accepted types in ProbMaskGater
            if MGA_PROB_APPROACH not in {"deterministic","gumbel","hard_st","bernoulli_detach"}:
                raise ValueError(f"MGA_PROB_APPROACH must be one of "
                                 f"{{'deterministic','gumbel','hard_st','bernoulli_detach'}}, got {MGA_PROB_APPROACH}")
            self.gater = ProbMaskGater(mode=MGA_PROB_APPROACH, 
                                       tau=1.0, 
                                       p_min=0.0, 
                                       threshold=0.5, 
                                       seed=None)


    @staticmethod
    def _ensure_4d_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 3:   # (B,H,W) → (B,1,H,W)
            mask = mask.unsqueeze(1)
        return mask

    def _masked_avg(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # returns (B, C) pooled descriptor
        b, c, h, w = x.shape
        if mask is None:
            return F.adaptive_avg_pool2d(x, 1).view(b, c)
        mask = self._ensure_4d_mask(mask)
        if self.use_sigmoid_mask:
            mask = mask.sigmoid()
        if mask.shape[1] == 1 and c > 1:
            mask = mask.expand(b, c, h, w)
        mean_mask = mask.mean(dim=(2, 3))  # (B, C)
        use_mask = (mean_mask.mean(dim=1) >= self.tiny_thr).to(x.dtype).view(b, 1, 1, 1)
        denom = mask.sum(dim=(2, 3), keepdim=False).clamp_min(self.eps)  # (B, C)
        mavg = (x * mask).sum(dim=(2, 3)) / denom  # (B, C)
        gap = F.adaptive_avg_pool2d(x, 1).view(b, c)
        return mavg * use_mask.view(b, 1) + gap * (1.0 - use_mask.view(b, 1))

    def _masked_max(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # returns (B, C) pooled descriptor
        b, c, h, w = x.shape
        if mask is None:
            return F.adaptive_max_pool2d(x, 1).view(b, c)
        mask = self._ensure_4d_mask(mask)
        if self.use_sigmoid_mask:
            mask = mask.sigmoid()
        if mask.shape[1] == 1 and c > 1:
            mask = mask.expand(b, c, h, w)
        # set invalid locations to very low values before max
        very_low = torch.finfo(x.dtype).min if x.dtype.is_floating_point else -1e9
        x_masked = torch.where(mask > 0.5, x, torch.as_tensor(very_low, dtype=x.dtype, device=x.device))
        mmax = F.adaptive_max_pool2d(x_masked, 1).view(b, c)
        # if all masked-out (still very_low), fallback to GAP for stability
        fallback = F.adaptive_avg_pool2d(x, 1).view(b, c)
        invalid = torch.isclose(mmax, torch.as_tensor(very_low, dtype=mmax.dtype, device=mmax.device))
        return torch.where(invalid, fallback, mmax)

    def _cam(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self._masked_avg(x, mask)   # (B, C)
        mx  = self._masked_max(x, mask)   # (B, C)
        # shared MLP on both, then sum
        y = self.cam_mlp(avg) + self.cam_mlp(mx)
        y = y.view(b, c, 1, 1).sigmoid().to(x.dtype)
        return x * y

    def _sam(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, _, H, W = x.shape
        # channel-wise max/avg → (B,1,H,W)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_avg = torch.mean(x, dim=1, keepdim=True)
        if mask is None:
            m = torch.zeros((b, 1, H, W), dtype=x.dtype, device=x.device)  # learn to ignore if mask absent
        else:
            m = self._ensure_4d_mask(mask)
            if (m.shape[-2], m.shape[-1]) != (H, W):
                m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
            if self.use_sigmoid_mask:
                m = m.sigmoid()
            m = m.to(x.dtype)
        cat = torch.cat([x_max, x_avg, m], dim=1)  # (B,3,H,W)
        att = self.sam_conv(cat).sigmoid().to(x.dtype)
        return x * att

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.beta)

    def forward(self, x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            assert len(x) == 2, "MaskCBAM expects [feature, mask]"
            feat, mask = x
        else:
            feat, mask = x, None
        assert isinstance(feat, torch.Tensor) and feat.dim() == 4
        
        # Apply gate to mask if probabilistic masks are enabled
        if os.getenv("MGA_PROB_MODE", False) and mask is not None:
            mask = self.gater(mask)
        
        # CAM → SAM
        cam_out = self._cam(feat, mask)
        sam_out = self._sam(cam_out, mask)
        # α-modulated residual
        a = self.alpha.to(sam_out.dtype)
        return feat + a * (sam_out - feat)

    def extra_repr(self) -> str:  # pragma: no cover - simple metadata
        return f"channels={self.C}, r={self.r}, spatial_k={self.k}, use_sigmoid_mask={self.use_sigmoid_mask}, tiny_mask_thr={self.tiny_thr}, eps={self.eps}, alpha={self.alpha.item():.4f}"