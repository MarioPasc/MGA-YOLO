from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MGA_OVERRIDES = {
    "seg_enable": True,
    "seg_bce_weight": 1.0,
    "seg_dice_weight": 1.0,
    "seg_scale_weights": [1.0, 1.0, 1.0],
    "seg_loss_lambda": 1.0,
    "seg_smooth": 1.0,
}

@dataclass
class SegLossConfig:
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    scale_weights: Sequence[float] = (1.0, 1.0, 1.0)
    smooth: float = 1.0
    loss_lambda: float = 1.0  # global multiplier
    enabled: bool = True


class SegmentationLoss(nn.Module):
    """
    Computes multi-scale segmentation loss (BCE + soft Dice) for mask logits.

    Expects:
        preds: dict {'p3': (B,1,H/8,W/8), 'p4': ..., 'p5': ...}
        targets: list[Tensor] length 3 aligned to strides (P3,P4,P5)
                 each (B,1,H/stride,W/stride) or (B,H,W)
    """

    def __init__(self, cfg: SegLossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.cfg.enabled:
            return torch.zeros((), device=next(iter(preds.values())).device), {}

        # Ordered scale mapping
        scale_keys = ["p3", "p4", "p5"]
        losses: Dict[str, float] = {}
        total = torch.zeros((), device=next(iter(preds.values())).device)

        for i, sk in enumerate(scale_keys):
            if sk not in preds or i >= len(targets):
                continue
            pred = preds[sk]  # (B,1,H,W)
            tgt = targets[i]
            if tgt.dim() == 3:
                tgt = tgt.unsqueeze(1)
            # Resize gt if minor mismatch (safety)
            if tgt.shape[-2:] != pred.shape[-2:]:
                tgt = F.interpolate(tgt.float(), size=pred.shape[-2:], mode="nearest")

            bce_loss = self.bce(pred, tgt.float())
            probs = torch.sigmoid(pred)
            intersection = (probs * tgt).sum(dim=(1, 2, 3))
            denom = probs.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + self.cfg.smooth
            dice_loss = 1.0 - (2.0 * intersection + self.cfg.smooth) / denom
            dice_loss = dice_loss.mean()

            w_scale = self.cfg.scale_weights[i] if i < len(self.cfg.scale_weights) else 1.0
            combined = w_scale * (self.cfg.bce_weight * bce_loss + self.cfg.dice_weight * dice_loss)
            total = total + combined

            losses[f"{sk}_bce"] = float(bce_loss.detach())
            losses[f"{sk}_dice"] = float(dice_loss.detach())
            losses[f"{sk}_combined"] = float(combined.detach())

        total = total * self.cfg.loss_lambda
        losses["seg_total"] = float(total.detach())
        return total, losses