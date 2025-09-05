from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER

@dataclass
class SegLossConfig:
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    scale_weights: Sequence[float] = (1.0, 1.0, 1.0)
    smooth: float = 1.0
    loss_lambda: float = 1.0
    enabled: bool = True
    # Unified Focal Loss toggles
    use_unified_focal: bool = False
    ufl_lambda: float = 0.5   # λ in LsUF
    ufl_delta: float = 0.6    # δ balance
    ufl_gamma: float = 0.5    # γ focus

class SegmentationLoss(nn.Module):
    """
    Multi-scale segmentation loss for MGA masks.
    Modes:
      - BCE + soft Dice (default)
      - Unified Focal Loss (UFL), symmetric variant for binary masks
        LsUF = λ·LmF + (1−λ)·LmFT
    """

    def __init__(self, cfg: SegLossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    @staticmethod
    def _dice_probs(probs: torch.Tensor, tgt: torch.Tensor, smooth: float) -> torch.Tensor:
        inter = (probs * tgt).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + smooth
        return 1.0 - (2.0 * inter + smooth) / denom

    @staticmethod
    def _lmf(logits: torch.Tensor, tgt: torch.Tensor, delta: float, gamma: float, eps: float = 1e-6) -> torch.Tensor:
        # Modified focal CE with stability: clamp pt and bases, compute in float32
        x = logits.float()
        t = tgt.float()
        probs = torch.sigmoid(x)
        pt = torch.where(t > 0.5, probs, 1.0 - probs).clamp(eps, 1.0 - eps)
        ce = F.binary_cross_entropy_with_logits(x, t, reduction="none").float()
        w = torch.where(t > 0.5, delta, 1.0 - delta).float()
        base = (1.0 - pt).clamp_min(eps)  # avoid 0^a and exploding d/dpt
        out = (base.pow(1.0 - gamma) * ce * w).mean()

        # debug
        if not torch.isfinite(out):
            from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER
            LOGGER.warning(f"[SegLoss/UFL] _lmf non-finite. "
                           f"logits[min,max]=({x.min().item():.3e},{x.max().item():.3e}) "
                           f"pt[min,max]=({pt.min().item():.3e},{pt.max().item():.3e}) "
                           f"ce_mean={ce.mean().item():.3e}")
        return out

    @staticmethod
    def _lmft(logits: torch.Tensor, tgt: torch.Tensor, delta: float, gamma: float, smooth: float, eps: float = 1e-6) -> torch.Tensor:
        # Modified focal Tversky with stability: guard denominator and base
        x = logits.float()
        t = tgt.float()
        p = torch.sigmoid(x)
        tp = (p * t).sum(dim=(1, 2, 3)).float()
        fn = (t * (1.0 - p)).sum(dim=(1, 2, 3)).float()
        fp = ((1.0 - t) * p).sum(dim=(1, 2, 3)).float()

        denom = (tp + delta * fn + (1.0 - delta) * fp + smooth).clamp_min(eps)
        mti = (tp + smooth) / denom
        base = (1.0 - mti).clamp_min(eps)  # avoid (1-mti)^γ with mti≈1 and γ<1
        out = base.pow(gamma).mean()

        # debug
        if not torch.isfinite(out):
            from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER
            LOGGER.warning(f"[SegLoss/UFL] _lmft non-finite. "
                           f"tp={tp.mean().item():.3e} fn={fn.mean().item():.3e} fp={fp.mean().item():.3e} "
                           f"mti[min,max]=({mti.min().item():.6f},{mti.max().item():.6f})")
        return out

    def forward(self, preds: Dict[str, torch.Tensor], targets: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.cfg.enabled:
            device = next(iter(preds.values())).device
            return torch.zeros((), device=device), {}

        scale_keys = ["p3", "p4", "p5"]
        total = torch.zeros((), device=next(iter(preds.values())).device, dtype=torch.float32)
        logs: Dict[str, float] = {}

        for i, sk in enumerate(scale_keys):
            if sk not in preds or i >= len(targets):
                continue
            pred = preds[sk]  # logits
            tgt = targets[i]
            if tgt.dim() == 3:
                tgt = tgt.unsqueeze(1)
            if tgt.shape[-2:] != pred.shape[-2:]:
                tgt = F.interpolate(tgt.float(), size=pred.shape[-2:], mode="nearest")

            w_scale = self.cfg.scale_weights[i] if i < len(self.cfg.scale_weights) else 1.0

            if self.cfg.use_unified_focal:
                # compute in float32 for stability even under AMP
                pred32 = pred.float()
                tgt32 = tgt.float()

                l_mf  = self._lmf(pred32, tgt32, self.cfg.ufl_delta, self.cfg.ufl_gamma, eps=1e-6)
                l_mft = self._lmft(pred32, tgt32, self.cfg.ufl_delta, self.cfg.ufl_gamma, self.cfg.smooth, eps=1e-6)
                combined = w_scale * (self.cfg.ufl_lambda * l_mf + (1.0 - self.cfg.ufl_lambda) * l_mft)

                # runtime logging
                from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER
                with torch.no_grad():
                    p = torch.sigmoid(pred32)
                    LOGGER.debug(f"[SegmentationLoss] Scale {sk}: "
                                f"logits[min,max]=({pred32.min().item():.3e},{pred32.max().item():.3e}) "
                                f"p[min,max]=({p.min().item():.3e},{p.max().item():.3e}); "
                                f"l_mf={float(l_mf):.4f}, l_mft={float(l_mft):.4f}, combined={float(combined):.4f}")
                    LOGGER.debug(f"[SegmentationLoss] (cfg: λ={self.cfg.ufl_lambda}, δ={self.cfg.ufl_delta}, γ={self.cfg.ufl_gamma})")

                logs[f"{sk}_bce"]  = float(l_mf.detach())
                logs[f"{sk}_dice"] = float(l_mft.detach())
            else:
                bce = self.bce(pred, tgt.float())
                dice = self._dice_probs(torch.sigmoid(pred), tgt.float(), self.cfg.smooth).mean()
                combined = w_scale * (self.cfg.bce_weight * bce + self.cfg.dice_weight * dice)
                logs[f"{sk}_bce"]  = float(bce.detach())
                logs[f"{sk}_dice"] = float(dice.detach())

            if not torch.isfinite(combined):
                from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER
                LOGGER.warning(f"[SegLoss] non-finite at {sk}. Forcing stop."); raise FloatingPointError("Segmentation loss became non-finite.")

            total = total + combined.float()
            logs[f"{sk}_combined"] = float(combined.detach())

        total = total * self.cfg.loss_lambda
        logs["seg_total"] = float(total.detach())
        return total, logs