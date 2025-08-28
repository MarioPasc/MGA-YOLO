from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
from types import SimpleNamespace

import torch
from torch import Tensor

from mga_yolo.external.ultralytics.ultralytics.nn.tasks import DetectionModel
from mga_yolo.external.ultralytics.ultralytics.nn.modules import Detect
from mga_yolo.nn.modules.segmentation import MGAMaskHead
from mga_yolo.nn.losses.segmentation import SegmentationLoss, SegLossConfig
from mga_yolo.external.ultralytics.ultralytics.utils.loss import v8DetectionLoss

class MGAModel(DetectionModel):
    """DetectionModel extension that returns dict with detection + multi-scale mask logits.

    Implementation: override `_predict_once` (used by base `forward`) to mirror YOLO logic while
    capturing outputs of MGAMaskHead layers. This preserves indexing semantics used by `m.f` and
    `self.save`, avoiding shape mismatches encountered when re-implementing the entire forward.
    """

    def __init__(self, cfg: Union[str, Dict[str, Any]] = "yolov8_mga.yaml", ch: int = 3, nc: int = 80, verbose: bool = True):
        self.mga_mask_indices: List[int] = []
        self.mga_scaled_names: Dict[int, str] = {}
        self.return_dict: bool = False  # disable dict output during DetectionModel initialization
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self._index_mask_heads()
        self.mga_scaled_names = self._assign_scale_names()
        self.return_dict = True

    def _index_mask_heads(self) -> None:
        if MGAMaskHead is None:
            return
        for i, m in enumerate(self.model):
            if isinstance(m, MGAMaskHead):
                # modules built by parse_model have attribute 'i' (their index)
                self.mga_mask_indices.append(getattr(m, 'i', i))

    def _assign_scale_names(self) -> Dict[int, str]:
        ordered = sorted(self.mga_mask_indices)
        return {idx: f"p{3 + k}" for k, idx in enumerate(ordered)}

    # --- Core override ---
    def _predict_once(self, x: Tensor, profile: bool = False, visualize: bool = False, embed=None):  # type: ignore[override]
        from mga_yolo.external.ultralytics.ultralytics.utils.plotting import feature_visualization  # local import for optional dependency
        seg_outs: Dict[str, Tensor] = {}
        y: List[Any] = []  # saved outputs (None for non-saved layers)
        dt = []  # profile timings (ignored unless profile=True)
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        embeddings: List[Tensor] = []
        for m in self.model:
            if m.f != -1:  # from earlier layers
                x_in = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x
            if profile:
                self._profile_one_layer(m, x_in, dt)  # type: ignore[attr-defined]
            x = m(x_in)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in self.mga_mask_indices:
                seg_outs[self.mga_scaled_names.get(m.i, f"mask_{m.i}")] = x
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        if self.return_dict:
            return {"det": x, "seg": seg_outs}
        return x

    # --- Loss override to integrate detection + segmentation during Ultralytics training loop ---
    def _ensure_criteria(self):
        if not hasattr(self, "det_criterion") or self.det_criterion is None:
            # Base detection loss (v8)
            self.det_criterion = v8DetectionLoss(self)
        if not hasattr(self, "seg_criterion") or self.seg_criterion is None:
            # Build seg loss from model.args if available
            a = getattr(self, "args", SimpleNamespace())
            seg_cfg = SegLossConfig(
                bce_weight=getattr(a, "seg_bce_weight", 1.0),
                dice_weight=getattr(a, "seg_dice_weight", 1.0),
                scale_weights=getattr(a, "seg_scale_weights", [1.0, 1.0, 1.0]),
                smooth=getattr(a, "seg_smooth", 1.0),
                loss_lambda=getattr(a, "seg_loss_lambda", 1.0),
                enabled=getattr(a, "seg_enable", True),
            )
            self.seg_criterion = SegmentationLoss(seg_cfg)

    def loss(self, batch, preds=None):  # type: ignore[override]
        """Compute combined detection + segmentation loss and return (loss, loss_items)."""
        self._ensure_criteria()
        img = batch["img"]
        preds = self.predict(img) if preds is None else preds
        # Split predictions
        if isinstance(preds, dict):
            det_preds = preds.get("det")
            seg_preds = preds.get("seg", {})
        else:
            det_preds, seg_preds = preds, {}

        # Detection loss
        det_loss, det_items = self.det_criterion(det_preds, batch)

        # Segmentation loss
        seg_total = det_loss.new_zeros(())
        seg_logs: Dict[str, float] = {}
        if isinstance(seg_preds, dict) and seg_preds and batch.get("masks_multi"):
            seg_total, seg_logs = self.seg_criterion(seg_preds, batch["masks_multi"])  # type: ignore[arg-type]

        total = det_loss + seg_total

        # Compose loss_items aligned with MGATrainer.loss_names
        li: List[float] = det_items.detach().cpu().tolist() if hasattr(det_items, "tolist") else list(det_items)
        for key in ["p3_bce", "p3_dice", "p4_bce", "p4_dice", "p5_bce", "p5_dice"]:
            li.append(float(seg_logs.get(key, 0.0)))
        li.append(float(seg_logs.get("seg_total", float(seg_total.detach().cpu()))))
        loss_items = torch.as_tensor(li, device=total.device)
        return total, loss_items