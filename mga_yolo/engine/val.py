from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
import numpy as np
import cv2

from mga_yolo.external.ultralytics.ultralytics.models.yolo.detect.val import DetectionValidator


class MGAValidator(DetectionValidator):
    """
    Validator that ignores seg logits for metrics (detection only for now).
    """

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Extend base preprocess to also move masks_multi to device for seg loss during training-val."""
        batch = super().preprocess(batch)
        try:
            if "masks_multi" not in batch:
                batch["masks_multi"] = []
            else:
                moved = []
                for t in batch["masks_multi"]:
                    try:
                        moved.append(t.to(self.device, non_blocking=True).float())
                    except Exception:
                        moved.append(t.float())
                batch["masks_multi"] = moved
        except Exception:
            # Non-critical in pure-detection mode
            pass
        return batch

    def postprocess(self, preds):
        if isinstance(preds, dict):
            preds_det = preds["det"]
            self.last_seg = preds.get("seg", {})
        else:
            preds_det = preds
            self.last_seg = {}
        return super().postprocess(preds_det)

    def plot_predictions(
        self,
        batch: Dict[str, Any],
        preds: List[Dict[str, torch.Tensor]],
        ni: int,
        max_det: Optional[int] = None,
    ) -> None:
        """Save standard detection plots and also write raw MGA mask previews per image."""
        super().plot_predictions(batch, preds, ni, max_det=max_det)
        # Save mask previews if present
        if not getattr(self, "last_seg", None):
            return
        out_dir = Path(self.save_dir) / f"val_batch{ni}_masks"
        out_dir.mkdir(parents=True, exist_ok=True)
        B = batch["img"].shape[0]
        for bi in range(B):
            # For each scale, save a grayscale mask image
            for sk, t in self.last_seg.items():
                m = t[bi]  # (1,H,W) or (C,H,W)
                if m.ndim == 3 and m.shape[0] == 1:
                    m = m.squeeze(0)
                m = torch.sigmoid(m).detach().cpu().numpy()
                if m.ndim == 3:
                    m = m[0]
                m_img = (m * 255).astype(np.uint8)
                cv2.imwrite(str(out_dir / f"{bi}_{sk}.png"), m_img)