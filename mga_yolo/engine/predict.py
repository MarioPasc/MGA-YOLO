from __future__ import annotations
from typing import Any, Dict

from ultralytics.models.yolo.detect.predict import DetectionPredictor


class MGAPredictor(DetectionPredictor):
    """
    Predictor that returns detections plus segmentation logits (raw).
    Post-processing for seg can be added later.
    """

    def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, dict):
            det = preds["det"]
            seg = preds.get("seg", {})
        else:
            det, seg = preds, {}
        results = super().postprocess(det, img, orig_imgs)
        # Attach seg tensors (raw) per batch element
        for r in results:
            r.mga_masks = {k: v[r.idx].cpu() for k, v in seg.items()}  # type: ignore[attr-defined]
        return results