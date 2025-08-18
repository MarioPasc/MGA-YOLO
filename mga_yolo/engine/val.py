from __future__ import annotations
from typing import Any, Dict

from ultralytics.models.yolo.detect.val import DetectionValidator


class MGAValidator(DetectionValidator):
    """
    Validator that ignores seg logits for metrics (detection only for now).
    """

    def postprocess(self, preds):
        if isinstance(preds, dict):
            preds_det = preds["det"]
            self.last_seg = preds.get("seg", {})
        else:
            preds_det = preds
            self.last_seg = {}
        return super().postprocess(preds_det)