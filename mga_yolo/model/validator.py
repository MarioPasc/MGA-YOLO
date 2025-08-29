from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable
from pathlib import Path

import torch
import numpy as np
import cv2
import os

from mga_yolo.external.ultralytics.ultralytics.models.yolo.detect.val import DetectionValidator


class MGAValidator(DetectionValidator):
    """
    Validator that ignores seg logits for metrics (detection only for now).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Feature-map capture config (validation-time only)
        self._fm_enabled: bool = str(os.getenv("MGA_SAVE_FM", "")).lower() in {"1", "true", "yes"}
        # Allow optional override of layers via env; default to post-MaskECA nodes
        layers_str = os.getenv("MGA_SAVE_LAYERS", "23,25,27").strip()
        self._fm_layers: List[int] = (
            [int(t) for t in layers_str.strip("[]").split(",") if t.strip()]
            if layers_str
            else [23, 25, 27]
        )
        self._fm_hooks: List[Any] = []
        self._fm_last: Dict[int, torch.Tensor] = {}

    def _ensure_fm_hooks(self) -> None:
        """Register forward hooks on selected layers once to capture outputs per batch."""
        if not self._fm_enabled or self._fm_hooks:
            return
        try:
            # Model is Ultralytics BaseModel; .model is nn.Sequential with modules having attribute 'i'
            seq = getattr(self.model, "model", None)
            if seq is None:
                return
            index_to_module: Dict[int, torch.nn.Module] = {}
            for m in seq:
                mi = getattr(m, "i", None)
                if isinstance(mi, int):
                    index_to_module[mi] = m

            for li in self._fm_layers:
                if li not in index_to_module:
                    continue
                mod = index_to_module[li]

                def make_hook(idx: int):
                    def _hook(module: torch.nn.Module, inp: Iterable[torch.Tensor], out: torch.Tensor):
                        try:
                            self._fm_last[idx] = out.detach()
                        except Exception:
                            pass

                    return _hook

                h = mod.register_forward_hook(make_hook(li))
                self._fm_hooks.append(h)
        except Exception:
            # Non-critical; simply disable if anything goes wrong
            self._fm_enabled = False

    def _clear_fm_cache(self) -> None:
        self._fm_last = {}

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
        # Ensure save_dir exists
        save_dir = Path(getattr(self, "save_dir", Path("runs/val/exp")))
        save_dir.mkdir(parents=True, exist_ok=True)

        # Prepare feature map saving for this batch if enabled
        if self._fm_enabled:
            self._ensure_fm_hooks()
            self._clear_fm_cache()

        # Plot standard predictions (bboxes)
        try:
            super().plot_predictions(batch, preds, ni, max_det=max_det)
        except Exception:
            # Non-critical for tests
            pass

        # Save feature maps captured by hooks for this batch
        if self._fm_enabled and self._fm_last:
            fm_dir = save_dir / f"val_batch{ni}_fm"
            fm_dir.mkdir(parents=True, exist_ok=True)

            # Resolve filenames per image
            if isinstance(batch, dict):
                img_tensor = batch.get("img", torch.empty(0))
                B = int(getattr(img_tensor, "shape", [0])[0]) if hasattr(img_tensor, "shape") else 0
                im_files = batch.get("im_file", [str(i) for i in range(B)])
            else:
                B, im_files = 0, []

            for li, t in sorted(self._fm_last.items()):
                try:
                    if not isinstance(t, torch.Tensor) or t.ndim < 3:
                        continue
                    bsz = t.shape[0] if t.ndim >= 3 else 1
                    for bi in range(min(B, bsz)):
                        stem = Path(im_files[bi]).stem if isinstance(im_files, (list, tuple)) and len(im_files) > bi else str(bi)
                        payload = {
                            "layer_index": int(li),
                            "shape": tuple(t[bi].shape),
                            "tensor": t[bi].detach().cpu(),
                        }
                        torch.save(payload, fm_dir / f"{stem}_{li}.pt")
                except Exception:
                    continue

        # Save mask previews if present
        if not getattr(self, "last_seg", None):
            return

        out_dir = save_dir / f"val_batch{ni}_masks"
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(batch, dict):
            img_tensor = batch.get("img", torch.empty(0))
            B = int(getattr(img_tensor, "shape", [0])[0]) if hasattr(img_tensor, "shape") else 0
            im_files = batch.get("im_file", [str(i) for i in range(B)])
        else:
            B, im_files = 0, []

        for bi in range(B):
            # For each scale, save a grayscale mask image
            for sk, t in self.last_seg.items():
                try:
                    m = t[bi]  # (1,H,W) or (C,H,W)
                    if m.ndim == 3 and m.shape[0] == 1:
                        m = m.squeeze(0)
                    m = torch.sigmoid(m).detach().cpu().numpy()
                    if m.ndim == 3:
                        m = m[0]
                    m_img = (m * 255).astype(np.uint8)
                    stem = Path(im_files[bi]).stem if isinstance(im_files, (list, tuple)) and len(im_files) > bi else str(bi)
                    cv2.imwrite(str(out_dir / f"{stem}_{sk}.png"), m_img)
                except Exception:
                    continue