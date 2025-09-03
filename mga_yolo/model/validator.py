from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import contextlib
import os
import torch
import cv2
import numpy as np
import pandas as pd # type: ignore
import yaml # type: ignore

from mga_yolo.external.ultralytics.ultralytics.models.yolo.detect.val import DetectionValidator
from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER


class MGAValidator(DetectionValidator):
    """
    Validator that captures and saves feature maps during validation.

    Saved tensors and their meanings (per image):
      - 23: MaskECA(P3) output. Refined P3 feature after mask-guided attention.
      - 25: MaskECA(P4) output. Refined P4 feature after mask-guided attention.
      - 27: MaskECA(P5) output. Refined P5 feature after mask-guided attention.
      - 280: Detect inputs[0] = P3 feature as fed into Detect (pre-head).
      - 281: Detect inputs[1] = P4 feature as fed into Detect (pre-head).
      - 282: Detect inputs[2] = P5 feature as fed into Detect (pre-head).

    Environment:
      MGA_SAVE_FM    ∈ {1,true,yes} enables saving. Default off.
      MGA_SAVE_LAYERS comma list of layer indices to hook (e.g., "23,25,27").

    Notes:
      - We always hook Detect’s inputs to guarantee features even if MaskECA
        modules don’t execute in the validator path.
      - Tensors are saved to:  <save_dir>/val_batch0_fm/
        (single folder; no per-batch FM folders).
    """

    def __init__(
        self,
        dataloader: Optional[Any] = None,
        save_dir: Optional[Path] = None,
        args: Optional[Any] = None,
        _callbacks: Optional[Any] = None,
    ) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self._fm_enabled: bool = str(os.getenv("MGA_SAVE_FM", "")).lower() in {"1", "true", "yes"}
        layers_str = os.getenv("MGA_SAVE_LAYERS", "23,25,27").strip()
        self._fm_layers: List[int] = [int(t) for t in layers_str.strip("[]").split(",") if t.strip()]

        self._fm_hooks: List[Any] = []
        self._fm_last: Dict[int, torch.Tensor] = {}
        self._saw_forward: bool = False
        self._registered_ids: Dict[int, int] = {}  # layer -> id(module)
        self._force_once_done: bool = False

        self._logged_once: bool = False
        LOGGER.info(f"[MGAValidator] init fm_enabled={self._fm_enabled} layers={self._fm_layers}")

    # -------------------------- helpers --------------------------

    def _cmp_t(self, a: torch.Tensor, b: torch.Tensor) -> str:
        if a is None or b is None:
            return "missing"
        same_obj = a.data_ptr() == b.data_ptr()
        if a.shape != b.shape:
            return f"shape_diff a{tuple(a.shape)} b{tuple(b.shape)}"
        # upcast to float32 for fair numeric compare
        a32 = a.detach().float().cpu()
        b32 = b.detach().float().cpu()
        diff = (a32 - b32).abs()
        maxd = float(diff.max().item()) if diff.numel() else 0.0
        meand = float(diff.mean().item()) if diff.numel() else 0.0
        return f"same_obj={same_obj} dtype(a,b)=({a.dtype},{b.dtype}) maxΔ={maxd:.3e} meanΔ={meand:.3e}"

    @staticmethod
    def _unwrap(m: Any) -> Any:
        return m.module if hasattr(m, "module") else m

    def _is_main(self) -> bool:
        return int(getattr(self, "rank", 0)) in (0, -1)

    def _to_device_mask(self, x: Any, B: int, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.float()
        # shape to (B,1,H,W)
        if x.ndim == 2:  # (H,W)
            x = x.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        elif x.ndim == 3:  # (B,H,W) or (C,H,W) with C==1 or 1xHxW
            if x.shape[0] == B:
                x = x.unsqueeze(1)
            elif x.shape[0] == 1:
                x = x.unsqueeze(1).repeat(B, 1, 1, 1)
            else:  # assume (C,H,W)
                x = x[:1].unsqueeze(0).repeat(B, 1, 1, 1)
        elif x.ndim == 4:
            pass
        else:
            x = x.view(B, 1, *x.shape[-2:])
        return x.to(device, dtype=dtype, non_blocking=True)

    # -------------------------- lifecycle --------------------------

    def __call__(self, trainer: Optional[Any] = None, model: Optional[Any] = None) -> Dict[str, Any]:
        LOGGER.info("[MGAValidator] __call__ entered")

        self._logged_once = False
        if model is None and trainer is not None:
            model = getattr(trainer, "model", None)
        if model is None:
            raise RuntimeError("[MGAValidator] No model provided to validator")

        self.model = self._unwrap(model)
        if trainer is not None:
            trainer.model = self.model  # ensure same instance
        LOGGER.info(f"[MGAValidator] using model id={id(self.model)} class={type(self.model).__name__}")

        if self._fm_enabled and not self._fm_hooks:
            self._ensure_fm_hooks()


        # in MGAValidator.__call__
        orig_format = getattr(self.args, "format", None)
        try:
            if getattr(self.args, "format", None):
                self.args.format = None  # ensure native PyTorch forward for hooks
            out = super().__call__(trainer=trainer, model=self.model)
        finally:
            if orig_format is not None:
                self.args.format = orig_format

        LOGGER.info("[MGAValidator] __call__ finished")
        return out

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self._fm_enabled:
            self._fm_last.clear()
        batch = super().preprocess(batch)

        # align masks_multi to device/dtype of images for consistent loss
        if "img" in batch:
            B = int(batch["img"].shape[0])
            img_dtype = batch["img"].dtype
            device = self.device
            mm = batch.get("masks_multi", None)
            if mm is not None:
                if isinstance(mm, dict):
                    mm = {k: self._to_device_mask(v, B, img_dtype, device) for k, v in mm.items()}
                elif isinstance(mm, (list, tuple)):
                    mm = [self._to_device_mask(v, B, img_dtype, device) for v in mm]
                else:
                    mm = self._to_device_mask(mm, B, img_dtype, device)
                batch["masks_multi"] = mm

                def _summ(v: Any) -> Any:
                    if v is None:
                        return "None"
                    if isinstance(v, (list, tuple)):
                        return [_summ(t) for t in v]
                    if isinstance(v, dict):
                        return {k: _summ(t) for k, t in v.items()}
                    if torch.is_tensor(v):
                        return (tuple(v.shape), str(v.dtype), str(v.device))
                    return type(v).__name__

                LOGGER.debug(f"[MGAValidator] masks_multi after preprocess: {_summ(batch['masks_multi'])}")
        return batch

    # -------------------------- hooks --------------------------

    def _ensure_fm_hooks(self) -> None:
        if not self._fm_enabled or self._fm_hooks:
            return

        base = self._unwrap(self.model)

        # Proof-of-forward hook on the whole model
        def _top_hook(_m: Any, _in: Any, _out: Any) -> None:
            if not self._saw_forward:
                self._saw_forward = True
                LOGGER.info("[MGAValidator] TOP forward seen.")
        self._fm_hooks.append(base.register_forward_hook(_top_hook))

        # Find sequential graph and modules by YOLO index
        seq = getattr(base, "model", None)
        if seq is None:
            LOGGER.warning("[MGAValidator] BaseModel.model missing; cannot hook.")
            return

        index_to_module: Dict[int, Any] = {getattr(m, "i", i): m for i, m in enumerate(seq) if hasattr(m, "forward")}
        LOGGER.info(
            f"[MGAValidator] Available indices: "
            f"{[(i, type(index_to_module[i]).__name__) for i in sorted(index_to_module)]}"
        )

        detect_idx: Optional[int] = None
        for i in sorted(index_to_module):
            if type(index_to_module[i]).__name__ == "Detect":
                detect_idx = i
                break

        if detect_idx is not None:
            det_mod = index_to_module[detect_idx]
            self._registered_ids[detect_idx] = id(det_mod)

            # PRE-hook: capture P3/P4/P5 BEFORE Detect mutates the list in-place
            def _detect_prehook(_m: Any, _in: Any) -> None:
                xs = _in[0] if isinstance(_in, (tuple, list)) else _in
                if isinstance(xs, (list, tuple)):
                    for si, t in enumerate(xs):
                        if torch.is_tensor(t):
                            key = detect_idx * 10 + si   # 280, 281, 282 if detect_idx==28
                            self._fm_last[key] = t.detach()

            self._fm_hooks.append(det_mod.register_forward_pre_hook(_detect_prehook))
            LOGGER.info(f"[MGAValidator] Hooked Detect PRE-inputs at layer {detect_idx} id={id(det_mod)}")
        else:
            LOGGER.warning("[MGAValidator] Detect layer not found; skipping Detect-input hook.")


        # Optionally hook requested layers (e.g., 23/25/27 = MaskECA at P3/P4/P5)
        registered, missing = [], []
        for li in self._fm_layers:
            mod = index_to_module.get(li)
            if mod is None:
                missing.append(li)
                continue
            self._registered_ids[li] = id(mod)

            def _f_hook(_m: Any, _in: Any, out: Any, idx: int = li) -> None:
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if torch.is_tensor(out):
                    self._fm_last[idx] = out.detach()

            self._fm_hooks.append(mod.register_forward_hook(_f_hook))
            LOGGER.info(f"[MGAValidator] Hooked layer {li}: {type(mod).__name__} id={id(mod)}")
            registered.append(li)

        LOGGER.info(f"[MGAValidator] Registered={registered} Missing={missing}")
        if not registered:
            LOGGER.warning("[MGAValidator] No per-layer hooks registered from MGA_SAVE_LAYERS.")

    # -------------------------- metrics + saving --------------------------

    # -------------------------- saving primitives --------------------------
    def _ensure_fm_tensors(self, batch: Dict[str, Any]) -> None:
        """
        Ensure self._fm_last is populated for the current batch by forcing a native
        forward if hooks did not trigger this step (e.g., scripted backend).
        """
        if self._fm_last:
            return
        imgs = batch.get("img", None)
        if imgs is None:
            return
        try:
            self.model.eval()
            use_amp = bool(getattr(self.args, "half", False)) and (self.device.type == "cuda")
            amp_ctx = torch.cuda.amp.autocast(enabled=use_amp) if use_amp else contextlib.nullcontext()
            with torch.no_grad(), amp_ctx:
                _ = self.model(imgs.to(self.device, non_blocking=True))
            LOGGER.debug("[MGAValidator] forced forward to capture feature maps.")
        except Exception as e:
            LOGGER.warning(f"[MGAValidator] forced forward failed: {e}")

    def _save_feature_maps(self, fm_dir: Path, batch: Dict[str, Any]) -> int:
        """
        Save captured feature maps for each image in batch to fm_dir.
        Returns the number of tensors written.
        """
        fm_dir.mkdir(parents=True, exist_ok=True)
        img_tensor = batch.get("img", torch.empty(0))
        B = int(getattr(img_tensor, "shape", [0])[0]) if hasattr(img_tensor, "shape") else 0
        im_files = batch.get("im_file", [str(i) for i in range(B)])
        n_saved = 0
        for li, t in sorted(self._fm_last.items()):
            if not torch.is_tensor(t) or t.ndim < 3:
                continue
            bsz = t.shape[0]
            for bi in range(min(B, bsz)):
                stem = Path(im_files[bi]).stem if isinstance(im_files, (list, tuple)) and len(im_files) > bi else str(bi)
                out_path = fm_dir / f"{stem}_{li}.pt"
                torch.save({"layer_index": int(li), "shape": tuple(t[bi].shape), "tensor": t[bi].cpu()}, out_path)
                n_saved += 1
        return n_saved

    def _img_tensor_to_bgr(self, t: torch.Tensor) -> np.ndarray:
        """
        Convert a single image tensor (C,H,W), float in [0,1], to uint8 BGR HxWxC.
        """
        if t.ndim != 3:
            raise ValueError(f"expected CHW, got shape={tuple(t.shape)}")
        x = t.detach().float().clamp(0, 1).cpu().numpy()  # CHW, 0..1
        x = (x * 255.0).round().astype(np.uint8)          # CHW, 0..255
        x = np.transpose(x, (1, 2, 0))                    # HWC
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        # tensor is RGB; cv2 expects BGR
        return x[:, :, ::-1]

    def _draw_dets(
        self,
        img_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        cls: np.ndarray,
        conf: np.ndarray,
        names: List[str],
    ) -> np.ndarray:
        """
        Draw axis-aligned boxes over an image. Modifies a copy and returns it.
        """
        out = img_bgr.copy()
        n = boxes_xyxy.shape[0]
        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int).tolist()
            c = int(cls[i]) if i < cls.shape[0] else 0
            s = float(conf[i]) if i < conf.shape[0] else 0.0
            label = f"{names[c] if 0 <= c < len(names) else c}:{s:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            y1t = max(y1, th + 3)
            cv2.rectangle(out, (x1, y1t - th - 4), (x1 + tw + 4, y1t), (0, 255, 0), -1)
            cv2.putText(out, label, (x1 + 2, y1t - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def _save_preds_and_masks(
        self,
        preds_dir: Path,
        batch: Dict[str, Any],
        preds: List[Dict[str, torch.Tensor]],
    ) -> Tuple[int, int]:
        """
        Save, for each image in batch:
          - predicted boxes overlaid on the validation image
          - available masks in self.last_seg as grayscale PNGs
        Returns (#overlay_images_saved, #mask_images_saved).
        """
        preds_dir.mkdir(parents=True, exist_ok=True)
        img_tensor = batch.get("img", torch.empty(0))
        B = int(getattr(img_tensor, "shape", [0])[0]) if hasattr(img_tensor, "shape") else 0
        im_files = batch.get("im_file", [str(i) for i in range(B)])
        n_overlays = 0
        n_masks = 0

        # save overlays using preds in resized image space (same as batch["img"])
        for bi in range(min(B, len(preds))):
            stem = Path(im_files[bi]).stem if len(im_files) > bi else str(bi)
            try:
                img_bgr = self._img_tensor_to_bgr(img_tensor[bi])
                p = preds[bi]
                boxes = p.get("bboxes", torch.zeros((0, 4))).detach().cpu().numpy()
                conf = p.get("conf", torch.zeros((0,))).detach().cpu().numpy()
                cls = p.get("cls", torch.zeros((0,))).detach().cpu().numpy()
                overlay = self._draw_dets(img_bgr, boxes, cls, conf, getattr(self, "names", []))
                cv2.imwrite(str(preds_dir / f"{stem}_pred.jpg"), overlay)
                n_overlays += 1
            except Exception as e:
                LOGGER.debug(f"[MGAValidator] overlay save failed for {stem}: {e}")

        # save masks if present (same tensor layout you used in plot_predictions)
        seg = getattr(self, "last_seg", {})
        if isinstance(seg, dict) and B > 0:
            for bi in range(B):
                stem = Path(im_files[bi]).stem if len(im_files) > bi else str(bi)
                for sk, t in seg.items():
                    try:
                        m = t[bi]
                        if m.ndim == 3 and m.shape[0] == 1:
                            m = m.squeeze(0)
                        m = torch.sigmoid(m).detach().cpu().numpy()
                        if m.ndim == 3:
                            m = m[0]
                        m_img = (m * 255).astype(np.uint8)
                        cv2.imwrite(str(preds_dir / f"{stem}_{sk}.png"), m_img)
                        n_masks += 1
                    except Exception as e:
                        LOGGER.debug(f"[MGAValidator] mask save failed for {stem}/{sk}: {e}")
        return n_overlays, n_masks

     # -------------------------- helpers --------------------------
    def _timepoints(self, total_epochs: int) -> np.ndarray:
        """
        Compute 4 evaluation timepoints at 25%, 50%, 75%, and 100% of training.
        Returns a 1D np.ndarray of epoch indices (1-based).
        """
        if total_epochs <= 0:
            return np.array([], dtype=int)
        q = max(total_epochs // 4, 1)
        return np.arange(q, total_epochs + q, step=q, dtype=int)
    # -------------------------- metrics + saving --------------------------

    def update_metrics(self, preds: Any, batch: Dict[str, Any]) -> None:
        super().update_metrics(preds, batch)

        # ---- save captured tensors to a single folder: val_batch0_fm ----
        save_dir = Path(getattr(self, "save_dir", Path("runs/val/exp")))
        
        # Only save FM maps for the last epoch
        # Which epoch are we in?
        results_csv = pd.read_csv(save_dir / "results.csv") if (save_dir / "results.csv").exists() else None
        current_epoch = int(results_csv["epoch"].max()) if results_csv is not None else 0
        # What is the max epoch?
        args_yaml = yaml.safe_load(open(save_dir / "args.yaml")) if (save_dir / "args.yaml").exists() else {}
        total_epochs = int(args_yaml.get("epochs", 0))
        
        # We are going to save the feature maps in 4 timepoints during training, to examine the evolution. 
        # One time at 25% trained, 50% trained, 75% trained, and 100% trained
        _timepoints = self._timepoints(total_epochs)
        

            
        if int(current_epoch+1) in _timepoints:
            _percentage = (current_epoch + 1) / total_epochs * 100
        
            if not self._logged_once:
                LOGGER.info(f"[MGAValidator] Epoch {current_epoch+1} in {_timepoints}. Saving feature maps at {_percentage}% trained network.")
                self._logged_once = True
                
            if not self._fm_enabled or not self._is_main():
                return

            # ---- enforce FM capture, then save under feature_maps/epoch_k/{fm,preds} ----
            base_dir = save_dir / "feature_maps" / f"epoch_{current_epoch+1}"
            fm_dir = base_dir / "fm"
            preds_dir = base_dir / "preds"

            self._ensure_fm_tensors(batch)
            LOGGER.debug(f"[MGAValidator] captured FM keys: {sorted(self._fm_last.keys())}")
            n_fm = self._save_feature_maps(fm_dir, batch)
            n_ov, n_ms = self._save_preds_and_masks(preds_dir, batch, preds if isinstance(preds, list) else [])
            LOGGER.info(f"[MGAValidator] wrote FM={n_fm} tensors, overlays={n_ov}, masks={n_ms} to {base_dir}")


    # -------------------------- postprocess/plots passthrough --------------------------    
    def postprocess(self, preds):
        if isinstance(preds, dict):
            preds_det = preds.get("det", preds)
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
        # keep default plots only; FM saving now happens in update_metrics
        LOGGER.debug("plot_predictions called for ni=%d (FM saving handled elsewhere).", ni)
        try:
            super().plot_predictions(batch, preds, ni, max_det=max_det)
        except Exception as e:
            LOGGER.debug("plot_predictions failed: %s", e)
        # Optional: save mask previews if self.last_seg exists
        if not getattr(self, "last_seg", None):
            return

        save_dir = Path(getattr(self, "save_dir", Path("runs/val/exp")))
        out_dir = save_dir / f"val_batch{ni}_masks"
        out_dir.mkdir(parents=True, exist_ok=True)

        img_tensor = batch.get("img", torch.empty(0))
        B = int(getattr(img_tensor, "shape", [0])[0]) if hasattr(img_tensor, "shape") else 0
        im_files = batch.get("im_file", [str(i) for i in range(B)])

        for bi in range(B):
            for sk, t in self.last_seg.items():
                try:
                    m = t[bi]
                    if m.ndim == 3 and m.shape[0] == 1:
                        m = m.squeeze(0)
                    m = torch.sigmoid(m).detach().cpu().numpy()
                    if m.ndim == 3:
                        m = m[0]
                    m_img = (m * 255).astype(np.uint8)
                    stem = Path(im_files[bi]).stem if len(im_files) > bi else str(bi)
                    cv2.imwrite(str(out_dir / f"{stem}_{sk}.png"), m_img)
                except Exception as e:
                    LOGGER.debug("Mask preview save failed: %s", e)
