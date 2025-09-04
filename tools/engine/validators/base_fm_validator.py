from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import contextlib
import os
import torch
import cv2
import numpy as np
import pandas as pd  # type: ignore
import yaml  # type: ignore

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER


class BaseFMValidator(DetectionValidator):
    """
    Feature-map validator compatible with pip Ultralytics and Ultralytics' final_eval() path-based call.

    Outputs per run:
      feature_maps/<epoch_tag>/
        ├─ fm/      # .pt tensors per image+layer
        └─ preds/   # bbox overlays only (no masks)

    Env (prefix BASE_FM_*):
      BASE_FM_SAVE   ∈ {1,true,yes}
      BASE_FM_LAYERS e.g. "15,18,21"   (defaults to 15,18,21)
      BASE_FM_MAX    int >=1           (defaults to 4)
    """

    def __init__(
        self,
        dataloader: Optional[Any] = None,
        save_dir: Optional[Path] = None,
        args: Optional[Any] = None,
        _callbacks: Optional[Any] = None,
    ) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self._fm_enabled: bool = str(os.getenv("BASE_FM_SAVE", "")).lower() in {"1", "true", "yes"}
        layers_str = os.getenv("BASE_FM_LAYERS", "15,18,21").strip()
        self._save_fm_max: int = int(os.getenv("BASE_FM_MAX", "4"))
        self._fm_layers: List[int] = [int(t) for t in layers_str.strip("[]").split(",") if t.strip()]

        self._fm_last: Dict[int, torch.Tensor] = {}
        self._saw_forward: bool = False
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._logged_once: bool = False

        LOGGER.info(f"[BaseFMValidator] init fm_enabled={self._fm_enabled} layers={self._fm_layers}")

    # -------------------------- helpers --------------------------

    @staticmethod
    def _unwrap(m: Any) -> Any:
        return m.module if hasattr(m, "module") else m

    def _is_main(self) -> bool:
        return int(getattr(self, "rank", 0)) in (0, -1)

    # -------------------------- lifecycle --------------------------

    def __call__(self, trainer: Optional[Any] = None, model: Optional[Any] = None) -> Dict[str, Any]:
        """
        Accepts:
          - a torch.nn.Module (normal case),
          - a YOLO object (has .model),
          - a str/Path to weights (Ultralytics final_eval passes Path).
        """
        self._logged_once = False

        # Resolve model
        m = model
        if m is None and trainer is not None:
            m = getattr(trainer, "model", None)

        # If we got a path-like or an object without .forward, try to load via YOLO
        if isinstance(m, (str, os.PathLike, Path)) or not hasattr(m, "forward"):
            try:
                from ultralytics import YOLO
                weights_path = str(m) if m is not None else None
                if weights_path is None:
                    raise RuntimeError("[BaseFMValidator] No model provided")
                y = YOLO(weights_path)
                m = y.model  # torch.nn.Module
                LOGGER.info(f"[BaseFMValidator] Loaded model from weights: {weights_path}")
            except Exception as e:
                raise RuntimeError(f"[BaseFMValidator] Could not load model from {m}: {e}") from e

        self.model = self._unwrap(m)
        if trainer is not None:
            trainer.model = self.model  # keep trainer consistent

        # Register hooks for this validation call
        if self._fm_enabled and not self._hook_handles:
            self._ensure_fm_hooks()

        # Ensure native forward so hooks fire
        orig_format = getattr(self.args, "format", None)
        try:
            if getattr(self.args, "format", None):
                self.args.format = None
            out = super().__call__(trainer=trainer, model=self.model)
        finally:
            if orig_format is not None:
                self.args.format = orig_format

        # Remove all hooks to keep checkpointing pickle-safe
        for h in self._hook_handles:
            with contextlib.suppress(Exception):
                h.remove()
        self._hook_handles.clear()
        return out

    # -------------------------- hooks --------------------------

    def _ensure_fm_hooks(self) -> None:
        if not self._fm_enabled or self._hook_handles:
            return

        base = self._unwrap(self.model)
        if not isinstance(base, torch.nn.Module):
            LOGGER.warning("[BaseFMValidator] Model is not a torch.nn.Module; skipping hooks.")
            return

        # Top-level hook to confirm forward executed
        def _top_hook(_m: Any, _in: Any, _out: Any) -> None:
            self._saw_forward = True

        self._hook_handles.append(base.register_forward_hook(_top_hook))

        # Access sequential modules
        seq = getattr(base, "model", None)
        if seq is None:
            LOGGER.warning("[BaseFMValidator] BaseModel.model missing; cannot hook.")
            return

        index_to_module: Dict[int, Any] = {getattr(m, "i", i): m for i, m in enumerate(seq) if hasattr(m, "forward")}

        # Detect head pre-hook to capture P3/P4/P5 inputs
        detect_idx: Optional[int] = None
        for i in sorted(index_to_module):
            if type(index_to_module[i]).__name__ == "Detect":
                detect_idx = i
                break

        if detect_idx is not None:
            det_mod = index_to_module[detect_idx]

            def _detect_prehook(_m: Any, _in: Any) -> None:
                xs = _in[0] if isinstance(_in, (tuple, list)) else _in
                if isinstance(xs, (list, tuple)):
                    for si, t in enumerate(xs):
                        if torch.is_tensor(t):
                            key = detect_idx * 10 + si  # e.g., 220,221,222 if Detect at 22
                            self._fm_last[key] = t.detach()

            self._hook_handles.append(det_mod.register_forward_pre_hook(_detect_prehook))
            LOGGER.info(f"[BaseFMValidator] Hooked Detect PRE-inputs at layer {detect_idx}")
        else:
            LOGGER.warning("[BaseFMValidator] Detect layer not found; skipping Detect-input hook.")

        # Optional per-layer hooks
        registered, missing = [], []
        for li in self._fm_layers:
            mod = index_to_module.get(li)
            if mod is None:
                missing.append(li)
                continue

            def _f_hook(_m: Any, _in: Any, out: Any, idx: int = li) -> None:
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if torch.is_tensor(out):
                    self._fm_last[idx] = out.detach()

            self._hook_handles.append(mod.register_forward_hook(_f_hook))
            registered.append(li)

        LOGGER.info(f"[BaseFMValidator] Registered layers={registered}, missing={missing}")

    # -------------------------- saving primitives --------------------------

    def _ensure_forward_once(self, batch: Dict[str, Any]) -> None:
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
        except Exception as e:
            LOGGER.debug(f"[BaseFMValidator] forced forward failed: {e}")

    def _save_feature_maps(self, fm_dir: Path, batch: Dict[str, Any]) -> int:
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
        if t.ndim != 3:
            raise ValueError(f"expected CHW, got shape={tuple(t.shape)}")
        x = t.detach().float().clamp(0, 1).cpu().numpy()
        x = (x * 255.0).round().astype(np.uint8)
        x = np.transpose(x, (1, 2, 0))
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        return x[:, :, ::-1]

    def _draw_dets(
        self,
        img_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        cls: np.ndarray,
        conf: np.ndarray,
        names: List[str],
    ) -> np.ndarray:
        out = img_bgr.copy()
        n = boxes_xyxy.shape[0]
        LOGGER.debug("[BaseFMValidator] Drawing %d boxes" % n)
        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int).tolist()
            c = int(cls[i]) if i < cls.shape[0] else 0
            s = float(conf[i]) if i < conf.shape[0] else 0.0
            label = f"{names[c] if 0 <= c < len(names) else c}:{s:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            y1t = max(y1, th + 3)
            cv2.rectangle(out, (x1, y1t - th - 4), (x1 + tw + 4), (0, 255, 0), -1)
            cv2.putText(out, label, (x1 + 2, y1t - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def _save_preds_overlays(
        self,
        preds_dir: Path,
        batch: Dict[str, Any],
        preds: List[Dict[str, torch.Tensor]],
    ) -> int:
        preds_dir.mkdir(parents=True, exist_ok=True)
        img_tensor = batch.get("img", torch.empty(0))
        B = int(getattr(img_tensor, "shape", [0])[0]) if hasattr(img_tensor, "shape") else 0
        im_files = batch.get("im_file", [str(i) for i in range(B)])
        n_overlays = 0

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
                LOGGER.debug(f"[BaseFMValidator] overlay save failed for {stem}: {e}")
        return n_overlays

    def _timepoints(self, total_epochs: int) -> np.ndarray:
        import numpy as _np
        if total_epochs <= 0 or self._save_fm_max <= 0:
            return _np.array([], dtype=int)
        q = max(total_epochs // self._save_fm_max, 1)
        return _np.arange(q, total_epochs + q, step=q, dtype=int)

    # -------------------------- metrics + saving --------------------------

    def update_metrics(self, preds: Any, batch: Dict[str, Any]) -> None:
        super().update_metrics(preds, batch)

        save_dir = Path(getattr(self, "save_dir", Path("runs/val/exp")))
        results_csv = save_dir / "results.csv"
        args_yaml = save_dir / "args.yaml"

        current_epoch: Optional[int] = None
        total_epochs: int = 0

        if results_csv.exists():
            try:
                df = pd.read_csv(results_csv)
                current_epoch = int(df["epoch"].max())
            except Exception:
                current_epoch = None

        if args_yaml.exists():
            try:
                a = yaml.safe_load(open(args_yaml, "r", encoding="utf-8")) or {}
                total_epochs = int(a.get("epochs", 0) or 0)
            except Exception:
                total_epochs = 0

        timepoints = self._timepoints(total_epochs)
        should_save = False

        if total_epochs > 0 and current_epoch is not None:
            if int(current_epoch + 1) in timepoints:
                should_save = True
                if not self._logged_once:
                    LOGGER.info(f"[BaseFMValidator] Saving feature maps at epoch {current_epoch+1}/{total_epochs}")
                    self._logged_once = True
        else:
            should_save = True
            if not self._logged_once:
                LOGGER.info("[BaseFMValidator] No epoch context. Saving feature maps for this validation run.")
                self._logged_once = True

        if not should_save or not self._fm_enabled or not self._is_main():
            return

        epoch_tag = f"epoch_{current_epoch+1}" if current_epoch is not None else "val_only"
        base_dir = save_dir / "feature_maps" / epoch_tag
        fm_dir = base_dir / "fm"
        preds_dir = base_dir / "preds"

        self._ensure_forward_once(batch)
        _ = self._save_feature_maps(fm_dir, batch)
        _ = self._save_preds_overlays(preds_dir, batch, preds if isinstance(preds, list) else [])
