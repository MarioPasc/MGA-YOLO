from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import contextlib
import os
import torch

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

        out = super().__call__(trainer=trainer, model=self.model)
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

                LOGGER.info(f"[MGAValidator] masks_multi after preprocess: {_summ(batch['masks_multi'])}")
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

    def update_metrics(self, preds: Any, batch: Dict[str, Any]) -> None:
        super().update_metrics(preds, batch)
        if not self._fm_enabled or not self._is_main():
            return

        # Diagnostics
        now_ids: Dict[int, int] = {}
        try:
            seq = getattr(self.model, "model", None)
            if seq is not None:
                for li in self._fm_layers:
                    if li < len(seq):
                        now_ids[getattr(seq[li], "i", li)] = id(seq[li])
        except Exception:
            pass
        LOGGER.info(
            f"[MGAValidator] diag saw_forward={self._saw_forward} "
            f"fm_keys={sorted(self._fm_last.keys())} ids_reg={self._registered_ids} ids_now={now_ids}"
        )

        # Emergency: force one forward once to populate hooks if empty
        if not self._fm_last and not self._force_once_done:
            imgs = batch.get("img", None)
            if imgs is not None:
                try:
                    try:
                        p = next(self.model.parameters())
                        LOGGER.info(
                            f"[MGAValidator] force fwd: imgs.dtype={imgs.dtype} dev={imgs.device} "
                            f"model.dtype={p.dtype} dev={p.device} amp_half={getattr(self.args,'half', False)}"
                        )
                    except Exception:
                        pass
                    self.model.eval()
                    use_amp = bool(getattr(self.args, "half", False)) and (self.device.type == "cuda")
                    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp) if use_amp else contextlib.nullcontext()
                    with torch.no_grad(), amp_ctx:
                        _ = self.model(imgs.to(self.device, non_blocking=True))
                    self._force_once_done = True
                    LOGGER.info("[MGAValidator] forced one forward to capture FMs.")
                except Exception as e:
                    LOGGER.warning(f"[MGAValidator] forced forward failed: {e}")

        # ---- save captured tensors to a single folder: val_batch0_fm ----
        save_dir = Path(getattr(self, "save_dir", Path("runs/val/exp")))
        out_dir = save_dir / "val_batch0_fm"
        out_dir.mkdir(parents=True, exist_ok=True)

        B = int(batch.get("img", torch.empty(0)).shape[0]) if "img" in batch else 0
        im_files = batch.get("im_file", [str(i) for i in range(B)])

        n_saved = 0
        LOGGER.info(f"[MGAValidator] captured keys this batch: {sorted(self._fm_last.keys())}")
        for li, t in sorted(self._fm_last.items()):
            if not torch.is_tensor(t) or t.ndim < 3:
                continue
            bsz = t.shape[0]
            for bi in range(min(B, bsz)):
                stem = Path(im_files[bi]).stem if isinstance(im_files, (list, tuple)) and len(im_files) > bi else str(bi)
                out_path = out_dir / f"{stem}_{li}.pt"
                torch.save(
                    {"layer_index": int(li), "shape": tuple(t[bi].shape), "tensor": t[bi].cpu()},
                    out_path,
                )
                n_saved += 1
        LOGGER.info(f"[MGAValidator] saved {n_saved} tensors to {out_dir}")

        # Diagnostics: check MaskECA vs Detect inputs
        pairs = [(23, 280), (25, 281), (27, 282)]
        for la, lb in pairs:
            ta = self._fm_last.get(la, None)
            tb = self._fm_last.get(lb, None)
            if isinstance(ta, torch.Tensor) and isinstance(tb, torch.Tensor):
                # compare per image 0 to keep it light
                try:
                    msg = self._cmp_t(ta[0], tb[0])
                except Exception as _:
                    msg = "compare_failed"
                LOGGER.info(f"[MGAValidator] compare {la} vs {lb}: {msg}")
            else:
                LOGGER.info(f"[MGAValidator] compare {la} vs {lb}: missing (keys={sorted(self._fm_last.keys())})")


    # -------------------------- postprocess/plots passthrough --------------------------

    
    # ---------- post/plot ----------
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
