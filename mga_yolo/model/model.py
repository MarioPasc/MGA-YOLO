from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER

import torch
from torch import Tensor

from mga_yolo.external.ultralytics.ultralytics.nn.tasks import DetectionModel
from mga_yolo.nn.modules.segmentation import MGAMaskHead

class MGAModel(DetectionModel):
    # Class-level attribute annotations for type checkers
    det_criterion: Any
    seg_criterion: Any
    mtl_log_vars: torch.nn.Parameter
    """DetectionModel extension that returns dict with detection + multi-scale mask logits.

    Implementation: override `_predict_once` (used by base `forward`) to mirror YOLO logic while
    capturing outputs of MGAMaskHead layers. This preserves indexing semantics used by `m.f` and
    `self.save`, avoiding shape mismatches encountered when re-implementing the entire forward.
    """

    def __init__(
        self,
        cfg: Union[str, Dict[str, Any]] = "yolov8_mga.yaml",
        ch: int = 3,
        nc: int = 80,
        verbose: bool = True,
    ) -> None:
        # state used by MGA
        self.mga_mask_indices: List[int] = []
        self.mga_scaled_names: Dict[int, str] = {}
        # criteria are built lazily in _ensure_criteria()
        self.det_criterion = None
        self.seg_criterion = None
        # During base DetectionModel init we don't want dict outputs
        self.return_dict = False
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self._index_mask_heads()
        self.mga_scaled_names = self._assign_scale_names()
        self.return_dict = True
        
    # --- Core override ---
    def _predict_once(self, x: Tensor, profile: bool = False, visualize: bool = True, embed=None) -> dict[str, torch.Tensor] | list[torch.Tensor] | torch.Tensor: # type: ignore[override]
        # local import for optional dependency
        from mga_yolo.external.ultralytics.ultralytics.utils.plotting import feature_visualization
        from pathlib import Path

        seg_outs: Dict[str, Tensor] = {}
        y: List[Any] = []  # saved outputs (None for non-saved layers)
        dt: List[float] = []  # profile timings (ignored unless profile=True)
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
                try:
                    # match Ultralytics behavior; allow any type for save_dir
                    sd: Any = visualize  # type: ignore[assignment]
                    feature_visualization(x, m.type, m.i, save_dir=sd)
                except Exception:
                    pass
            if m.i in self.mga_mask_indices:
                seg_outs[self.mga_scaled_names.get(m.i, f"mask_{m.i}")] = x
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        LOGGER.debug(
            "[MGA][_predict_once] return_dict=%s det_type=%s seg_keys=%s",
            self.return_dict,
            type(x).__name__,
            list(seg_outs.keys()),
        )
        if self.return_dict:
            return {"det": x, "seg": seg_outs}
        return x

    # --- Loss override to integrate detection + segmentation during Ultralytics training loop ---
    def _ensure_criteria(self) -> None:
        """
        Build detection+segmentation criteria once and attach Kendall log-variance
        parameters on the model for 2 tasks: [det, seg].
        """
        from types import SimpleNamespace
        import torch
        from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER
        if not hasattr(self, "det_criterion") or self.det_criterion is None:
            from mga_yolo.external.ultralytics.ultralytics.utils.loss import v8DetectionLoss
            self.det_criterion = v8DetectionLoss(self)
            LOGGER.info(f"[MGA] det_criterion created with initial device={self.det_criterion.device}")
        if not hasattr(self, "seg_criterion") or self.seg_criterion is None:
            from mga_yolo.nn.losses.segmentation import SegmentationLoss, SegLossConfig
            args = getattr(self, "args", SimpleNamespace())
            cfg = SegLossConfig(
                bce_weight=getattr(args, "bce_weight", 1.0),
                dice_weight=getattr(args, "dice_weight", 1.0),
                scale_weights=getattr(args, "scale_weights", [1.0, 1.0, 1.0]),
                smooth=getattr(args, "smooth", 1.0),
                loss_lambda=getattr(args, "loss_lambda", 1.0),
                enabled=getattr(args, "enabled", True),
                use_unified_focal=getattr(args, "use_unified_focal", False),
                ufl_lambda=getattr(args, "ufl_lambda", 0.5),
                ufl_delta=getattr(args, "ufl_delta", 0.6),
                ufl_gamma=getattr(args, "ufl_gamma", 0.5),
            )
            self.seg_criterion = SegmentationLoss(cfg)
            LOGGER.info(f"[MGA] seg_criterion created (enabled={cfg.enabled})")
        if not hasattr(self, "mtl_log_vars"):
            # s = log(σ^2); initialize to 0 → weight e^{-s}=1
            self.mtl_log_vars = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def loss(self, batch: dict, preds=None):
        """
        Ultralytics contract in train/val: loss(batch, preds=None|postprocessed).
        If preds are postprocessed (list of dicts), recompute raw head outputs.
        Applies det+seg with Kendall weighting. Returns (total_loss, loss_items).
        """
        import torch
        from typing import Any, Dict, List
        from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER

        self._ensure_criteria()

        # ---- helper: identify raw vs postprocessed preds -----------------------
        def _first_tensor(obj):
            if isinstance(obj, torch.Tensor):
                return obj
            if isinstance(obj, (list, tuple)):
                for o in obj:
                    t = _first_tensor(o)
                    if t is not None:
                        return t
            if isinstance(obj, dict):
                for v in obj.values():
                    t = _first_tensor(v)
                    if t is not None:
                        return t
            return None

        def _looks_postprocessed(p) -> bool:
            # Typical val-time preds: List[Dict] with keys {'bboxes','conf','cls',...}
            if isinstance(p, list) and p and isinstance(p[0], dict):
                k = set(p[0].keys())
                return bool({"bboxes", "conf", "cls"} & k)
            return False

        # ---- normalize/produce predictions -------------------------------------
        if preds is None or _looks_postprocessed(preds):
            # Train path gives preds=None; Val path gives postprocessed preds → recompute raw
            imgs = batch["img"]
            preds = self._predict_once(imgs)

        if isinstance(preds, dict):
            det_preds = preds.get("det", None)
            seg_preds = preds.get("seg", {}) if isinstance(preds.get("seg", {}), dict) else {}
            if det_preds is None:
                for k in ("pred", "preds", "out", "det_out"):
                    if k in preds:
                        det_preds = preds[k]
                        break
            if det_preds is None:
                raise RuntimeError("[MGA] det_preds missing in forward outputs.")
        else:
            det_preds, seg_preds = preds, {}

        # ---- device sync for loss internals ------------------------------------
        ref_t = _first_tensor(det_preds)
        if ref_t is None:
            raise RuntimeError("[MGA] Could not find a tensor in det_preds.")
        dev = ref_t.device

        crit = self.det_criterion
        try:
            crit.device = dev
            if hasattr(crit, "proj"):      crit.proj = crit.proj.to(dev)
            if hasattr(crit, "stride"):    crit.stride = crit.stride.to(dev)
            if hasattr(crit, "bbox_loss"): crit.bbox_loss = crit.bbox_loss.to(dev)
        except Exception as e:
            LOGGER.info(f"[MGA] det_criterion device sync warning: {e}")
        try:
            self.seg_criterion.to(dev)
        except Exception:
            pass

        # ---- compute losses -----------------------------------------------------
        det_loss, det_items = self.det_criterion(det_preds, batch)

        seg_total = det_loss.new_zeros(())
        seg_logs: Dict[str, float] = {}
        if isinstance(seg_preds, dict) and seg_preds and batch.get("masks_multi"):
            seg_total, seg_logs = self.seg_criterion(seg_preds, batch["masks_multi"])  # type: ignore[arg-type]

        # Kendall: L = e^{-s_det} L_det + s_det + e^{-s_seg} L_seg + s_seg
        s_det, s_seg = self.mtl_log_vars[0], self.mtl_log_vars[1]
        total = torch.exp(-s_det) * det_loss + s_det + torch.exp(-s_seg) * seg_total + s_seg

        # ---- pack loss_items for progress bar ----------------------------------
        li: List[float] = det_items.detach().cpu().tolist() if hasattr(det_items, "tolist") else list(det_items)
        for key in ("p3_bce", "p3_dice", "p4_bce", "p4_dice", "p5_bce", "p5_dice"):
            li.append(float(seg_logs.get(key, 0.0)))
        li.append(float(seg_logs.get("seg_total", float(seg_total.detach().cpu()))))
        loss_items = torch.as_tensor(li, device=total.device)
        return total, loss_items

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
