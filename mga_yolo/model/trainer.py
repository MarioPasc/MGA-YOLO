from __future__ import annotations
from typing import Any, Dict, List

from copy import deepcopy
import torch
import torch.nn as nn

from mga_yolo.external.ultralytics.ultralytics.models.yolo.detect.train import DetectionTrainer
from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER, RANK

from mga_yolo.nn.losses.segmentation import SegmentationLoss, SegLossConfig


class MGATrainer(DetectionTrainer):
    """
    Trainer that understands MGAModel forward dict outputs.
    Detection loss only (seg outputs passed through for logging / future loss).
    """

    def set_model_attributes(self) -> None:
        """
        Extend parent setup. Also attach learnable homoscedastic-uncertainty
        parameters (log-variances) to the model so the optimizer sees them.
        """
        super().set_model_attributes()
        # attach 2-task log-variance vector [det, seg] on the model
        import torch, torch.nn as nn
        m = getattr(self, "model", None)
        if m is not None and not hasattr(m, "mtl_log_vars"):
            # s_i = log(sigma_i^2); initialized to 0 => weight ~ 1.0
            m.mtl_log_vars = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        # lightweight feedback on MGA heads
        try:
            mask_info = getattr(m, "mga_mask_indices", []) or []
            if mask_info:
                LOGGER.info(f"[MGA] Detected MGAMaskHead layers at indices: {mask_info}")
        except Exception:
            pass

        base_names = ["box", "cls", "dfl"]
        extra_names = ["p3_bce", "p3_dice", "p4_bce", "p4_dice", "p5_bce", "p5_dice", "seg_total"]
        for n in extra_names:
            if n not in base_names:
                base_names.append(n)
        self.loss_names = base_names


    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Dataset building unchanged, just delegate."""
        return super().build_dataset(img_path, mode, batch)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):  # type: ignore[override]
        """Return an MGAModel so forward outputs include segmentation logits for training/val."""
        from mga_yolo.model.model import MGAModel
        nc = self.data["nc"]
        ch = self.data["channels"]
        model = MGAModel(cfg or self.args.model, nc=nc, ch=ch, verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = super().preprocess_batch(batch)
        # Ensure masks_multi exists and move to device for seg loss
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
        return batch

    def train_step(self, batch):
        return super().train_step(batch)  # rely on model.loss
    
    def get_validator(self):
        from mga_yolo.model.validator import MGAValidator
        v = MGAValidator(self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks)
        try:
            v.data = self.data
        except Exception:
            pass
        LOGGER.info("[MGATrainer] Using MGAValidator")
        return v

    # --- Logging extensions -------------------------------------------------
    def save_metrics(self, metrics):
        """
        Append per-epoch metrics to results.csv and log learned task weights.
        """
        row = {"epoch": float(self.epoch + 1)}
        row.update(self._collect_train_epoch_losses())
        
        try:
            row.update(self._collect_alpha_params())
        except Exception:
            pass
        try:
            row.update(self._collect_gamma_beta_params())
        except Exception:
            pass
        
        if isinstance(metrics, dict):
            row.update(self._collect_val_epoch_losses(metrics))
            for k, v in metrics.items():
                if k not in row:
                    try:
                        row[k] = float(v)
                    except Exception:
                        continue
        # learned uncertainty terms
        try:
            lv = getattr(self.model, "mtl_log_vars", None)
            if isinstance(lv, torch.Tensor) and lv.numel() >= 2:
                row["mtl/sigma2_det"] = float(torch.exp(lv[0]).detach().cpu())
                row["mtl/sigma2_seg"] = float(torch.exp(lv[1]).detach().cpu())
                row["mtl/w_det"] = float(torch.exp(-lv[0]).detach().cpu())
                row["mtl/w_seg"] = float(torch.exp(-lv[1]).detach().cpu())
        except Exception:
            pass

        import csv
        from pathlib import Path
        csv_path = Path(getattr(self, "save_dir", ".")) / "results.csv"
        header_order = [
            "epoch",
            "train/det/total","train/det/box","train/det/dfl","train/det/cls",
            "train/seg/total","train/seg/p3_bce","train/seg/p3_dice","train/seg/p4_bce","train/seg/p4_dice","train/seg/p5_bce","train/seg/p5_dice",
            "val/det/total","val/det/box","val/det/dfl","val/det/cls",
            "val/seg/total","val/seg/p3_bce","val/seg/p3_dice","val/seg/p4_bce","val/seg/p4_dice","val/seg/p5_bce","val/seg/p5_dice",
            "mtl/sigma2_det","mtl/sigma2_seg","mtl/w_det","mtl/w_seg"
        ]
        extras = [k for k in row.keys() if k not in header_order]
        header = header_order + sorted(extras)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = {k: row.get(k, None) for k in header}
            if write_header:
                csv.DictWriter(f, fieldnames=list(w.keys())).writeheader()
            csv.DictWriter(f, fieldnames=list(w.keys())).writerow(w)
    
    def _collect_train_epoch_losses(self) -> Dict[str, float]:
        """
        Build a dict of per-epoch TRAIN losses using Ultralytics' running mean `self.tloss`
        and the order in `self.loss_names`.

        Expected names from your criterion(): ["box","cls","dfl","p3_bce","p3_dice","p4_bce","p4_dice","p5_bce","p5_dice","seg_total"].
        This function is defensive if names are missing or reordered.
        """
        out: Dict[str, float] = {}
        if not hasattr(self, "tloss") or self.tloss is None or not hasattr(self, "loss_names"):
            return out

        # Convert tloss to a name->value mapping
        vals = self.tloss.detach().cpu().tolist() if hasattr(self.tloss, "detach") else list(self.tloss)
        names = list(self.loss_names)
        # pad/trim to align
        if isinstance(vals, (int, float)):
            vals = [float(vals)]
        if len(vals) < len(names):
            vals = list(vals) + [0.0] * (len(names) - len(vals))
        elif len(vals) > len(names):
            vals = vals[:len(names)]
        d = {k: float(v) for k, v in zip(names, vals)}

        # Detection components
        def pick(m: Dict[str, float], *keys: str) -> float:
            for k in keys:
                if k in m:
                    return m[k]
            return 0.0
        
        
        box = pick(d, "box", "box_loss")
        cls_ = pick(d, "cls", "cls_loss")
        dfl = pick(d, "dfl", "dfl_loss")
        det_total = box + cls_ + dfl

        # Segmentation components
        seg_total = d.get("seg_total", 0.0)
        p3_bce  = d.get("p3_bce", 0.0)
        p3_dice = d.get("p3_dice", 0.0)
        p4_bce  = d.get("p4_bce", 0.0)
        p4_dice = d.get("p4_dice", 0.0)
        p5_bce  = d.get("p5_bce", 0.0)
        p5_dice = d.get("p5_dice", 0.0)

        out.update({
            "train/det/total": det_total,
            "train/det/box":   box,
            "train/det/dfl":   dfl,
            "train/det/cls":   cls_,
            "train/seg/total": seg_total,
            "train/seg/p3_bce": p3_bce,
            "train/seg/p3_dice": p3_dice,
            "train/seg/p4_bce": p4_bce,
            "train/seg/p4_dice": p4_dice,
            "train/seg/p5_bce": p5_bce,
            "train/seg/p5_dice": p5_dice,
        })
        return out


    def _collect_val_epoch_losses(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Build a dict of per-epoch VALIDATION losses from the validator metrics dict.
        We support multiple key variants to be robust:
        - box loss: one of ['val/box','val/box_loss','box_loss']
        - cls loss: ['val/cls','val/cls_loss','cls_loss']
        - dfl loss: ['val/dfl','val/dfl_loss','dfl_loss']
        - seg total: ['val/seg_total','val/Loss','seg_total']
        - seg per-scale: ['val/p3_bce', 'p3_bce'], etc.
        """
        def first(*keys: str) -> float:
            for k in keys:
                v = metrics.get(k, None)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        pass
            return 0.0

        # Detection parts
        box = first("val/box", "val/box_loss", "box_loss")
        cls_ = first("val/cls", "val/cls_loss", "cls_loss")
        dfl = first("val/dfl", "val/dfl_loss", "dfl_loss")
        det_total = box + cls_ + dfl

        # Segmentation total (support 'val/Loss' as some validators expose this)
        seg_total = first("val/seg_total", "val/Loss", "seg_total")

        # Segmentation per-scale (K→val/K fallback)
        p3_bce  = first("val/p3_bce",  "p3_bce")
        p3_dice = first("val/p3_dice", "p3_dice")
        p4_bce  = first("val/p4_bce",  "p4_bce")
        p4_dice = first("val/p4_dice", "p4_dice")
        p5_bce  = first("val/p5_bce",  "p5_bce")
        p5_dice = first("val/p5_dice", "p5_dice")

        return {
            "val/det/total": det_total,
            "val/det/box":   box,
            "val/det/dfl":   dfl,
            "val/det/cls":   cls_,
            "val/seg/total": seg_total,
            "val/seg/p3_bce": p3_bce,
            "val/seg/p3_dice": p3_dice,
            "val/seg/p4_bce": p4_bce,
            "val/seg/p4_dice": p4_dice,
            "val/seg/p5_bce": p5_bce,
            "val/seg/p5_dice": p5_dice,
        }

    def _collect_alpha_params(self) -> Dict[str, float]:
        """Return {'alpha_P3': a3, 'alpha_P4': a4, 'alpha_P5': a5} if present, else zeros.

        It inspects MaskECA modules in EMA model if available, otherwise the current model.
        """
        out: Dict[str, float] = {"alpha_P3": 0.0, "alpha_P4": 0.0, "alpha_P5": 0.0}
        try:
            model = self.ema.ema if getattr(self, "ema", None) and getattr(self.ema, "ema", None) is not None else self.model
            if model is None:
                return out
            # Late import to avoid circular imports on trainer init
            from mga_yolo.nn.modules.masked_eca import MaskECA  # type: ignore
            has_any = False
            found = []  # list of tuples (name_or_None, channels_or_None, alpha_value)
            for m in model.modules():
                if isinstance(m, MaskECA):
                    has_any = True
                    name = getattr(m, "scale_name", None)
                    ch = getattr(getattr(m, "cfg", None), "channels", None)
                    try:
                        a = getattr(m, "alpha")  # property tensor
                        alpha_val = float(a.detach().cpu()) if hasattr(a, "detach") else float(a)
                    except Exception:
                        alpha_val = None
                    if alpha_val is not None:
                        found.append((name, ch, alpha_val))

            if not has_any:
                return {"alpha_P3": 0.0, "alpha_P4": 0.0, "alpha_P5": 0.0}

            # First fill by explicit names if any
            for name, ch, alpha_val in found:
                if name in ("P3", "P4", "P5"):
                    out[f"alpha_{name}"] = alpha_val

            # Fallback: if some remain 0.0 and we have 3 values, assign by ascending channels
            remaining_keys = [k for k, v in out.items() if v == 0.0]
            if remaining_keys and len(found) >= 1:
                # sort by ch (None last), then by alpha to ensure deterministic order
                found_sorted = sorted(found, key=lambda t: (float('inf') if t[1] is None else t[1]))
                map_keys = ["alpha_P3", "alpha_P4", "alpha_P5"]
                for (name, ch, alpha_val), key in zip(found_sorted, map_keys):
                    out[key] = alpha_val
        except Exception:
            # keep defaults
            pass
        return out

    def _collect_gamma_beta_params(self) -> Dict[str, float]:
        """Aggregate gamma/beta statistics from MaskSPADE modules if present.

        Returns keys like 'spade/P3/gamma_mean', 'spade/P3/gamma_std', same for beta.
        If absent, returns zeros for P3/P4/P5.
        """
        out: Dict[str, float] = {}
        # initialize zeros to keep headers stable
        def init_scale(scale: str):
            out[f"spade/{scale}/gamma_mean"] = 0.0
            out[f"spade/{scale}/gamma_std"] = 0.0
            out[f"spade/{scale}/beta_mean"] = 0.0
            out[f"spade/{scale}/beta_std"] = 0.0
        for s in ("P3", "P4", "P5"):
            init_scale(s)

        try:
            model = self.ema.ema if getattr(self, "ema", None) and getattr(self.ema, "ema", None) is not None else self.model
            if model is None:
                return out
            from mga_yolo.nn.modules.masked_spade import MaskSPADE  # type: ignore
            any_found = False
            # Collect running stats of conv_gamma/beta weights as a proxy for modulation scale
            for m in model.modules():
                if isinstance(m, MaskSPADE):
                    any_found = True
                    scale = getattr(m, "scale_name", "")
                    if scale not in ("P3", "P4", "P5"):
                        # map by channels if possible
                        ch_val = getattr(getattr(m, "cfg", None), "channels", None)
                        try:
                            ch_key = int(ch_val) if ch_val is not None else -1
                        except Exception:
                            ch_key = -1
                        scale = {256: "P3", 512: "P4", 1024: "P5"}.get(ch_key, "P3")
                    # Use weight stats in absence of per-batch activations
                    gw = m.conv_gamma.weight.detach().float().view(-1)
                    gb = m.conv_beta.weight.detach().float().view(-1)
                    out[f"spade/{scale}/gamma_mean"] = float(gw.mean())
                    out[f"spade/{scale}/gamma_std"] = float(gw.std(unbiased=False))
                    out[f"spade/{scale}/beta_mean"] = float(gb.mean())
                    out[f"spade/{scale}/beta_std"] = float(gb.std(unbiased=False))
            return out if any_found else out
        except Exception:
            return out

    def save_model(self) -> None:
        """
        Save pure-state checkpoints to avoid pickling any classes.
        Produces weights/last.pt and weights/best.pt.
        """
        from mga_yolo.external.ultralytics.ultralytics.utils.patches import torch_save  # Ultralytics wrapper

        (self.save_dir / "weights").mkdir(parents=True, exist_ok=True)
        last_pt = self.save_dir / "weights" / "last.pt"
        best_pt = self.save_dir / "weights" / "best.pt"

        # Gather state
        model_state = self.model.state_dict()
        ema = getattr(self, "ema", None)
        ema_state = (getattr(ema, "ema", None) or ema)
        ema_state = ema_state.state_dict() if ema_state is not None else None
        opt_state = self.optimizer.state_dict() if getattr(self, "optimizer", None) else None

        # Minimal, JSON-serializable metadata
        meta = {
            "epoch": int(getattr(self, "epoch", -1)),
            "best_fitness": float(getattr(self, "best_fitness", 0.0)),
            "imgsz": getattr(self.args, "imgsz", None),
            "overrides": dict(vars(getattr(self.model, "args", type("A", (), {})())))  # to plain dict if present
        }

        ckpt = {
            "model_state": model_state,
            "ema_state": ema_state,
            "optimizer_state": opt_state,
            "metadata_json": __import__("json").dumps(meta),
        }

        torch_save(ckpt, last_pt)
        torch_save(ckpt, best_pt)


    # Match Ultralytics API: print a richer header including our extra loss names
    def progress_string(self) -> str:  # type: ignore[override]
        names = tuple(self.loss_names) if hasattr(self, "loss_names") else ("box_loss", "cls_loss", "dfl_loss")
        return ("\n" + "%11s" * (4 + len(names))) % (
            "Epoch",
            "GPU_mem",
            *names,
            "Instances",
            "Size",
        )

    def final_eval(self):  # type: ignore[override]
        """Run final evaluation using the in-memory model to avoid checkpoint dependency."""
        try:
            model = self.ema.ema if getattr(self, "ema", None) and getattr(self.ema, "ema", None) is not None else self.model
            if model is None or self.validator is None:
                return
            self.validator.args.plots = getattr(self.args, "plots", False)
            self.metrics = self.validator(model=model)
            # Align with Ultralytics: drop 'fitness' if present and trigger callback
            if isinstance(self.metrics, dict):
                self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")
        except Exception as e:
            try:
                LOGGER.warning(f"[MGA] final_eval failed: {e}")
            except Exception:
                pass