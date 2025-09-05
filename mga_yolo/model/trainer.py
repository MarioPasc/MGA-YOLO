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
        super().set_model_attributes()
        if not hasattr(self, "seg_loss"):
            self.init_losses()
        # Lightweight console feedback about MGA specifics
        try:
            m = getattr(self, "model", None)
            mask_info: List[int] = getattr(m, "mga_mask_indices", []) or []
            if mask_info:
                LOGGER.info(f"[MGA] Detected MGAMaskHead layers at indices: {mask_info}")
        except Exception:
            pass

    def init_losses(self) -> None:
        """
        Initialize loss objects.
        Call after model is built. DetectionTrainer usually builds self.compute_loss.
        """
        # Parent prepares detection loss via self.model.init_criterion() internally.
        # We just add segmentation.
        args = self.args
        seg_cfg = SegLossConfig(
            bce_weight=getattr(args, "seg_bce_weight", 1.0),
            dice_weight=getattr(args, "seg_dice_weight", 1.0),
            scale_weights=getattr(args, "seg_scale_weights", [1.0, 1.0, 1.0]),
            smooth=getattr(args, "seg_smooth", 1.0),
            loss_lambda=getattr(args, "seg_loss_lambda", 1.0),
            enabled=getattr(args, "seg_enable", True),
        )
        self.seg_loss = SegmentationLoss(seg_cfg)
        # Console feedback on segmentation loss configuration
        try:
            LOGGER.info(
                "[MGA] SegmentationLoss configured: enabled=%s, bce=%.3f, dice=%.3f, lambda=%.3f, smooth=%.3f, scale_weights=%s",
                seg_cfg.enabled,
                seg_cfg.bce_weight,
                seg_cfg.dice_weight,
                seg_cfg.loss_lambda,
                seg_cfg.smooth,
                list(seg_cfg.scale_weights),
            )
        except Exception:
            pass
        # Extend loss names for logging (order matters)
        # Force canonical detection names to align with MGAModel.loss items: [box, cls, dfl]
        base_names = ["box", "cls", "dfl"]
        extra_names = ["p3_bce", "p3_dice", "p4_bce", "p4_dice", "p5_bce", "p5_dice", "seg_total"]
        # Avoid duplicates if re-init
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
    
    def criterion(self, preds, batch):  # overrides DetectionTrainer.criterion
        """
        Compute combined detection + segmentation loss.
        preds: dict {'det': det_preds, 'seg': {...}} or raw detection output.
        """
        if isinstance(preds, dict):
            det_preds = preds["det"]
            seg_preds = preds.get("seg", {})
        else:
            det_preds = preds
            seg_preds = {}

        # 1. Detection loss (calls parent logic via self.compute_loss)
        det_loss, det_loss_items = self.compute_loss(det_preds, batch)
        # det_loss_items tensor length matches base detection loss_names subset

        # 2. Segmentation loss (if enabled)
        seg_total = torch.zeros_like(det_loss)
        seg_logs: Dict[str, float] = {}
        if isinstance(seg_preds, dict) and seg_preds and batch.get("masks_multi"):
            seg_total, seg_logs = self.seg_loss(seg_preds, batch["masks_multi"])

        total = det_loss + seg_total

        # Assemble logging items aligned to self.loss_names
        # Start with detection components already in det_loss_items
        loss_item_list = det_loss_items.tolist() if hasattr(det_loss_items, "tolist") else list(det_loss_items)
        # Append segmentation components in order (bce/dice per scale + seg_total)
        for key in ["p3_bce", "p3_dice", "p4_bce", "p4_dice", "p5_bce", "p5_dice"]:
            loss_item_list.append(seg_logs.get(key, 0.0))
        loss_item_list.append(seg_logs.get("seg_total", float(seg_total.detach())))

        # Convert back to tensor on same device for trainer logging expectations
        self.loss_items = torch.as_tensor(loss_item_list, device=total.device)
        return total

    def train_step(self, batch: Dict[str, Any]):
        """
        Override to handle dict preds seamlessly while reusing mixed precision, etc.
        """
        with torch.cuda.amp.autocast(enabled=self.amp):
            preds = self.model(batch["img"])
            loss = self.criterion(preds, batch)
        self.scaler.scale(loss).backward()
        return loss
    
    def get_validator(self):
        from mga_yolo.model.validator import MGAValidator
        v = MGAValidator(self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks)
        try:
            v.data = self.data
        except Exception:
            pass
        LOGGER.info("[MGATrainer] Using MGAValidator")
        return v


    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        # Re-init seg loss if resuming (optional safeguard)
        if not hasattr(self, "seg_loss"):
            self.init_losses()

    # Replace checkpoint serialization with a minimal, pickle-safe writer (state_dict-only)
    def save_model(self):  # type: ignore[override]
        try:
            if not getattr(self.args, "save", True):
                return
            # Helper to recursively sanitize objects for pickle/torch.save
            from types import SimpleNamespace
            import pathlib
            import numbers

            def _strip_all_hooks(m: nn.Module) -> None:
                # Remove all forward/pre/backward hooks so the module is picklable
                for sub in m.modules():
                    d = getattr(sub, "_forward_hooks", None)
                    if isinstance(d, dict): d.clear()
                    d = getattr(sub, "_forward_pre_hooks", None)
                    if isinstance(d, dict): d.clear()
                    d = getattr(sub, "_backward_hooks", None)
                    if isinstance(d, dict): d.clear()

            # Build a torch.save-compatible checkpoint so downstream tools can load it
            model_to_save = deepcopy(self.model).half().cpu()
            _strip_all_hooks(model_to_save)

            ema_to_save = None
            if getattr(self, "ema", None) and getattr(self.ema, "ema", None) is not None:
                ema_to_save = deepcopy(self.ema.ema).half().cpu()
                _strip_all_hooks(ema_to_save)

            ckpt = {
                "epoch": int(self.epoch),
                "best_fitness": float(self.best_fitness or 0.0),
                "model": model_to_save,          # full, picklable
                "ema": ema_to_save,              # optional
                "optimizer": (self.optimizer.state_dict() if getattr(self, "optimizer", None) else None),
                "train_args": vars(self.args),
                "train_metrics": self.metrics or {},
            }
            torch.save(ckpt, self.last)
        except Exception as e:
            try:
                LOGGER.warning(f"[MGA] Skipping checkpoint save due to error: {e}")
            except Exception:
                pass
            return

    # --- Logging extensions -------------------------------------------------
    def save_metrics(self, metrics):  # type: ignore[override]
        """
        Write ONE file: results.csv, containing:
        - epoch number
        - TRAIN losses: train/det/{total,box,dfl,cls}, train/seg/{total,p3_bce,p3_dice,p4_bce,p4_dice,p5_bce,p5_dice}
        - VAL losses  : val/det/{total,box,dfl,cls},   val/seg/{total,p3_bce,p3_dice,p4_bce,p4_dice,p5_bce,p5_dice}
        - Plus any additional keys from `metrics` (e.g., precision/recall/mAP, learning rates), preserved as-is.
        """
        # Build the row
        row: Dict[str, float] = {"epoch": float(self.epoch + 1)}

        # TRAIN epoch means from tloss/loss_names
        row.update(self._collect_train_epoch_losses())

        # Static parameters snapshot (per-epoch):
        # - MaskECA alphas, if any MaskECA present
        # - MaskSPADE gamma/beta stats, if any MaskSPADE present
        try:
            row.update(self._collect_alpha_params())
        except Exception:
            pass
        try:
            row.update(self._collect_gamma_beta_params())
        except Exception:
            pass

        # VAL epoch metrics returned by validator
        if isinstance(metrics, dict):
            row.update(self._collect_val_epoch_losses(metrics))
            # Also preserve any additional metrics the validator produced (e.g., 'metrics/mAP50(B)')
            # without overwriting our keys.
            for k, v in metrics.items():
                if k not in row:
                    try:
                        row[k] = float(v)
                    except Exception:
                        # keep only numeric values in results.csv
                        continue

        # Write results.csv (append; create header on first write)
        try:
            import csv
            from pathlib import Path
            save_dir = Path(getattr(self, "save_dir", "."))
            csv_path = save_dir / "results.csv"

            # Stable header order: epoch, TRAIN..., VAL..., ALPHAS..., then any extra metrics
            train_keys = [
                "train/det/total", "train/det/box", "train/det/dfl", "train/det/cls",
                "train/seg/total",
                "train/seg/p3_bce", "train/seg/p3_dice",
                "train/seg/p4_bce", "train/seg/p4_dice",
                "train/seg/p5_bce", "train/seg/p5_dice",
            ]
            val_keys = [
                "val/det/total", "val/det/box", "val/det/dfl", "val/det/cls",
                "val/seg/total",
                "val/seg/p3_bce", "val/seg/p3_dice",
                "val/seg/p4_bce", "val/seg/p4_dice",
                "val/seg/p5_bce", "val/seg/p5_dice",
            ]
            alpha_keys = ["alpha_P3", "alpha_P4", "alpha_P5"]
            spade_keys = [
                "spade/P3/gamma_mean", "spade/P3/gamma_std", "spade/P3/beta_mean", "spade/P3/beta_std",
                "spade/P4/gamma_mean", "spade/P4/gamma_std", "spade/P4/beta_mean", "spade/P4/beta_std",
                "spade/P5/gamma_mean", "spade/P5/gamma_std", "spade/P5/beta_mean", "spade/P5/beta_std",
            ]
            fixed = ["epoch"] + train_keys + val_keys + alpha_keys + spade_keys
            extras = [k for k in row.keys() if k not in fixed]
            header = fixed + sorted(extras)

            write_header = not csv_path.exists()
            with csv_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                # Ensure all header fields exist in row (fill missing with 0.0)
                complete_row = {h: row.get(h, 0.0) for h in header}
                w.writerow(complete_row)
        except Exception as e:
            try:
                LOGGER.warning(f"[MGA] Failed writing results.csv: {e}")
            except Exception:
                pass

    def _log_losses_csv(self) -> None:
        # Deprecated: we now write everything into results.csv
        return


    def after_epoch(self):  # type: ignore[override]
        super().after_epoch()
        self._log_losses_csv()

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
