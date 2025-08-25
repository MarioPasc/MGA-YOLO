from __future__ import annotations
from typing import Any, Dict, List
import torch

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
        base_names = getattr(self, "loss_names", ["box", "cls", "dfl"])
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
        from mga_yolo.engine.model import MGAModel
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
        """Return MGAValidator for validation so we can also save masks/boxes."""
        from .val import MGAValidator
        # Ensure base loss names exist for val logging
        if not hasattr(self, "loss_names") or not self.loss_names:
            self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return MGAValidator(self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks)

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

            def _sanitize(obj: Any) -> Any:
                # Primitives and None
                if obj is None or isinstance(obj, (str, bool, int, float)):
                    return obj
                # Numeric types (numpy) fallback
                if isinstance(obj, numbers.Number):
                    return float(obj)
                # Paths -> str
                if isinstance(obj, (pathlib.Path, )):
                    return str(obj)
                # Tensors/state dicts are fine as-is (torch handles pickling)
                if isinstance(obj, torch.Tensor):
                    return obj
                # SimpleNamespace / IterableSimpleNamespace -> dict
                if isinstance(obj, SimpleNamespace):
                    return {k: _sanitize(v) for k, v in vars(obj).items()}
                # Dicts
                if isinstance(obj, dict):
                    return {str(k): _sanitize(v) for k, v in obj.items()}
                # Sequences
                if isinstance(obj, (list, tuple, set)):
                    t = type(obj)
                    seq = [_sanitize(v) for v in obj]
                    return seq if t is list else (tuple(seq) if t is tuple else list(seq))
                # Fallback to string to avoid pickling custom classes
                try:
                    return str(obj)
                except Exception:
                    return None
            # Build a torch.save-compatible checkpoint so downstream tools can load it
            from copy import deepcopy
            from mga_yolo.external.ultralytics.ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16
            import io, torch as _torch

            # Build a minimal tensor-only checkpoint (no model objects)
            ema_sd = None
            if getattr(self, "ema", None) is not None and getattr(self.ema, "ema", None) is not None:
                try:
                    ema_sd = deepcopy(self.ema.ema).half().state_dict()
                except Exception:
                    ema_sd = None

            ckpt = {
                "epoch": int(self.epoch),
                "best_fitness": float(self.best_fitness or 0.0),
                "model": None,
                "ema": None,
                "ema_state_dict": ema_sd,
                "model_state_dict": (deepcopy(self.model).float().state_dict() if getattr(self, "model", None) else None),
                "updates": int(getattr(self.ema, "updates", 0)) if getattr(self, "ema", None) else 0,
                "optimizer": None,
                "train_args": _sanitize(vars(self.args)),
                "train_metrics": _sanitize(self.metrics or {}),
            }

            buffer = io.BytesIO()
            _torch.save(ckpt, buffer)
            data = buffer.getvalue()
            self.last.write_bytes(data)
            if self.best_fitness is None or self.best_fitness == ckpt["best_fitness"]:
                self.best.write_bytes(data)
            if (self.save_period > 0) and (self.epoch % self.save_period == 0):
                (self.wdir / f"epoch{self.epoch}.pt").write_bytes(data)
        except Exception as e:
            try:
                LOGGER.warning(f"[MGA] Skipping checkpoint save due to error: {e}")
            except Exception:
                pass
            return

    # --- Logging extensions -------------------------------------------------
    def save_metrics(self, metrics):  # type: ignore[override]
        """Persist Ultralytics results.csv then append our per-epoch MGA loss breakdown to loss_log.csv."""
        super().save_metrics(metrics)
        try:
            if not getattr(self.args, 'save', True):
                return
            if not hasattr(self, 'loss_names') or self.tloss is None:
                return
            import csv
            from pathlib import Path
            save_dir = Path(getattr(self, 'save_dir', '.'))
            csv_path = save_dir / 'loss_log.csv'
            header = ['epoch'] + list(self.loss_names)
            # tloss is the running mean of self.loss_items across the epoch
            vals = self.tloss.detach().cpu().tolist() if hasattr(self.tloss, 'detach') else list(self.tloss)
            # Ensure list length matches header-1; pad or trim defensively
            if isinstance(vals, (int, float)):
                vals = [float(vals)]
            if len(vals) < len(self.loss_names):
                vals = list(vals) + [0.0] * (len(self.loss_names) - len(vals))
            if len(vals) > len(self.loss_names):
                vals = vals[: len(self.loss_names)]
            row = [self.epoch + 1] + [float(x) for x in vals]
            write_header = not csv_path.exists()
            with csv_path.open('a', newline='') as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(header)
                w.writerow(row)
        except Exception:
            pass  # non-critical
    def _log_losses_csv(self) -> None:
        """Append current loss_items tensor (aligned to self.loss_names) to a CSV file under save_dir."""
        try:
            if not getattr(self.args, 'save', True):  # honor save flag
                return
            import csv
            from pathlib import Path
            if not hasattr(self, 'loss_names') or not hasattr(self, 'loss_items'):
                return
            save_dir = Path(getattr(self, 'save_dir', '.'))
            csv_path = save_dir / 'loss_log.csv'
            header = ['epoch'] + list(self.loss_names)
            row = [self.epoch] + [float(x) for x in self.loss_items.detach().cpu().tolist()]
            write_header = not csv_path.exists()
            with csv_path.open('a', newline='') as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(header)
                w.writerow(row)
        except Exception:
            pass  # non-critical

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