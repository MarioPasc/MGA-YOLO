from __future__ import annotations
from typing import Any, Dict
import torch

from mga_yolo.external.ultralytics.ultralytics.models.yolo.detect.train import DetectionTrainer

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

    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = super().preprocess_batch(batch)
        # Ensure masks_multi exists (or empty list fallback)
        if "masks_multi" not in batch:
            batch["masks_multi"] = []
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
        """Use parent validator (it will ignore extra seg outputs)."""
        return super().get_validator()

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        # Re-init seg loss if resuming (optional safeguard)
        if not hasattr(self, "seg_loss"):
            self.init_losses()

    # Disable checkpoint serialization if save flag is False (useful for unit tests)
    def save_model(self):  # type: ignore[override]
        if not getattr(self.args, 'save', True):
            return
        return super().save_model()

    # --- Logging extensions -------------------------------------------------
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