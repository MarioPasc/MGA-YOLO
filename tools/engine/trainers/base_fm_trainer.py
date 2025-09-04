from __future__ import annotations
"""
Minimal custom trainer to integrate feature-map validation and write per-component losses to results.csv.

Adds the columns:
  train/box_loss, train/cls_loss, train/dfl_loss,
  val/box_loss,   val/cls_loss,   val/dfl_loss

Keeps Ultralytics' default behavior for everything else.
"""

from typing import Any, Dict, Mapping, Sequence

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER


class BaseFMTrainer(DetectionTrainer):
    """Custom trainer that injects BaseFMValidator and augments CSV logging."""

    # ---------- validator injection ----------

    def get_validator(self) -> Any:
        from tools.engine.validators.base_fm_validator import BaseFMValidator
        v = BaseFMValidator(self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks)
        try:
            v.data = self.data
        except Exception:
            pass
        LOGGER.info("[BaseFMTrainer] Using BaseFMValidator")
        return v

    # ---------- CSV enrichment ----------

    def _collect_train_epoch_losses(self) -> Dict[str, float]:
        """
        Map running mean `self.tloss` to explicit train losses.

        Expected loss_names contain at least ['box','cls','dfl'] for detection.
        Defensive to missing or reordered names.
        """
        out: Dict[str, float] = {}
        if not hasattr(self, "tloss") or self.tloss is None or not hasattr(self, "loss_names"):
            return out

        names: Sequence[str] = list(getattr(self, "loss_names", []))  # e.g., ('box','cls','dfl',...)
        vals_raw = self.tloss
        try:
            vals: Sequence[float] = vals_raw.detach().cpu().tolist()  # torch.Tensor -> list
        except Exception:
            # already a list/tuple/number
            if isinstance(vals_raw, (float, int)):
                vals = [float(vals_raw)]
            else:
                vals = list(vals_raw)

        # pad/trim to align lengths
        if len(vals) < len(names):
            vals = list(vals) + [0.0] * (len(names) - len(vals))
        elif len(vals) > len(names):
            vals = list(vals[: len(names)])

        d = {k: float(v) for k, v in zip(names, vals)}

        def pick(m: Mapping[str, float], *keys: str) -> float:
            for k in keys:
                if k in m:
                    return float(m[k])
            return 0.0

        box = pick(d, "box", "box_loss")
        cls_ = pick(d, "cls", "cls_loss")
        dfl = pick(d, "dfl", "dfl_loss")

        out.update({
            "train/box_loss": box,
            "train/cls_loss": cls_,
            "train/dfl_loss": dfl,
        })
        return out

    def _collect_val_epoch_losses(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract explicit VALIDATION losses from validator metrics dict.
        Supports multiple key variants for robustness.
        """
        def first(*keys: str) -> float:
            for k in keys:
                v = metrics.get(k, None)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        continue
            return 0.0

        box = first("val/box", "val/box_loss", "box_loss")
        cls_ = first("val/cls", "val/cls_loss", "cls_loss")
        dfl = first("val/dfl", "val/dfl_loss", "dfl_loss")

        return {
            "val/box_loss": box,
            "val/cls_loss": cls_,
            "val/dfl_loss": dfl,
        }

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Augment `metrics` with per-component train/val losses, then defer to Ultralytics writer.
        This preserves default columns (epoch, time, lr/pg*, metrics/*, etc.) and adds:
          train/box_loss, train/cls_loss, train/dfl_loss, val/box_loss, val/cls_loss, val/dfl_loss
        """
        # Inject explicit train losses from running average of the epoch
        train_parts = self._collect_train_epoch_losses()
        metrics.update(train_parts)

        # Inject explicit val losses from validator
        val_parts = self._collect_val_epoch_losses(metrics)
        metrics.update(val_parts)

        # Delegate to default CSV writer (will add columns for any numeric keys present)
        super().save_metrics(metrics)
