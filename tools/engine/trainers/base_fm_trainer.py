from __future__ import annotations
"""
BaseFMTrainer
- Integrates BaseFMValidator.
- Writes per-component losses to results.csv:
  train/box_loss, train/cls_loss, train/dfl_loss,
  val/box_loss,   val/cls_loss,   val/dfl_loss.

Rationale:
- Training loop maintains a running mean tensor `self.tloss` over batches.
- During val (inside training), validator accumulates component losses and then calls
  `trainer.label_loss_items(val_loss_vector, prefix="val")`.
- Overriding `label_loss_items` controls BOTH train and val CSV columns, while leaving Ultralytics'
  writer (`save_metrics`) unchanged.
"""

from typing import Any, Dict, Iterable, Mapping, Sequence

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER


class BaseFMTrainer(DetectionTrainer):
    """Custom trainer that injects BaseFMValidator and labels loss components for CSV logging."""

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

    # ---------- loss column labeling for CSV ----------

    def label_loss_items(self, loss_items: Any | None = None, prefix: str = "train") -> Dict[str, float] | list[str]:
        """
        Map a detection loss vector to explicit CSV columns.
        Works for both train (`self.tloss`) and val (validator calls with prefix='val').

        Expected for YOLOv8 detection:
            loss_items shape == (3,) in order [box, cls, dfl].

        Behavior:
            - If `loss_items` is None, return the list of column names for header construction.
            - If a scalar or unexpected length, fall back to a single '{prefix}/Loss' column.
        """
        names = [f"{prefix}/box_loss", f"{prefix}/cls_loss", f"{prefix}/dfl_loss"]
        if loss_items is None:
            return names  # header during trainer setup

        # Normalize to a flat list of floats
        try:
            # torch.Tensor -> list
            if hasattr(loss_items, "detach"):
                vals: Sequence[float] = loss_items.detach().cpu().tolist()
            else:
                vals = list(loss_items) if isinstance(loss_items, (list, tuple)) else [float(loss_items)]
        except Exception:
            vals = [float(loss_items)]

        # Detection: 3 components [box, cls, dfl]
        if len(vals) == 3:
            return {names[0]: float(vals[0]), names[1]: float(vals[1]), names[2]: float(vals[2])}

        # Fallbacks for other tasks or unexpected lengths
        if len(vals) == 1:
            return {f"{prefix}/Loss": float(vals[0])}
        # Generic labeling loss_0, loss_1, ...
        return {f"{prefix}/loss_{i}": float(v) for i, v in enumerate(vals)}


    # ---------- CSV enrichment ----------

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

        out.update({
            "train/box":   box,
            "train/dfl":   dfl,
            "train/cls":   cls_,
        })
        return out


    def _collect_val_epoch_losses(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Build a dict of per-epoch VALIDATION losses from the validator metrics dict.
        We support multiple key variants to be robust:
        - box loss: one of ['val/box','val/box_loss','box_loss']
        - cls loss: ['val/cls','val/cls_loss','cls_loss']
        - dfl loss: ['val/dfl','val/dfl_loss','dfl_loss']
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

        return {
            "val/box":   box,
            "val/dfl":   dfl,
            "val/cls":   cls_,
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
