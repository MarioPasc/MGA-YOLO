from __future__ import annotations
"""
Train with pip Ultralytics using BaseFMTrainer (integrated feature-map validation).

Examples:
  python cli/ultra_train_basefm.py \
    --cfg configs/ultra_defaults.yaml \
    --weights yolov8n.pt \
    --data path/to/data.yaml \
    --imgsz 640 --epochs 100 \
    --project runs/train --name exp_basefm

Notes:
- Add BASE_FM_* keys in your YAML to control FM capture, e.g.:
    BASE_FM_SAVE: true
    BASE_FM_LAYERS: "15,18,21"
    BASE_FM_MAX: 4
- These are exported to environment before training.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore

from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Local utilities
from tools.engine.config_env import apply_env_from_config
from tools.engine.trainers.base_fm_trainer import BaseFMTrainer


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ValueError(f"Config must be a mapping. Got type={type(d)} from {p}")
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ultralytics training with BaseFMTrainer")
    ap.add_argument("--cfg", required=True, help="Path to YAML with Ultralytics overrides")
    ap.add_argument("--weights", default="yolov8n.pt", help="Initial weights or model name")
    # Allow quick overrides on CLI (optional)
    ap.add_argument("--data", default=None, help="data.yaml path")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    # Load YAML and export BASE_FM_* to environment
    raw_cfg = _load_yaml(args.cfg)
    cleaned_cfg = apply_env_from_config(raw_cfg, prefix="BASE_FM_", overwrite=True)

    # Merge optional quick overrides
    for k in ("data", "imgsz", "epochs", "batch", "device", "project", "name"):
        v = getattr(args, k)
        if v is not None:
            cleaned_cfg[k] = v

    LOGGER.info("[BaseFM CLI] Cleaned overrides (sans BASE_FM_*):")
    for k, v in cleaned_cfg.items():
        if k.startswith("BASE_FM_"):
            continue
        LOGGER.info(f"  {k}: {v}")

    # Start training with custom trainer
    model = YOLO(args.weights, task="detect")
    model.train(trainer=BaseFMTrainer, **cleaned_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
