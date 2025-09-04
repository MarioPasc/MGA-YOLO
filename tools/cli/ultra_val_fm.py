from __future__ import annotations
"""
Standalone CLI to run pip-installed Ultralytics validation with a custom feature-map validator.

Usage:
  python cli/ultra_val_fm.py --weights runs/detect/train/weights/best.pt --data data.yaml --imgsz 512 \
      --save-fm yes --layers 23,25,27 --fm-max 4

Notes:
- Imports `ultralytics` from the active environment.
- Exports MGA_* environment variables for the validator.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

try:
    from ultralytics import YOLO
except Exception:
    print("[ultra_val_fm] Install ultralytics: pip install ultralytics", file=sys.stderr)
    raise

# Local validator
try:
    from tools.validators.base_fm_validator import BaseFMValidator
except Exception:
    print("[ultra_val_fm] Could not import tools.custom_validator", file=sys.stderr)
    raise


def _to_env(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate with feature-map capture using pip Ultralytics.")
    p.add_argument("--weights", required=True, help="Path to model weights (.pt or model name)")
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--device", default=None, help="CUDA device id or 'cpu'")
    p.add_argument("--save-fm", default="yes", help="Enable FM saving: {yes,no,1,0,true,false}")
    p.add_argument("--layers", default="23,25,27", help="Comma-separated layer indices to hook")
    p.add_argument("--fm-max", type=int, default=4, help="Max timepoints to save across training; ignored for plain val")
    p.add_argument("--project", default=None, help="Override project directory")
    p.add_argument("--name", default=None, help="Override run name")
    args = p.parse_args(argv)

    # Export validator env
    os.environ["SAVE_FM"] = _to_env(args.save_fm) or "yes"
    os.environ["SAVE_LAYERS"] = _to_env(args.layers) or "23,25,27"
    os.environ["SAVE_FM_MAX"] = str(int(args.fm_max))

    print(f"[ultra_val_fm] ENV SAVE_FM={os.environ['SAVE_FM']}")
    print(f"[ultra_val_fm] ENV SAVE_LAYERS={os.environ['SAVE_LAYERS']}")
    print(f"[ultra_val_fm] ENV SAVE_FM_MAX={os.environ['SAVE_FM_MAX']}")

    yolo = YOLO(args.weights, task="detect")
    print(f"[ultra_val_fm] Loaded weights: {args.weights}")

    kw: Dict[str, Any] = dict(data=args.data, imgsz=args.imgsz, batch=args.batch)
    if args.device is not None:
        kw["device"] = args.device
    if args.project is not None:
        kw["project"] = args.project
    if args.name is not None:
        kw["name"] = args.name

    print("[ultra_val_fm] Starting validation with BaseFMValidator...")
    yolo.val(validator=BaseFMValidator, **kw)
    print("[ultra_val_fm] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
