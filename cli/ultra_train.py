from __future__ import annotations

"""
Standalone CLI to train with the pip-installed `ultralytics` package from a YAML config.

Usage:
  python cli/ultra_train.py --cfg /path/to/defaults.yaml

Notes:
- This script intentionally imports `ultralytics` from the active environment, not the vendored copy.
- Any config keys starting with 'MGA_' are exported as environment variables and removed from the training kwargs.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    raise


def _to_env_string(value: Any) -> str | None:
    """Convert arbitrary config value to an environment-safe string (or None to skip)."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def apply_env_and_clean(cfg: Dict[str, Any], prefix: str = "MGA_", overwrite: bool = True) -> Dict[str, Any]:
    """Export prefix-matching keys as environment variables and return config without them."""
    cleaned = dict(cfg)
    exported = 0
    kept = 0
    for k in list(cleaned.keys()):
        if not isinstance(k, str) or not k.startswith(prefix):
            continue
        v = cleaned.pop(k)
        vs = _to_env_string(v)
        if vs is None:
            print(f"[ultra_train] ENV skip (None): {k}")
            continue
        if not overwrite and k in os.environ:
            kept += 1
            print(f"[ultra_train] ENV keep existing: {k}={os.environ.get(k, '')}")
            continue
        os.environ[k] = vs
        exported += 1
        shown = vs if len(vs) <= 256 else vs[:253] + "..."
        print(f"[ultra_train] ENV set: {k}={shown}")
    if exported or kept:
        print(f"[ultra_train] ENV summary: set={exported}, kept_existing={kept}")
    return cleaned


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping/dict, got {type(data)}")
    return data  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train using pip ultralytics from a YAML config.")
    parser.add_argument("--cfg", required=True, help="Path to YAML configuration file")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.cfg)
    cfg = apply_env_and_clean(cfg, prefix="MGA_", overwrite=True)

    model_path = cfg.pop("model", None)
    if not model_path:
        print("[ultra_train] 'model' is required in the YAML.", file=sys.stderr)
        return 2

    # Remove pure control keys not accepted by ultralytics.train kwargs
    task = cfg.pop("task", None)
    cfg.pop("mode", None)

    try:
        from ultralytics import YOLO  # pip-installed package
    except Exception as e:  # pragma: no cover
        print("[ultra_train] Failed to import ultralytics. Install it with: pip install ultralytics", file=sys.stderr)
        raise

    print(f"[ultra_train] Initializing model from: {model_path}")
    yolo = YOLO(model_path, task=task)

    print("[ultra_train] Starting training...")
    results = yolo.train(**cfg)
    # Print a compact summary if dict-like
    if isinstance(results, dict):
        keys = ", ".join(sorted(results.keys()))
        print(f"[ultra_train] Training finished. Result keys: {keys}")
    else:
        print("[ultra_train] Training finished.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
