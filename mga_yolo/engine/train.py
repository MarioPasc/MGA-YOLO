from __future__ import annotations
from typing import Any, Dict

import torch

from mga_yolo.external.ultralytics.ultralytics import YOLO
from mga_yolo.utils.env import apply_mga_env_from_config


def train(config: Dict[str, Any]) -> Dict[str, Any] | None:
	"""
	Entry point to launch MGA-YOLO training given a merged config dict.

	Expected keys (non-exhaustive):
	- model: path to model YAML or weights
	- task:  training task (defaults to 'mga' if absent)
	- any other ultralytics YOLO.train kwargs (epochs, imgsz, data, batch, etc.)
	"""
	cfg = dict(config) if config else {}
	# 1) Export MGA_* keys to environment and clean them from cfg
	cfg = apply_mga_env_from_config(cfg, prefix="MGA_", overwrite=True)
	if "model" not in cfg:
		raise ValueError("config must include 'model' entry (path to model YAML/weights)")
	model_path = str(cfg.pop("model"))
	task = str(cfg.get("task", "mga"))

	# Device default if not provided
	if "device" not in cfg:
		cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

	# Instantiate YOLO.
	yolo = YOLO(model_path, task=task, verbose=bool(cfg.get("verbose", True)))

	results = yolo.train(**cfg)

	# Prefer returning validator metrics if present
	try:
		return results
	except Exception:
		return results if isinstance(results, dict) else None