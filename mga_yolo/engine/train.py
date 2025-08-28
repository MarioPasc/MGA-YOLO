from __future__ import annotations
from typing import Any, Dict

import torch

from mga_yolo.external.ultralytics.ultralytics import YOLO


def train(config: Dict[str, Any]) -> Dict[str, Any] | None:
	"""
	Entry point to launch MGA-YOLO training given a merged config dict.

	Expected keys (non-exhaustive):
	- model: path to model YAML or weights
	- task:  training task (defaults to 'mga' if absent)
	- any other ultralytics YOLO.train kwargs (epochs, imgsz, data, batch, etc.)
	"""
	cfg = dict(config) if config else {}
	if "model" not in cfg:
		raise ValueError("config must include 'model' entry (path to model YAML/weights)")
	model_path = str(cfg.pop("model"))
	task = str(cfg.get("task", "mga"))

	# Device default if not provided
	if "device" not in cfg:
		cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

	# Remove control keys not meant for YOLO.train kwargs
	cfg.pop("mode", None)

	# Instantiate YOLO. Some environments may not include 'mga' in task map at _new time.
	# Fallback to 'detect' and then set task back to 'mga' prior to training.
	try:
		yolo = YOLO(model_path, task=task, verbose=bool(cfg.get("verbose", True)))
	except NotImplementedError:
		yolo = YOLO(model_path, task="detect", verbose=bool(cfg.get("verbose", True)))
		# Ensure downstream trainer selection uses MGA path
		setattr(yolo, "task", "mga")

	results = yolo.train(**cfg)

	# Prefer returning validator metrics if present
	try:
		trainer = getattr(yolo, "trainer", None)
		validator = getattr(trainer, "validator", None) if trainer is not None else None
		return getattr(validator, "metrics", None)
	except Exception:
		return results if isinstance(results, dict) else None