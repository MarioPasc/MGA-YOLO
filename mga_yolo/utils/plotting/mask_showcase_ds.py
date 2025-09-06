"""mask_showcase_ds.py

Generate side‑by‑side comparison plots of binary mask downsampling strategies
for the three YOLO feature map resolutions P3 (68x68), P4 (34x34), P5 (17x17).

Methods showcased (mirroring _downsample_mask logic):
  - nearest
  - area
  - maxpool
  - pyrdown
  - skeleton_bresenham (connectivity preserving)

Three PNGs are produced (one per pyramid level) each with a 1xN panel where
N = number of methods. Only the first subplot carries a y‑axis label in the
form "P{level}\n(HxW)". All other axes have no axis labels or ticks.

Usage:
  python -m mga_yolo.utils.plotting.mask_showcase_ds MASK_PATH OUT_DIR \
		 [--prefix PREFIX] [--thresh 0.0] [--no-bridge]

The input mask can be:
  - An image file (read via Pillow) – any non‑zero pixel becomes 1
  - A .npy file containing a 2D array

Matplotlib styling reuses `configure_matplotlib` from `model_comparison`.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from mga_yolo.utils.plotting.model_comparison import configure_matplotlib  # reuse global style
from mga_yolo.utils.mask_downsample import (
	DownsampleConfig,
	downsample_preserve_connectivity,
)

LOGGER = logging.getLogger(__name__)

# Methods explicitly mentioned in the reference _downsample_mask implementation
DS_METHODS: List[str] = ["nearest", "area", "maxpool", "pyrdown", "skeleton_bresenham"]


def _binary_load(path: Path) -> np.ndarray:
	"""Load a mask (image or .npy) and return a uint8 {0,1} array."""
	if path.suffix.lower() == ".npy":
		arr = np.load(path)
	else:
		img = Image.open(path).convert("L")
		arr = np.array(img)
	if arr.ndim != 2:
		raise ValueError("Mask must be 2D (HxW).")
	return (arr > 0).astype(np.uint8)


def _ensure_target_stride(h: int, w: int, target: int) -> int:
	"""Compute stride such that ceil(h/stride)=target. Prefer exact division.

	Raises if resulting width mismatch would differ from target (expects square-ish).
	"""
	stride = math.ceil(h / target)
	if math.ceil(w / stride) != target:
		raise ValueError(
			f"Width/stride mismatch: original=({h},{w}) target={target} stride={stride} -> (ceil={math.ceil(h/stride)},{math.ceil(w/stride)})"
		)
	return stride


def _downsample(mask: np.ndarray, stride: int, method: str, thresh: float, bridge: bool) -> np.ndarray:
	"""Apply a single downsampling method, returning a uint8 {0,1} mask."""
	h, w = mask.shape
	nh, nw = math.ceil(h / stride), math.ceil(w / stride)
	if method == "nearest":
		return cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
	if method == "area":
		small = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
		out = (small > thresh).astype(np.uint8)
		if bridge:
			kernel = np.ones((3, 3), np.uint8)
			out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1).astype(np.uint8)
		return out.astype(np.uint8)
	if method == "maxpool":
		pad_h = (stride - (h % stride)) % stride
		pad_w = (stride - (w % stride)) % stride
		if pad_h or pad_w:
			mp = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
		else:
			mp = mask
		H2, W2 = mp.shape
		view = mp.reshape(H2 // stride, stride, W2 // stride, stride)
		out = view.max(axis=(1, 3)).astype(np.uint8)
		return out
	if method == "pyrdown":
		# only if power of two; else fallback to nearest resize for parity
		s = stride
		if s & (s - 1) == 0 and s > 1:
			out = mask.copy()
			while s > 1:
				out = cv2.pyrDown(out).astype(np.uint8)
				s //= 2
			out = (out > 0).astype(np.uint8)
			if bridge:
				kernel = np.ones((3, 3), np.uint8)
				out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1).astype(np.uint8)
			return out.astype(np.uint8)
		LOGGER.debug("pyrdown stride %d not power-of-two; using nearest fallback", stride)
		return cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
	if method == "skeleton_bresenham":
		cfg = DownsampleConfig(factor=stride, method="skeleton_bresenham", threshold=max(thresh, 0.2), close_diagonals=bridge)
		return downsample_preserve_connectivity(mask, cfg)
	raise ValueError(f"Unknown method '{method}'")


def _downsample_all(mask: np.ndarray, target: int, thresh: float, bridge: bool) -> Dict[str, np.ndarray]:
	"""Downsample mask to target resolution with all methods; returns dict method->mask."""
	h, w = mask.shape
	stride = _ensure_target_stride(h, w, target)
	out: Dict[str, np.ndarray] = {}
	for m in DS_METHODS:
		ds = _downsample(mask, stride, m, thresh, bridge)
		# Harmonize output to target (crop/resize if any rounding difference)
		if ds.shape != (target, target):  # enforce expected square size
			ds = cv2.resize(ds, (target, target), interpolation=cv2.INTER_NEAREST)
		out[m] = ds.astype(np.uint8)
	return out


def _plot_panel(res_dict: Dict[str, np.ndarray], level_label: str, size: int, out_path: Path) -> None:
	"""Create 1xN panel plot for a pyramid level."""
	configure_matplotlib()
	methods = list(res_dict.keys())
	n = len(methods)
	fig, axes = plt.subplots(1, n, figsize=(n * 1.4, 1.4), constrained_layout=True)
	if n == 1:
		axes = [axes]  # type: ignore
	for idx, m in enumerate(methods):
		ax = axes[idx]
		ax.imshow(res_dict[m], cmap="gray", interpolation="nearest")
		ax.set_title(m, fontsize=7)
		ax.set_xticks([])
		ax.set_yticks([])
		if idx == 0:
			ax.set_ylabel(f"{level_label}\n({size}x{size})")
		else:
			ax.set_ylabel("")
	fig.savefig(out_path)
	plt.close(fig)
	LOGGER.info("Saved %s", out_path)


def generate_showcase(mask_path: Path, out_dir: Path, prefix: str | None, thresh: float = 0.0, bridge: bool = True) -> List[Path]:
	"""High-level driver returning list of created figure paths."""
	mask = _binary_load(mask_path)
	h, w = mask.shape
	LOGGER.info("Loaded mask %s shape=%s", mask_path, (h, w))

	# Expected YOLO strides relative to original: 8,16,32 -> target sizes 64,32,16 for 544 input.
	targets: List[Tuple[str, int]] = [("P3", 64), ("P4", 32), ("P5", 16)]
	out_dir.mkdir(parents=True, exist_ok=True)
	if prefix is None:
		prefix = mask_path.stem

	paths: List[Path] = []
	for level, size in targets:
		try:
			res_dict = _downsample_all(mask, size, thresh=thresh, bridge=bridge)
		except ValueError as e:
			LOGGER.error("Skipping %s: %s", level, e)
			continue
		out_path = out_dir / f"{prefix}_{level.lower()}.png"
		_plot_panel(res_dict, level, size, out_path)
		paths.append(out_path)
	return paths


def _build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Showcase mask downsampling methods for YOLO pyramid levels")
	p.add_argument("mask", type=Path, help="Path to binary mask image or .npy file")
	p.add_argument("out", type=Path, help="Output directory for figures")
	p.add_argument("--prefix", type=str, default=None, help="Filename prefix (default: mask stem)")
	p.add_argument("--thresh", type=float, default=0.0, help="Threshold for area method")
	p.add_argument("--no-bridge", action="store_true", help="Disable 3x3 closing bridge")
	p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
	return p


def main() -> None:  # pragma: no cover - CLI utility
	ap = _build_argparser()
	args = ap.parse_args()
	level = logging.WARNING if args.verbose == 0 else logging.INFO if args.verbose == 1 else logging.DEBUG
	logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
	paths = generate_showcase(args.mask, args.out, args.prefix, thresh=args.thresh, bridge=not args.no_bridge)
	for p in paths:
		print(p)


if __name__ == "__main__":  # pragma: no cover
	main()

