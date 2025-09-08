"""Visualization utilities for inspecting mask processing stages.

Generates, for a single image instance, two sets of per-level visualizations:

1. Downsampled ground-truth mask probabilities using three methods (avgpool, area, nearest)
   Layout: 1 row x 3 columns -> one PNG per level (e.g., downsample_P3.png)
2. Probabilistic gating (Gumbel-Sigmoid) applied to a predicted probability mask for
   several temperature (tau) values. Layout: 1 row x 5 columns -> one PNG per level
   (e.g., gumbel_P3.png)

Assumptions & Notes
-------------------
* Feature pyramid strides follow standard YOLO: P3=8, P4=16, P5=32.
* Temperature sweep: user requested 0..1.0 step 0.2 with 5 columns. A 0..1 inclusive
  sweep at step 0.2 would yield 6 values. Here we assume the intended set is
  {0.2, 0.4, 0.6, 0.8, 1.0} (exclude 0 because tau > 0 is required by ProbMaskGater).
* Input predicted mask paths can be .npy (float array in [0,1]) or image files (will
  be converted to [0,1] grayscale floats).
* Ground-truth mask is treated as binary; any non-zero is foreground.

Usage (example)
---------------
python -m mga_yolo.utils.plotting.mask_process_visualized \
  --image path/to/img.jpg \
  --mask  path/to/mask.png \
  --pred-p3 preds/p3.npy \
  --pred-p4 preds/p4.npy \
  --pred-p5 preds/p5.npy \
  --outdir outputs/mask_vis --seed 123

Outputs
-------
downsample_P{3,4,5}.png and gumbel_P{3,4,5}.png inside --outdir
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import torch

from mga_yolo.utils.mask_utils import MaskUtils
from mga_yolo.nn.modules.probmaskgater import ProbMaskGater
from mga_yolo.utils.plotting.model_comparison import configure_matplotlib


STRIDES: Dict[str, int] = {"P3": 8, "P4": 16, "P5": 32}
TAUS: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
DS_METHODS: Tuple[str, ...] = ("avgpool", "area", "nearest")


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (m > 0).astype(np.uint8)


def _load_pred_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # potential CHW -> take first channel
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"Could not read predicted mask image: {path}")
        arr = arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    # Normalize/clamp to [0,1] with safe constant handling
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max > 1.0 or a_min < 0.0:
        denom = (a_max - a_min)
        if denom > 0:
            arr = (arr - a_min) / denom
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def visualize_downsample(mask: np.ndarray, outdir: Path, level: str, add_colorbar: bool = False, save_arrays: bool = False) -> Path:
    stride = STRIDES[level]
    fig, axes = plt.subplots(1, len(DS_METHODS), figsize=(4 * len(DS_METHODS), 4), constrained_layout=True)
    if len(DS_METHODS) == 1:
        axes = [axes]
    last_img = None
    for ax, method in zip(axes, DS_METHODS):
        prob = MaskUtils.downsample_mask_prob(mask, stride=stride, method=method)
        last_img = ax.imshow(prob, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"{level} {method}\n{prob.shape[1]}x{prob.shape[0]}")
        ax.axis('off')
        if save_arrays:
            np.save(outdir / f"downsample_{level}_{method}.npy", prob.astype(np.float32))
    if add_colorbar and last_img is not None:
        plt.colorbar(mappable=last_img, ax=axes[-1], fraction=0.046, pad=0.04)
    fig.suptitle(f"Downsample Mask Prob - {level} (stride={stride})")
    out_path = outdir / f"downsample_{level}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def visualize_gumbel(pred: np.ndarray, outdir: Path, level: str, seed: int | None, add_colorbar: bool = False, save_arrays: bool = False) -> Path:
    """Visualize gating outputs and residuals with a temperature arrow annotation.

    Layout:
        Row 0: [Pre-Gating Mask] [tau samples increasing ->]
        Row 1: [Residual Maps]   [residuals]
    Only right-most column in each row gets a colorbar if add_colorbar=True.
    A horizontal double-headed arrow between rows labels the temperature sweep.
    """
    stride = STRIDES[level]
    ncols = len(TAUS) + 1  # +1 for pre-gate reference
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 2 * 4), constrained_layout=True)
    device = torch.device("cpu")

    pre = pred.astype(np.float32)
    # Row 0 col 0: pre-gating reference
    im_pre = axes[0, 0].imshow(pre, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Pre-Gating Mask")
    axes[0, 0].axis('off')
    axes[1, 0].imshow(np.zeros_like(pre), cmap="seismic", vmin=-1.0, vmax=1.0)
    axes[1, 0].set_title("Residual Maps")
    axes[1, 0].axis('off')
    if save_arrays:
        np.save(outdir / f"gumbel_{level}_pre.npy", pre)

    gated_maps = []
    residuals = []
    with torch.no_grad():
        lvl_offset = {"P3": 0, "P4": 1, "P5": 2}[level]
        base_seed = (seed + lvl_offset) if seed is not None else None
        p_tensor = torch.from_numpy(pre).to(device)
        for j, tau in enumerate(TAUS, start=1):
            gater = ProbMaskGater(mode='gumbel', tau=float(tau), seed=base_seed)
            gater.train()
            gate = gater(p_tensor.unsqueeze(0).unsqueeze(0))
            gate_np = gate.squeeze().cpu().numpy().astype(np.float32)
            gated_maps.append(gate_np)
            residuals.append(gate_np - pre)
            if save_arrays:
                np.save(outdir / f"gumbel_{level}_tau{tau:.2f}.npy", gate_np)

    # Determine residual color scaling symmetrically
    if residuals:
        abs_max = max(abs(r).max() for r in residuals) + 1e-8
    else:
        abs_max = 1.0

    # Plot gated and residual columns (no per-column titles now)
    last_gate_img = None
    last_res_img = None
    for j, tau in enumerate(TAUS, start=1):
        gate_np = gated_maps[j - 1]
        last_gate_img = axes[0, j].imshow(gate_np, cmap="magma", vmin=0.0, vmax=1.0)
        axes[0, j].axis('off')
        resid_np = residuals[j - 1]
        last_res_img = axes[1, j].imshow(resid_np, cmap="seismic", vmin=-abs_max, vmax=abs_max)
        axes[1, j].axis('off')
        if save_arrays:
            np.save(outdir / f"gumbel_{level}_resid_tau{TAUS[j-1]:.2f}.npy", resid_np.astype(np.float32))

    # Add colorbars only on right-most column for each row
    if add_colorbar and last_gate_img is not None and last_res_img is not None:
        plt.colorbar(last_gate_img, ax=axes[0, -1], fraction=0.046, pad=0.04)
        plt.colorbar(last_res_img, ax=axes[1, -1], fraction=0.046, pad=0.04)

    # Temperature arrow annotation between rows (from first tau column to last tau column)
    fig.canvas.draw()  # ensure layout positions are computed
    if ncols > 1:
        # Use positions of first tau column (col=1) and last column
        left_bbox = axes[0, 1].get_position()
        right_bbox = axes[0, -1].get_position()
        top_ax_bbox = axes[0, 1].get_position()
        bottom_ax_bbox = axes[1, 1].get_position()
        y_mid = (top_ax_bbox.y0 + bottom_ax_bbox.y1) / 2.0
        x_left = left_bbox.x0
        x_right = right_bbox.x1
        y_offset: float = 0.01
        arrow = FancyArrowPatch((x_left, y_mid - y_offset), (x_right, y_mid - y_offset),
                                transform=fig.transFigure,
                                arrowstyle='<->', lw=4, color='black', mutation_scale=20)
        fig.add_artist(arrow)
        fontsize: int =10
        fig.text(x_left, y_mid - (y_offset+0.03), f"{TAUS[0]:.1f}", ha='left', va='center', fontsize=fontsize)
        fig.text((x_left + x_right) / 2.0, y_mid - (y_offset+0.03), r'$\tau$ (Temperature)', ha='center', va='center', fontsize=fontsize)
        fig.text(x_right, y_mid - (y_offset+0.03), f"{TAUS[-1]:.1f}", ha='right', va='center', fontsize=fontsize)

    fig.suptitle(f"Gumbel Gate Samples and Residuals - {level} (stride={stride})")
    out_path = outdir / f"gumbel_{level}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(image_path: Path, mask_path: Path, pred_paths: Dict[str, Path], outdir: Path, seed: int | None, add_colorbar: bool, save_arrays: bool) -> None:
    configure_matplotlib()  # Apply global fancy style
    _ensure_outdir(outdir)
    _ = _load_image(image_path)  # currently unused; reserved if overlay desired later
    mask = _load_mask(mask_path)

    # Downsample visualizations use the same original GT mask for all levels.
    for lvl in ("P3", "P4", "P5"):
        ds_out = visualize_downsample(mask, outdir, lvl, add_colorbar=add_colorbar, save_arrays=save_arrays)
        print(f"Saved {ds_out}")

    # Gumbel gating per level on predicted probability masks.
    for lvl in ("P3", "P4", "P5"):
        pred_path = pred_paths[lvl]
        pred = _load_pred_mask(pred_path)
        g_out = visualize_gumbel(pred, outdir, lvl, seed, add_colorbar=add_colorbar, save_arrays=save_arrays)
        print(f"Saved {g_out}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize mask downsampling and gumbel gating per feature level.")
    p.add_argument('--image', type=Path, required=True, help='Path to original image (RGB).')
    p.add_argument('--mask', type=Path, required=True, help='Path to ground-truth binary mask image.')
    p.add_argument('--pred-p3', type=Path, required=True, help='Path to predicted probability mask for P3 (npy or image).')
    p.add_argument('--pred-p4', type=Path, required=True, help='Path to predicted probability mask for P4 (npy or image).')
    p.add_argument('--pred-p5', type=Path, required=True, help='Path to predicted probability mask for P5 (npy or image).')
    p.add_argument('--outdir', type=Path, default=Path('mask_process_outputs'), help='Output directory for PNGs and (optionally) arrays.')
    p.add_argument('--seed', type=int, default=None, help='Seed for gumbel sampling reproducibility.')
    p.add_argument('--colorbar', action='store_true', help='Add colorbars to subplots.')
    p.add_argument('--save-arrays', action='store_true', help='Also save raw .npy arrays for each visualization.')
    return p


def main() -> None:
    args = build_argparser().parse_args()
    pred_paths = {"P3": args.pred_p3, "P4": args.pred_p4, "P5": args.pred_p5}
    run(args.image, args.mask, pred_paths, args.outdir, args.seed, add_colorbar=args.colorbar, save_arrays=args.save_arrays)


if __name__ == '__main__':  # pragma: no cover
    main()

