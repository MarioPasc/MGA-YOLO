# add imports
from typing import Callable, Optional, Dict

try:
    # Prefer Ultralytics logger for consistent formatting
    from ultralytics.utils import LOGGER  # type: ignore
except Exception:  # fallback to std logging if ultralytics is unavailable
    import logging

    LOGGER = logging.getLogger("plot_results")
    if not LOGGER.handlers:
        logging.basicConfig(level=logging.INFO)
from typing import Optional, Callable, Dict


def plot_results(
    file: str = "path/to/results.csv",
    dir: str = "",
    segment: bool = False,
    pose: bool = False,
    classify: bool = False,
    on_plot: Optional[Callable] = None,
    smooth_sigma: float = 3.0,
    separator_offset: float = 0.7,  
):

    """
    Publication-style training figure distinguishing:
      - Detection Loss (row): total, box, dfl, cls
      - Segmentation Loss (2 rows): seg total spanning both rows in col 0; P3/P4/P5 with Dice (row 1) and BCE (row 2)
      - Detection Performance (row): precision, recall, mAP@0.50, mAP@0.50:0.95

    The main source is results*.csv in save_dir. Multiple files are overlaid.
    """
    # Local imports
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from matplotlib import gridspec
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter1d

    # -------------------------
    # Styling (science + IEEE)
    # -------------------------
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee"])
        plt.rcParams.update({
            "font.size": 8,
            "font.family": "serif",
            "font.serif": "Times",
            "text.usetex": True,
        })
    except Exception as e:
        LOGGER.warning(f"Science/IEEE style not available ({e}); falling back to default Matplotlib.")
        plt.rcParams.update({
            "font.size": 8,
            "font.family": "serif",
            "font.serif": "Times",
        })

    # -------------------------
    # Resolve paths and files
    # -------------------------
    save_dir = Path(file).parent if file else Path(dir)
    files = list(save_dir.glob("results*.csv")) if save_dir.exists() else []
    if file:
        p = Path(file)
        if p.exists() and p not in files:
            files.insert(0, p)
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."

    # -------------------------
    # Column alias resolution
    # -------------------------
    ALIASES = {
        # x-axis
        "x": ["epoch", "Epoch", "step", "Step", "iter", "iteration"],

        # Detection losses
        "det_total": ["train/det/total", "val/det/total", "train/total_loss", "loss/det_total",
                      "det_total", "train/loss", "loss/total", "total_loss"],
        "det_box":   ["train/det/box", "val/det/box", "train/box", "val/box",
                      "loss/box", "train/box_loss", "box_loss", "box"],
        "det_dfl":   ["train/det/dfl", "val/det/dfl", "train/dfl", "val/dfl",
                      "loss/dfl", "train/dfl_loss", "dfl_loss", "dfl"],
        "det_cls":   ["train/det/cls", "val/det/cls", "train/cls", "val/cls",
                      "loss/cls", "train/cls_loss", "cls_loss", "cls"],

        # Segmentation totals
        "seg_total": ["train/seg/total", "train/seg_total", "val/seg/total", "val/seg_total",
                      "loss/seg_total", "seg_total", "loss/seg", "train/seg_loss", "seg_loss"],

        # Segmentation scales — Dice and BCE (accept *bca* as synonym if present)
        "p3_dice": ["train/seg/p3_dice", "val/seg/p3_dice", "p3_dice", "seg/p3_dice", "loss/p3_dice"],
        "p4_dice": ["train/seg/p4_dice", "val/seg/p4_dice", "p4_dice", "seg/p4_dice", "loss/p4_dice"],
        "p5_dice": ["train/seg/p5_dice", "val/seg/p5_dice", "p5_dice", "seg/p5_dice", "loss/p5_dice"],

        "p3_bce":  ["train/seg/p3_bce",  "val/seg/p3_bce",  "p3_bce",  "train/seg/p3_bca", "p3_bca", "seg/p3_bca"],
        "p4_bce":  ["train/seg/p4_bce",  "val/seg/p4_bce",  "p4_bce",  "train/seg/p4_bca", "p4_bca", "seg/p4_bca"],
        "p5_bce":  ["train/seg/p5_bce",  "val/seg/p5_bce",  "p5_bce",  "train/seg/p5_bca", "p5_bca", "seg/p5_bca"],

        # Metrics
        "precision": ["metrics/precision(B)", "metrics/precision", "precision", "Precision"],
        "recall":    ["metrics/recall(B)",    "metrics/recall",    "recall",    "Recall"],
        "map50":     ["metrics/mAP50(B)",     "metrics/mAP50",     "mAP50",     "map50"],
        "map5095":   ["metrics/mAP50-95(B)",  "metrics/mAP50-95",  "mAP50-95",  "map50-95", "mAP@0.50:0.95"],
    }

    def _get_col_name(header: list, candidates: list) -> Optional[str]:
        s = {h.strip(): h for h in header}
        for c in candidates:
            if c in s:
                return s[c]
        normalized = {h.strip().lower().replace(" ", ""): h for h in header}
        for c in candidates:
            key = c.strip().lower().replace(" ", "")
            if key in normalized:
                return normalized[key]
        return None

    def _resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        hdr = [h.strip() for h in df.columns]
        resolved = {}
        for k, cands in ALIASES.items():
            resolved[k] = _get_col_name(hdr, cands)
            if resolved[k] is None and k not in {"p3_dice", "p4_dice", "p5_dice", "p3_bce", "p4_bce", "p5_bce"}:
                LOGGER.warning(f"[plot_results] Missing column for '{k}'. Tried: {cands}")
        return resolved

    # Colors and unified-legend collector
    ACTUAL_COLOR = "#009988"
    SMOOTH_COLOR = "#EE7733"
    legend_items: dict[str, Line2D] = {}

    def _plot_one(ax, x, y, label: str, title: str, sigma: float = 3.0):
        """Plot raw (solid) and smoothed (dashed) with fixed colors and collect legend entries once per label."""
        y = np.asarray(y, dtype=float)
        # actual
        ln_actual, = ax.plot(x, y, marker=".", linewidth=1.2, markersize=3.2,
                             color=ACTUAL_COLOR, label=label)
        # smoothed
        if len(y) >= 5 and sigma > 0:
            ys = gaussian_filter1d(y, sigma=sigma)
            ln_smooth, = ax.plot(x, ys, linestyle="--", linewidth=1.2,
                                 color=SMOOTH_COLOR, label=f"{label} (smooth)")
            # add proxy for smooth if first time
            if f"{label} (smooth)" not in legend_items:
                legend_items[f"{label} (smooth)"] = Line2D([0], [0], linestyle="--", color=SMOOTH_COLOR, linewidth=1.2)
        # add proxy for actual if first time
        if label not in legend_items:
            legend_items[label] = Line2D([0], [0], linestyle="-", color=ACTUAL_COLOR, marker=".", linewidth=1.2,
                                         markersize=3.2)

        ax.set_title(title, pad=2.0)
        ax.grid(True, which="major", alpha=0.25)
        ax.tick_params(axis="both", which="both", length=2)

    # --------------------------------------------
    # Figure layout (exact schematic with centered separators)
    # --------------------------------------------
    # Rows:
    #  0  Detection row (4 panels)
    #  1  Spacer
    #  2  Seg row 1 (Seg total + P3/P4/P5 Dice)
    #  3  Seg row 2 (       —      P3/P4/P5 BCE )
    #  4  Spacer
    #  5  Metrics row (4 panels)
    fig = plt.figure(figsize=(7.6, 8.2), constrained_layout=False)
    height_ratios = [1.20, 0.35, 1.25, 1.25, 0.40, 1.15]
    outer = gridspec.GridSpec(
        nrows=len(height_ratios), ncols=4, figure=fig,
        height_ratios=height_ratios, wspace=0.55, hspace=0.80
    )

    # Detection row (row 0)
    ax_det_total = fig.add_subplot(outer[0, 0])
    ax_det_box   = fig.add_subplot(outer[0, 1])
    ax_det_dfl   = fig.add_subplot(outer[0, 2])
    ax_det_cls   = fig.add_subplot(outer[0, 3])

    # Segmentation rows (rows 2 and 3)
    ax_seg_total = fig.add_subplot(outer[2:4, 0])  # span two rows in col 0
    ax_p3_dice   = fig.add_subplot(outer[2, 1])
    ax_p4_dice   = fig.add_subplot(outer[2, 2])
    ax_p5_dice   = fig.add_subplot(outer[2, 3])
    ax_p3_bce    = fig.add_subplot(outer[3, 1])
    ax_p4_bce    = fig.add_subplot(outer[3, 2])
    ax_p5_bce    = fig.add_subplot(outer[3, 3])

    # Metrics row (row 5)
    ax_prec     = fig.add_subplot(outer[5, 0])
    ax_rec      = fig.add_subplot(outer[5, 1])
    ax_map50    = fig.add_subplot(outer[5, 2])
    ax_map5095  = fig.add_subplot(outer[5, 3])

    # --------------------------------------------
    # Horizontal separators with configurable offset
    #   - Lines are placed at the center of each spacer row (1 and 4),
    #     shifted by 'separator_offset' * spacer_height.
    #   - Positive offset moves the line upward (towards the previous block).
    # --------------------------------------------
    hr = np.array(height_ratios, dtype=float)
    hr_norm = hr / hr.sum()

    def _row_center_y(row_idx: int) -> float:
        """Figure Y of the vertical center of a given GridSpec row."""
        y_top = 1.0 - float(hr_norm[:row_idx].sum())
        y_bottom = 1.0 - float(hr_norm[:row_idx + 1].sum())
        return 0.5 * (y_top + y_bottom)

    def _sep_y(row_idx: int, bias: float) -> float:
        """Separator Y at row center plus 'bias' times that row's normalized height."""
        center = _row_center_y(row_idx)
        shift = bias * float(hr_norm[row_idx])
        y = center + shift
        # clamp slightly inside the figure
        return min(0.98, max(0.02, y))

    # spacer rows are 1 (between Detection and Segmentation) and 4 (between Segmentation and Metrics)
    y_between_det_seg = _sep_y(1, separator_offset)
    y_between_seg_met = _sep_y(4, separator_offset)

    for y in (y_between_det_seg, y_between_seg_met):
        fig.lines.append(plt.Line2D([0.06, 0.98], [y, y], transform=fig.transFigure, linewidth=0.9, alpha=0.7))

    # --------------------------------------------------
    # Read and plot (overlay multiple results*.csv)
    # --------------------------------------------------
    any_curve = False
    for f in files:
        try:
            df = pd.read_csv(f)
            keys = _resolve_columns(df)
            x_name = keys["x"] or df.columns[0]
            x = df[x_name].values

            # Detection row
            for ax, key, ttl in [
                (ax_det_total, "det_total", "Detection total"),
                (ax_det_box,   "det_box",   "Box loss"),
                (ax_det_dfl,   "det_dfl",   "DFL loss"),
                (ax_det_cls,   "det_cls",   "Cls loss"),
            ]:
                cname = keys.get(key)
                if cname is None:
                    ax.set_title(ttl + " (missing)"); ax.grid(True, alpha=0.15)
                else:
                    _plot_one(ax, x, df[cname], label=f.stem, title=ttl, sigma=smooth_sigma); any_curve = True

            # Segmentation rows
            cname = keys.get("seg_total")
            if cname is None:
                ax_seg_total.set_title("Seg total (missing)"); ax_seg_total.grid(True, alpha=0.15)
            else:
                _plot_one(ax_seg_total, x, df[cname], label=f.stem, title="Seg total", sigma=smooth_sigma); any_curve = True

            for ax, key, ttl in [
                (ax_p3_dice, "p3_dice", "P3 Dice"),
                (ax_p4_dice, "p4_dice", "P4 Dice"),
                (ax_p5_dice, "p5_dice", "P5 Dice"),
            ]:
                cname = keys.get(key)
                if cname is None:
                    ax.set_title(ttl + " (missing)"); ax.grid(True, alpha=0.15)
                else:
                    _plot_one(ax, x, df[cname], label=f.stem, title=ttl, sigma=smooth_sigma); any_curve = True

            for ax, key, ttl in [
                (ax_p3_bce, "p3_bce", "P3 BCE"),
                (ax_p4_bce, "p4_bce", "P4 BCE"),
                (ax_p5_bce, "p5_bce", "P5 BCE"),
            ]:
                cname = keys.get(key)
                if cname is None:
                    ax.set_title(ttl + " (missing)"); ax.grid(True, alpha=0.15)
                else:
                    _plot_one(ax, x, df[cname], label=f.stem, title=ttl, sigma=smooth_sigma); any_curve = True

            # Metrics row
            for ax, key, ttl in [
                (ax_prec,   "precision", "Precision"),
                (ax_rec,    "recall",    "Recall"),
                (ax_map50,  "map50",     r"mAP@0.50"),
                (ax_map5095,"map5095",   r"mAP@0.50:0.95"),
            ]:
                cname = keys.get(key)
                if cname is None:
                    ax.set_title(ttl + " (missing)"); ax.grid(True, alpha=0.15)
                else:
                    _plot_one(ax, x, df[cname], label=f.stem, title=ttl, sigma=smooth_sigma); any_curve = True

        except Exception as e:
            LOGGER.error(f"[plot_results] Plotting error for {f}: {e}")

    # --------------------------------------------
    # Cosmetics: labels and unified legend
    # --------------------------------------------
    # X labels only where useful (bottom row of each block)
    for ax in [ax_det_total, ax_det_box, ax_det_dfl, ax_det_cls,
               ax_p3_bce, ax_p4_bce, ax_p5_bce,
               ax_prec, ax_rec, ax_map50, ax_map5095]:
        ax.set_xlabel("Epoch")

    # Global legend (bottom, outside all plots)
    if legend_items:
        handles = list(legend_items.values())
        labels = list(legend_items.keys())
        ncols = max(2, min(len(labels), 6))
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.015),
                   ncol=ncols, frameon=False, handlelength=2.5, columnspacing=1.6)

    # Tighten and save (extra bottom space for legend)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.12, wspace=0.55, hspace=0.82)
    fname = save_dir / "results.png"
    try:
        fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0.02)
    finally:
        plt.close(fig)

    if on_plot:
        on_plot(fname)
    LOGGER.info(f"[plot_results] Saved {fname}")



if __name__ == "__main__":
    plot_results(file="/home/mpascual/research/code/MGA-YOLO/mga_yolo/external/ultralytics/tests/tmp/runs/mga/test_mga_train_v8_segloss11/results.csv")