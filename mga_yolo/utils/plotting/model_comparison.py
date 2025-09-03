
"""
model_comparison.py

Utilities to compare N training runs across models from CSV logs and a YAML spec.
Generates three images:
  1) Detection losses (2x4 grid): train row then val row
  2) Segmentation losses (4x4 grid, only if present): train rows (2) then val rows (2)
  3) Validation performance (1x4): Precision, Recall, mAP@50, mAP@50-95

Author: synthetic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -----------------------------
# Global plotting configuration
# -----------------------------

def configure_matplotlib() -> None:
    """
    Configure matplotlib and scienceplots with LaTeX and requested typography.
    Falls back gracefully if LaTeX or scienceplots are not available.
    """
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science'])  # base science style
    except Exception as e:
        logging.warning("scienceplots not available: %s. Continuing with default style.", e)

    # Requested typography
    plt.rcParams.update({
        'figure.dpi': 600,
        'font.size': 8,
        'font.family': 'serif',
        'font.serif': ['Times'],
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'savefig.bbox': 'tight',
    })
    # LaTeX text rendering
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    except Exception as e:
        logging.warning("LaTeX not available: %s. Falling back to non-LaTeX text.", e)
        plt.rcParams['text.usetex'] = False


# -----------------------------
# Data model
# -----------------------------

class ModelType(str, Enum):
    """Type of log schema expected for the series."""
    BASELINE = "baseline"
    MGA = "mga"


@dataclass
class StyleConfig:
    """Plot styling for a series."""
    color: Optional[str] = None
    linestyle: Optional[str] = None
    linewidth: Optional[float] = None
    marker: Optional[str] = None

    def to_matplotlib_kwargs(self) -> Dict[str, object]:
        """Convert to matplotlib.plot kwargs, omitting None values."""
        out: Dict[str, object] = {}
        if self.color is not None:
            out['color'] = self.color
        if self.linestyle is not None:
            out['linestyle'] = self.linestyle
        if self.linewidth is not None:
            out['linewidth'] = self.linewidth
        if self.marker is not None:
            out['marker'] = self.marker
        return out


@dataclass
class DataSeriesConfig:
    """Configuration for a single data series loaded from YAML."""
    name: str
    type: ModelType
    input_paths: List[Path]
    style: StyleConfig = field(default_factory=StyleConfig)


# -----------------------------
# YAML loading
# -----------------------------

def _parse_series_mapping(series_id: str, payload: Mapping) -> DataSeriesConfig:
    """
    Parse a series mapping. Supports two shapes:
      A) {'name': 'Model A', 'type': 'baseline', 'input': [...], 'style': {...}}
      B) {'type': 'baseline', 'input': [...], 'style': {...}} with series_id as display name
    """
    name = payload.get('name', series_id)
    other = payload.get('other', {})  # tolerate 'other' nesting
    type_str = (payload.get('type') or other.get('type'))
    if type_str is None:
        raise ValueError(f"Series '{series_id}': 'type' must be set to 'baseline' or 'mga'.")
    try:
        mtype = ModelType(type_str.lower())
    except Exception as e:
        raise ValueError(f"Series '{series_id}': invalid type '{type_str}'.") from e

    # Accept both 'input' and 'inputs'
    inputs = payload.get('input') or payload.get('inputs')
    if inputs is None or not isinstance(inputs, (list, tuple)) or len(inputs) == 0:
        raise ValueError(f"Series '{series_id}': 'input' must be a non-empty list of CSV paths.")

    # Style may be under 'style' or inside 'style' as a dict. Some users may use 'colour'.
    style_raw = payload.get('style', {}) or {}
    # Normalize key 'colour' -> 'color'
    color = style_raw.get('color', style_raw.get('colour'))
    style = StyleConfig(
        color=color,
        linestyle=style_raw.get('linestyle'),
        linewidth=style_raw.get('linewidth'),
        marker=style_raw.get('marker'),
    )

    input_paths = [Path(str(p)).expanduser() for p in inputs]
    return DataSeriesConfig(name=name, type=mtype, input_paths=input_paths, style=style)


def load_config(yaml_path: Path) -> List[DataSeriesConfig]:
    """
    Load a YAML file and return a list of DataSeriesConfig.
    The YAML can be either:
      - {'series': {'id1': {...}, 'id2': {...}}}
      - {'id1': {...}, 'id2': {...}}  (top-level mapping)
      - {'series': [{...}, {...}]}    (list with explicit 'name' fields)
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        spec = yaml.safe_load(f)

    series_cfgs: List[DataSeriesConfig] = []

    if 'series' in spec:
        series_node = spec['series']
        if isinstance(series_node, Mapping):
            for sid, payload in series_node.items():
                series_cfgs.append(_parse_series_mapping(str(sid), payload))
        elif isinstance(series_node, list):
            for idx, payload in enumerate(series_node):
                name = payload.get('name', f'series_{idx+1}')
                series_cfgs.append(_parse_series_mapping(name, payload))
        else:
            raise ValueError("'series' must be a mapping or list.")
    else:
        # Assume top-level mapping of series
        for sid, payload in spec.items():
            series_cfgs.append(_parse_series_mapping(str(sid), payload))

    return series_cfgs


# -----------------------------
# Loading and harmonizing CSVs
# -----------------------------

# Canonical column names we will compute to harmonize schemas
DET_TRAIN_COLS = ['train/det/total', 'train/det/box', 'train/det/dfl', 'train/det/cls']
DET_VAL_COLS   = ['val/det/total',   'val/det/box',   'val/det/dfl',   'val/det/cls']

SEG_TRAIN_COLS = ['train/seg/total',
                  'train/seg/p3_dice', 'train/seg/p4_dice', 'train/seg/p5_dice',
                  'train/seg/p3_bce',  'train/seg/p4_bce',  'train/seg/p5_bce']
SEG_VAL_COLS   = ['val/seg/total',
                  'val/seg/p3_dice', 'val/seg/p4_dice', 'val/seg/p5_dice',
                  'val/seg/p3_bce',  'val/seg/p4_bce',  'val/seg/p5_bce']

VAL_METRICS = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """In-place convert columns to numeric if present."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')


def _harmonize_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    For 'baseline' logs, compute canonical detection totals and rename columns to canonical ones.
    Baseline has no segmentation information.
    """
    df = df.copy()
    # Ensure epoch is int
    if 'epoch' not in df.columns:
        raise ValueError("Baseline CSV missing 'epoch' column.")
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce').astype('Int64')

    # Convert detection components to numeric
    base_train = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
    base_val   = ['val/box_loss',   'val/cls_loss',   'val/dfl_loss']
    _coerce_numeric(df, base_train + base_val + VAL_METRICS)

    # Compute totals
    df['train/det/box'] = df.get('train/box_loss')
    df['train/det/cls'] = df.get('train/cls_loss')
    df['train/det/dfl'] = df.get('train/dfl_loss')
    df['train/det/total'] = df[['train/det/box', 'train/det/cls', 'train/det/dfl']].sum(axis=1, min_count=1)

    df['val/det/box'] = df.get('val/box_loss')
    df['val/det/cls'] = df.get('val/cls_loss')
    df['val/det/dfl'] = df.get('val/dfl_loss')
    df['val/det/total'] = df[['val/det/box', 'val/det/cls', 'val/det/dfl']].sum(axis=1, min_count=1)

    keep_cols = ['epoch'] + DET_TRAIN_COLS + DET_VAL_COLS + [c for c in VAL_METRICS if c in df.columns]
    return df[keep_cols].dropna(subset=['epoch']).sort_values('epoch')


def _harmonize_mga(df: pd.DataFrame) -> pd.DataFrame:
    """
    For 'mga' logs, copy already present canonical names and ensure numeric types.
    """
    df = df.copy()
    if 'epoch' not in df.columns:
        raise ValueError("MGA CSV missing 'epoch' column.")
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce').astype('Int64')

    # Some MGA logs include duplicate summary columns at the end; prefer canonical ones if present.
    # Coerce numeric for all potentially used columns.
    all_cols = set(DET_TRAIN_COLS + DET_VAL_COLS + SEG_TRAIN_COLS + SEG_VAL_COLS + VAL_METRICS)
    _coerce_numeric(df, list(all_cols))

    # If segmentation totals are missing but components exist, compute totals
    if 'train/seg/total' not in df.columns:
        seg_train_components = [c for c in SEG_TRAIN_COLS if c != 'train/seg/total' and c in df.columns]
        if seg_train_components:
            df['train/seg/total'] = df[seg_train_components].sum(axis=1, min_count=1)
    if 'val/seg/total' not in df.columns:
        seg_val_components = [c for c in SEG_VAL_COLS if c != 'val/seg/total' and c in df.columns]
        if seg_val_components:
            df['val/seg/total'] = df[seg_val_components].sum(axis=1, min_count=1)

    keep = ['epoch'] + [c for c in (DET_TRAIN_COLS + DET_VAL_COLS + SEG_TRAIN_COLS + SEG_VAL_COLS + VAL_METRICS) if c in df.columns]
    return df[keep].dropna(subset=['epoch']).sort_values('epoch')


def load_series_frames(series: DataSeriesConfig) -> List[pd.DataFrame]:
    """
    Load and harmonize all CSVs for a series according to its type.
    Returns a list of dataframes aligned on canonical column names.
    """
    frames: List[pd.DataFrame] = []
    for p in series.input_paths:
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        df = pd.read_csv(p)
        if series.type == ModelType.BASELINE:
            dfh = _harmonize_baseline(df)
        else:
            dfh = _harmonize_mga(df)
        frames.append(dfh)
    return frames


# -----------------------------
# Aggregation across replicas
# -----------------------------

def aggregate_by_epoch(frames: List[pd.DataFrame], columns: List[str]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Compute per-epoch mean and std across a list of frames.
    Returns (mean_df, std_df). If len(frames) == 1, std_df is None.
    Only columns present across frames are kept.
    """
    if len(frames) == 0:
        raise ValueError("No frames provided.")
    # Intersect available columns
    common_cols = set(['epoch']).union(*(set(df.columns) for df in frames))
    desired = ['epoch'] + [c for c in columns if c in common_cols]
    # Concatenate with a 'rep' index
    cat = []
    for i, df in enumerate(frames):
        sub = df[[c for c in desired if c in df.columns]].copy()
        sub['__rep__'] = i
        cat.append(sub)
    all_df = pd.concat(cat, ignore_index=True)
    # Group by epoch
    mean_df = all_df.groupby('epoch', as_index=False).mean(numeric_only=True)
    if len(frames) > 1:
        std_df = all_df.groupby('epoch', as_index=False).std(numeric_only=True).rename(columns={c: f"{c}__std" for c in desired if c != 'epoch'})
        return mean_df, std_df
    return mean_df, None


# -----------------------------
# Plotting primitives
# -----------------------------

def _plot_series_with_optional_error(ax: plt.Axes,
                                     x: np.ndarray,
                                     y: np.ndarray,
                                     yerr: Optional[np.ndarray],
                                     label: str,
                                     style: StyleConfig) -> None:
    """
    Plot mean line and optional whiskers as standard deviation error bars.
    """
    kwargs = style.to_matplotlib_kwargs()
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, label=label, capsize=2, linewidth=kwargs.get('linewidth', 1.0),
                    linestyle=kwargs.get('linestyle', '-'), color=kwargs.get('color', None),
                    marker=kwargs.get('marker', None), markersize=2, alpha=0.9)
    else:
        ax.plot(x, y, label=label, **kwargs)


def _finalize_axes(ax: plt.Axes, title: str, xlabel: str = "Epoch", ylabel: Optional[str] = None) -> None:
    """Set title and axis labels."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# -----------------------------
# Figure 1: Detection losses
# -----------------------------

def fig_detection(series_cfgs: List[DataSeriesConfig],
                  all_frames: Dict[str, List[pd.DataFrame]]) -> plt.Figure:
    """
    Build the detection losses figure with shape 2x4.
    """
    fig = plt.figure(constrained_layout=True, figsize=(10, 5))
    gs = GridSpec(2, 4, figure=fig)

    axes = np.empty((2, 4), dtype=object)
    for r in range(2):
        for c in range(4):
            axes[r, c] = fig.add_subplot(gs[r, c])

    titles = ["Total detection loss", "Box loss", "DFL loss", "CLS loss"]
    train_cols = DET_TRAIN_COLS
    val_cols   = DET_VAL_COLS

    # Plot each series
    for s in series_cfgs:
        frames = all_frames[s.name]
        # Aggregate for detection columns only
        mean_df, std_df = aggregate_by_epoch(frames, train_cols + val_cols)
        x = mean_df['epoch'].to_numpy()

        # Row 0: train
        for idx, col in enumerate(train_cols):
            if col in mean_df.columns:
                y = mean_df[col].to_numpy()
                yerr = std_df[f"{col}__std"].to_numpy() if std_df is not None and f"{col}__std" in std_df.columns else None
                _plot_series_with_optional_error(axes[0, idx], x, y, yerr, s.name, s.style)
                _finalize_axes(axes[0, idx], f"Train • {titles[idx]}", ylabel="Loss" if idx == 0 else None)
        # Row 1: val
        for idx, col in enumerate(val_cols):
            if col in mean_df.columns:
                y = mean_df[col].to_numpy()
                yerr = std_df[f"{col}__std"].to_numpy() if std_df is not None and f"{col}__std" in std_df.columns else None
                _plot_series_with_optional_error(axes[1, idx], x, y, yerr, s.name, s.style)
                _finalize_axes(axes[1, idx], f"Val • {titles[idx]}", ylabel="Loss" if idx == 0 else None)

    # Single legend outside
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
    return fig


# -----------------------------
# Figure 2: Segmentation losses (optional)
# -----------------------------

def fig_segmentation(series_cfgs: List[DataSeriesConfig],
                     all_frames: Dict[str, List[pd.DataFrame]]) -> Optional[plt.Figure]:
    """
    Build the segmentation losses figure with a 4x4 grid.
    Returns None if no series contains segmentation columns.
    """
    # Determine if any series has segmentation columns
    any_seg = False
    for s in series_cfgs:
        frames = all_frames[s.name]
        cols_present = set().union(*(set(f.columns) for f in frames))
        if any(c in cols_present for c in SEG_TRAIN_COLS + SEG_VAL_COLS):
            any_seg = True
            break
    if not any_seg:
        return None

    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    gs = GridSpec(4, 4, figure=fig)

    # Build axes with rowspan for column 0 in training and validation blocks
    axes = np.empty((4, 4), dtype=object)
    # Train totals span rows 0-1, col 0
    axes[0, 0] = fig.add_subplot(gs[0:2, 0])
    # Val totals span rows 2-3, col 0
    axes[2, 0] = fig.add_subplot(gs[2:4, 0])
    # The rest are 1x1
    for r in range(4):
        for c in range(4):
            if (r in (0, 1) and c == 0) or (r in (2, 3) and c == 0):
                continue
            axes[r, c] = fig.add_subplot(gs[r, c])

    # Titles and mapping per cell
    col_titles = {
        (0, 0): "Train • Seg total",
        (0, 1): "Train • P3 Dice",
        (0, 2): "Train • P4 Dice",
        (0, 3): "Train • P5 Dice",
        (1, 1): "Train • P3 BCE",
        (1, 2): "Train • P4 BCE",
        (1, 3): "Train • P5 BCE",
        (2, 0): "Val • Seg total",
        (2, 1): "Val • P3 Dice",
        (2, 2): "Val • P4 Dice",
        (2, 3): "Val • P5 Dice",
        (3, 1): "Val • P3 BCE",
        (3, 2): "Val • P4 BCE",
        (3, 3): "Val • P5 BCE",
    }
    col_map = {
        (0, 0): 'train/seg/total',
        (0, 1): 'train/seg/p3_dice',
        (0, 2): 'train/seg/p4_dice',
        (0, 3): 'train/seg/p5_dice',
        (1, 1): 'train/seg/p3_bce',
        (1, 2): 'train/seg/p4_bce',
        (1, 3): 'train/seg/p5_bce',
        (2, 0): 'val/seg/total',
        (2, 1): 'val/seg/p3_dice',
        (2, 2): 'val/seg/p4_dice',
        (2, 3): 'val/seg/p5_dice',
        (3, 1): 'val/seg/p3_bce',
        (3, 2): 'val/seg/p4_bce',
        (3, 3): 'val/seg/p5_bce',
    }

    for s in series_cfgs:
        frames = all_frames[s.name]
        # If a series lacks seg columns entirely, skip plotting it on this figure
        cols_present = set().union(*(set(f.columns) for f in frames))
        subset_cols = [v for v in col_map.values() if v in cols_present]
        if not subset_cols:
            continue

        mean_df, std_df = aggregate_by_epoch(frames, subset_cols)
        x = mean_df['epoch'].to_numpy()

        for pos, col in col_map.items():
            if col not in mean_df.columns:
                continue
            r, c = pos
            y = mean_df[col].to_numpy()
            yerr = std_df[f"{col}__std"].to_numpy() if std_df is not None and f"{col}__std" in std_df.columns else None
            _plot_series_with_optional_error(axes[r, c], x, y, yerr, s.name, s.style)
            _finalize_axes(axes[r, c], col_titles[pos], ylabel="Loss" if c == 0 else None)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
    return fig


# -----------------------------
# Figure 3: Validation performance
# -----------------------------

def fig_performance(series_cfgs: List[DataSeriesConfig],
                    all_frames: Dict[str, List[pd.DataFrame]]) -> plt.Figure:
    """
    Build the validation performance figure with shape 1x4.
    """
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), constrained_layout=True)

    titles = ["Precision (val)", "Recall (val)", "mAP@50 (val)", "mAP@50-95 (val)"]
    cols   = VAL_METRICS

    for s in series_cfgs:
        frames = all_frames[s.name]
        # These metrics are single series over epochs; we show evolution with mean±std if replicas
        mean_df, std_df = aggregate_by_epoch(frames, cols)
        x = mean_df['epoch'].to_numpy()

        for idx, col in enumerate(cols):
            if col not in mean_df.columns:
                continue
            y = mean_df[col].to_numpy()
            yerr = std_df[f"{col}__std"].to_numpy() if std_df is not None and f"{col}__std" in std_df.columns else None
            _plot_series_with_optional_error(axes[idx], x, y, yerr, s.name, s.style)
            _finalize_axes(axes[idx], titles[idx], ylabel=None)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
    return fig


# -----------------------------
# Public API
# -----------------------------

def compare_models(config_path: Path, out_dir: Path, prefix: Optional[str] = None) -> Tuple[Path, Optional[Path], Path]:
    """
    Entry point. Load YAML, read CSV logs, and write three figures to disk.

    Parameters
    ----------
    config_path : Path
        Path to YAML configuration describing series, inputs, style, and type.
    out_dir : Path
        Output directory for the generated PNGs.
    prefix : Optional[str]
        Optional filename prefix for outputs. Defaults to YAML stem.

    Returns
    -------
    Tuple[Path, Optional[Path], Path]
        Paths to detection figure, segmentation figure or None, and performance figure.
    """
    logging.info("Loading configuration: %s", config_path)
    cfgs = load_config(Path(config_path))

    logging.info("Config contains %d series.", len(cfgs))
    for s in cfgs:
        logging.info("Series '%s' | type=%s | replicas=%d", s.name, s.type.value, len(s.input_paths))

    # Load frames for each series
    all_frames: Dict[str, List[pd.DataFrame]] = {}
    for s in cfgs:
        all_frames[s.name] = load_series_frames(s)

    # Configure plotting
    configure_matplotlib()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = Path(config_path).stem

    # Figures
    det_fig = fig_detection(cfgs, all_frames)
    det_path = out_dir / f"{prefix}_detection.png"
    det_fig.savefig(det_path)
    plt.close(det_fig)

    seg_fig = fig_segmentation(cfgs, all_frames)
    seg_path: Optional[Path] = None
    if seg_fig is not None:
        seg_path = out_dir / f"{prefix}_segmentation.png"
        seg_fig.savefig(seg_path)
        plt.close(seg_fig)

    perf_fig = fig_performance(cfgs, all_frames)
    perf_path = out_dir / f"{prefix}_performance.png"
    perf_fig.savefig(perf_path)
    plt.close(perf_fig)

    logging.info("Wrote figures to %s", out_dir)
    return det_path, seg_path, perf_path


# -----------------------------
# CLI helper
# -----------------------------

def _setup_logging(verbosity: int = 1) -> None:
    """Configure logging level. verbosity=0: WARNING, 1: INFO, 2+: DEBUG"""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    """
    Simple CLI:
      python -m model_comparison /path/to/config.yaml /path/to/outdir [prefix] [verbosity]
    """
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m model_comparison CONFIG_YAML OUT_DIR [PREFIX] [VERBOSITY]", flush=True)
        sys.exit(1)
    cfg = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    prefix = sys.argv[3] if len(sys.argv) >= 4 else None
    verb = int(sys.argv[4]) if len(sys.argv) >= 5 else 1
    _setup_logging(verb)
    compare_models(cfg, outdir, prefix)


if __name__ == "__main__":
    main()
