"""pareto_performance_size.py

Generate Pareto trade-off plots (performance vs model size) across multiple
models, scales and folds using experiment folders containing `results.csv` and
`profiling.yaml` (profiling file provides size metrics like parameters/FLOPs).

Folder naming convention (flexible):
  <ModelName>_<scale>_fold<k>
Example:
  MaskCBAM_l_fold1, MaskSPADE_m_fold2, YOLOv8_s_fold3

YAML configuration (example):
root_dir: /path/to/experiments
out_dir: /path/to/output
models:
  - MaskedCBAM:
	- color: "#F94902"
	- marker: "s"
	- label: "CBAM"
  - YOLOv8:
	- color: "#AAAAAA"
scales: ['n','s','m','l']
folds: [1,2,3]
metrics_to_plot: ['metrics/mAP50(B)','metrics/mAP50-95(B)']
size_to_plot: ['model_parameters','model_gflops640']
method: 'max'   # or 'mean'

The script aggregates each (model, scale) across folds, computing the chosen
aggregation (mean or max over epochs) per metric within each fold, then mean±std
across folds. Size columns are assumed constant across epochs (first non-null).

Requirement adjustment: produce ONE IMAGE PER (metric, size) pair instead of a
grid. Each figure includes all model-scale points, metric std whiskers across
folds, annotated scale letter, connecting lines per model across scales, and
the global Pareto front (non-dominated: minimal size, maximal metric).

Matplotlib style is unified via `configure_matplotlib` imported from
`model_comparison`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Any, cast

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mga_yolo.utils.plotting.model_comparison import configure_matplotlib  # reuse project style

# Known aliases for size keys seen in profiling files
SIZE_ALIASES = {
	"model_gflops640": "model_gflops_640",
	"gflops640": "model_gflops_640",
	"gflops@640": "model_gflops_640",
	"gflops": "model_gflops",
	"params": "model_parameters",
}

# Optional model-name aliases (folder → config key)
DEFAULT_MODEL_ALIASES = {
	"yolov8": "BaseYOLO",
	"yolo": "BaseYOLO",
}
# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------


@dataclass
class ModelStyle:
	color: Optional[str] = None
	linestyle: str = "-"
	marker: Optional[str] = None
	markersize: float = 4.0
	label: Optional[str] = None

	def mpl_kwargs(self) -> Dict[str, object]:
		out: Dict[str, object] = {}
		if self.color: out['color'] = self.color
		if self.linestyle: out['linestyle'] = self.linestyle
		if self.marker: out['marker'] = self.marker
		if self.markersize: out['markersize'] = self.markersize
		return out


@dataclass
class PointStats:
	model_key: str
	scale: str
	metric_values: Dict[str, float]        # mean across folds
	metric_stds: Dict[str, float]          # std across folds
	size_values: Dict[str, float]          # mean size (should be constant)


# --------------------------------------------------------------------------------------
# YAML parsing (flexible list-of-dicts style used in repo)
# --------------------------------------------------------------------------------------


def _flatten_style_list(style_list: Sequence[Mapping]) -> Dict[str, object]:
	flat: Dict[str, object] = {}
	for item in style_list:
		if not isinstance(item, Mapping):
			continue
		for k, v in item.items():
			flat[str(k)] = v
	return flat


def load_performance_size_config(path: Path) -> Dict[str, object]:
	with open(path, 'r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f)
	required = ['root_dir', 'out_dir', 'models', 'scales', 'folds', 'metrics_to_plot', 'size_to_plot']
	for r in required:
		if r not in cfg:
			raise ValueError(f"Config missing required key: {r}")
	cfg.setdefault('method', 'mean')

	# Parse models list into mapping model_key -> ModelStyle
	model_styles: Dict[str, ModelStyle] = {}
	for entry in cfg['models']:
		if not isinstance(entry, Mapping):
			continue
		for model_key, style_seq in entry.items():
			style_dict = _flatten_style_list(style_seq if isinstance(style_seq, list) else [])
			color = style_dict.get('color')
			linestyle = style_dict.get('linestyle', '-')
			marker = style_dict.get('marker')
			markersize_raw = style_dict.get('markersize', 4.0)
			try:
				markersize = float(markersize_raw)  # type: ignore[arg-type]
			except Exception:  # pragma: no cover - fallback
				markersize = 4.0
			label = style_dict.get('label', model_key)
			model_styles[str(model_key)] = ModelStyle(
				color=str(color) if isinstance(color, str) else None,
				linestyle=str(linestyle),
				marker=str(marker) if isinstance(marker, str) else None,
				markersize=markersize,
				label=str(label) if isinstance(label, str) else str(model_key),
			)
	cfg['model_styles'] = model_styles

	# Optional explicit model aliases in YAML
	# Example:
	# model_aliases:
	#   yolov8: BaseYOLO
	model_aliases = cfg.get('model_aliases', {}) or {}
	if not isinstance(model_aliases, dict):
		logging.warning("Ignoring non-dict model_aliases in config.")
		model_aliases = {}
	cfg['model_aliases'] = {str(k).lower(): str(v) for k, v in model_aliases.items()}
	return cfg


# --------------------------------------------------------------------------------------
# Experiment directory scanning and data extraction
# --------------------------------------------------------------------------------------


def _canonical_model_key(name: str) -> str:
	return name.lower().replace('masked', 'mask')  # normalize for matching


def discover_runs(root: Path) -> List[Path]:
	runs = [p for p in root.iterdir() if p.is_dir()]
	logging.debug("discover_runs: %d dirs at %s", len(runs), root)
	return runs


def parse_run_dir(dir_path: Path) -> Optional[Tuple[str, str, int]]:
	"""Return (model_base, scale, fold) or None if pattern mismatched."""
	name = dir_path.name
	parts = name.split('_')
	if len(parts) < 3:
		return None
	fold_token = parts[-1]
	if not fold_token.startswith('fold'):
		return None
	try:
		fold = int(fold_token.replace('fold', ''))
	except ValueError:
		return None
	scale = parts[-2]
	model_base = '_'.join(parts[:-2])
	logging.debug("parse_run_dir: %s → model=%s scale=%s fold=%d", dir_path.name, model_base, scale, fold)
	return model_base, scale, fold


def load_results_csv(path: Path) -> Optional[pd.DataFrame]:
	csv_path = path / 'results.csv'
	if not csv_path.exists():
		logging.warning("results.csv missing in %s", path.name)
		return None
	try:
		df = pd.read_csv(csv_path)
		logging.debug("Loaded %s with %d rows, %d cols", csv_path.name, len(df), df.shape[1])
		return df
	except Exception as e:
		logging.warning("Failed reading %s: %s", csv_path, e)
		return None


def extract_fold_metrics(df: pd.DataFrame, metrics: Sequence[str], method: str) -> Dict[str, float]:
	"""Compute per-fold metric summary (performance metrics only) from one run.

	Size metrics are sourced from profiling.yaml separately.
	"""
	out_metrics: Dict[str, float] = {}
	for m in metrics:
		if m not in df.columns:
			logging.debug("Metric column missing in results.csv: %s", m)
			continue
		series = pd.to_numeric(df[m], errors='coerce').dropna()
		if series.empty:
			logging.debug("Metric column empty after coercion: %s", m)
			continue
		out_metrics[m] = float(series.max()) if method == 'max' else float(series.mean())
		logging.debug("Metric %s (%s over epochs) = %.6f", m, method, out_metrics[m])
	return out_metrics


def _normalize_size_key(k: str) -> str:
	k2 = SIZE_ALIASES.get(k, k)
	# Also normalize common variants
	k2 = k2.replace(" ", "_").replace("-", "_")
	return k2

def load_profiling_sizes(run_dir: Path, size_cols: Sequence[str]) -> Dict[str, float]:
	"""Load size metrics from profiling.yaml (flat or nested)."""
	profiling_path = run_dir / 'profiling.yaml'
	if not profiling_path.exists():
		logging.warning("profiling.yaml missing in %s", run_dir.name)
		return {}
	try:
		with open(profiling_path, 'r', encoding='utf-8') as f:
			data = yaml.safe_load(f)
	except Exception as e:  # pragma: no cover
		logging.warning("Failed to parse profiling.yaml in %s: %s", run_dir.name, e)
		return {}
	# Normalize requested size keys once
	normalized_targets = {_normalize_size_key(s) for s in size_cols}
	results: Dict[str, float] = {}

	def recurse(obj: Any, path: str = "") -> None:
		if isinstance(obj, Mapping):
			for k, v in obj.items():
				new_path = f"{path}.{k}" if path else str(k)
				recurse(v, new_path)
		elif isinstance(obj, (list, tuple)):
			for i, v in enumerate(obj):
				recurse(v, f"{path}[{i}]")
		else:
			leaf_key = path.split('.')[-1]
			leaf_key_norm = _normalize_size_key(leaf_key)
			if leaf_key_norm in normalized_targets and isinstance(obj, (int, float)):
				results[leaf_key_norm] = float(obj)

	recurse(data)
	for s in normalized_targets:
		if s not in results:
			logging.debug("Profiling size key missing: %s in %s", s, run_dir.name)
		else:
			logging.debug("Profiling %s = %.6f in %s", s, results[s], run_dir.name)
	return results


def aggregate_across_folds(per_fold: List[Tuple[Dict[str, float], Dict[str, float]]], metrics: Sequence[str], size_cols: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
	"""Aggregate list of (metric_dict, size_dict) across folds.

	Returns: (metric_means, metric_stds, size_means).
	Missing metrics are skipped (only computed over folds where present).
	"""
	metric_values: Dict[str, List[float]] = {m: [] for m in metrics}
	size_values: Dict[str, List[float]] = {s: [] for s in size_cols}
	for m_dict, s_dict in per_fold:
		for m, v in m_dict.items():
			if m in metric_values:
				metric_values[m].append(v)
		for s, v in s_dict.items():
			if s in size_values:
				size_values[s].append(v)

	metric_means: Dict[str, float] = {}
	metric_stds: Dict[str, float] = {}
	for m, vals in metric_values.items():
		if len(vals) == 0:
			continue
		arr = np.asarray(vals, dtype=float)
		metric_means[m] = float(arr.mean())
		metric_stds[m] = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0

	size_means: Dict[str, float] = {}
	for s, vals in size_values.items():
		if len(vals) == 0:
			continue
		arr = np.asarray(vals, dtype=float)
		size_means[s] = float(arr.mean())

	return metric_means, metric_stds, size_means


# --------------------------------------------------------------------------------------
# Pareto computation
# --------------------------------------------------------------------------------------


def compute_pareto(points: List[PointStats], size_key: str, metric_key: str) -> List[PointStats]:
	"""Return non-dominated points (min size, max metric)."""
	# Filter points with both values present
	valid = [p for p in points if size_key in p.size_values and metric_key in p.metric_values]
	logging.debug("compute_pareto: valid points=%d for size=%s metric=%s", len(valid), size_key, metric_key)
	# Sort by size ascending, metric descending to make scan easy
	valid.sort(key=lambda p: (p.size_values[size_key], -p.metric_values[metric_key]))
	pareto: List[PointStats] = []
	best_metric = -np.inf
	for p in valid:
		metric = p.metric_values[metric_key]
		if metric > best_metric + 1e-12:  # strictly better metric
			pareto.append(p)
			best_metric = metric
	return pareto


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def plot_one(metric: str, size_key: str, points: List[PointStats], model_styles: Mapping[str, ModelStyle], out_path: Path) -> Path:
	configure_matplotlib()
	fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)

	# Group by model
	by_model: Dict[str, List[PointStats]] = {}
	for p in points:
		if metric in p.metric_values and size_key in p.size_values:
			by_model.setdefault(p.model_key, []).append(p)

	# Scatter points
	for plist in by_model.values():
		for p in plist:
			style = model_styles.get(p.model_key, ModelStyle())
			x = p.size_values[size_key]
			y = p.metric_values[metric]
			yerr = p.metric_stds.get(metric, 0.0)
			ax.errorbar([x], [y], yerr=[[yerr], [yerr]], fmt=style.marker or 'o', color=style.color,
				markersize=style.markersize, elinewidth=0.8, capsize=2, alpha=0.95, linestyle='None')
			ax.text(x, y, f" {p.scale}", va='center', ha='left', fontsize=7)

	# Connect per-model sequences ordered by size
	for mkey, plist in by_model.items():
		if len(plist) < 2:
			continue
		plist.sort(key=lambda p: p.size_values[size_key])
		style = model_styles.get(mkey, ModelStyle())
		ax.plot([p.size_values[size_key] for p in plist], [p.metric_values[metric] for p in plist],
			color=style.color, linestyle=style.linestyle, linewidth=1.0, alpha=0.6)
	logging.info("Plotted figure: %s vs %s | models=%d points=%d",
				 metric, size_key, len(by_model), sum(len(v) for v in by_model.values()))

	# Pareto front overlay
	pareto = compute_pareto(points, size_key=size_key, metric_key=metric)
	pareto = [p for p in pareto if metric in p.metric_values and size_key in p.size_values]
	if len(pareto) >= 2:
		pareto_sorted = sorted(pareto, key=lambda p: p.size_values[size_key])
		ax.plot([p.size_values[size_key] for p in pareto_sorted],
			[p.metric_values[metric] for p in pareto_sorted],
			color='black', linestyle='--', linewidth=1.2, alpha=0.8, label='Pareto')

	ax.set_xlabel(size_key.replace('_', ' '))
	ax.set_ylabel(metric)
	if 'param' in size_key.lower() or 'flop' in size_key.lower():
		ax.set_xscale('log')
	ax.grid(True, alpha=0.3)

	# Legend
	import matplotlib.lines as mlines
	handles = []
	labels = []
	for mkey, style in model_styles.items():
		if mkey not in by_model:
			continue
		h = mlines.Line2D([], [], marker=style.marker or 'o', color=style.color, linestyle='None', markersize=style.markersize)
		handles.append(h)
		labels.append(style.label or mkey)
	if len(pareto) >= 2:
		handles.append(mlines.Line2D([], [], color='black', linestyle='--', label='Pareto'))
		labels.append('Pareto')
	if handles:
		ax.legend(handles, labels, loc='best', fontsize=7)

	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path)
	plt.close(fig)
	logging.info('Saved %s', out_path)
	return out_path


def plot_multiple(points: List[PointStats], model_styles: Mapping[str, ModelStyle], metrics: Sequence[str], sizes: Sequence[str], out_dir: Path) -> List[Path]:
	outputs: List[Path] = []
	for metric in metrics:
		for size_key in sizes:
			outfile = out_dir / f"pareto_{metric.replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')}__vs__{size_key}.png"
			outputs.append(plot_one(metric, size_key, points, model_styles, outfile))
	return outputs


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------


def build_pareto_from_config(cfg_path: Path) -> Path:
	cfg = load_performance_size_config(Path(cfg_path))
	root = Path(str(cfg['root_dir'])).expanduser()
	out_dir = Path(str(cfg['out_dir'])).expanduser()

	if not isinstance(cfg['metrics_to_plot'], (list, tuple)):
		raise TypeError('metrics_to_plot must be a list')
	if not isinstance(cfg['size_to_plot'], (list, tuple)):
		raise TypeError('size_to_plot must be a list')
	if not isinstance(cfg['folds'], (list, tuple)):
		raise TypeError('folds must be a list')
	if not isinstance(cfg['scales'], (list, tuple)):
		raise TypeError('scales must be a list')

	metrics: List[str] = [str(m) for m in cast(Sequence[Any], cfg['metrics_to_plot'])]
	# Normalize size keys early so downstream uses a single canonical form
	sizes: List[str] = [_normalize_size_key(str(s)) for s in cast(Sequence[Any], cfg['size_to_plot'])]
	method_obj: Any = cfg.get('method', 'mean')
	method: str = str(method_obj).lower()
	folds: List[int] = [int(f) for f in cast(Sequence[Any], cfg['folds'])]
	scales: List[str] = [str(s) for s in cast(Sequence[Any], cfg['scales'])]
	model_styles: Dict[str, ModelStyle] = dict(cast(Dict[str, ModelStyle], cfg['model_styles']))  # type: ignore

	logging.info("Scanning experiment root: %s", root)
	runs = discover_runs(root)
	logging.info("Found %d run directories", len(runs))

	# Map canonical name for quick lookup
	canonical_to_modelkey: Dict[str, str] = {}
	for mk in model_styles.keys():
		canonical_to_modelkey[_canonical_model_key(mk)] = mk
		# allow variant without 'mask' prefix as fallback
		if mk.lower().startswith('masked'):
			canonical_to_modelkey[_canonical_model_key(mk).replace('mask', '')] = mk
	# Add default aliases only for models present in styles
	model_aliases_raw = cfg.get('model_aliases', {})
	if not isinstance(model_aliases_raw, dict):
		model_aliases_raw = {}
	model_aliases: Dict[str, str] = {str(k): str(v) for k, v in model_aliases_raw.items()}
	for alias, target in DEFAULT_MODEL_ALIASES.items():
		if target in model_styles and alias not in model_aliases:
			model_aliases[alias] = target

	collected: List[PointStats] = []
	# Build nested dictionary: (model_key, scale) -> list of per-fold tuples
	accumulator: Dict[Tuple[str, str], List[Tuple[Dict[str, float], Dict[str, float]]]] = {}

	for rdir in runs:
		parsed = parse_run_dir(rdir)
		if parsed is None:
			logging.debug("Skipping dir (pattern mismatch): %s", rdir.name)
			continue
		model_base, scale, fold = parsed
		if scale not in scales or fold not in folds:
			logging.debug("Skipping dir (scale/fold not requested): %s", rdir.name)
			continue
		canon = _canonical_model_key(model_base)
		# find model key by prefix match among canonical mapping, then aliases
		matched_key: Optional[str] = None
		for ck, original in sorted(canonical_to_modelkey.items(), key=lambda x: -len(x[0])):
			if canon.startswith(ck):
				matched_key = original
				break
		if matched_key is None:
			# Try aliases: if canon starts with alias, map to target style key
			for alias, target in model_aliases.items():
				if canon.startswith(alias.lower()):
					matched_key = target
					logging.debug("Alias matched: %s → %s (dir=%s)", alias, target, rdir.name)
					break
		if matched_key is None:
			logging.debug("Skipping dir (unmatched model): %s (canon=%s)", rdir.name, canon)
			continue
		df = load_results_csv(rdir)
		if df is None:
			continue
		fold_metrics = extract_fold_metrics(df, metrics, method)
		if not fold_metrics:
			logging.debug("Skipping dir (no usable metrics after extraction): %s", rdir.name)
			continue
		fold_sizes = load_profiling_sizes(rdir, sizes)
		if not any(k in fold_sizes for k in sizes):
			logging.debug("Skipping dir (no requested size keys in profiling): %s", rdir.name)
			# Still store metrics so other plots without these sizes work
		accumulator.setdefault((matched_key, scale), []).append((fold_metrics, fold_sizes))

	# Aggregate
	for (model_key, scale), per_fold in accumulator.items():
		metric_means, metric_stds, size_means = aggregate_across_folds(per_fold, metrics, sizes)
		if not metric_means:
			logging.debug("No metric means for %s %s", model_key, scale)
		if not size_means:
			logging.debug("No size means for %s %s", model_key, scale)
		collected.append(PointStats(model_key=model_key, scale=scale,
									metric_values=metric_means, metric_stds=metric_stds,
									size_values=size_means))

	if not collected:
		raise RuntimeError("No data collected; ensure directory names and config match.")

	out_dir.mkdir(parents=True, exist_ok=True)
	outputs = plot_multiple(collected, model_styles, metrics, sizes, out_dir)
	return outputs[-1]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _setup_logging(verbosity: int = 1):
	level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
	logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(message)s')


def main():  # pragma: no cover - CLI utility
	import sys
	if len(sys.argv) < 2:
		print("Usage: python -m mga_yolo.utils.plotting.pareto_performance_size CONFIG_YAML [VERBOSITY]", flush=True)
		sys.exit(1)
	cfg_path = Path(sys.argv[1])
	verb = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
	_setup_logging(verb)
	build_pareto_from_config(cfg_path)


if __name__ == '__main__':  # pragma: no cover
	main()
