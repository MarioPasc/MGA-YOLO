"""Experiment orchestrator for running a grid of Ultralytics (base YOLO) trainings.

This mirrors the MGA orchestrator (`mga_yolo/scripts/performance_comparison.py`) but
targets the plain Ultralytics pipeline via `tools/cli/train.py` (BaseFMTrainer capable).

Config YAML (example):
  cfg_root: configs               # root containing hyperparams/ (same layout as MGA) (required)
  kfold_root: /data/kfold_splits  # directory with fold_1/, fold_2/, ... each containing fold_i.yaml (required)
  output_root: results/base       # where to store runs (default: results)
  experiment_name: ultra_grid     # (default: experiment)
  scales: [n, s, m]               # list of YOLO scale suffixes (e.g. n,s,m,l,x) (required)
  gpu: [cuda:0, cuda:1]           # list of GPU device strings (default: [cuda:0])
  models:                         # list of single-key mappings (same pattern as MGA orchestrator)
	- Base:
		weights: "yolov8{scale}.pt"   # template or direct path/name. `{scale}` is formatted per scale.
		hyp_cfg: ultra_defaults.yaml  # file under cfg_root/hyperparams/

Semantics:
  For each (model x scale x fold) we launch `python tools/cli/train.py` with:
	--cfg <hyp_cfg>
	--weights <resolved_weights>
	--data <fold_yaml>
	--device <gpu>
	--project <output_root>/<experiment_name>
	--name <model>_<scale>_fold<k>

Concurrency:
  Per-GPU slot scheduling (configurable via --gpu-slots). Each scheduled process
  streams stdout; epoch progress lines are annotated with job tag & device.

Dry run:
  Use --dry-run to list planned experiment tags and exit.

Notes:
  * If `weights` (after formatting) does not exist on disk, it's passed verbatim to Ultralytics;
	this allows automatic download of official pretrained weights.
  * You can alternatively supply `weights_template` instead of `weights`; both are supported.
  * Any BASE_FM_* keys inside the hyp YAML will be exported by the train CLI to control
	feature map capture (see tools/cli/train.py docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List
import re

import yaml  # type: ignore
import typer
from ultralytics.utils import LOGGER

# Repository root (assumes this file is at <root>/tools/scripts/)
REPO_ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(add_completion=False, help="Run Ultralytics base model comparison grid.")


# ---------------------------- Data Structures ---------------------------- #

@dataclass
class Experiment:
	model_name: str
	scale: str
	fold_index: int
	weights: str  # resolved (formatted) weights string/path
	hyp_cfg: Path
	fold_yaml: Path
	device: str | None = None
	idx: int = 0

	def tag(self) -> str:
		return f"{self.model_name}-yolov8{self.scale}-fold{self.fold_index}".replace("/", "_")


@dataclass
class GPUState:
	device: str
	max_slots: int
	active: dict[int, subprocess.Popen] = field(default_factory=dict)

	def free_slots(self) -> int:
		finished = [jid for jid, proc in self.active.items() if proc.poll() is not None]
		for jid in finished:
			self.active.pop(jid, None)
		return self.max_slots - len(self.active)


# ---------------------------- Helpers ---------------------------- #

def _load_yaml(path: Path) -> dict:
	with path.open("r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def _resolve_weights(template: str, scale: str) -> str:
	# Allow either implicit formatting or pass-through.
    
	try:
		return template.format(scale=scale)
	except Exception:  # noqa: BLE001
		return template


def build_experiments(cfg_path: Path) -> list[Experiment]:
	cfg = _load_yaml(cfg_path)
	cfg_root = Path(cfg["cfg_root"]).expanduser().resolve()
	kfold_root = Path(cfg["kfold_root"]).expanduser().resolve()
	scales: list[str] = [str(s) for s in cfg.get("scales", [])]
	raw_models: list[dict] = cfg.get("models", [])

	# Discover folds
	folds: list[int] = []
	for d in sorted(kfold_root.glob("fold_*")):
		if not d.is_dir():
			continue
		try:
			idx = int(d.name.split("_")[-1])
		except ValueError:
			continue
		if (d / f"fold_{idx}.yaml").is_file():
			folds.append(idx)

	experiments: list[Experiment] = []
	gid = 0
	for model_map in raw_models:
		for model_name, spec in model_map.items():
			hyp_cfg = cfg_root / "hyperparams" / spec["hyp_cfg"]
			weights_template = spec.get("weights") or spec.get("weights_template") or spec.get("model_cfg")
			if not weights_template:
				raise ValueError(f"Model spec for '{model_name}' must define 'weights' or 'weights_template'.")
			for scale in scales:
				resolved_weights = _resolve_weights(str(weights_template), scale)
				for fold_idx in folds:
					fold_yaml = kfold_root / f"fold_{fold_idx}" / f"fold_{fold_idx}.yaml"
					experiments.append(
						Experiment(
							model_name=model_name,
							scale=scale,
							fold_index=fold_idx,
							weights=resolved_weights,
							hyp_cfg=hyp_cfg,
							fold_yaml=fold_yaml,
							idx=gid,
						)
					)
					gid += 1
	return experiments


def prepare_run(exp: Experiment, output_root: Path, experiment_name: str) -> dict[str, Any]:
	"""Return dynamic values for this run (project/name)."""
	project_root = output_root / experiment_name
	project_root.mkdir(parents=True, exist_ok=True)
	run_name = f"{exp.model_name}_{exp.scale}_fold{exp.fold_index}"
	return {"project": project_root, "name": run_name}


# ------------------------- Subprocess Management ------------------------- #

def launch_subprocess(exp: Experiment, run_meta: dict[str, Any], device: str, python: str) -> subprocess.Popen:
	"""Launch training via tools.cli.train module to ensure proper package imports."""
	project: Path = run_meta["project"]
	name: str = run_meta["name"]
	cmd: list[str] = [
		python,
		"-m",
		"tools.cli.train",
		"--cfg",
		str(exp.hyp_cfg),
		"--weights",
		exp.weights,
		"--data",
		str(exp.fold_yaml),
		"--device",
		device,
		"--project",
		str(project),
		"--name",
		name,
	]
	env = os.environ.copy()
	env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if device.startswith("cuda:") else ""
	# Prepend repo root to PYTHONPATH for safety
	env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH','')}" if env.get("PYTHONPATH") else str(REPO_ROOT)
	return subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


_EPOCH_PATTERNS = [
	re.compile(r"[Ee]poch\s*(\d+)(?:\s*/\s*(\d+))?"),
	re.compile(r"\b(\d+)/(\d+)\b"),
]


def _extract_epoch(line: str) -> str | None:
	for pat in _EPOCH_PATTERNS:
		m = pat.search(line)
		if m:
			cur = m.group(1)
			total = m.group(2) if m.lastindex and m.lastindex >= 2 and m.group(2) else None
			return f"{cur}/{total}" if total else cur
	return None


def stream_output(tag: str, proc: subprocess.Popen, device: str) -> None:
	assert proc.stdout is not None
	for raw in proc.stdout:
		line = raw.rstrip("\n")
		epoch = _extract_epoch(line)
		if epoch:
			LOGGER.info(f"[{tag}][{device}][epoch={epoch}] {line}")
		else:
			LOGGER.info(f"[{tag}][{device}] {line}")
	proc.wait()


# ------------------------------ Typer Command ----------------------------- #

@app.command()
def run(
	config: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
	gpu_slots: int = typer.Option(2, help="Concurrent experiments per GPU"),
	python_exec: str = typer.Option(sys.executable, help="Python interpreter to use for subprocesses"),
	dry_run: bool = typer.Option(False, help="Print planned experiments and exit"),
) -> None:
	"""Execute all experiments maximizing GPU utilization for base Ultralytics models."""
	exp_list = build_experiments(config)
	if not exp_list:
		typer.echo("No experiments found.")
		raise typer.Exit(0)

	exp_cfg = _load_yaml(config)
	output_root = Path(exp_cfg.get("output_root", "results")).expanduser().resolve()
	experiment_name = exp_cfg.get("experiment_name", "experiment")
	gpus = exp_cfg.get("gpu") or ["cuda:0"]
	gpu_states = [GPUState(device=g, max_slots=gpu_slots) for g in gpus]

	pending: queue.Queue[Experiment] = queue.Queue()
	for e in exp_list:
		pending.put(e)

	LOGGER.info(f"Planning {pending.qsize()} experiments across GPUs: {', '.join(gpus)}")
	if dry_run:
		for e in exp_list:
			typer.echo(e.tag())
		raise typer.Exit(0)

	live_threads: list[threading.Thread] = []
	run_meta: dict[int, dict[str, Any]] = {e.idx: prepare_run(e, output_root, experiment_name) for e in exp_list}
	job_processes: dict[int, tuple[Experiment, subprocess.Popen, GPUState]] = {}

	def schedule_loop():
		while not pending.empty() or any(gs.active for gs in gpu_states):
			# Schedule new jobs
			for gs in gpu_states:
				while gs.free_slots() > 0 and not pending.empty():
					exp = pending.get()
					meta = run_meta[exp.idx]
					proc = launch_subprocess(exp, meta, gs.device, python_exec)
					gs.active[exp.idx] = proc
					job_processes[exp.idx] = (exp, proc, gs)
					tag = exp.tag()
					t = threading.Thread(target=stream_output, args=(tag, proc, gs.device), daemon=True)
					t.start()
					live_threads.append(t)
					LOGGER.info(f"[SCHEDULED][{gs.device}] {tag} (slots left: {gs.free_slots()})")

			# Check completions
			finished_ids: list[int] = []
			for jid, (exp, proc, gs) in list(job_processes.items()):
				if proc.poll() is not None:
					ret = proc.returncode
					status = "OK" if ret == 0 else f"FAIL({ret})"
					LOGGER.info(f"[DONE][{gs.device}] {exp.tag()} -> {status}")
					finished_ids.append(jid)
			for jid in finished_ids:
				job_processes.pop(jid, None)
			time.sleep(1.0)

	schedule_loop()
	for t in live_threads:
		t.join()
	LOGGER.info("All experiments completed.")


if __name__ == "__main__":  # pragma: no cover
	app()
