"""Experiment orchestrator: run all (model x scale x fold) trainings maximizing GPU usage.

Reads an experiment config YAML (see `exp_cfg.yaml`) containing:
  cfg_root: root directory containing model + hyperparam configs
  kfold_root: directory with k-fold subfolders (fold_1, fold_2, ... each containing fold_i.yaml)
  output_root: base results directory
  scales: list of model scales (e.g. ['n','s','m','l'])
  models: list of mappings { ModelName: { model_cfg: ..., hyp_cfg: ... } }
  gpu: list of GPU identifiers (e.g. ['cuda:0','cuda:1'])

For every combination it launches training using `mga_yolo.engine.train.train`.
Concurrency strategy:
  * Maintain a per-GPU slot count (default 2 slots per GPU – configurable via env MGA_GPU_SLOTS or CLI).
  * Launch up to that number of concurrent processes per GPU.
  * As soon as a process finishes, schedule the next pending job on the freed slot.

Each job merges the hyperparameter YAML with dynamic fields:
  - model: path to model config (cfg_root/models/<model_cfg>)
  - model_scale: current scale
  - data: path to fold_i/fold_i.yaml
  - name/project: augmented to include model, scale, fold
  - device: assigned GPU string (e.g. 'cuda:0')

Resumes and checkpoints: we respect existing 'project' path inside hyp YAML if present, but by default
we override/extend to ensure unique experiment directories under output_root.

Logging: prints concise status lines on scheduling, start, finish, and failure.
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
import yaml # type: ignore
import typer

from mga_yolo.scripts import LOGGER
from mga_yolo.engine.train import train as train_entry  # noqa: F401 (imported for context)

app = typer.Typer(add_completion=False, help="Run performance comparison grid of experiments.")


@dataclass
class Experiment:
    model_name: str
    scale: str
    fold_index: int
    model_cfg: Path
    hyp_cfg: Path
    fold_yaml: Path
    device: str | None = None
    idx: int = 0  # global ordering index

    def tag(self) -> str:
        return f"{self.model_name}-yolov8{self.scale}-fold{self.fold_index}".replace("/", "_")


@dataclass
class GPUState:
    device: str
    max_slots: int
    active: dict[int, subprocess.Popen] = field(default_factory=dict)

    def free_slots(self) -> int:
        # Clean finished
        finished = [jid for jid, proc in self.active.items() if proc.poll() is not None]
        for jid in finished:
            self.active.pop(jid, None)
        return self.max_slots - len(self.active)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_experiments(cfg_path: Path) -> list[Experiment]:
    cfg = load_yaml(cfg_path)
    cfg_root = Path(cfg["cfg_root"]).expanduser().resolve()
    kfold_root = Path(cfg["kfold_root"]).expanduser().resolve()
    scales: list[str] = [str(s) for s in cfg.get("scales", [])]
    raw_models: list[dict] = cfg.get("models", [])

    # Discover folds (fold_* directories with YAML inside)
    folds: list[int] = []
    for d in sorted(kfold_root.glob("fold_*")):
        if d.is_dir():
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
            model_cfg = cfg_root / 'models' / spec['model_cfg']
            hyp_cfg = cfg_root / 'hyperparams' / spec['hyp_cfg']
            for scale in scales:
                for fold_idx in folds:
                    fold_yaml = kfold_root / f"fold_{fold_idx}" / f"fold_{fold_idx}.yaml"
                    experiments.append(Experiment(
                        model_name=model_name,
                        scale=scale,
                        fold_index=fold_idx,
                        model_cfg=model_cfg,
                        hyp_cfg=hyp_cfg,
                        fold_yaml=fold_yaml,
                        idx=gid,
                    ))
                    gid += 1
    return experiments


def prepare_config(exp: Experiment, output_root: Path, experiment_name: str) -> dict[str, Any]:
        """Merge base hyperparameters with dynamic fields for this experiment.

        Output directory layout required:
            {output_root}/{experiment_name}/{model}_{scale}_fold{fold}/

        YOLO uses project/name => project/name/* for run artifacts, so we set:
            project = {output_root}/{experiment_name}
            name    = {model}_{scale}_fold{fold}
        """
        hyp = load_yaml(exp.hyp_cfg)
        project_root = Path(output_root) / experiment_name
        project_root.mkdir(parents=True, exist_ok=True)
        run_name = f"{exp.model_name}_{exp.scale}_fold{exp.fold_index}"
        hyp.update(
                {
                        "model": str(exp.model_cfg),
                        "model_scale": exp.scale,
                        "data": str(exp.fold_yaml),
                        "name": run_name,
                        "project": str(project_root),
                }
        )
        return hyp


def launch_subprocess(exp: Experiment, cfg: dict[str, Any], device: str, python: str) -> subprocess.Popen:
    # We'll run a small inline python snippet to call train() to avoid needing new scripts.
    # Pass config via YAML dumped to a temp file under the experiment project directory.
    import tempfile, json
    project_dir = Path(cfg["project"]).expanduser().resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=project_dir)
    cfg_with_device = dict(cfg, device=device)
    json.dump(cfg_with_device, tmp_cfg)
    tmp_cfg.flush()
    tmp_cfg.close()
    code = (
        "import json,sys;"
        "from mga_yolo.engine.train import train;"
        f"cfg=json.load(open(r'{tmp_cfg.name}','r'));"
        "train(cfg)"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if device.startswith("cuda:") else ""
    return subprocess.Popen([python, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


_EPOCH_PATTERNS = [
    re.compile(r"[Ee]poch\s*(\d+)(?:\s*/\s*(\d+))?"),
    re.compile(r"\b(\d+)/(\d+)\b"),
]


def _extract_epoch(line: str) -> str | None:
    """Try to extract epoch progress as 'current/total' from a log line."""
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


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    gpu_slots: int = typer.Option(2, help="Concurrent experiments per GPU"),
    python_exec: str = typer.Option(sys.executable, help="Python interpreter to use for subprocesses"),
    dry_run: bool = typer.Option(False, help="Print planned experiments and exit"),
) -> None:
    """Execute all experiments maximizing GPU utilization."""
    exp_list = build_experiments(config)
    if not exp_list:
        typer.echo("No experiments found.")
        raise typer.Exit(0)
    exp_cfg = load_yaml(config)
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
    job_cfgs: dict[int, dict[str, Any]] = {e.idx: prepare_config(e, output_root, experiment_name) for e in exp_list}
    job_processes: dict[int, tuple[Experiment, subprocess.Popen, GPUState]] = {}

    def schedule_loop():
        while not pending.empty() or any(gs.active for gs in gpu_states):
            # Attempt to schedule new jobs
            for gs in gpu_states:
                while gs.free_slots() > 0 and not pending.empty():
                    exp = pending.get()
                    cfg = job_cfgs[exp.idx]
                    proc = launch_subprocess(exp, cfg, gs.device, python_exec)
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
    LOGGER.info("All experiments completed.")

    schedule_loop()
    for t in live_threads:
        t.join()


if __name__ == "__main__":  # pragma: no cover
    app()
