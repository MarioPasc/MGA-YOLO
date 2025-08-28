import sys
from pathlib import Path
from typing import Any, Dict, List

import typer
import yaml # type: ignore
import warnings

import torch

from mga_yolo.engine.train import train as run_train


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _parse_overrides(argv: List[str]) -> Dict[str, Any]:
    """Parse unknown CLI args like --key value into a dict with typed values.

    Rules:
    - Keys must start with '--'.
    - Values are required (space-separated). Example: --lr0 0.001
    - Values are parsed using yaml.safe_load to support ints/floats/bools/lists.
    """
    overrides: Dict[str, Any] = {}
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if not tok.startswith("--"):
            raise typer.BadParameter(f"Unexpected argument '{tok}'. Use --key value pairs after --cfg.")
        key = tok[2:]
        if not key:
            raise typer.BadParameter("Empty option name '--' is not allowed.")
        if i + 1 >= n or argv[i + 1].startswith("--"):
            raise typer.BadParameter(f"Option '--{key}' requires a value.")
        raw_val = argv[i + 1]
        try:
            val = yaml.safe_load(raw_val)
        except Exception:
            val = raw_val  # fallback to string
        overrides[key] = val
        i += 2
    return overrides


def _merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(base) if base else {}
    for k, v in overrides.items():
        if k not in cfg:
            warnings.warn(f"Override key '{k}' not present in config; proceeding anyway.")
        cfg[k] = v
    # Normalize known aliases
    if "seg_scale_weight" in cfg and "seg_scale_weights" not in cfg:
        cfg["seg_scale_weights"] = cfg["seg_scale_weight"]
    return cfg


@app.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def main(
    ctx: typer.Context,
    cfg: Path = typer.Option(..., "--cfg", exists=True, file_okay=True, dir_okay=False, readable=True,
                              help="Path to YAML configuration for training."),
):
    """Train MGA-YOLO using a YAML config, allowing arbitrary --key value overrides after --cfg.

    Examples:
    - Base:    mga_train --cfg configs/defaults/defaults.yaml
    - Override mga_train --cfg configs/defaults/defaults.yaml --name Exp1 --lr0 0.001 --epochs 50
    """
    # Load base config
    with cfg.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    # Parse unknown overrides from CLI
    extra_args = list(ctx.args)
    overrides = _parse_overrides(extra_args)
    merged = _merge_config(base_cfg, overrides)

    # Default device if not provided
    if "device" not in merged:
        merged["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Delegate to engine-level entry point
    results = run_train(merged)

    # Minimal exit info
    typer.echo("Training finished.")
    raise typer.Exit(code=0)


if __name__ == "__main__":  # pragma: no cover
    app()
