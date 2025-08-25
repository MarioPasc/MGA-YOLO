from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mga_yolo.engine.checkpoint import rebuild_mga_model_from_minimal_ckpt


app = typer.Typer(help="MGA-YOLO checkpoint utilities")


@app.command()
def load(
    ckpt: Path = typer.Argument(..., exists=True, readable=True, help="Path to minimal .pt checkpoint"),
    cfg: Optional[Path] = typer.Option(None, "--cfg", help="Model YAML path; if omitted, try to infer from ckpt args"),
    summary: bool = typer.Option(True, "--summary/--no-summary", help="Print a short model summary after load"),
):
    """Load a minimal MGA checkpoint and optionally print a short model summary."""
    cfg_yaml: Path | dict
    if cfg is None:
        # Try to infer from checkpoint contents
        import torch
        raw = torch.load(ckpt, map_location='cpu')
        args = raw.get('train_args', {}) if isinstance(raw, dict) else {}
        yaml_file = (args or {}).get('model') or (args or {}).get('yaml_file')
        if yaml_file and Path(yaml_file).exists():
            cfg_yaml = Path(yaml_file)
        else:
            raise typer.BadParameter("--cfg is required if YAML path is not stored in checkpoint train_args")
    else:
        cfg_yaml = cfg

    model, raw = rebuild_mga_model_from_minimal_ckpt(ckpt, cfg_yaml)
    typer.echo(f"Loaded model: {type(model).__name__}")
    if summary:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        typer.echo(f"Parameters: total={total:,}, trainable={trainable:,}")
        # Print a few head keys presence
        sd = model.state_dict()
        keys = [k for k in sd.keys() if k.startswith('model.25')][:5]
        if keys:
            typer.echo("Sample head keys:")
            for k in keys:
                typer.echo(f"  - {k}: {tuple(sd[k].shape)}")


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
