"""
Command-line entrypoint: `python -m mga_yolo.cli.train --config <yaml>`.
"""
from __future__ import annotations

import typer
from rich import print

from mga_yolo.cfg.defaults import MGAConfig
from mga_yolo.engine.trainer import MaskGuidedTrainer

app = typer.Typer(help="Train MGA-YOLO with mask-guided CBAM hooks")


@app.command()
def main(config: str):
    cfg = MGAConfig.load(config)
    print("[bold cyan]Loaded config:[/]\n", cfg)
    trainer = MaskGuidedTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    app()
