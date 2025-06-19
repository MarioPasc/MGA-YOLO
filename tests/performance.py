"""
Performance comparison between base YOLOv8 and MGA-YOLO.

Usage:
    python -m tests.performance --mode base --config path/to/config.yaml
    python -m tests.performance --mode mga --config path/to/config.yaml

This script allows training and comparing the standard YOLOv8 model against 
the Mask-Guided Attention version using the same configuration file.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from mga_yolo.cfg.defaults import MGAConfig
from mga_yolo.engine.trainer import MaskGuidedTrainer
from mga_yolo.external.ultralytics.ultralytics import YOLO

console = Console()
app = typer.Typer(help="Compare base YOLOv8 vs MGA-YOLO performance")


def train_base_yolo(cfg: MGAConfig) -> tuple[float, YOLO]:
    """Train a standard YOLOv8 model without mask-guided attention."""
    banner = r"""
╔════════════════════════════════════════════════╗
║               Base YOLO Training               ║
╚════════════════════════════════════════════════╝
"""
    print(f"[cyan]{banner}[/cyan]")
    
    start_time = time.time()
    
    # Initialize the YOLO model
    model = YOLO(cfg.model_cfg)
    
    # Log configuration
    print(f"[bold cyan]Training Configuration:[/bold cyan]")
    print(f"  Model: {cfg.model_cfg}")
    print(f"  Data: {cfg.data_yaml}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Image Size: {cfg.imgsz}")
    print(f"  Batch Size: {cfg.batch}")
    print(f"  Device: {cfg.device}")
    
    # Train the model
    model.train(
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name=f"{cfg.name}_base",
        iou=cfg.iou,
        **cfg.augmentation_config,
    )
    
    elapsed = time.time() - start_time
    print(f"[bold green]Base YOLO training completed in {elapsed:.2f} seconds[/bold green]")
    
    return elapsed, model


def train_mga_yolo(cfg: MGAConfig) -> tuple[float, YOLO]:
    """Train an MGA-YOLO model with mask-guided attention."""
    print(f"[bold cyan]Training MGA-YOLO model...[/bold cyan]")
    
    start_time = time.time()
    
    # Initialize MGA-YOLO trainer
    trainer = MaskGuidedTrainer(cfg)
    
    # Train the model
    model = trainer.train()
    
    elapsed = time.time() - start_time
    print(f"[bold green]MGA-YOLO training completed in {elapsed:.2f} seconds[/bold green]")
    
    return elapsed, model


def compare_results(base_time: float, mga_time: float, cfg: MGAConfig) -> None:
    """Display a comparison table of results."""
    table = Table(title="Training Performance Comparison")
    
    table.add_column("Model", style="cyan")
    table.add_column("Training Time (s)", style="green")
    table.add_column("Configuration", style="yellow")
    
    table.add_row(
        "Base YOLOv8", 
        f"{base_time:.2f}", 
        f"epochs={cfg.epochs}, imgsz={cfg.imgsz}, batch={cfg.batch}"
    )
    
    table.add_row(
        "MGA-YOLO", 
        f"{mga_time:.2f}", 
        f"epochs={cfg.epochs}, imgsz={cfg.imgsz}, batch={cfg.batch}, target_layers={cfg.target_layers}"
    )
    
    console.print(table)


@app.command()
def main(
    mode: str = typer.Option(
        "compare", 
        "--mode", "-m",
        help="Training mode: base YOLOv8, MGA-YOLO, or compare both",
        case_sensitive=False,
    ),
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
):
    """Run performance tests comparing base YOLOv8 and MGA-YOLO."""
    # Validate mode manually
    valid_modes = ["base", "mga", "compare"]
    if mode.lower() not in valid_modes:
        print(f"[bold red]Error: mode must be one of {valid_modes}[/bold red]")
        raise typer.Exit(code=1)
    
    mode = mode.lower()  # Normalize to lowercase
    cfg = MGAConfig.load(config)
    
    base_time = 0.0
    mga_time = 0.0
    
    if mode in ["base", "compare"]:
        base_time, base_model = train_base_yolo(cfg)
    
    if mode in ["mga", "compare"]:
        mga_time, mga_model = train_mga_yolo(cfg)
    
    if mode == "compare":
        compare_results(base_time, mga_time, cfg)


if __name__ == "__main__":
    app()