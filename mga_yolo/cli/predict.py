"""
CLI helper to run MGA-YOLO inference.

Example
-------
python -m mga_yolo.cli.predict \
       --config examples/stenosis_mga.yaml \
       --weights runs/mga-yolo/exp/weights/best.pt \
       --images data/stenosis/val/*.png \
       --save-feature-maps \
       --feature-dir runs/feature_vis
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import typer
from rich import print

from mga_yolo.cfg.defaults import MGAConfig
from mga_yolo.engine.inference import MaskGuidedInference

app = typer.Typer(help="Inference with MGA-YOLO (optionally saving feature maps)")


@app.command()
def main(
    config: str = typer.Option(..., help="Path to YAML config for MGA-YOLO"),
    weights: str = typer.Option(..., help="Path to trained .pt weights"),
    images: List[str] = typer.Option(..., help="One or more image/dir/glob patterns"),
    save_feature_maps: bool = typer.Option(
        True, help="Save mask/feat_before/feat_after figures"
    ),
    feature_dir: str = typer.Option(
        "feature_maps", help="Directory to store the figures"
    ),
    conf: float = typer.Option(0.25, help="Confidence threshold"),
):
    cfg = MGAConfig.load(config)
    predictor = MaskGuidedInference(
        cfg,
        weight_path=weights,
        save_feature_maps=save_feature_maps,
        feature_dir=feature_dir,
    )

    # resolve globs & folders
    expanded: list[str] = []
    for pat in images:
        p = Path(pat)
        if p.is_dir():
            expanded.extend(str(x) for x in p.iterdir() if x.is_file())
        else:
            expanded.extend(str(x) for x in p.parent.glob(p.name))

    print(f"[bold cyan]Running inference on {len(expanded)} file(s)…[/]")
    predictor(expanded, conf=conf)
    print("[bold green]Done![/]")


if __name__ == "__main__":
    app()
