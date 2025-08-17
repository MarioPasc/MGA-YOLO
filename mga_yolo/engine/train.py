"""MGA-YOLO custom training entrypoint.

Usage (example):
    python -m mga_yolo.engine.train \
        --model configs/models/yolov8_test_segment_heads.yaml \
        --data configs/data/data.yaml \
        --epochs 1 --imgsz 640

This is a thin wrapper around Ultralytics' YOLO class to ensure that custom
MGA modules are imported before model construction.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MGA-YOLO Train")
    p.add_argument("--model", type=str, required=True, help="Path to MGA YAML model file")
    p.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", type=str, default="runs/mga")
    p.add_argument("--name", type=str, default="exp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info(f"[MGA] Starting training with model={args.model}")
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
