"""K-Fold dataset splitter creating YOLO-compatible directory trees with symlinks.

This utility receives:
 1. A parent dataset directory containing the subfolders:
	   images/  (all images)
	   labels/  (YOLO txt annotation files matching image basenames)
	   masks/   (segmentation masks – NOT split, referenced globally)
 2. Number of folds (k)
 3. Output directory where k fold subdirectories will be created.

For each fold i (1-indexed) we create the structure (using "fold_{i}" naming):

{out}/
  fold_1/
	fold_1.yaml
	images/
	  train/
	  val/
	  test/         (empty unless later populated – created for Ultralytics config completeness)
	labels/
	  train/
	  val/
	  test/

The YAML file contains (paths are relative to the provided --out directory):

dataset: <absolute path to original dataset root (where masks/ lives)>
path: <absolute path to the output directory>
train: fold_1/images/train
val: fold_1/images/val
test: fold_1/images/test
masks_dir: masks
names: {0: stenosis}

Images and labels inside each fold are *symlinks* pointing back to the originals – no data
duplication. Validation set for fold i is the i-th slice; training set is the union of all other
folds. Test set is created empty (can be manually populated) – kept because downstream configs
expect a 'test' key. You may later decide to reuse val as test; in that case simply symlink.

Usage (CLI):
	python -m mga_yolo.utils.kfold_splitter \
		--dataset /path/to/dataset_root \
		--k 5 \
		--out /path/to/output_folds \
		--seed 42

Requirements satisfied by existing project dependencies (PyYAML, Typer).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import random
import shutil
from typing import Sequence

import typer
import yaml  # type: ignore

APP_NAME = "kfold-splitter"


IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

DEFAULTS = {
    "dataset": Path("/home/mpascual/research/datasets/angio/arcade_mga"),
    "k": 3,
    "out": Path("/home/mpascual/research/datasets/angio/kfold"),
    "seed": 42,
    "force": False,
}


@dataclass(frozen=True)
class FoldAssignment:
	"""Indices for one fold (validation subset) relative to master list of images."""

	fold_index: int  # 1-indexed
	val_indices: list[int]


def list_images(images_dir: Path) -> list[Path]:
	"""Return a sorted list of image files under images_dir with accepted extensions.

	Sorting ensures deterministic ordering before shuffling.
	"""

	if not images_dir.is_dir():  # pragma: no cover - defensive
		raise FileNotFoundError(f"Images directory not found: {images_dir}")
	images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
	images.sort()
	if not images:
		raise ValueError(f"No image files with extensions {IMAGE_EXTENSIONS} found in {images_dir}")
	return images


def validate_labels(images: Sequence[Path], labels_dir: Path) -> list[Path]:
	"""Return a list of label paths corresponding to each image.

	Raises if any expected label is missing to avoid silent training issues.
	"""

	missing: list[str] = []
	labels: list[Path] = []
	for img in images:
		label = labels_dir / (img.stem + ".txt")
		if not label.exists():
			missing.append(label.name)
		labels.append(label)
	if missing:
		raise FileNotFoundError(
			f"Missing {len(missing)} label files in '{labels_dir}': first few: {missing[:5]}"
		)
	return labels


def kfold_indices(n: int, k: int) -> list[FoldAssignment]:
	"""Compute k folds returning validation indices for each fold."""

	if k < 2:
		raise ValueError("k must be >= 2 for k-fold splitting")
	if k > n:
		raise ValueError(f"k={k} cannot exceed number of samples ({n})")
	fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
	indices = list(range(n))
	assignments: list[FoldAssignment] = []
	start = 0
	for i, size in enumerate(fold_sizes, start=1):
		end = start + size
		assignments.append(FoldAssignment(fold_index=i, val_indices=indices[start:end]))
		start = end
	return assignments


def ensure_clean_output_dir(out_dir: Path, force: bool) -> None:
	"""Create or (if --force) clear the output directory."""

	if out_dir.exists():
		if any(out_dir.iterdir()):
			if not force:
				raise FileExistsError(
					f"Output directory '{out_dir}' is not empty. Use --force to overwrite contents."
				)
			# Clear directory
			for p in out_dir.iterdir():
				if p.is_dir() and not p.is_symlink():
					shutil.rmtree(p)
				else:
					p.unlink()
	else:
		out_dir.mkdir(parents=True, exist_ok=True)


def relative_symlink(target: Path, link_path: Path) -> None:
	"""Create a relative symlink at link_path pointing to target (file)."""

	link_path.parent.mkdir(parents=True, exist_ok=True)
	if link_path.exists():  # pragma: no cover - defensive cleanup
		link_path.unlink()
	rel_target = Path(os_path_rel := path_rel(target, link_path.parent))
	link_path.symlink_to(rel_target)


def path_rel(target: Path, start: Path) -> str:
	"""Return a POSIX relative path from start to target."""

	return os.path.relpath(target, start).replace(os.sep, "/")


def write_fold_yaml(
	fold_dir: Path, dataset_root: Path, out_root: Path, fold_index: int, yaml_name: str = "fold_{}.yaml"
) -> Path:
	"""Write the fold YAML file returning its path."""

	yaml_path = fold_dir / yaml_name.format(fold_index)
	data = {
		"dataset": str(dataset_root.resolve()),
		"path": str(out_root.resolve()),
		"train": f"fold_{fold_index}/images/train",
		"val": f"fold_{fold_index}/images/val",
		"test": f"fold_{fold_index}/images/test",
		"masks_dir": "masks",
		"names": {0: "stenosis"},
	}
	with yaml_path.open("w", encoding="utf-8") as f:
		yaml.safe_dump(data, f, sort_keys=False)
	return yaml_path


def populate_fold(
	fold_assignment: FoldAssignment,
	images: Sequence[Path],
	labels: Sequence[Path],
	out_root: Path,
) -> None:
	"""Create symlink structure for a single fold."""

	i = fold_assignment.fold_index
	fold_dir = out_root / f"fold_{i}"
	# Create directories
	for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
		(fold_dir / sub).mkdir(parents=True, exist_ok=True)

	val_set = set(fold_assignment.val_indices)
	for idx, (img, lbl) in enumerate(zip(images, labels)):
		subset = "val" if idx in val_set else "train"
		img_link = fold_dir / "images" / subset / img.name
		lbl_link = fold_dir / "labels" / subset / lbl.name
		relative_symlink(img, img_link)
		relative_symlink(lbl, lbl_link)

def main(
	dataset: Path = typer.Option(
		DEFAULTS["dataset"],
		exists=True,
		file_okay=False,
		dir_okay=True,
		readable=True,
		help="Dataset root containing images/, labels/, masks/ (default from DEFAULTS)",
	),
	k: int = typer.Option(DEFAULTS["k"], min=2, help="Number of folds"),
	out: Path = typer.Option(
		DEFAULTS["out"], file_okay=False, dir_okay=True, help="Output directory for folds"
	),
	seed: int = typer.Option(DEFAULTS["seed"], help="Random seed for shuffling"),
	force: bool = typer.Option(DEFAULTS["force"], help="Overwrite existing output dir contents"),
) -> None:
	"""Generate K-fold symlink splits and YAML config files (all params have defaults)."""

	images_dir = dataset / "images"
	labels_dir = dataset / "labels"
	masks_dir = dataset / "masks"
	for required in [images_dir, labels_dir, masks_dir]:
		if not required.is_dir():
			raise typer.BadParameter(f"Required directory missing: {required}")

	ensure_clean_output_dir(out, force=force)

	images = list_images(images_dir)
	labels = validate_labels(images, labels_dir)

	rng = random.Random(seed)
	indices = list(range(len(images)))
	rng.shuffle(indices)
	images = [images[i] for i in indices]
	labels = [labels[i] for i in indices]

	assignments = kfold_indices(len(images), k)
	for assignment in assignments:
		populate_fold(assignment, images, labels, out)
		write_fold_yaml(out / f"fold_{assignment.fold_index}", dataset, out, assignment.fold_index)

	typer.echo(
		f"Created {k} folds in '{out}'. Each fold_X/ has YAML + symlinked train/val (test empty)."
	)


if __name__ == "__main__":  # pragma: no cover
	typer.run(main)
