"""mask_showcase_precomputed.py

Assemble side‑by‑side comparison panels of pre‑computed downsampled masks.

Directory layout expected:

    INPUT_DIR/
        nearest/
            someprefix_p3.png
            someprefix_p4.png
            someprefix_p5.png
        area/
            ...
        maxpool/
        pyrdown/
        skeleton_bresenham/
        (any other methods ...)

Each method directory must contain images for three pyramid levels whose
filenames end with: "_p3.png", "_p4.png", "_p5.png" (case‑insensitive). The
prefix part may vary per method; only the suffix pattern matters.

Output mirrors `mask_showcase_ds.generate_showcase`: one PNG per level (P3, P4,
P5) with columns = methods. The leftmost subplot carries the composite ylabel
"P{level}\n(HxW)"; all axes are tickless.

Differences vs. the original on-demand downsampling script:
    * Images are used exactly as found (no binarization / thresholding).
    * Display uses interpolation='none' (no pixel smoothing or replication).
    * Method column order is fixed: nearest, area, maxpool, pyrdown, skeleton_bresenham.

Usage:
    python -m mga_yolo.utils.plotting.mask_showcase_precomputed INPUT_DIR OUT_DIR \
        [--prefix PREFIX] [--keep-order]

If --prefix is omitted the stem of the first discovered *_p3.png file is used.
Order of methods is alphabetical unless --keep-order is passed, in which case
the natural filesystem iteration order (per Path.iterdir) is preserved.

Matplotlib styling reuses `configure_matplotlib` from `model_comparison`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mga_yolo.utils.plotting.model_comparison import configure_matplotlib  # style

LOGGER = logging.getLogger(__name__)

LEVEL_SUFFIXES: Dict[str, str] = {"P3": "_p3.png", "P4": "_p4.png", "P5": "_p5.png"}
# Fixed desired visualization order
METHOD_VIS_ORDER: List[str] = [
    "nearest",
    "area",
    "maxpool",
    "pyrdown",
    "skeleton_bresenham",
]


def _gather_methods(root: Path) -> List[Path]:
    """Return list of child directories (methods) under root (non-empty)."""
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory '{root}' does not exist or is not a directory")
    methods = [p for p in root.iterdir() if p.is_dir()]
    if not methods:
        raise ValueError(f"No method subdirectories found in '{root}'")
    return methods


def _load_mask_image(path: Path) -> np.ndarray:
    """Load an image as grayscale uint8 array (no binarization)."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    return arr


def _find_level_image(method_dir: Path, suffix: str) -> Optional[Path]:
    """Find first file in method_dir that endswith given suffix (case-insensitive)."""
    suffix_lower = suffix.lower()
    for p in method_dir.iterdir():
        if p.is_file() and p.name.lower().endswith(suffix_lower):
            return p
    return None


def _collect_level_images(
    method_dirs: Sequence[Path], level: str, suffix: str
) -> Dict[str, Tuple[np.ndarray, Path]]:
    """For a given pyramid level, map method -> (mask_array, source_path).

    Missing images raise a ValueError listing absent methods.
    """
    missing: List[str] = []
    out: Dict[str, Tuple[np.ndarray, Path]] = {}
    for d in method_dirs:
        img_path = _find_level_image(d, suffix)
        if img_path is None:
            missing.append(d.name)
            continue
        try:
            arr = _load_mask_image(img_path)
        except Exception as e:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load {img_path}: {e}") from e
        out[d.name] = (arr, img_path)
    if missing:
        raise ValueError(
            f"Missing level '{level}' image(s) with suffix '{suffix}' for methods: {', '.join(missing)}"
        )
    return out


def _plot_panel(
    level: str,
    data: Dict[str, Tuple[np.ndarray, Path]],
    out_path: Path,
) -> None:
    """Create 1xN panel for one pyramid level using provided mask arrays."""
    configure_matplotlib()
    # Preserve the fixed visualization order filtering to available methods
    methods = [m for m in METHOD_VIS_ORDER if m in data]
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.4, 1.4), constrained_layout=True)
    if n == 1:
        axes = [axes]  # type: ignore
    # Determine representative size from first method
    first_arr = data[methods[0]][0]
    h, w = first_arr.shape
    for idx, m in enumerate(methods):
        arr, src = data[m]
        ax = axes[idx]
        ax.imshow(arr, cmap="gray", interpolation="none")
        ax.set_title(m, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.set_ylabel(f"{level}\n({h}x{w})")
        else:
            ax.set_ylabel("")
    fig.savefig(out_path)
    plt.close(fig)
    LOGGER.info("Saved %s", out_path)


def generate_showcase_precomputed(
    input_dir: Path,
    out_dir: Path,
    prefix: Optional[str] = None,
    keep_order: bool = False,
) -> List[Path]:
    """High-level driver assembling panels from precomputed per-method images.

    Returns list of produced figure paths (one per pyramid level).
    """
    method_dirs = _gather_methods(input_dir)
    # Map name -> path, then build ordered list following METHOD_VIS_ORDER (ignore keep_order flag now)
    name_to_dir: Dict[str, Path] = {p.name: p for p in method_dirs}
    ordered: List[Path] = [name_to_dir[n] for n in METHOD_VIS_ORDER if n in name_to_dir]
    missing = [n for n in METHOD_VIS_ORDER if n not in name_to_dir]
    if missing:
        LOGGER.warning("Missing method directories (will be omitted): %s", ", ".join(missing))
    method_dirs = ordered  # only show requested ones
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive prefix if not provided: take stem of first discovered P3 image
    if prefix is None:
        for d in method_dirs:
            p3_img = _find_level_image(d, LEVEL_SUFFIXES["P3"])
            if p3_img is not None:
                prefix = p3_img.stem.rsplit("_p3", 1)[0]
                break
        if prefix is None:
            raise ValueError("Could not infer prefix (no *_p3.png found in any method directory)")

    outputs: List[Path] = []
    for level, suffix in LEVEL_SUFFIXES.items():
        try:
            data = _collect_level_images(method_dirs, level, suffix)
        except ValueError as e:
            LOGGER.warning("Skipping level %s: %s", level, e)
            continue
        out_path = out_dir / f"{prefix}_{level.lower()}.png"
        _plot_panel(level, data, out_path)
        outputs.append(out_path)
    return outputs


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Assemble comparison panels from precomputed per-method mask images"
    )
    p.add_argument("input", type=Path, help="Input directory containing method subfolders")
    p.add_argument("out", type=Path, help="Output directory for generated panels")
    p.add_argument(
        "--prefix", type=str, default=None, help="Output filename prefix (default: inferred from *_p3.png)"
    )
    p.add_argument(
        "--keep-order",
        action="store_true",
        help="Keep filesystem order of method directories instead of alphabetical",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)"
    )
    return p


def main() -> None:  # pragma: no cover - CLI utility
    ap = _build_argparser()
    args = ap.parse_args()
    level = logging.WARNING if args.verbose == 0 else logging.INFO if args.verbose == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    paths = generate_showcase_precomputed(
        input_dir=args.input,
        out_dir=args.out,
        prefix=args.prefix,
        keep_order=args.keep_order,
    )
    for p in paths:
        print(p)


if __name__ == "__main__":  # pragma: no cover
    main()
