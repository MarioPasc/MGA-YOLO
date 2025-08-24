# Algorithmic prototype for connectivity-preserving downsampling of a binary vessel mask.
# It implements two options:
#  1) "skeleton_bresenham": topology-preserving downsample by thinning to a skeleton,
#     mapping to the coarse grid, and rasterizing connections via Bresenham lines.
#  2) "gaussian_maxpool": antialiased soft downsample (Gaussian blur + block max/avg),
#     which can be thresholded; good as a soft target but doesn't *guarantee* connectivity.
#
# The script will:
#   - load the sample mask from /mnt/data/arcadetest_p1_v1_00001.png
#   - run the downsampling for strides 8, 16, 32 with "skeleton_bresenham"
#   - display results and report connected-component counts.
#
# Notes for integration:
#   - The implementation is pure NumPy; skeletonization prefers scikit-image if present,
#     and falls back to a Zhang–Suen thinning in NumPy if not available.
#   - Bresenham rasterization guarantees that any edge in the fine skeleton graph is
#     represented as an 8-connected path in the coarse grid.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Sequence

import numpy as np
import cv2
import concurrent.futures as futures
from PIL import Image
import matplotlib.pyplot as plt

# Try to import optional libs; provide fallbacks if unavailable.
try:
    from skimage.morphology import skeletonize as skimage_skeletonize
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

def _skeletonize_binary(bin_img: np.ndarray) -> np.ndarray:
    """
    Topology-preserving thinning to obtain 1-pixel-wide skeleton.

    Prefers scikit-image's implementation if available; otherwise, uses a NumPy
    Zhang–Suen thinning fallback.

    Parameters
    ----------
    bin_img : np.ndarray
        Boolean or {0,1} array.

    Returns
    -------
    np.ndarray
        Boolean skeleton of the same shape.
    """
    if bin_img.dtype != bool:
        bin_img = bin_img > 0

    if _HAS_SKIMAGE:
        return skimage_skeletonize(bin_img)

    # ---------- Zhang–Suen thinning fallback ----------
    img = bin_img.copy().astype(np.uint8)
    changed = True

    def neighbors(y, x, arr):
        # clockwise P2..P9 around (y,x)
        return [
            arr[y-1, x],     # P2
            arr[y-1, x+1],   # P3
            arr[y,   x+1],   # P4
            arr[y+1, x+1],   # P5
            arr[y+1, x],     # P6
            arr[y+1, x-1],   # P7
            arr[y,   x-1],   # P8
            arr[y-1, x-1],   # P9
        ]

    def transitions(ns):
        # number of 0->1 transitions in circular sequence
        n = 0
        for i in range(8):
            n += (ns[i] == 0 and ns[(i+1) % 8] == 1)
        return n

    img_padded = np.pad(img, 1, mode="constant", constant_values=0)
    while changed:
        changed = False
        for iter_idx in [0, 1]:
            to_delete = []
            for y in range(1, img_padded.shape[0]-1):
                for x in range(1, img_padded.shape[1]-1):
                    P1 = img_padded[y, x]
                    if P1 == 0:
                        continue
                    ns = neighbors(y, x, img_padded)
                    B = sum(ns)
                    A = transitions(ns)
                    if A != 1 or B < 2 or B > 6:
                        continue
                    if iter_idx == 0:
                        if ns[0] * ns[2] * ns[4] != 0:
                            continue
                        if ns[2] * ns[4] * ns[6] != 0:
                            continue
                    else:
                        if ns[0] * ns[2] * ns[6] != 0:
                            continue
                        if ns[0] * ns[4] * ns[6] != 0:
                            continue
                    to_delete.append((y, x))
            if to_delete:
                changed = True
                for (y, x) in to_delete:
                    img_padded[y, x] = 0
    return img_padded[1:-1, 1:-1].astype(bool)


def _bresenham_line(p0: Tuple[int, int], p1: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Bresenham line between integer grid points inclusive.
    Returns list of (y, x) coordinates.
    """
    y0, x0 = p0
    y1, x1 = p1
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)

    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1

    y, x = y0, x0
    points = [(y, x)]
    if dx > dy:
        err = dx // 2
        while x != x1:
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
            points.append((y, x))
    else:
        err = dy // 2
        while y != y1:
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
            points.append((y, x))
    return points


@dataclass
class DownsampleConfig:
    """
    Configuration for connectivity-preserving downsampling.

    Attributes
    ----------
    factor : int
        Integer stride / downsampling factor (e.g., 8, 16, 32).
    method : str
        'skeleton_bresenham' or 'gaussian_maxpool'.
    threshold : float
        Threshold for soft masks in 'gaussian_maxpool' (0..1). Ignored otherwise.
    close_diagonals : bool
        If True, run a 3x3 binary closing to ensure 8-connectivity after rasterization.
    """
    factor: int
    method: str = "skeleton_bresenham"  # or: 'area', 'maxpool', 'pyrdown', 'gaussian_maxpool'
    threshold: float = 0.2
    close_diagonals: bool = True


def downsample_preserve_connectivity(mask: np.ndarray, cfg: DownsampleConfig) -> np.ndarray:
    """
    Downsample a binary mask by an integer factor, preserving connectivity.

    For 'skeleton_bresenham', we:
      1) skeletonize the mask
      2) project skeleton pixels to coarse grid via floor division by factor
      3) rasterize edges between 8-neighbor pairs in the fine skeleton using Bresenham
      4) (optional) close 3x3 to ensure diagonal joins

    For 'gaussian_maxpool', we:
      1) apply Gaussian blur with sigma=factor/2 to antialias
      2) block-max/avg-pool with kernel=stride=factor
      3) threshold to binary

    Parameters
    ----------
    mask : np.ndarray
        Input binary mask (H, W), values in {0,1} or bool.
    cfg : DownsampleConfig
        Configuration instance.

    Returns
    -------
    np.ndarray
        Downsampled binary mask of shape (ceil(H/factor), ceil(W/factor)), dtype=uint8.
    """
    assert cfg.factor >= 1 and int(cfg.factor) == cfg.factor, "factor must be a positive integer"
    if mask.dtype != bool:
        bin_mask = mask > 0
    else:
        bin_mask = mask

    H, W = bin_mask.shape
    Hc = (H + cfg.factor - 1) // cfg.factor
    Wc = (W + cfg.factor - 1) // cfg.factor

    # Fast paths that still preserve thin structures reasonably well
    if cfg.method == "area":
        nh, nw = Hc, Wc
        small = cv2.resize(bin_mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_AREA)
        out = (small > cfg.threshold).astype(np.uint8)
        if cfg.close_diagonals:
            kernel = np.ones((3, 3), np.uint8)
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
        return out

    if cfg.method == "maxpool":
        k = cfg.factor
        pad_h = (k - (H % k)) % k
        pad_w = (k - (W % k)) % k
        if pad_h or pad_w:
            mp = np.pad(bin_mask.astype(np.uint8), ((0, pad_h), (0, pad_w)), mode="constant")
        else:
            mp = bin_mask.astype(np.uint8)
        H2, W2 = mp.shape
        view = mp.reshape(H2 // k, k, W2 // k, k)
        out = view.max(axis=(1, 3)).astype(np.uint8)
        if cfg.close_diagonals:
            kernel = np.ones((3, 3), np.uint8)
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
        return out

    if cfg.method == "pyrdown":
        s = cfg.factor
        out = bin_mask.astype(np.uint8)
        if s & (s - 1) == 0 and s > 1:
            while s > 1:
                out = cv2.pyrDown(out)
                s //= 2
            out = (out > 0).astype(np.uint8)
            if cfg.close_diagonals:
                kernel = np.ones((3, 3), np.uint8)
                out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
            return out

    if cfg.method == "skeleton_bresenham":
        skel = _skeletonize_binary(bin_mask)
        out = np.zeros((Hc, Wc), dtype=np.uint8)
        # Offsets for 8-neighborhood
        ns = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        skel_idx = np.argwhere(skel)
        skel_set = set(map(tuple, skel_idx.tolist()))
        # Map skeleton points to coarse grid
        def to_coarse(p):
            y, x = p
            return (y // cfg.factor, x // cfg.factor)

        visited_edges = set()
        for y, x in skel_idx:
            p = (y, x)
            for dy, dx in ns:
                q = (y + dy, x + dx)
                if q not in skel_set:
                    continue
                # undirected edge; avoid double-drawing
                e = tuple(sorted([p, q]))
                if e in visited_edges:
                    continue
                visited_edges.add(e)

                pc = to_coarse(p)
                qc = to_coarse(q)
                # draw Bresenham line in coarse grid
                for yy, xx in _bresenham_line(pc, qc):
                    if 0 <= yy < Hc and 0 <= xx < Wc:
                        out[yy, xx] = 1

        if cfg.close_diagonals:
            # 3x3 binary closing (dilation then erosion) to fix diagonal "holes"
            # Implemented via convolution-like neighborhood check
            padded = np.pad(out, 1, mode="constant", constant_values=0)
            # dilate
            dil = np.zeros_like(padded)
            for i in range(1, padded.shape[0] - 1):
                for j in range(1, padded.shape[1] - 1):
                    if np.any(padded[i-1:i+2, j-1:j+2] > 0):
                        dil[i, j] = 1
            # erode
            ero = np.zeros_like(dil)
            for i in range(1, dil.shape[0] - 1):
                for j in range(1, dil.shape[1] - 1):
                    if np.all(dil[i-1:i+2, j-1:j+2] > 0):
                        ero[i, j] = 1
            out = ero[1:-1, 1:-1].astype(np.uint8)

        return out.astype(np.uint8)

    elif cfg.method == "gaussian_maxpool":
        try:
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(bin_mask.astype(np.float32), sigma=cfg.factor / 2.0, mode="reflect")
        except Exception:
            # crude box blur if scipy is not available
            k = cfg.factor
            pad = k // 2
            padded = np.pad(bin_mask.astype(np.float32), pad, mode="reflect")
            kernel = np.ones((k, k), dtype=np.float32) / (k * k)
            # naive convolution
            H2, W2 = bin_mask.shape
            blurred = np.zeros_like(bin_mask, dtype=np.float32)
            for i in range(H2):
                for j in range(W2):
                    patch = padded[i:i+k, j:j+k]
                    blurred[i, j] = float(np.sum(patch * kernel))

        # block max/avg pool with kernel=stride=factor
        out = np.zeros((Hc, Wc), dtype=np.float32)
        for i in range(Hc):
            for j in range(Wc):
                y0, x0 = i * cfg.factor, j * cfg.factor
                y1, x1 = min(y0 + cfg.factor, H), min(x0 + cfg.factor, W)
                out[i, j] = np.max(blurred[y0:y1, x0:x1])  # max-pool; could be avg
        return (out >= cfg.threshold).astype(np.uint8)

    else:
        raise ValueError(f"Unknown method: {cfg.method}")


def downsample_batch(
    masks: Sequence[np.ndarray],
    factor: int,
    method: str = "area",
    threshold: float = 0.0,
    close_diagonals: bool = True,
    max_workers: int = 0,
) -> List[np.ndarray]:
    """Downsample a batch of binary masks in parallel.

    Args:
        masks: Sequence of HxW binary arrays.
        factor: Integer downsample factor.
        method: One of 'area', 'maxpool', 'pyrdown', 'skeleton_bresenham', 'gaussian_maxpool'.
        threshold: Threshold used by some methods.
        close_diagonals: Apply small 3x3 closing to keep 8-connectivity.
        max_workers: If > 0, use ThreadPool with this number of workers.

    Returns:
        List of downsampled binary masks.
    """
    cfg = DownsampleConfig(factor=factor, method=method, threshold=threshold, close_diagonals=close_diagonals)

    def _run(m):
        return downsample_preserve_connectivity(m, cfg)

    if max_workers and max_workers > 0:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_run, masks))
    else:
        return [ _run(m) for m in masks ]


def connected_components_count(bin_img: np.ndarray, connectivity: int = 2) -> int:
    """
    Count connected components in a binary image.

    Parameters
    ----------
    bin_img : np.ndarray
        Binary image with values {0,1}.
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity (default).

    Returns
    -------
    int
        Number of connected components (excluding background).
    """
    try:
        from skimage.measure import label
        lab = label(bin_img, connectivity=connectivity)
        return lab.max()
    except Exception:
        # simple BFS fallback
        bin_img = (bin_img > 0).astype(np.uint8)
        H, W = bin_img.shape
        visited = np.zeros_like(bin_img, dtype=bool)
        comps = 0
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 2:
            nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for i in range(H):
            for j in range(W):
                if bin_img[i, j] == 0 or visited[i, j]:
                    continue
                comps += 1
                # BFS
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    y, x = stack.pop()
                    for dy, dx in nbrs:
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W and bin_img[yy, xx] and not visited[yy, xx]:
                            visited[yy, xx] = True
                            stack.append((yy, xx))
        return comps


def main():
    # --------- Demo on the provided image ---------
    img_path = "/mnt/data/arcadetest_p1_v1_00001.png"
    img = Image.open(img_path).convert("L")
    mask = (np.array(img) > 0).astype(np.uint8)

    factors = [8, 16, 32]
    results = {}

    for f in factors:
        cfg = DownsampleConfig(factor=f, method="skeleton_bresenham", threshold=0.2, close_diagonals=True)
        ds = downsample_preserve_connectivity(mask, cfg)
        results[f] = ds
        # Save for download
        Image.fromarray((ds * 255).astype(np.uint8)).save(f"/mnt/data/mask_down_f{f}.png")

    # Show the original and the downsampled masks
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("Original (512x512)")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    for idx, f in enumerate(factors, start=2):
        plt.subplot(2, 2, idx)
        ds = results[f]
        plt.title(f"Downsample factor {f} -> {ds.shape[1]}x{ds.shape[0]}")
        # Upscale for visualization so it's visible
        viz = Image.fromarray((ds * 255).astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
        plt.imshow(viz, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Report component counts
    for f in factors:
        comps = connected_components_count(results[f], connectivity=2)
        print(f"factor={f}: components(8-connected) = {comps}")

    print("\nSaved outputs:")
    for f in factors:
        print(f" - [f={f}] mask_down_f{f}.png -> sandbox:/mnt/data/mask_down_f{f}.png")
