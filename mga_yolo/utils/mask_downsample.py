# fast_mask_downsample.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
import os
import numpy as np
import cv2

# Optional deps
_HAS_SKIMAGE = False
_HAS_SCIPY = False
try:
    from skimage.morphology import thin as _sk_thin, skeletonize as _sk_skeletonize
    from skimage.measure import block_reduce as _block_reduce
    _HAS_SKIMAGE = True
except Exception:
    _block_reduce = None  # type: ignore

try:
    from scipy.ndimage import maximum_filter as _maximum_filter  # noqa: F401
    _HAS_SCIPY = True
except Exception:
    pass

# ---------- backends ----------

def _skeletonize_fast(bin_img: np.ndarray) -> np.ndarray:
    """
    Connectivity-preserving thinning to 1 px width.

    Preference order:
      1) OpenCV ximgproc.thinning (ZHANGSUEN or GUOHALL)
      2) scikit-image thin/skeletonize
      3) NumPy Zhang–Suen (rarely used)
    """
    img = (bin_img > 0).astype(np.uint8)
    # 1) OpenCV ximgproc if available
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        ttype = cv2.ximgproc.THINNING_ZHANGSUEN  # or GUOHALL
        sk = cv2.ximgproc.thinning(img, thinningType=ttype)
        return sk.astype(bool)
    # 2) scikit-image
    if _HAS_SKIMAGE:
        # thin is slightly faster; skeletonize is more conservative
        sk = _sk_thin(img.astype(bool))
        return sk.astype(bool)
    # 3) Fallback: compact vectorized Zhang–Suen
    # Adapted for speed: operate on uint8 with convolution-like neighbor reads
    I = img.copy()
    changed = True
    pad = np.pad(I, 1, mode="constant")
    while changed:
        changed = False
        for k in (0, 1):
            P2 = pad[:-2, 1:-1]; P3 = pad[:-2, 2:];  P4 = pad[1:-1, 2:];  P5 = pad[2:, 2:]
            P6 = pad[2:, 1:-1];  P7 = pad[2:, 0:-2]; P8 = pad[1:-1, 0:-2]; P9 = pad[:-2, 0:-2]
            B = (P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9)
            A = ((P2 == 0) & (P3 == 1)).astype(np.uint8)
            A += ((P3 == 0) & (P4 == 1)); A += ((P4 == 0) & (P5 == 1)); A += ((P5 == 0) & (P6 == 1))
            A += ((P6 == 0) & (P7 == 1)); A += ((P7 == 0) & (P8 == 1)); A += ((P8 == 0) & (P9 == 1))
            A += ((P9 == 0) & (P2 == 1))
            m = (pad[1:-1,1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1)
            if k == 0:
                m &= ~((P2 & P4 & P6) | (P4 & P6 & P8))
            else:
                m &= ~((P2 & P4 & P8) | (P2 & P6 & P8))
            if m.any():
                pad[1:-1,1:-1][m] = 0
                changed = True
    return pad[1:-1,1:-1].astype(bool)


def _edges_from_skeleton(skel: np.ndarray) -> np.ndarray:
    """
    Return N x 4 int array of (y0, x0, y1, x1) for undirected 8-neighbor edges.
    Uses four directional shifts to avoid Python per-pixel loops.
    """
    s = skel.astype(bool)
    H, W = s.shape
    edges: List[np.ndarray] = []

    def _pairs(dy: int, dx: int) -> np.ndarray:
        # crop to avoid wrap-around from roll
        y0a, y0b = (0, H - dy) if dy >= 0 else (-dy, H)
        x0a, x0b = (0, W - dx) if dx >= 0 else (-dx, W)
        y1a, y1b = (dy, H) if dy >= 0 else (0, H + dy)
        x1a, x1b = (dx, W) if dx >= 0 else (0, W + dx)
        a = s[y0a:y0b, x0a:x0b]
        b = s[y1a:y1b, x1a:x1b]
        m = a & b
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            return np.empty((0, 4), np.int32)
        y0 = ys + y0a; x0 = xs + x0a
        y1 = ys + y1a; x1 = xs + x1a
        return np.stack([y0, x0, y1, x1], axis=1).astype(np.int32)

    # minimal set of undirected directions
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        e = _pairs(dy, dx)
        if e.size:
            edges.append(e)
    if not edges:
        return np.empty((0, 4), np.int32)
    E = np.vstack(edges)
    return E  # (N,4)


@dataclass
class DownsampleConfig:
    factor: int
    method: str = "skeleton_bresenham"
    threshold: float = 0.2
    close_diagonals: bool = True


def downsample_preserve_connectivity(mask: np.ndarray, cfg: DownsampleConfig) -> np.ndarray:
    """
    Connectivity-preserving downsample. Output shape = ceil(H/f) x ceil(W/f), uint8 {0,1}.
    """
    assert cfg.factor >= 1 and int(cfg.factor) == cfg.factor
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    Hc = (H + cfg.factor - 1) // cfg.factor
    Wc = (W + cfg.factor - 1) // cfg.factor

    # Fast paths
    if cfg.method == "area":
        small = cv2.resize(m, (Wc, Hc), interpolation=cv2.INTER_AREA)
        out = (small > cfg.threshold).astype(np.uint8)
        if cfg.close_diagonals:
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        return out

    if cfg.method == "maxpool":
        # reshape fast path if divisible; else use block_reduce if present
        k = cfg.factor
        pad_h = (k - (H % k)) % k; pad_w = (k - (W % k)) % k
        mp = np.pad(m, ((0, pad_h), (0, pad_w))) if (pad_h or pad_w) else m
        H2, W2 = mp.shape
        view = mp.reshape(H2 // k, k, W2 // k, k)
        out = view.max(axis=(1, 3)).astype(np.uint8)
        if cfg.close_diagonals:
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        return out

    if cfg.method == "gaussian_maxpool":
        # antialias in C++ then block max in C/NumPy
        sigma = float(cfg.factor) / 2.0
        blurred = cv2.GaussianBlur(m.astype(np.float32), ksize=(0, 0),
                                   sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        if _block_reduce is not None:
            pooled = _block_reduce(blurred, block_size=(cfg.factor, cfg.factor), func=np.max)
        else:
            # fallback: resize with INTER_AREA approximates average; we want max → approximate by dilation
            k = cfg.factor
            pooled = cv2.dilate(blurred, np.ones((k, k), np.float32))[::k, ::k]
        return (pooled >= cfg.threshold).astype(np.uint8)

    # skeleton_bresenham
    strict = os.getenv("MGA_SKELETON_STRICT", "0").lower() in {"1", "true", "yes"}
    if not strict:
        # Occupancy + optional 3x3 close
        return downsample_preserve_connectivity(m, DownsampleConfig(cfg.factor, "maxpool",
                                                                    cfg.threshold, cfg.close_diagonals))

    # Strict path
    sk = _skeletonize_fast(m)
    if not sk.any():
        return np.zeros((Hc, Wc), np.uint8)

    # Map skeleton pixels to coarse grid
    yc, xc = np.nonzero(sk)
    pc = np.stack([yc // cfg.factor, xc // cfg.factor], axis=1)
    out = np.zeros((Hc, Wc), np.uint8)
    out[pc[:, 0], pc[:, 1]] = 1

    # Build edges and rasterize them on coarse grid using cv2.line (Bresenham inside)
    edges = _edges_from_skeleton(sk)
    if edges.size:
        for y0, x0, y1, x1 in edges:
            p0 = (int(x0 // cfg.factor), int(y0 // cfg.factor))
            p1 = (int(x1 // cfg.factor), int(y1 // cfg.factor))
            if p0 == p1:
                continue
            cv2.line(out, p0, p1, 1, 1)

    if cfg.close_diagonals:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    return out.astype(np.uint8)


def downsample_preserve_connectivity_multi(mask: np.ndarray,
                                           factors: Sequence[int],
                                           method: str = "skeleton_bresenham",
                                           threshold: float = 0.2,
                                           close_diagonals: bool = True) -> Dict[int, np.ndarray]:
    """
    Multi-factor variant with single skeletonization when strict skeleton is requested.
    """
    if method != "skeleton_bresenham":
        return {f: downsample_preserve_connectivity(mask, DownsampleConfig(f, method, threshold, close_diagonals))
                for f in factors}

    strict = os.getenv("MGA_SKELETON_STRICT", "0").lower() in {"1", "true", "yes"}
    if not strict:
        return {f: downsample_preserve_connectivity(mask, DownsampleConfig(f, "maxpool", threshold, close_diagonals))
                for f in factors}

    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    sk = _skeletonize_fast(m)
    if not sk.any():
        return {f: np.zeros(((H + f - 1)//f, (W + f - 1)//f), np.uint8) for f in factors}

    edges = _edges_from_skeleton(sk)
    ys, xs = np.nonzero(sk)
    results: Dict[int, np.uint8] = {}

    for f in factors:
        Hc = (H + f - 1) // f; Wc = (W + f - 1) // f
        out = np.zeros((Hc, Wc), np.uint8)
        # nodes
        out[(ys // f), (xs // f)] = 1
        # edges
        if edges.size:
            for y0, x0, y1, x1 in edges:
                p0 = (int(x0 // f), int(y0 // f))
                p1 = (int(x1 // f), int(y1 // f))
                if p0 == p1:
                    continue
                cv2.line(out, p0, p1, 1, 1)
        if close_diagonals:
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        results[f] = out
    return results


def connected_components_count(bin_img: np.ndarray, connectivity: int = 2) -> int:
    """Count connected components excluding background, using OpenCV if present."""
    img = (bin_img > 0).astype(np.uint8)
    if connectivity == 2:
        conn = cv2.CV_8U
        n, _ = cv2.connectedComponents(img, connectivity=8, ltype=conn)
    else:
        n, _ = cv2.connectedComponents(img, connectivity=4)
    return int(n - 1)
