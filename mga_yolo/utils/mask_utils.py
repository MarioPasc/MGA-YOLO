import os, math, cv2, numpy as np
from typing import Optional

from pathlib import Path
from typing import Optional

from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER
from mga_yolo.utils.mask_downsample import DownsampleConfig, downsample_preserve_connectivity

PROB_MODE = os.getenv("MGA_MASK_PROB", False)

class MaskUtils:
    @staticmethod
    def downsample_mask_prob(mask: np.ndarray, stride: int, method: str = "area") -> np.ndarray:
        """
        Downsample a binary mask to a probability mask in [0,1] by block average.
        - 'area' -> cv2.INTER_AREA (equiv. promedio espacial)
        - 'avgpool' -> promedio por bloques exacto cuando stride divide H,W
        - 'nearest' -> retorno {0,1}, sólo para compatibilidad/velocidad
        Returns float32 in [0,1], shape ≈ ceil(H/stride) x ceil(W/stride).
        """
        if stride <= 1:
            return mask.astype(np.float32)

        # fuerza binaria de entrada, pero como 0/1 float
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)

        h, w = mask.shape
        nh, nw = math.ceil(h / stride), math.ceil(w / stride)

        if method == "avgpool":
            pad_h = (stride - (h % stride)) % stride
            pad_w = (stride - (w % stride)) % stride
            if pad_h or pad_w:
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
                h, w = mask.shape
            view = mask.reshape(h // stride, stride, w // stride, stride).astype(np.float32)
            prob = view.mean(axis=(1, 3))  # promedio de 0/1 -> prob
            return prob.astype(np.float32)

        if method == "nearest":
            out = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            return out.astype(np.float32)

        # por defecto: area
        out = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)  # ya produce fracciones
        return np.clip(out.astype(np.float32), 0.0, 1.0)


    @staticmethod
    def infer_mask_path(im_file: str, data_root: Optional[str], masks_dir: Optional[str]) -> Optional[Path]:
        if data_root is None or masks_dir is None:
            return None
        stem = Path(im_file).stem
        # Support common mask extensions
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            p = Path(data_root) / masks_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    @staticmethod
    def downsample_mask(mask: np.ndarray, stride: int) -> np.ndarray:
        """Downsample a binary mask by stride using configurable methods.

        Env MGA_MASK_METHOD controls algorithm:
          - 'nearest'  : super fast but low quality (for CI/smoke)
          - 'area'     : fast high-quality (INTER_AREA + >0 + optional close)
          - 'maxpool'  : block-wise max pooling
          - 'pyrdown'  : repeated pyrDown (power-of-two strides)
          - otherwise  : use connectivity-preserving utility (skeleton_bresenham)
        Env MGA_MASK_BRIDGE=0 disables 3x3 closing bridge (default on)
        Env MGA_MASK_THRESH sets threshold for 'area' method (default 0.0, i.e., >0)
        """
        method = os.getenv("MGA_MASK_METHOD", "skeleton_bresenham").lower()
        bridge = os.getenv("MGA_MASK_BRIDGE", "1") not in {"0", "false", "False"}
        thresh = float(os.getenv("MGA_MASK_THRESH", "0.0"))

        # unify dtype to uint8 binary {0,1}
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)

        if stride <= 1:
            return mask

        h, w = mask.shape
        nh, nw = math.ceil(h / stride), math.ceil(w / stride)

        if method == "nearest":
            return cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        if method == "area":
            small = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
            out = (small > thresh).astype(np.uint8)
            if bridge:
                kernel = np.ones((3, 3), np.uint8)
                out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
            return out

        if method == "maxpool":
            pad_h = (stride - (h % stride)) % stride
            pad_w = (stride - (w % stride)) % stride
            if pad_h or pad_w:
                mask_pad = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            else:
                mask_pad = mask
            H2, W2 = mask_pad.shape
            view = mask_pad.reshape(H2 // stride, stride, W2 // stride, stride)
            return view.max(axis=(1, 3)).astype(np.uint8)

        if method == "pyrdown":
            s = stride
            out = mask.copy()
            # only valid if stride is power of two; otherwise fall back
            if s & (s - 1) == 0:
                while s > 1:
                    out = cv2.pyrDown(out)
                    s //= 2
                out = (out > 0).astype(np.uint8)
                if bridge:
                    kernel = np.ones((3, 3), np.uint8)
                    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
                return out

        # Default/fallback: call connectivity-preserving utility
        try:
            return downsample_preserve_connectivity(
                mask,
                DownsampleConfig(
                    factor=stride,
                    method="skeleton_bresenham",
                    threshold=thresh if thresh > 0 else 0.2,
                    close_diagonals=bridge,
                ),
            )
        except Exception as e:  # pragma: no cover
            LOGGER.debug(
                f"Connectivity-preserving downsample failed (stride={stride}, method={method}): {e}; using INTER_NEAREST"
            )

        return cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)