from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch


def _safe_torch_load(path: Path) -> Dict[str, Any]:
    """Load a checkpoint dict from disk on CPU with robust error handling."""
    ckpt: Dict[str, Any]
    with open(path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu")  # type: ignore[assignment]
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint at {path} is not a dict. Got: {type(ckpt)!r}")
    return ckpt


def rebuild_mga_model_from_minimal_ckpt(
    ckpt_path: str | Path,
    cfg_yaml: Union[str, Path, Dict[str, Any]],
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Rebuild an MGAModel from a minimal checkpoint that stores only state_dict tensors.

    This accepts checkpoints saved by MGATrainer.save_model that contain one or both keys:
      - 'ema_state_dict': state dict for the EMA weights
      - 'model_state_dict': state dict for the raw model

    Args:
        ckpt_path: Path to the minimal checkpoint (.pt).
        cfg_yaml: Model YAML used to re-instantiate the network topology.

    Returns:
        (model, ckpt): The reconstructed model (eval, FP32, on CPU) and the raw checkpoint dict.
    """
    from mga_yolo.engine.model import MGAModel

    p = Path(ckpt_path)
    ckpt = _safe_torch_load(p)

    if "ema_state_dict" not in ckpt and "model_state_dict" not in ckpt:
        raise KeyError("Minimal MGA checkpoint missing 'ema_state_dict' and 'model_state_dict'.")

    # Choose state dict (prefer EMA)
    raw_sd = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict")
    if not isinstance(raw_sd, dict):  # basic sanity
        raise TypeError("state_dict in minimal MGA checkpoint must be a dict")

    # Infer number of classes (nc) from checkpoint or args to avoid Detect head size mismatches
    # Priority: explicit 'nc' in train_args -> infer from cls conv weights -> default to 80
    nc = None
    args = ckpt.get("train_args", {}) or {}
    if isinstance(args, dict):
        nc = args.get("nc")
    if not isinstance(nc, int) or nc <= 0:
        # Try to infer from Detect head class conv weights, usually keys like 'model.25.cv3.0.2.weight'
        for k, v in raw_sd.items():
            if isinstance(v, torch.Tensor) and ".cv3." in k and k.endswith(".2.weight") and v.ndim == 4:
                nc = int(v.shape[0])
                break
    if not isinstance(nc, int) or nc <= 0:
        nc = 80

    # Build model skeleton from YAML (path or dict) with inferred nc
    cfg_arg: Any = cfg_yaml  # can be a path-like or a dict
    model = MGAModel(cfg_arg, nc=nc, verbose=False)

    # Prepare a filtered state dict that only includes matching parameter/buffer shapes and casts dtypes
    curr_sd = model.state_dict()
    filtered_sd: Dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if k in curr_sd and isinstance(v, torch.Tensor):
            tgt = curr_sd[k]
            if tuple(tgt.shape) == tuple(v.shape):
                # Cast to target dtype/device (CPU) for safe load
                vv = v.detach().to(dtype=tgt.dtype, device="cpu")
                filtered_sd[k] = vv

    # Load weights non-strictly to allow minor key mismatches across versions
    load_res = model.load_state_dict(filtered_sd, strict=False)
    # torch.nn.Module.load_state_dict returns an IncompatibleKeys with attributes
    # 'missing_keys' and 'unexpected_keys' rather than a tuple
    missing = getattr(load_res, "missing_keys", [])
    unexpected = getattr(load_res, "unexpected_keys", [])
    # Non-fatal: keep going even if there are mismatched keys
    _ = (missing, unexpected)

    # Attach args if present for downstream config merge compatibility
    args = ckpt.get("train_args", {})
    if isinstance(args, dict):
        try:
            model.args = args
        except Exception:
            pass

    # Place in eval() FP32 on CPU, aligning with Ultralytics post-load behavior
    model = model.to("cpu").float().eval()
    return model, ckpt
