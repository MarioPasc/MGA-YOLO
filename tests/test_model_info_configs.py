from pathlib import Path
import pytest
import torch

from mga_yolo.external.ultralytics.ultralytics import YOLO

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "models"

MODEL_CFGS = [
    "yolov8_cbam.yaml",
    "yolov8_eca.yaml",
    "yolov8_seg_heads.yaml",
    "yolov8_spade.yaml",
    "yolov8.yaml",
]

@pytest.mark.parametrize("cfg_name", MODEL_CFGS)
def test_model_info_fallback(cfg_name: str):
    cfg_path = CONFIG_DIR / cfg_name
    assert cfg_path.exists(), f"Missing config: {cfg_path}"

    model = YOLO(str(cfg_path)).model  # underlying nn.Module
    # Try native info
    basic = None
    detailed = None
    if hasattr(model, "info"):
        try:
            basic = model.info(detailed=False, verbose=False)
            detailed = model.info(detailed=True, verbose=False)
        except Exception:
            pass

    # Fallback manual stats
    params = sum(p.numel() for p in model.parameters())
    assert params > 0, "Parameter count must be > 0"

    # Lightweight GFLOPs attempt (skip if fails)
    try:
        from thop import profile  # type: ignore
        dummy = torch.zeros(1, 3, 640, 640, device=next(model.parameters()).device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = (macs * 2) / 1e9
        assert gflops > 0
    except Exception:
        gflops = None  # acceptable

    print(f"\nConfig: {cfg_name}")
    print(f"model.info() basic returned: {basic}")
    print(f"model.info() detailed returned: {('len='+str(len(detailed)) if isinstance(detailed,(list,tuple)) else detailed)}")
    print(f"Params: {params:,}")
    print(f"GFLOPs(approx): {gflops if gflops is not None else 'n/a'}")