import torch
import pytest
from mga_yolo.external.ultralytics.ultralytics import YOLO
import traceback
import sys

def _log(msg: str) -> None:
    print(f"[TEST][MGA] {msg}")


def test_mga_model_build_forward():
    _log("Starting model build forward test")
    try:
        model = YOLO('configs/models/yolov8_test_segment_heads.yaml', verbose=True)
    except Exception as e:
        _log("Model creation FAILED")
        traceback.print_exc()
        pytest.fail(f"YOLO model creation failed: {e}")

    _log(f"Model created: type={type(model.model).__name__}, task={getattr(model,'task',None)}")
    # Dump mask head indices
    mask_layers = []
    for i, m in enumerate(getattr(model.model, 'model', [])):
        if m.__class__.__name__ == 'MGAMaskHead':
            mask_layers.append(i)
    _log(f"Detected MGAMaskHead layers at indices: {mask_layers or 'NONE'}")

    # Ensure MGA task selected
    assert model.task == 'mga', f"Expected task 'mga' got {model.task}" 

    x = torch.randn(2,3,640,640)
    _log("Running forward pass ...")
    try:
        out = model.model(x)
    except Exception as e:
        _log("Forward pass FAILED")
        traceback.print_exc()
        pytest.fail(f"Forward failed: {e}")
    _log(f"Forward output type: {type(out)}")
    assert isinstance(out, dict), 'MGAModel forward should return dict'
    assert 'det' in out and 'seg' in out
    _log(f"Seg keys: {list(out['seg'].keys())}")
    assert len(out['seg']) == 3, f"Expected 3 segmentation scales, got {len(out['seg'])}"
    for k,v in out['seg'].items():
        _log(f"Seg[{k}] shape={tuple(v.shape)} min={float(v.min()):.4f} max={float(v.max()):.4f}")
        assert v.ndim == 4 and v.shape[0] == 2


def test_mga_predict(tmp_path):
    _log("Starting predict test")
    model = YOLO('configs/models/yolov8_test_segment_heads.yaml', verbose=False)
    img = (torch.rand(1,3,640,640)*255).byte().permute(0,2,3,1).numpy()[0]
    results = model.predict(source=[img], verbose=False)
    _log(f"Predict returned {len(results)} result(s)")
    assert len(results) == 1
    r = results[0]
    _log(f"Result attributes: has mga_masks={hasattr(r,'mga_masks')}")
    assert hasattr(r, 'mga_masks')
    _log(f"Number of mga_masks: {len(r.mga_masks)}")
    assert len(r.mga_masks) == 3


@pytest.mark.skip(reason='Requires dataset with masks for full loss test')
def test_mga_train_one_iter(tmp_path):
    # This would require a minimal synthetic dataset with masks; placeholder for future.
    pass
