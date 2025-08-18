import torch
import pytest
from ultralytics import YOLO


def test_mga_model_build_forward():
    model = YOLO('configs/models/yolov8_test_segment_heads.yaml')
    # Ensure MGA task selected
    assert model.task == 'mga'
    out = model.model(torch.randn(2,3,640,640))
    assert isinstance(out, dict), 'MGAModel forward should return dict'
    assert 'det' in out and 'seg' in out
    assert len(out['seg']) == 3, f"Expected 3 segmentation scales, got {len(out['seg'])}"
    for k,v in out['seg'].items():
        assert v.ndim == 4 and v.shape[0] == 2


def test_mga_predict(tmp_path):
    model = YOLO('configs/models/yolov8_test_segment_heads.yaml')
    # Single random image inference
    img = (torch.rand(1,3,640,640)*255).byte().permute(0,2,3,1).numpy()[0]
    results = model.predict(source=[img], verbose=False)
    assert len(results) == 1
    r = results[0]
    assert hasattr(r, 'mga_masks')
    assert len(r.mga_masks) == 3


@pytest.mark.skip(reason='Requires dataset with masks for full loss test')
def test_mga_train_one_iter(tmp_path):
    # This would require a minimal synthetic dataset with masks; placeholder for future.
    pass
