from pathlib import Path
import torch

from mga_yolo.engine.model import MGAModel
from mga_yolo.engine.checkpoint import rebuild_mga_model_from_minimal_ckpt


def test_rebuild_mga_model_from_minimal_ckpt(tmp_path: Path):
    """Create a tiny minimal checkpoint from a fresh MGAModel and reload it via the helper."""
    yaml_path = Path('configs/models/yolov8_test_segment_heads.yaml').resolve()
    assert yaml_path.exists(), 'Model YAML not found'

    # Build a small model with 1 class to keep Detect head compact
    base = MGAModel(str(yaml_path), nc=1, verbose=False)
    sd = base.state_dict()

    ckpt_path = tmp_path / 'minimal.pt'
    torch.save({
        'epoch': 0,
        'ema': None,
        'model': None,
        'ema_state_dict': None,
        'model_state_dict': sd,
        'train_args': {'nc': 1, 'model': str(yaml_path)},
        'train_metrics': {},
        'best_fitness': 0.0,
        'optimizer': None,
        'updates': 0,
    }, ckpt_path)

    re_model, raw = rebuild_mga_model_from_minimal_ckpt(ckpt_path, yaml_path)
    assert hasattr(re_model, 'state_dict')
    # Compare a couple of parameters for equality
    k = 'model.0.conv.weight'
    assert k in sd and k in re_model.state_dict()
    t0, t1 = sd[k].cpu(), re_model.state_dict()[k].cpu()
    assert t0.shape == t1.shape
    assert torch.allclose(t0, t1)
