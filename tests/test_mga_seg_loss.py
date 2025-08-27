from types import SimpleNamespace
from pathlib import Path
import torch
import pytest
from mga_yolo.engine.model import MGAModel
from mga_yolo.engine.train import MGATrainer
from mga_yolo.external.ultralytics.ultralytics import YOLO


class _DummyScaler:
    def scale(self, loss):
        class _Obj:
            def __init__(self, l):
                self.l = l
            def backward(self):
                self.l.backward()
        return _Obj(loss)


class DummyTrainer(MGATrainer):  # type: ignore[misc]
    def __init__(self):
        # Bypass parent __init__ entirely to avoid dataset building
        self.args = SimpleNamespace(
            save=False,
            seg_bce_weight=1.0,
            seg_dice_weight=1.0,
            seg_scale_weights=[1.0, 1.0, 1.0],
            seg_smooth=1.0,
            seg_loss_lambda=1.0,
            seg_enable=True,
        )
        self.device = torch.device('cpu')
        self.amp = False
        self.scaler = _DummyScaler()
        # base detection loss names mimic DetectionTrainer
        self.loss_names = ["box", "cls", "dfl"]
        # Provide a dummy compute_loss returning zero detection loss
        def _compute_loss(preds, batch):
            z = torch.zeros(1, requires_grad=True)
            # detection returns (loss, items tensor)
            return z, torch.zeros(3)
        self.compute_loss = _compute_loss  # type: ignore[assignment]
        self.epoch = 0
        self.save_dir = '.'
        self.init_losses()


def test_mga_segmentation_loss_integration():
    B = 2
    imgsz = 128
    model = MGAModel('configs/models/yolov8_test_segment_heads.yaml')
    trainer = DummyTrainer()
    trainer.model = model

    img = torch.randn(B, 3, imgsz, imgsz)
    preds = model(img)

    # Build synthetic masks aligned to strides 8,16,32
    masks_multi = []
    for s in (8, 16, 32):
        h = imgsz // s
        w = imgsz // s
        masks_multi.append((torch.rand(B, 1, h, w) > 0.5).float())
    batch = {
        'img': img,
        'masks_multi': masks_multi,
        'cls': torch.zeros((0,), dtype=torch.int64),
        'bboxes': torch.zeros((0, 4), dtype=torch.float32),
        'batch_idx': torch.zeros((0,), dtype=torch.int64),
    }
    total_loss = trainer.criterion(preds, batch)
    assert total_loss.requires_grad, 'Loss should require grad.'
    assert any(n == 'seg_total' for n in trainer.loss_names), 'seg_total not registered in loss names.'
    seg_total_idx = trainer.loss_names.index('seg_total')
    assert trainer.loss_items[seg_total_idx].item() >= 0.0
    # Ensure individual scale metrics logged
    for key in ['p3_bce', 'p4_bce', 'p5_bce', 'p3_dice', 'p4_dice', 'p5_dice']:
        assert key in trainer.loss_names, f'{key} missing in loss names.'


@pytest.mark.slow
def test_mga_detect_and_seg_train_10_epochs(tmp_path):
    # End-to-end: train MGA for 10 epochs using provided dataset YAML; save losses and val visuals
    data_yaml = Path('configs/data/data.yaml').resolve()
    assert data_yaml.exists(), "configs/data/data.yaml not found"
    model = YOLO('configs/models/yolov8_test_segment_heads.yaml', task='mga', verbose=True)
    # Force CPU for CI; enable saving to capture CSV and validation plots
    # Pick device dynamically: prefer CUDA if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    results = model.train(
        data=str(data_yaml),
        task='mga',
        epochs=4,
        imgsz=512,
        batch=2,
        device=device,
        workers=0,
        save=True,
        val=True,
        plots=True,
        seg_enable=True,
        seg_bce_weight=1.0,
        seg_dice_weight=1.0,
        seg_scale_weights=[1.0, 1.0, 1.0],
        seg_loss_lambda=1.0,
        seg_smooth=1.0,
        verbose=True,
        name='test_mga_train_v8_segloss',
        save_dir=str(tmp_path),
        
        # https://docs.ultralytics.com/guides/yolo-data-augmentation/#introduction
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        #degrees=0.0,
        #translate=0.0,
        #scale=0.0,
        #shear=0.0,
        #perspective=0.0,
        #flipud=0.0,
        #fliplr=0.0,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        erasing=0.0,
        copy_paste=0.0,
    )
    # Basic assertions: training completed and artifacts exist
    import time
    time.sleep(0.5)  # give I/O a moment
    assert results is not None
    save_dir = Path(model.trainer.save_dir)
    assert (save_dir / 'results.csv').exists(), 'results.csv missing'
    # Check some validation outputs (pred plots and mask previews)
    pred_samples = list(save_dir.glob('val_batch*_pred.jpg'))
    mask_dirs = list(save_dir.glob('val_batch*_masks'))
    assert pred_samples, 'No validation prediction images saved'
    assert mask_dirs, 'No validation mask previews saved'
