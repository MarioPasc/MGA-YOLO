import os
from pathlib import Path
import numpy as np
import torch
from mga_yolo.external.ultralytics.ultralytics import YOLO


def create_synthetic_dataset(root: Path):
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'labels').mkdir(parents=True, exist_ok=True)
    (root / 'masks').mkdir(parents=True, exist_ok=True)
    # create 2 images 64x64 with one white square and matching mask & bbox
    for i in range(2):
        img = np.zeros((64,64,3), dtype=np.uint8)
        mask = np.zeros((64,64), dtype=np.uint8)
        y0,x0 = 16,16
        y1,x1 = 48,48
        img[y0:y1,x0:x1,1] = 255
        mask[y0:y1,x0:x1] = 255
        import cv2
        cv2.imwrite(str(root / 'images' / f'im{i}.png'), img)
        cv2.imwrite(str(root / 'masks' / f'im{i}.png'), mask)
        # YOLO label: class 0, center x,y,w,h normalized
        cx = (x0 + x1)/2/64
        cy = (y0 + y1)/2/64
        w = (x1 - x0)/64
        h = (y1 - y0)/64
        with open(root / 'labels' / f'im{i}.txt','w') as f:
            f.write(f"0 {cx} {cy} {w} {h}\n")
    data_yaml = root / 'data.yaml'
    data_yaml.write_text(f"""# synthetic
path: {root}
train: images
val: images
names: {{0: obj}}
dataset: {root}
masks_dir: masks
""")
    return data_yaml


def test_mga_train_one_epoch(tmp_path):
    data_yaml = create_synthetic_dataset(tmp_path)
    model = YOLO('configs/models/yolov8_test_segment_heads.yaml')
    results = model.train(data=str(data_yaml), epochs=1, imgsz=64, batch=2, lr0=1e-3, device='cpu', verbose=False, save=False, val=False)
    assert results is not None
