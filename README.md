# MGA-YOLO 

*Mask-Guided Attention for Ultralytics YOLO*

[![Python](https://img.shields.io/badge/python-3.10%20|%203.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MGA-YOLO is a research-grade fork of Ultralytics-YOLO that injects **mask-guided Convolutional Block Attention Modules (CBAM)** into the feature pyramid (layers P3 / P4 / P5). By fusing binary vessel (or organ) masks with spatial attention, MGA-YOLO focuses learning on anatomically relevant regions and consistently boosts detection accuracy in medical and other cluttered domains.

> **Lead author:** Mario Pascual González · *[mpascual@uma.es](mailto:mpascual@uma.es)* · **ICAI Resarch Group, University of Málaga**
> 
> **Paper draft:** coming soon (working title *“Mask-Guided YOLO for Coronary Stenosis Detection”*)

![15](/assets/feature_maps/arcadetest_p45_v45_00045_layer-model-15.png)
![18](/assets/feature_maps/arcadetest_p45_v45_00045_layer-model-18.png)
![21](/assets/feature_maps/arcadetest_p45_v45_00045_layer-model-21.png)
> Figure 1. *Masked feature maps after MGA-CBAM module for YOLOv8n*. Images used are from [ARCADE: Automatic Region-based Coronary Artery Disease diagnostics using x-ray angiography imagEs Dataset](https://doi.org/10.5281/zenodo.10390295)

---

## Table of Contents

- [MGA-YOLO](#mga-yolo)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Dataset \& Masks](#dataset--masks)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Comparison](#comparison)
  - [Inference](#inference)
  - [Project Layout](#project-layout)
  - [Citing MGA-YOLO](#citing-mga-yolo)
  - [Contributing \& Support](#contributing--support)


## Quick Start

```bash
# 1 ▸ set up an env with CUDA ≥ 11.8 & PyTorch ≥ 2.1
conda create -n mga_yolo python=3.10
conda activate mga_yolo

# 2 ▸ install the package (editable mode for research)
git clone https://github.com/MarioPasc/MGA-YOLO.git
cd MGA-YOLO
pip install -e .

# 3 ▸ train on your coronary-angiography dataset
python -m mga_yolo.cli.train --config configs/MGAConfig.yaml
```

---

## Installation

```bash
git clone https://github.com/MarioPasc/MGA-YOLO.git
cd MGA-YOLO
pip install -e .
```

> **Requirements**
> • Python 3.10 / 3.11 • PyTorch ≥ 2.1 • CUDA 11.7 / 11.8 (optional for GPU)

---

## Dataset & Masks

MGA-YOLO needs the classic YOLO detection tuples **plus** a binary mask per image.

```
my_dataset/
├─ images/
│  ├─ train/000123.jpg
│  └─ val/ ...
├─ labels/                 # YOLO txt files
│  ├─ train/000123.txt
│  └─ val/ ...
masks/                  # binary PNG or JPG
├─ 000123.png           # 1→object, 0→background
└─ ...
```

Mask discovery heuristics (old-style, prefix match, numeric ID, etc.) are implemented in `mga_yolo/io/mask_io.py`; just keep identical stems (`000123.jpg` ↔ `000123.png`) and you are safe.

---

## Configuration

Create a YAML (or start from `configs/template.yaml`):

```yaml
# configs/coronary_stenosis.yaml
model_cfg:  yolov8n.pt
data_yaml:  data/coronary.yaml
masks_dir:  data/coronary/masks

# Training
epochs: 150
imgsz:  640
device: 0             # GPU id or 'cpu'
batch:  8
lr0:    1e-3
weight_decay: 5e-4

# MGA
target_layers: ["15", "18", "21"]      # P3/P4/P5
reduction_ratio:        16
sam_cam_fusion:         add            # {add, concat, multiply}
mga_pyramid_fusion:     add
visualize_features:     true           # save one PNG/layer
```

Any argument accepted by Ultralytics can be added here as well.

---

## Training

```bash
python -m mga_yolo.cli.train \
       --config configs/coronary_stenosis.yaml \
```

During the first epoch you should see log lines like

```
[HookManager] registered MGA hook on layer 15
[HookManager] saved PNG -> runs/feature_vis/000123_P3.png
```

which confirm that masks and CBAM are active.

---

## Comparison

You can compare the performance of the base YOLO model with the MGA-YOLO model by using the `compare` flag with this test. 

```bash

cd MGA-YOLO
python -m tests.performance --config path/to/MGAConfig.yaml \
                            --mode [compare / mga / base] \ 
```

---

## Inference

```bash
# image
python -m mga_yolo.cli.predict --config path/to/MGAConfig.yaml \
                               --weights runs/exp/weights/best.pt \
                               --images example.png \
                               --save-feature-maps \
                               --feature-dir path/to/feature_dir
```

Dataset path is given under `MGAConfig.yaml`, we therefore only need the image name to run inference (as long as the image is in the dataset). 

---

## Project Layout

```
MGA-YOLO/
├─ mga_yolo/
│  ├─ cli/                 # thin wrappers around Ultralytics
│  ├─ cfg/                 # dataclass-based config + validators
│  ├─ nn/
│  │  ├─ mga_cbam.py       # ⟵ MaskGuidedCBAM (SAM+CAM+mask)
│  │  └─ cbam.py           # baseline CBAM
│  ├─ hooks.py             # forward-hook manager
│  ├─ trainer.py           # MGADetectionTrainer subclass
│  ├─ external/            # ultralytics package
│  └─ utils/               # visualisation, logging
├─ configs/                # ready-to-run YAMLs
├─ tests/                  # pytest + health checks
└─ docs/                   # sphinx (optional)
```

---

## Citing MGA-YOLO

```bibtex
@software{pascual2025mgayolo,
  author  = {Pascual González, Mario},
  title   = {Mask-Guided YOLO for Coronary Stenosis Detection”},
  year    = {2025},
  url     = {https://github.com/MarioPasc/MGA-YOLO},
  note    = {MIT License}
}
```

---

## Contributing & Support

**Pull requests** are welcome. For questions open an issue or email *[mpascual@uma.es](mailto:mpascual@uma.es)*.

---

© 2025 Mario Pascual González — released under the MIT License.
