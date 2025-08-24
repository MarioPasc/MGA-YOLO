# MGA-YOLO Change Log (session summary)

## Added / Created

- `configs/models/yolov8_mga.yaml`
  - Added custom MGA architecture YAML (initial YOLO11-style then adapted to YOLOv8-style by user) introducing placeholder MGA components (`MGAMaskHead`, planned `MGAAttention`) and integrating them at P3/P4/P5 feature levels.
  - Goal: Provide a declarative model definition to extend baseline YOLO with mask-guided attention pipeline and per-scale mask heads.

- `mga_yolo/nn/modules/segmentation.py`
  - Implemented `MGAMaskHead` module (config dataclass + lightweight Conv->Norm->Act->Dropout->Conv head) producing coarse per-scale mask logits.
  - Goal: Supply a reusable, typed, easily extensible segmentation head for FPN scales (P3/P4/P5) to support mask-guided attention and future loss integration.

- `mga_yolo/nn/modules/__init__.py`
  - Added logging setup (`LOGGER`) for MGA segmentation namespace and later extended to expose custom modules.
  - Goal: Centralize debug logging for custom MGA components.

- `mga_yolo/utils/dataloader.py`
  - Added thin wrapper `build_dataloader` with debug logging hook.
  - Goal: Allow future custom sampling / curriculum logic while keeping current behavior identical to Ultralytics default.

- `mga_yolo/engine/train.py`
  - Added CLI training entrypoint wrapping Ultralytics `YOLO` to ensure MGA modules are imported before model instantiation.
  - Goal: Simplify launching training with custom MGA YAML and dataset configuration.

- `mga_yolo/engine/model.py`, `mga_yolo/engine/predict.py`, `mga_yolo/engine/val.py`
  - Introduced `MGAModel` (subclass of `DetectionModel`) returning a dict `{det: ..., seg: {...}}` with raw per-scale mask logits plus detection output.
  - Added `MGAPredictor` attaching `mga_masks` dict to each `Results` object and `MGAValidator` ignoring segmentation masks for current metrics.
  - Goal: Provide a native end-to-end forward path without reliance on forward hooks.

- `mga_yolo/nn/losses/segmentation.py`
  - Implemented multi-scale segmentation loss (`SegmentationLoss`) combining BCE-with-logits and soft Dice per scale (P3/P4/P5) with configurable weights (`SegLossConfig`).
  - Goal: Enable integration of coarse mask supervision into total training loss via `MGATrainer`.

- `tests_mga/test_mga_basic.py`, `tests_mga/test_mga_train.py`
  - Added smoke tests for: (a) model build & forward dict outputs (3 mask scales), (b) prediction path producing `mga_masks`, (c) synthetic dataset creator (future 1-epoch training smoke – currently minimal and may require installing `numpy`, `opencv-python`).
  - Goal: Establish an initial regression harness for MGA functionality.

## Modified

- `ultralytics/ultralytics/models/yolo/model.py`
  - Added automatic task override (`task='mga'`) when model filename (or YAML content) indicates MGA usage ("mga" in stem or presence of `MGAMaskHead`).
  - Extended `task_map` with an `mga` entry referencing `MGAModel`, `MGATrainer`, `MGAPredictor`, `MGAValidator`.
  - NOTE: Current iterative edits introduced circular import/initialization issues causing `from ultralytics import YOLO` to fail; planned resolution is to revert most invasive changes and re-apply a minimal, low-risk task injection (see Pending section below).

- `mga_yolo/engine/train.py`
  - Extended to initialize and aggregate segmentation losses (`SegmentationLoss`) alongside standard detection loss (BCE + Dice per scale + global weighting) producing extra loss item logging keys: `p3_bce`, `p3_dice`, `p4_bce`, `p4_dice`, `p5_bce`, `p5_dice`, `seg_total`.
  - Goal: Fuse segmentation supervision into training loop transparently.

- `mga_yolo/engine/model.py`
  - Tweaked forward pass implementation to always return dict and simplify stored outputs while collecting mask logits by layer index.
  - Goal: Provide stable contract for downstream trainer/predictor.

- (Planned fix not yet applied) `ultralytics/ultralytics/data/dataset.py`
  - Identified duplicate `__getitem__` definition shadowing mask-loading logic; pending clean-up to ensure only enhanced version with `masks_multi` assembly remains (will finalize in next iteration).

- `ultralytics/ultralytics/nn/modules/__init__.py`
  - Appended try/except import of `MGAMaskHead` so YAML parser can resolve the custom layer name.
  - Goal: Register custom MGA module within Ultralytics module namespace for YAML-driven model building.

- `ultralytics/ultralytics/data/dataset.py`
  - Added logic to: (1) infer per-image mask path from dataset root + `masks_dir`, (2) load grayscale mask, (3) perform naive stride-based downsampling for P3/P4/P5 (8/16/32) using nearest resize, (4) attach list `masks_multi` to each sample, (5) extend `collate_fn` to pad & stack multi-scale masks, (6) added helper methods `_infer_mask_path` and `_downsample_mask`, required imports (`math`, `torch.nn.functional as F`).
  - Goal: Provide training batches with multi-scale coarse masks aligned to detection feature map scales for future segmentation loss / attention.
  - NOTE: Current downsampling is a placeholder; will be replaced with connectivity-preserving method (`downsample_preserve_connectivity`) in a subsequent change per user request.

## Intent / Pending (not yet implemented in code above)

- Implement `MGAAttention` and potential fused detection+segmentation loss integration.
- Add segmentation loss computation using `masks_multi` inside training loop.
  (Partially implemented now via `SegmentationLoss`; next step: verify with real dataset & add segmentation metrics such as per-scale Dice/mIoU.)
- Resolve circular import introduced by extensive edits in `yolo/model.py` by reverting to upstream base and applying a minimal MGA task hook (safer path) OR refactoring remaining `yolo.*` references to lazy local imports.
- Remove obsolete duplicate dataset `__getitem__` and finalize connectivity-preserving downsampling integration.


## Segmentation loss

This section documents the end-to-end integration of a multi-scale segmentation loss into MGA-YOLO, detailing where the loss is defined, how ground-truth masks are loaded and prepared, which components consume the loss, and how it is wired into the training loop.

### Overview

- Goal: Supervise the three MGA coarse mask heads (P3/P4/P5) with a composite loss (BCE with logits + soft Dice), aggregated and combined with the base YOLO detection loss during training.
- Scales: P3, P4, P5 feature maps (typical strides 8/16/32) produce one-channel mask logits each.
- Outputs: The model forward returns a dict {det, seg} where seg is a mapping {p3, p4, p5 -> mask logits}.

### Where the loss is defined (mga_yolo)

- `mga_yolo/nn/losses/segmentation.py`
  - Defines `SegLossConfig` (typed, dataclass) with weights for BCE, Dice, per-scale weighting, smoothing, global lambda, and enable flag.
  - Implements `SegmentationLoss(nn.Module)` that:
    - Expects preds dict keys: `p3`, `p4`, `p5` with shapes (B, 1, H, W) per scale.
    - Expects targets as a list of tensors aligned to the scales (B, 1, Hs, Ws). Targets are resized with nearest if shapes slightly differ.
    - Computes:
      - BCE: `torch.nn.BCEWithLogitsLoss(reduction="mean")` on logits vs. binary targets.
      - Soft Dice: classic sigmoid(pred) against binary masks, per-batch averaged, with smoothing.
      - Weighted combination per scale, summed across scales, multiplied by `loss_lambda` for final total.
    - Returns `(total_loss, logs_dict)` with per-scale bce/dice/combined plus overall `seg_total`.

### Which components use the loss (mga_yolo)

- `mga_yolo/engine/train.py`
  - Class `MGATrainer(DetectionTrainer)` integrates segmentation loss alongside detection loss.
  - Key methods:
    - `init_losses()` constructs `SegmentationLoss` from `self.args` if available or sensible defaults, and extends `self.loss_names` with: `p3_bce`, `p3_dice`, `p4_bce`, `p4_dice`, `p5_bce`, `p5_dice`, `seg_total` (in this order).
    - `preprocess_batch()` guarantees `batch["masks_multi"]` exists (falling back to an empty list when absent) so the criterion can branch cleanly.
    - `criterion(preds, batch)`
      - Handles both dict and raw outputs; for dict expects `{det, seg}` and passes `det` into the base `compute_loss` while computing segmentation loss with `seg` and `batch["masks_multi"]` when present.
      - Aggregates detection and segmentation losses (`total = det_loss + seg_total`).
      - Builds `self.loss_items` tensor aligned to `self.loss_names` for Ultralytics logging.
    - `save_model()` respects `args.save=False` to avoid serialization during unit tests.
    - `after_epoch()` appends a CSV row to `loss_log.csv` under the experiment `save_dir` including `epoch` and all loss items (only when `args.save=True`).

- `mga_yolo/engine/model.py`
  - Class `MGAModel(DetectionModel)` overrides `_predict_once` to mirror YOLO’s internal forward, collecting outputs from `MGAMaskHead` layers and mapping them to scale names (`p3`, `p4`, `p5`).
  - Returns `{"det": x, "seg": {"p3": ..., "p4": ..., "p5": ...}}` by default to the trainer’s `criterion`.

### How ground-truth masks are loaded and aligned (vendored ultralytics)

- `mga_yolo/external/ultralytics/ultralytics/data/dataset.py`
  - Enhancements in dataset loading pipeline (inside `YOLODataset.__getitem__` and `collate_fn`):
    - `_infer_mask_path`: infers a per-image mask by combining the configured dataset root (`dataset:`) and `masks_dir:` from the data YAML with the image stem and common mask extensions.
    - Mask loading: loads grayscale mask; converts to binary; downscales to strides {8, 16, 32} via `_downsample_mask`.
    - `_downsample_mask`: prefers connectivity-preserving downsample (if `DownsampleConfig` and `downsample_preserve_connectivity` are available from `mga_yolo.utils.mask_downsample`) and falls back to nearest-neighbor resize.
    - `masks_multi`: attaches a Python list of three tensors (one per scale) to each sample.
    - `collate_fn`: pads across the batch and stacks each scale to shape (B, 1, Hs, Ws), producing `batch["masks_multi"]` as a 3-element list of batched tensors.

### Task registration and minimal framework changes (vendored ultralytics)

- `mga_yolo/external/ultralytics/ultralytics/cfg/__init__.py`
  - Registered a dedicated default dataset mapping for the MGA task: `TASK2DATA["mga"] = "coco8.yaml"` (acts as a safe baseline; users should override with their own `data.yaml`).

- `mga_yolo/external/ultralytics/ultralytics/models/yolo/model.py`
  - Extended `YOLO.task_map` to include the MGA route:
    - `model`: `MGAModel`
    - `trainer`: lazily imported `MGATrainer` to avoid circular imports (`__import__('mga_yolo.engine.train', fromlist=['MGATrainer']).MGATrainer`).
    - `validator`: `MGAValidator`
    - `predictor`: `MGAPredictor`
  - The constructor auto-detects MGA usage from the model path (contains "mga") or YAML content (contains `MGAMaskHead`) and sets `task="mga"` accordingly.

### Training loop flow

1) Data loader provides batches with images and `masks_multi` (list of (B,1,Hs,Ws) tensors for P3/P4/P5) when mask files are present; otherwise `masks_multi` may be empty and segmentation loss is skipped.
2) `MGAModel` forward returns dict output with detection and segmentation logits per scale.
3) `MGATrainer.criterion` computes detection loss through Ultralytics’ `compute_loss` and, if enabled and targets exist, computes segmentation loss through `SegmentationLoss`.
4) The total training loss is the sum of detection and segmentation components; detailed segmentation metrics are appended to `self.loss_items` and written to CSV after each epoch if saving.

### Tests and verification

- `tests/test_mga_seg_loss.py`
  - Unit-level integration test using a `DummyTrainer` subclass that bypasses dataset setup and exercises `MGATrainer.criterion` with synthetic `masks_multi` tensors aligned to P3/P4/P5. Verifies:
    - Loss requires gradients and is positive/finite.
    - `loss_names` include the segmentation keys and `seg_total`.
  - Complements existing smoke tests for model build/predict/train.

### Notes

- CSV logging (`loss_log.csv`) is written only when `args.save=True` to keep tests fast and side-effect-free.
- Users should set `configs/data/data.yaml` with correct `dataset:` and `masks_dir:`; `names` or `nc` are required by Ultralytics to validate datasets.
- The connectivity-preserving downsample path is optional; it will be used when the helper is importable, otherwise nearest-neighbor resize is applied.

