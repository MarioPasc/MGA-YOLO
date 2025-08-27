Title: Mask-aware augmentation added to MGA-YOLO data pipeline

Date: 2025-08-27

Summary
- Load the binary segmentation mask before transforms and propagate it through spatial augmentations.
- Apply the same geometry to masks as to images/bboxes (Mosaic, RandomPerspective, LetterBox, RandomFlip, MixUp, CutMix).
- Downsample the augmented mask to P3/P4/P5 (strides 8/16/32) after transforms and attach as masks_multi.
- Add optional debug saving of augmented image and mask to verify correctness.

Files changed
- ultralytics/data/dataset.py
  - In YOLODataset.update_labels_info: preload per-image binary mask into labels["bin_mask"].
  - In YOLODataset.__getitem__: when bin_mask exists post-augment, downsample to strides (8,16,32) and attach labels['masks_multi'].
  - Add optional debug saver controlled by env vars MGA_SAVE_AUG_MASKS and MGA_SAVE_MAX.

- ultralytics/data/augment.py
  - Mosaic (_mosaic3/_mosaic4/_mosaic9): tile labels['bin_mask'] alongside image tiles.
  - RandomPerspective.__call__: warp labels['bin_mask'] with same matrix using nearest interpolation.
  - RandomFlip.__call__: flip labels['bin_mask'] consistently.
  - LetterBox.__call__: resize and pad labels['bin_mask'] with nearest/zero padding.
  - MixUp._mix_transform: OR-combine masks when present.
  - CutMix._mix_transform: copy-cut region from donor mask.

Verification
- Export augmented samples by setting env var:
  - MGA_SAVE_AUG_MASKS=/tmp/aug_preview  # directory where to save augmented masks/images
  - MGA_SAVE_MAX=50                      # optional limit of saved pairs
- Run a short training or a DataLoader preview to generate files in that folder.
- Inspect *_img.png and *_mask.png pairs. They must align: mask contours should map onto the vessel pixels after augmentation.

Notes
- Mask downsampling method is controlled via env variables in YOLODataset._downsample_mask:
  - MGA_MASK_METHOD: nearest | area | maxpool | pyrdown | skeleton_bresenham (default)
  - MGA_MASK_BRIDGE: 1/0 to enable 3x3 morphological close bridge (area/pyrdown methods)
  - MGA_MASK_THRESH: threshold for area method
- The Albumentations transform is color-only by default here; spatial ops are handled by our geometry-aware transforms above.

Why
- Previously, masks were loaded after transforms, so they were not augmented consistently with images/bboxes. This change ensures parity and correctness for the segmentation head supervision.
