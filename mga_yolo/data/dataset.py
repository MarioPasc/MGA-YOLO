from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Dict, List, Optional

import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mga_yolo.utils.mask_utils import MaskUtils
from mga_yolo.external.ultralytics.ultralytics.data.dataset import YOLODataset
from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER
from mga_yolo.external.ultralytics.ultralytics.utils.instance import Instances
from mga_yolo.external.ultralytics.ultralytics.utils.ops import resample_segments



class MGADataset(YOLODataset):

    def __init__(self, *args, data: Optional[Dict] = None, task: str = "mga", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: Dict) -> Dict:
        """
        Update label format for different tasks.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)

        # Load per-image binary mask before any transforms so it can be augmented together
        try:
            data_cfg = getattr(self, "data", {}) or {}
            data_root = data_cfg.get("dataset", None)
            masks_dir = data_cfg.get("masks_dir", None)
            mask_path = MaskUtils.infer_mask_path(label.get("im_file", ""), data_root, masks_dir)
            if mask_path is not None and mask_path.exists():
                raw = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    label["bin_mask"] = (raw > 0).astype(np.uint8)
        except Exception as e:  # pragma: no cover
            LOGGER.debug(f"Mask preload failed for {label.get('im_file','')}: {e}")

        return label

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        # If an augmented binary mask is present, downsample to model scales
        bin_mask = sample.get("bin_mask", None)
        if isinstance(bin_mask, np.ndarray):
            # NOTE: Images get letterboxed to match the stride multiple of 8,16,32
            # For example, our images get resized to 544x544, which is divisible by 32
            # The problem is that here, we make the mask downsampling assuming the original
            # mask is aligned with the image after letterboxing. We have to resize the mask to
            # 544x544 first before downsampling.
            H, W = sample.get("ori_shape", (512, 512))[0] + self.stride, sample.get("ori_shape", (512, 512))[1] + self.stride

            LOGGER.debug(f"Sample attributes: {json.dumps({k: str(v) for k,v in sample.items() if k != 'img'}, indent=2)}")
            LOGGER.debug(f"Mask for index {index} resized to {W}x{H} before downsampling")
            # Align mask size with letterboxed image size (depends on stride)
            base = cv2.resize((bin_mask > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA)
            base = np.clip(base, 0.0, 1.0)
            
            MGA_PROB_MODE = os.getenv("MGA_PROB_MODE", False)

            multi_masks: List[torch.Tensor] = []
            for s in (8, 16, 32):
                if MGA_PROB_MODE:
                    ds = MaskUtils.downsample_mask_prob(base, s, method=os.getenv("MGA_MASK_METHOD", "area"))
                else:
                    # compat: binaria clásica
                    ds = MaskUtils.downsample_mask(bin_mask, s)

                multi_masks.append(torch.from_numpy(ds[None, ...]))  # (1,Hs,Ws)
            sample["masks_multi"] = multi_masks


            # Optional debugging: save augmented image/mask pairs
            out_dir = os.getenv("MGA_SAVE_AUG_MASKS", "")
            if out_dir:
                max_saves = int(os.getenv("MGA_SAVE_MAX", "0") or 0)
                try:
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    do_save = max_saves <= 0 or (self._aug_save_count < max_saves)
                    if do_save:
                        stem = Path(sample.get("im_file", f"idx_{index}")).stem
                        # Save mask (uint8 0/255)
                        m = (bin_mask * 255).astype(np.uint8)
                        cv2.imwrite(str(Path(out_dir) / f"{stem}_mask.png"), m)
                        # Save image as uint8 BGR
                        img_t: torch.Tensor = sample.get("img")  # (C,H,W)
                        if isinstance(img_t, torch.Tensor):
                            img_np = (img_t.detach().cpu().numpy().transpose(1, 2, 0))
                            if img_np.dtype != np.uint8:
                                # Format._format_img kept uint8; but if float, scale back
                                img_show = np.clip(img_np, 0, 255).astype(np.uint8)
                            else:
                                img_show = img_np
                            # If image is CHW in RGB/BGR already, trust as BGR for saving
                            cv2.imwrite(str(Path(out_dir) / f"{stem}_img.png"), img_show)
                        self._aug_save_count += 1
                except Exception as e:  # pragma: no cover
                    LOGGER.debug(f"Failed saving augmented mask/image for index {index}: {e}")
        return sample

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        # Stack multi-scale masks if present
        if any('masks_multi' in b for b in batch):
            # Transpose list-of-lists: scales x batch
            scales = len(batch[0].get('masks_multi', []))
            stacked = []
            for si in range(scales):
                tensors = [b['masks_multi'][si] for b in batch if 'masks_multi' in b]
                if tensors:
                    # Pad to max H,W in this scale across batch for stacking
                    max_h = max(t.shape[1] for t in tensors)
                    max_w = max(t.shape[2] for t in tensors)
                    padded = []
                    for t in tensors:
                        if t.shape[1] != max_h or t.shape[2] != max_w:
                            pad_h = max_h - t.shape[1]
                            pad_w = max_w - t.shape[2]
                            t = F.pad(t, (0, pad_w, 0, pad_h), value=0.0)
                        padded.append(t)
                    stacked.append(torch.stack(padded, 0))  # (B,1,H,W)
            if stacked:
                new_batch['masks_multi'] = stacked  # list[Tensor] length=3
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch