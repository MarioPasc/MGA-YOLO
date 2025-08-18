from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor, nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Detect
from mga_yolo.nn.modules.seg import MGAMaskHead



class MGAModel(DetectionModel):
    """
    MGAModel wraps a YOLO-style DetectionModel and augments forward() to also return
    per-scale segmentation logits produced by MGAMaskHead layers defined in the YAML.
    Output:
        If self.return_dict (or force_dict=True): {'det': detection_output, 'seg': {'p3': T, 'p4': T, 'p5': T}}
        Else: standard detection output (backward compatible).
    """

    def __init__(self, cfg: Union[str, Dict[str, Any]] = "yolov8_mga.yaml", ch: int = 3, nc: int = 80, verbose: bool = True):
        # Predefine attributes used inside forward() before parent calls any forward during init (e.g. stride calc)
        self.mga_mask_indices: List[int] = []
        self.mga_scaled_names: Dict[int, str] = {}
    # Disable dict output during base __init__ (stride/shape probes expect tensor)
    self.return_dict: bool = False
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self._index_mask_heads()
    self.mga_scaled_names = self._assign_scale_names()
    # Enable dict outputs for normal operation
    self.return_dict = True

    def _index_mask_heads(self) -> None:
        if MGAMaskHead is None:
            return
        for i, m in enumerate(self.model):
            if isinstance(m, MGAMaskHead):
                self.mga_mask_indices.append(i)

    def _assign_scale_names(self) -> Dict[int, str]:
        """
        Heuristic: map mask head indices to p3/p4/p5 by ascending spatial size (first = p3, etc.).
        More robust mapping could inspect source layer stride.
        """
        ordered = list(self.mga_mask_indices)
        return {idx: f"p{3 + k}" for k, idx in enumerate(ordered)}

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        augment: bool = False,
        profile: bool = False,
        visualize: bool = False,
        force_dict: bool = False,
    ) -> Union[Tensor, Dict[str, Any]]:
        """Forward pass producing detection output plus raw mask logits.

        Returns a dict by default for MGA models unless force_dict is False and return_dict disabled.
        """
        if augment:
            raise NotImplementedError("Augment not yet implemented for MGAModel.")
        saved: List[Tensor] = []  # stored layer outputs
        seg_outs: Dict[str, Tensor] = {}
        for i, m in enumerate(self.model):
            if isinstance(m.f, int):
                x_in: Any = saved[m.f] if m.f != -1 else x
            else:
                x_in = [x if j == -1 else saved[j] for j in m.f]
            x = m(x_in)
            if i in self.save:
                saved.append(x)
            else:
                saved.append(x)  # keep for potential later index safety
            if i in self.mga_mask_indices:
                seg_outs[self.mga_scaled_names.get(i, f"mask_{i}")] = x
        det_out: Tensor = x  # last layer output (Detect)
        if self.return_dict or force_dict:
            return {"det": det_out, "seg": seg_outs}
        return det_out