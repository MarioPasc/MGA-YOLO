"""Masked Efficient Channel Attention (MaskECA).

Implements a Mask-guided variant of ECA-Net (CVPR 2020) for lightweight channel attention.

Core idea:
- Standard ECA computes channel-wise attention weights by:
  1) Global Average Pooling (GAP) across HxW to get a C-dimensional descriptor.
  2) A fast 1D convolution (no dimensionality reduction) over the C-dim vector
	 to capture local cross-channel interaction.
  3) Sigmoid to produce channel weights, multiplied back to the feature map.

- MaskECA replaces GAP with masked average pooling when a spatial mask is available, i.e.:
	  v_c = sum_{h,w}(x_{c,h,w} * m_{h,w}) / sum_{h,w}(m_{h,w})
  falling back to GAP if the mask area is tiny or mask is None. This allows the module to
  emphasize channels activated on salient spatial regions.

References:
- Q. Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks," CVPR 2020.

Notes:
- Kernel size k is adaptively computed from the channel dimension C using the standard rule:
	  k = odd(round(|log2(C)/gamma + b|)) with gamma=2, b=1 by default.
  We build the 1D conv with that k at init time, using the input channel count provided by
  the model parser (which infers it from the graph). If the actual runtime input has a
  different C (should not happen under Ultralytics' parser), we recompute the conv lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MaskECA"]


def _odd(k: int) -> int:
	return k if k % 2 == 1 else k + 1


def eca_kernel_size(channels: int, gamma: float = 2.0, b: float = 1.0, k_min: int = 3, k_max: int = 15) -> int:
	"""Compute ECA 1D kernel size given channels C.

	Ensures an odd kernel size clamped to [k_min, k_max].
	"""
	if channels <= 0:
		return k_min
	k = int(abs((channels.bit_length() - 1) / gamma + b))  # approx log2(C) via bit_length
	k = _odd(max(k_min, min(k, k_max)))
	return k


@dataclass
class MaskECAConfig:
	channels: int
	gamma: float = 2.0
	b: float = 1.0
	k_min: int = 3
	k_max: int = 15
	use_sigmoid_mask: bool = True  # apply sigmoid before using mask
	tiny_mask_threshold: float = 1e-4  # fallback to GAP if mean(mask) below this
	eps: float = 1e-6


class MaskECA(nn.Module):
	"""Mask-guided Efficient Channel Attention.

	Args:
		channels: Number of input channels (C). Used to determine the ECA kernel size.
		gamma: ECA gamma parameter for adaptive kernel sizing.
		b: ECA b parameter for adaptive kernel sizing.
		k_min: Minimum kernel size (odd).
		k_max: Maximum kernel size (odd).
		use_sigmoid_mask: Whether to apply sigmoid() to the incoming mask before pooling.
		tiny_mask_threshold: If the average mask value is below this, fallback to global avg pooling.
		eps: Numerical stability epsilon.

	Forward inputs:
		- Either a single tensor x: (B, C, H, W), or a list/tuple [x, mask],
		  where mask: (B, 1, H, W) or (B, H, W). If mask is provided, masked average pooling is used.

	Returns:
		Refined tensor with same shape as x: (B, C, H, W).
	"""

	def __init__(
		self,
		channels: int,
		gamma: float = 2.0,
		b: float = 1.0,
		k_min: int = 3,
		k_max: int = 15,
		use_sigmoid_mask: bool = True,
		tiny_mask_threshold: float = 1e-4,
		eps: float = 1e-6,
	) -> None:
		super().__init__()
		self.cfg = MaskECAConfig(
			channels=channels,
			gamma=gamma,
			b=b,
			k_min=k_min,
			k_max=k_max,
			use_sigmoid_mask=use_sigmoid_mask,
			tiny_mask_threshold=tiny_mask_threshold,
			eps=eps,
		)
		k = eca_kernel_size(channels, gamma=gamma, b=b, k_min=k_min, k_max=k_max)
		self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

	def _maybe_rebuild_conv(self, channels: int) -> None:
		"""Rebuild the 1D conv if runtime channels differ from configured channels.

		This should rarely happen; Ultralytics parser passes the correct channels.
		"""
		if channels == self.cfg.channels:
			return
		k = eca_kernel_size(channels, gamma=self.cfg.gamma, b=self.cfg.b, k_min=self.cfg.k_min, k_max=self.cfg.k_max)
		# Preserve existing parameter dtype/device when rebuilding
		weight = self.conv1d.weight
		self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False).to(
			device=weight.device, dtype=weight.dtype
		)
		self.cfg.channels = channels

	def _pool(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
		"""Return (B, C) pooled descriptor using masked avg if available else global avg."""
		b, c, h, w = x.shape
		if mask is None:
			return F.adaptive_avg_pool2d(x, 1).view(b, c)
		if mask.dim() == 3:
			mask = mask.unsqueeze(1)
		if self.cfg.use_sigmoid_mask:
			mask = mask.sigmoid()
		# Broadcast mask to channels
		if mask.shape[1] == 1 and c > 1:
			mask = mask.expand(b, c, h, w)
		# Fallback to GAP if mask area too small
		mean_mask = mask.mean(dim=(2, 3), keepdim=False).mean(dim=1)  # (B,)
		if (mean_mask < self.cfg.tiny_mask_threshold).any():
			gap = F.adaptive_avg_pool2d(x, 1).view(b, c)
			# blend: use masked where valid, GAP otherwise
			mask_sum = mask.sum(dim=(2, 3)).clamp_min(self.cfg.eps)  # (B,C)
			masked = (x * mask).sum(dim=(2, 3)) / mask_sum  # (B,C)
			# ensure blending is done in the same dtype as features to avoid AMP type mismatches
			valid = (mean_mask >= self.cfg.tiny_mask_threshold).to(x.dtype).unsqueeze(1)  # (B,1)
			one = torch.ones(1, dtype=x.dtype, device=x.device)
			return masked * valid + gap * (one - valid)
		# Standard masked average pooling
		mask_sum = mask.sum(dim=(2, 3)).clamp_min(self.cfg.eps)  # (B,C)
		return (x * mask).sum(dim=(2, 3)) / mask_sum

	def forward(self, x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
		if isinstance(x, (list, tuple)):
			assert len(x) == 2, "MaskECA expects [feature, mask] as inputs"
			feat, mask = x  # type: ignore[assignment]
		else:
			feat, mask = x, None  # type: ignore[assignment]

		assert isinstance(feat, torch.Tensor) and feat.dim() == 4, "feature must be (B,C,H,W)"
		b, c, _, _ = feat.shape
		self._maybe_rebuild_conv(c)

		# 1) pooled descriptor: (B, C)
		y = self._pool(feat, mask)  # (B, C)
		# 2) local cross-channel interaction via fast 1D conv
		y = y.unsqueeze(1)  # (B, 1, C)
		# align dtype with conv parameters to be robust under half precision
		y = y.to(self.conv1d.weight.dtype)
		y = self.conv1d(y)  # (B, 1, C)
		# 3) sigmoid to get channel weights and re-scale features
		w = y.squeeze(1).sigmoid().view(b, c, 1, 1)
		return feat * w

	def extra_repr(self) -> str:  # pragma: no cover - simple metadata
		c = self.cfg
		return (
			f"C={c.channels}, gamma={c.gamma}, b={c.b}, k_min={c.k_min}, k_max={c.k_max}, "
			f"sigmoid_mask={c.use_sigmoid_mask}, tiny_thr={c.tiny_mask_threshold}"
		)
