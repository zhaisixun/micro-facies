"""Padding mask utilities for variable-length whole-well batches."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Build a bool mask (B, L) where True marks valid (non-padded) positions."""
    if lengths.ndim != 1:  # lengths.shape = (B,)  样表真实长度
        raise ValueError(f"lengths must be 1D, got shape {tuple(lengths.shape)}")
    idx = torch.arange(max_len, device=lengths.device)  # idx.shape = (L,) 索引长度为max_len
    return idx.unsqueeze(0) < lengths.unsqueeze(1)  # idx.unsqueeze(0).shape = (1, L), lengths.unsqueeze(1).shape = (B, 1), 比较结果的shape = (B, L)，返回一个bool mask (B, L)，True表示非padding位置


def mask_channels_first(mask: torch.Tensor) -> torch.Tensor:
    """(B, L) bool -> (B, 1, L) float multiplier."""
    return mask.unsqueeze(1).to(dtype=torch.float32)  # mask.shape = (B, L), mask.unsqueeze(1).shape = (B, 1, L), 返回一个float mask (B, 1, L)


def apply_feature_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out padded positions in (B, C, L) feature maps."""
    return x * mask_channels_first(mask)


def downsample_valid_mask(
    mask: torch.Tensor,
    kernel_size: int,
    stride: int,
) -> torch.Tensor:
    """Downsample mask after a strided conv.

    A position stays valid only when the entire receptive field is valid,
    so padded zeros cannot leak into real depth points.
    """
    inv = (~mask).float().unsqueeze(1)
    bad = F.max_pool1d(inv, kernel_size=kernel_size, stride=stride, padding=0)
    return (bad.squeeze(1) < 0.5)


def masked_adaptive_avg_pool1d(
    x: torch.Tensor,
    mask: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Average pool (B, C, L) using only valid positions in each segment."""
    bsz, channels, length = x.shape
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    if output_size == 1:
        m = mask_channels_first(mask)
        denom = m.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x * m).sum(dim=-1, keepdim=True) / denom

    if length % output_size != 0:
        pad = output_size - (length % output_size)
        x = F.pad(x, (0, pad))
        mask = F.pad(mask, (0, pad))
        length = x.shape[-1]

    seg_len = length // output_size
    x_seg = x.view(bsz, channels, output_size, seg_len)
    m_seg = mask.view(bsz, 1, output_size, seg_len).to(x.dtype)
    denom = m_seg.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return (x_seg * m_seg).sum(dim=-1) / denom.squeeze(-1)


def resize_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Nearest-neighbor resize for bool masks (B, L) -> (B, size)."""
    if mask.shape[-1] == size:
        return mask
    resized = F.interpolate(
        mask.unsqueeze(1).float(),
        size=size,
        mode="nearest",
    ).squeeze(1)
    return resized > 0.5
