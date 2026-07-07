"""Collate functions for well-log datasets."""

from typing import List, Tuple

import torch


def collate_whole_well(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length whole-well samples to a rectangular batch.

    Each item is (x, y, weight):
        x: (C, L)
        y: (L,) int64, invalid depth points already set to ignore_index
        weight: scalar float

    Returns:
        x_pad: (B, C, L_max)
        y_pad: (B, L_max)
        weights: (B,)
        lengths: (B,) actual sequence length before padding
    """
    if not batch:
        raise ValueError("collate_whole_well received an empty batch.")

    xs, ys, ws = zip(*batch)   # xs是(C, L)的tensor, ys是(L,)的tensor, ws是标量float
    lengths = torch.tensor([x.shape[-1] for x in xs], dtype=torch.long)  # 每个样本的实际长度
    max_len = int(lengths.max().item())  # 一个batch中最大的长度
    num_channels = xs[0].shape[0]  # 通道数
    batch_size = len(xs)  # 样本数

    x_pad = xs[0].new_zeros(batch_size, num_channels, max_len)  # 创建一个全零的batch_size x num_channels x max_len的tensor
    y_pad = ys[0].new_full((batch_size, max_len), fill_value=ignore_index, dtype=torch.long)  # 创建一个全为ignore_index的batch_size x max_len的tensor
    weights = torch.stack([w.reshape(()) for w in ws])

    for i, (x, y) in enumerate(zip(xs, ys)):
        seq_len = x.shape[-1]  # 当前样本的实际长度
        if y.shape[0] != seq_len:  
            raise ValueError(
                f"Feature length {seq_len} does not match label length {y.shape[0]} in batch item {i}."
            )
        x_pad[i, :, :seq_len] = x  # 将当前样本的特征填充到x_pad中，padding部分是0
        y_pad[i, :seq_len] = y  # 将当前样本的标签填充到y_pad中，padding部分是ignore_index=-100

    return x_pad, y_pad, weights, lengths  # 返回填充后的特征、标签、权重和长度
