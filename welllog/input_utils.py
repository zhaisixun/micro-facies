"""Convert well-log sliding windows to model input tensors."""

import numpy as np
import torch
import torch.nn.functional as F


def prepare_welllog_input(
    x_win: np.ndarray,
    input_size: int,
    use_1d_conv: bool,
    task_mode: str = "classification",
) -> torch.Tensor:
    """Map window array (window_size, num_features) to model input.

    2D mode: (C, L, 1) -> bilinear -> (C, input_size, input_size)
    1D mode: (C, L) -> optional linear resize -> (C, input_size)
    """
    x = torch.from_numpy(x_win.T).float()
    if use_1d_conv:
        if task_mode == "segmentation" and x.shape[-1] != input_size:
            raise ValueError(
                "1D segmentation requires input_size == window_size; "
                f"got input length {x.shape[-1]} and input_size {input_size}."
            )
        if x.shape[-1] != input_size:
            x = F.interpolate(
                x.unsqueeze(0),
                size=input_size,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
        return x

    x = x.unsqueeze(-1)
    x = F.interpolate(
        x.unsqueeze(0),
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return x


def prepare_welllog_batch(
    batch_np: np.ndarray,
    input_size: int,
    use_1d_conv: bool,
    task_mode: str = "classification",
) -> torch.Tensor:
    """Map batch array (B, C, L) to model input."""
    batch_t = torch.from_numpy(batch_np).float()
    if use_1d_conv:
        if task_mode == "segmentation" and batch_t.shape[-1] != input_size:
            raise ValueError(
                "1D segmentation requires input_size == window_size; "
                f"got input length {batch_t.shape[-1]} and input_size {input_size}."
            )
        if batch_t.shape[-1] != input_size:
            batch_t = F.interpolate(
                batch_t,
                size=input_size,
                mode="linear",
                align_corners=False,
            )
        return batch_t

    batch_t = batch_t.unsqueeze(-1)
    batch_t = F.interpolate(
        batch_t,
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    )
    return batch_t
