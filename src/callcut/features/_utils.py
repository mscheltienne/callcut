"""Internal utility functions for feature extraction."""

from __future__ import annotations

import torch
from torch import Tensor


def median_filter_1d(x: Tensor, kernel_size: int) -> Tensor:
    """Apply 1D median filter with zero padding.

    This implementation uses zero (constant) padding at boundaries to match the
    behavior of :func:`scipy.signal.medfilt`.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(T,)`` or ``(B, T)``.
    kernel_size : int
        Size of the median filter kernel. Must be a positive odd integer.
        If even, it will be incremented by ``1``.

    Returns
    -------
    filtered : Tensor
        Filtered tensor of the same shape as input.
    """
    # ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    # handle 1D input
    squeeze = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True

    pad = kernel_size // 2

    # zero padding to match scipy.signal.medfilt behavior
    x_padded = torch.nn.functional.pad(x, (pad, pad), mode="constant", value=0)

    # unfold to get sliding windows: (B, T, kernel_size)
    windows = x_padded.unfold(dimension=1, size=kernel_size, step=1)

    # median along last dimension
    result = windows.median(dim=-1).values

    if squeeze:
        result = result.squeeze(0)

    return result
