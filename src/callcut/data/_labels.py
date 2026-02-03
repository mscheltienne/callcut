"""Label generation utilities for frame-level segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from callcut.utils._checks import check_type

if TYPE_CHECKING:
    from numpy.typing import NDArray


def intervals_to_frame_labels(intervals: NDArray, times: Tensor) -> Tensor:
    """Convert annotation intervals to per-frame binary labels.

    For each frame, the label is ``1.0`` if the frame's time falls within any
    annotation interval, and ``0.0`` otherwise.

    Parameters
    ----------
    intervals : array of shape ``(n_intervals, 2)``
        Annotation intervals as ``(start, stop)`` times in seconds.
        Can be empty (shape ``(0, 2)``).
    times : Tensor of shape ``(n_frames,)``
        Time axis in seconds, typically from a feature extractor.

    Returns
    -------
    labels : Tensor of shape ``(n_frames,)``
        Binary labels (``0.0`` or ``1.0``) for each frame.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> intervals = np.array([[1.0, 2.0], [3.5, 4.0]])
    >>> times = torch.linspace(0, 5, 100)
    >>> labels = intervals_to_frame_labels(intervals, times)
    >>> labels.shape
    torch.Size([100])
    >>> labels.sum()  # frames within [1,2] and [3.5,4]
    tensor(30.)
    """
    check_type(times, (Tensor,), "times")
    if times.dim() != 1:
        raise ValueError(
            f"Argument 'times' must be 1D, got shape {tuple(times.shape)}."
        )

    labels = torch.zeros(len(times), dtype=torch.float32, device=times.device)

    if len(intervals) == 0:
        return labels

    for start, stop in intervals:
        mask = (times >= start) & (times <= stop)
        labels[mask] = 1.0

    return labels
