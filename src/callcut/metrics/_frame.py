"""Frame-level metric computation."""

from __future__ import annotations

import numpy as np
from torch import Tensor

from callcut.metrics._types import FrameMetrics
from callcut.utils._checks import check_type


def compute_frame_metrics(
    probabilities: Tensor, labels: Tensor, *, threshold: float = 0.5
) -> FrameMetrics:
    """Compute frame-level precision, recall, and F1 score.

    Frame-level metrics evaluate detection at the individual frame granularity: each
    frame is classified as either containing a call (positive) or not (negative), and
    compared against the ground truth labels.

    Parameters
    ----------
    probabilities : Tensor
        Predicted probabilities of shape ``(n_frames,)``, with values in ``[0, 1]``.
        Typically from :func:`~callcut.inference.predict_recording`.
    labels : Tensor
        Ground truth labels of shape ``(n_frames,)``, with values ``0`` (no call)
        or ``1`` (call). Typically from :func:`~callcut.data.intervals_to_frame_labels`.
    threshold : float
        Classification threshold. Frames with ``probability >= threshold`` are
        classified as positive (call). Default is ``0.5``.

    Returns
    -------
    metrics : FrameMetrics
        Frame-level detection metrics including TP, FP, FN, TN, precision, recall, and
        F1 score.

    Notes
    -----
    Frame-level metrics are useful for quick sanity checks during training
    but can be misleading for event detection. A model might achieve high
    frame-level F1 by correctly predicting the middle of calls while missing
    boundaries, or by predicting many short false alarms.

    For final evaluation, event-level metrics (from
    :func:`~callcut.metrics.compute_event_metrics`) are generally more
    meaningful.

    Examples
    --------
    >>> import torch
    >>> probs = torch.tensor([0.1, 0.8, 0.9, 0.7, 0.2, 0.1])
    >>> labels = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    >>> metrics = compute_frame_metrics(probs, labels, threshold=0.5)
    >>> metrics.precision
    1.0
    >>> metrics.recall
    1.0
    """
    check_type(probabilities, (Tensor,), "probabilities")
    check_type(labels, (Tensor,), "labels")
    check_type(threshold, ("numeric",), "threshold")

    if not 0 <= threshold <= 1:
        raise ValueError(f"Argument 'threshold' must be in [0, 1], got {threshold}.")

    # Convert to numpy for computation
    probs_np = probabilities.detach().cpu().numpy().astype(np.float64)
    labels_np = labels.detach().cpu().numpy().astype(np.float64)

    if probs_np.ndim != 1:
        raise ValueError(f"probabilities must be 1D, got shape {probs_np.shape}.")
    if labels_np.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels_np.shape}.")
    if probs_np.size != labels_np.size:
        raise ValueError(
            f"probabilities and labels must have same length. "
            f"Got {probs_np.size} and {labels_np.size}."
        )

    # Threshold predictions
    predictions = (probs_np >= threshold).astype(np.uint8)
    labels_binary = (labels_np >= 0.5).astype(np.uint8)

    # Compute confusion matrix elements
    tp = int(((predictions == 1) & (labels_binary == 1)).sum())
    fp = int(((predictions == 1) & (labels_binary == 0)).sum())
    fn = int(((predictions == 0) & (labels_binary == 1)).sum())
    tn = int(((predictions == 0) & (labels_binary == 0)).sum())

    # Compute metrics with epsilon to avoid division by zero
    eps = 1e-12
    precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall + eps)
        if (precision + recall) > 0
        else 0.0
    )

    return FrameMetrics(
        n_frames=probs_np.size,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
    )
