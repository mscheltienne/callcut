"""Boundary accuracy metric computation."""

from __future__ import annotations

import numpy as np

from callcut.inference import Interval
from callcut.metrics._types import BoundaryAccuracy, Match


def compute_boundary_accuracy(
    ground_truth: list[Interval], predictions: list[Interval], matches: list[Match]
) -> BoundaryAccuracy:
    """Compute onset/offset timing errors for matched events.

    Measures how accurately the predicted call boundaries align with the
    ground truth boundaries. Only matched events (true positives) are
    included in this computation.

    Parameters
    ----------
    ground_truth : list of Interval
        Ground truth call intervals.
    predictions : list of Interval
        Predicted call intervals.
    matches : list of Match
        Matches between ground truth and predictions, typically from
        :class:`~callcut.metrics.IoUMatcher`.

    Returns
    -------
    accuracy : BoundaryAccuracy
        Boundary accuracy statistics including onset and offset errors in milliseconds,
        plus summary statistics (median, mean absolute, 95th percentile).

    Notes
    -----
    Errors are computed as ``predicted - ground_truth``:

    - **Positive onset error**: Prediction started too late (detected late)
    - **Negative onset error**: Prediction started too early (detected early)
    - **Positive offset error**: Prediction ended too late (extended too long)
    - **Negative offset error**: Prediction ended too early (cut off too soon)

    The summary statistics help characterize the overall boundary accuracy:

    - **Median**: Typical signed error (positive = systematic late bias)
    - **Mean absolute**: Average magnitude of errors
    - **95th percentile (of absolute)**: Worst-case bound for most errors

    Examples
    --------
    >>> from callcut.inference import Interval
    >>> from callcut.metrics import IoUMatcher, compute_boundary_accuracy
    >>>
    >>> gt = [Interval(1.0, 2.0), Interval(3.0, 4.0)]
    >>> pred = [Interval(1.02, 1.98), Interval(3.01, 4.05)]  # small errors
    >>>
    >>> matcher = IoUMatcher(iou_threshold=0.2)
    >>> matches = matcher.match(gt, pred)
    >>>
    >>> accuracy = compute_boundary_accuracy(gt, pred, matches)
    >>> accuracy.onset_mean_abs_ms  # average onset error ~15ms
    15.0
    """
    n_matches = len(matches)

    if n_matches == 0:
        return BoundaryAccuracy(
            n_matches=0,
            onset_errors_ms=np.array([], dtype=np.float64),
            offset_errors_ms=np.array([], dtype=np.float64),
        )

    onset_errors: list[float] = []
    offset_errors: list[float] = []

    for match in matches:
        gt = ground_truth[match.gt_index]
        pred = predictions[match.pred_index]

        # Error = predicted - ground_truth (in milliseconds)
        onset_error_ms = (pred.onset - gt.onset) * 1000.0
        offset_error_ms = (pred.offset - gt.offset) * 1000.0

        onset_errors.append(onset_error_ms)
        offset_errors.append(offset_error_ms)

    return BoundaryAccuracy(
        n_matches=n_matches,
        onset_errors_ms=np.array(onset_errors, dtype=np.float64),
        offset_errors_ms=np.array(offset_errors, dtype=np.float64),
    )
