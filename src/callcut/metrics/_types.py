"""Data types for metric results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Match:
    """A match between a ground truth and predicted interval.

    Parameters
    ----------
    gt_index : int
        Index of the matched ground truth interval.
    pred_index : int
        Index of the matched predicted interval.
    iou : float
        Intersection over Union score for this match.

    Examples
    --------
    >>> match = Match(gt_index=0, pred_index=2, iou=0.75)
    >>> match.iou
    0.75
    """

    gt_index: int
    pred_index: int
    iou: float

    def __repr__(self) -> str:
        return (
            f"Match(gt_index={self.gt_index}, "
            f"pred_index={self.pred_index}, "
            f"iou={self.iou:.4f})"
        )


@dataclass(frozen=True)
class EventMetrics:
    """Event-level detection metrics.

    These metrics evaluate detection at the call/event level: each ground truth call is
    either matched to a prediction (true positive) or missed (false negative), and each
    prediction is either matched (true positive) or a false alarm (false positive).

    Parameters
    ----------
    n_ground_truth : int
        Total number of ground truth events.
    n_predicted : int
        Total number of predicted events.
    tp : int
        True positives (correctly matched predictions).
    fp : int
        False positives (predictions without a matching ground truth).
    fn : int
        False negatives (ground truth events without a matching prediction).
    precision : float
        Precision = TP / (TP + FP). Of the predicted calls, what fraction are real?
    recall : float
        Recall = TP / (TP + FN). Of the real calls, what fraction were detected?
    f1 : float
        F1 score = 2 * precision * recall / (precision + recall).
        Harmonic mean of precision and recall.

    Examples
    --------
    >>> metrics = EventMetrics(
    ...     n_ground_truth=10,
    ...     n_predicted=8,
    ...     tp=7,
    ...     fp=1,
    ...     fn=3,
    ...     precision=0.875,
    ...     recall=0.7,
    ...     f1=0.778,
    ... )
    >>> metrics.precision
    0.875
    """

    n_ground_truth: int
    n_predicted: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    def __repr__(self) -> str:
        return (
            f"EventMetrics(tp={self.tp}, fp={self.fp}, fn={self.fn}, "
            f"precision={self.precision:.3f}, recall={self.recall:.3f}, "
            f"f1={self.f1:.3f})"
        )


@dataclass(frozen=True)
class FrameMetrics:
    """Frame-level detection metrics.

    These metrics evaluate detection at the individual frame level: each frame is
    classified as either containing a call or not.

    Parameters
    ----------
    n_frames : int
        Total number of frames evaluated.
    tp : int
        True positives (call frames correctly predicted as calls).
    fp : int
        False positives (non-call frames incorrectly predicted as calls).
    fn : int
        False negatives (call frames incorrectly predicted as non-calls).
    tn : int
        True negatives (non-call frames correctly predicted as non-calls).
    precision : float
        Precision = TP / (TP + FP).
    recall : float
        Recall = TP / (TP + FN).
    f1 : float
        F1 score = 2 * precision * recall / (precision + recall).

    Examples
    --------
    >>> metrics = FrameMetrics(
    ...     n_frames=1000,
    ...     tp=150,
    ...     fp=20,
    ...     fn=30,
    ...     tn=800,
    ...     precision=0.882,
    ...     recall=0.833,
    ...     f1=0.857,
    ... )
    """

    n_frames: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float

    def __repr__(self) -> str:
        return (
            f"FrameMetrics(tp={self.tp}, fp={self.fp}, fn={self.fn}, tn={self.tn}, "
            f"precision={self.precision:.3f}, recall={self.recall:.3f}, "
            f"f1={self.f1:.3f})"
        )


def _compute_error_stats(errors: NDArray) -> tuple[float, float, float]:
    """Compute summary statistics for an array of errors.

    Returns (median, mean_abs, p95) or (nan, nan, nan) if empty.
    """
    if errors.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(errors)),
        float(np.mean(np.abs(errors))),
        float(np.percentile(np.abs(errors), 95)),
    )


@dataclass
class BoundaryAccuracy:
    """Boundary (onset/offset) accuracy statistics for matched events.

    Measures how accurately the predicted call boundaries match the ground truth
    boundaries. Only computed for matched events (true positives).

    Errors are computed as ``predicted - ground_truth``, so:

    - Positive onset error = prediction started too late
    - Negative onset error = prediction started too early
    - Positive offset error = prediction ended too late
    - Negative offset error = prediction ended too early

    Parameters
    ----------
    n_matches : int
        Number of matched event pairs used for computing errors.
    onset_errors_ms : NDArray
        Signed onset errors in milliseconds (predicted - ground_truth).
    offset_errors_ms : NDArray
        Signed offset errors in milliseconds (predicted - ground_truth).

    Attributes
    ----------
    onset_median_ms : float
        Median onset error in milliseconds.
    onset_mean_abs_ms : float
        Mean absolute onset error in milliseconds.
    onset_p95_ms : float
        95th percentile of absolute onset error in milliseconds.
    offset_median_ms : float
        Median offset error in milliseconds.
    offset_mean_abs_ms : float
        Mean absolute offset error in milliseconds.
    offset_p95_ms : float
        95th percentile of absolute offset error in milliseconds.

    Examples
    --------
    >>> import numpy as np
    >>> accuracy = BoundaryAccuracy(
    ...     n_matches=50,
    ...     onset_errors_ms=np.array([10.0, -5.0, 15.0, -8.0]),
    ...     offset_errors_ms=np.array([20.0, -10.0, 5.0, -15.0]),
    ... )
    >>> accuracy.onset_median_ms
    2.5
    """

    n_matches: int
    onset_errors_ms: NDArray = field(repr=False)
    offset_errors_ms: NDArray = field(repr=False)

    # Computed statistics (set in __post_init__)
    onset_median_ms: float = field(init=False)
    onset_mean_abs_ms: float = field(init=False)
    onset_p95_ms: float = field(init=False)
    offset_median_ms: float = field(init=False)
    offset_mean_abs_ms: float = field(init=False)
    offset_p95_ms: float = field(init=False)

    def __post_init__(self) -> None:
        # Compute onset statistics
        median, mean_abs, p95 = _compute_error_stats(self.onset_errors_ms)
        self.onset_median_ms = median
        self.onset_mean_abs_ms = mean_abs
        self.onset_p95_ms = p95

        # Compute offset statistics
        median, mean_abs, p95 = _compute_error_stats(self.offset_errors_ms)
        self.offset_median_ms = median
        self.offset_mean_abs_ms = mean_abs
        self.offset_p95_ms = p95

    def __repr__(self) -> str:
        return (
            f"BoundaryAccuracy(n_matches={self.n_matches}, "
            f"onset_median_ms={self.onset_median_ms:.1f}, "
            f"offset_median_ms={self.offset_median_ms:.1f})"
        )
