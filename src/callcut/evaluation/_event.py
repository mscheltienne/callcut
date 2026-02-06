"""Event-level metric computation."""

from __future__ import annotations

from callcut.evaluation._types import EventMetrics, Interval, Match
from callcut.evaluation._utils import _precision_recall_f1


def compute_event_metrics(
    ground_truth: list[Interval], predictions: list[Interval], matches: list[Match]
) -> EventMetrics:
    r"""Compute event-level precision, recall, and F1 score.

    Event-level metrics evaluate detection at the call/event granularity: each ground
    truth event is either correctly detected (true positive) or missed (false negative),
    and each prediction is either correct (true positive) or a false alarm (false
    positive).

    Parameters
    ----------
    ground_truth : list of Interval
        Ground truth call intervals.
    predictions : list of Interval
        Predicted call intervals.
    matches : list of Match
        Matches between ground truth and predictions, typically from
        :class:`~callcut.evaluation.IoUMatcher`.

    Returns
    -------
    metrics : EventMetrics
        Event-level detection metrics including TP, FP, FN, precision, recall, and F1
        score.

    Notes
    -----
    The metrics are computed as:

    - **True Positives (TP)**: Number of matched pairs. Each match represents
      a ground truth event that was correctly detected.
    - **False Positives (FP)**: Predictions without a match. These are
      "false alarms" - the model predicted a call where there was none.
    - **False Negatives (FN)**: Ground truth events without a match. These
      are "missed detections" - real calls that the model failed to detect.

    Precision and recall are computed as:

    .. math::

        \\text{Precision} = \\frac{TP}{TP + FP}

        \\text{Recall} = \\frac{TP}{TP + FN}

        F_1 = \\frac{2 \\cdot \\text{Precision} \\cdot \\text{Recall}}
              {\\text{Precision} + \\text{Recall}}

    Examples
    --------
    >>> from callcut.evaluation import Interval, IoUMatcher, compute_event_metrics
    >>>
    >>> gt = [Interval(0.0, 1.0), Interval(2.0, 3.0), Interval(4.0, 5.0)]
    >>> pred = [Interval(0.1, 0.9), Interval(2.1, 3.1)]  # missed one
    >>>
    >>> matcher = IoUMatcher(iou_threshold=0.2)
    >>> matches = matcher.match(gt, pred)
    >>>
    >>> metrics = compute_event_metrics(gt, pred, matches)
    >>> metrics.tp
    2
    >>> metrics.fn  # one ground truth was missed
    1
    >>> metrics.recall
    0.666...
    """
    n_gt = len(ground_truth)
    n_pred = len(predictions)
    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp

    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)

    return EventMetrics(
        n_ground_truth=n_gt,
        n_predicted=n_pred,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )
