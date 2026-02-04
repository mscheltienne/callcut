"""Interval matching algorithms for event-level evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from callcut.evaluation._types import Interval, Match
from callcut.utils._checks import check_type


class BaseIntervalMatcher(ABC):
    """Abstract base class for interval matching strategies.

    Matchers pair predicted intervals with ground truth intervals for evaluation
    purposes. Different matching strategies may use different criteria (IoU,
    boundary tolerance, etc.).

    Examples
    --------
    Create a custom matcher by subclassing:

    >>> class BoundaryMatcher(BaseIntervalMatcher):
    ...     def __init__(self, tolerance_ms: float = 50.0):
    ...         self._tolerance_ms = tolerance_ms
    ...
    ...     def match(self, ground_truth, predictions):
    ...         # Match based on onset boundary tolerance
    ...         ...
    """

    @abstractmethod
    def match(
        self, ground_truth: list[Interval], predictions: list[Interval]
    ) -> list[Match]:
        """Match predicted intervals to ground truth intervals.

        Parameters
        ----------
        ground_truth : list of Interval
            Ground truth call intervals.
        predictions : list of Interval
            Predicted call intervals.

        Returns
        -------
        matches : list of Match
            List of matches, where each match pairs a ground truth index
            with a prediction index. Each ground truth and prediction can
            appear in at most one match.
        """


class IoUMatcher(BaseIntervalMatcher):
    """Match intervals using greedy IoU (Intersection over Union).

    This matcher pairs ground truth and predicted intervals based on their overlap. It
    uses a greedy algorithm that prioritizes high-IoU matches.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU required for a valid match. Pairs with IoU below this threshold are
        not matched. Default is ``0.2``.

    Notes
    -----
    The matching algorithm:

    1. Compute IoU for all (ground_truth, prediction) pairs.
    2. Sort pairs by IoU in descending order.
    3. Greedily assign matches, starting with the highest IoU pair.
    4. Each ground truth and prediction can only be matched once.
    5. Pairs with IoU < threshold are not matched.

    Examples
    --------
    >>> from callcut.evaluation import Interval
    >>> gt = [Interval(0.0, 1.0), Interval(2.0, 3.0)]
    >>> pred = [Interval(0.1, 0.9), Interval(2.1, 3.1)]
    >>> matcher = IoUMatcher(iou_threshold=0.2)
    >>> matches = matcher.match(gt, pred)
    >>> len(matches)
    2
    """

    def __init__(self, iou_threshold: float = 0.2) -> None:
        check_type(iou_threshold, ("numeric",), "iou_threshold")
        if not 0 <= iou_threshold <= 1:
            raise ValueError(
                f"Argument 'iou_threshold' must be in [0, 1], got {iou_threshold}."
            )
        self._iou_threshold = float(iou_threshold)

    @property
    def iou_threshold(self) -> float:
        """Minimum IoU for a valid match.

        :type: :class:`float`
        """
        return self._iou_threshold

    @staticmethod
    def _interval_iou(a: Interval, b: Interval) -> float:
        r"""Compute Intersection over Union (IoU) between two intervals.

        Parameters
        ----------
        a : Interval
            First interval.
        b : Interval
            Second interval.

        Returns
        -------
        iou : float
            Intersection over Union, in range ``[0, 1]``.

        Notes
        -----
        The IoU is computed as:

        .. math::

            \text{IoU} = \frac{\text{intersection}}{\text{union}}

        where intersection is the overlap duration and union is the total duration
        covered by both intervals (without double-counting the overlap).
        """
        # Compute intersection
        intersection_start = max(a.onset, b.onset)
        intersection_end = min(a.offset, b.offset)
        intersection = max(0.0, intersection_end - intersection_start)

        # Compute union
        union = a.duration + b.duration - intersection

        # Avoid division by zero (both intervals have zero duration)
        if union <= 0:
            return 0.0

        return intersection / union

    def match(
        self, ground_truth: list[Interval], predictions: list[Interval]
    ) -> list[Match]:
        """Match predicted intervals to ground truth using greedy IoU matching.

        Parameters
        ----------
        ground_truth : list of Interval
            Ground truth call intervals.
        predictions : list of Interval
            Predicted call intervals.

        Returns
        -------
        matches : list of Match
            List of matches. Each match contains the indices of matched
            ground truth and prediction intervals, plus the IoU score.
        """
        if len(ground_truth) == 0 or len(predictions) == 0:
            return []

        # Compute all pairwise IoUs
        iou_pairs: list[tuple[float, int, int]] = []
        for gi, gt in enumerate(ground_truth):
            for pi, pred in enumerate(predictions):
                iou = self._interval_iou(gt, pred)
                iou_pairs.append((iou, gi, pi))

        # Sort by IoU descending
        iou_pairs.sort(key=lambda x: x[0], reverse=True)

        # Greedy matching
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        matches: list[Match] = []

        for iou, gi, pi in iou_pairs:
            # Stop if IoU is below threshold (list is sorted)
            if iou < self._iou_threshold:
                break

            # Skip if either already matched
            if gi in matched_gt or pi in matched_pred:
                continue

            # Create match
            matches.append(Match(gt_index=gi, pred_index=pi, iou=iou))
            matched_gt.add(gi)
            matched_pred.add(pi)

        return matches

    def __repr__(self) -> str:
        return f"IoUMatcher(iou_threshold={self._iou_threshold})"
