"""Decoders for converting frame probabilities to time intervals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from callcut.evaluation._types import Interval
from callcut.utils._checks import check_type

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor


class BaseDecoder(ABC):
    """Abstract base class for probability-to-interval decoders.

    Subclasses must implement :meth:`decode` to convert frame-level probabilities
    into a list of detected call intervals.

    Examples
    --------
    Create a custom decoder by subclassing:

    >>> class SimpleThresholdDecoder(BaseDecoder):
    ...     def __init__(self, threshold: float = 0.5):
    ...         self._threshold = threshold
    ...
    ...     def decode(self, times, probabilities):
    ...         # Implementation here
    ...         ...
    """

    @abstractmethod
    def decode(self, times: Tensor, probabilities: Tensor) -> list[Interval]:
        """Convert frame probabilities to a list of time intervals.

        Parameters
        ----------
        times : Tensor of shape ``(n_frames,)``
            Time axis of shape ``(n_frames,)`` in seconds, from the feature extractor.
        probabilities : Tensor of shape ``(n_frames,)``
            Per-frame probabilities of shape ``(n_frames,)``, values in ``[0, 1]``,
            from :meth:`~callcut.nn.BaseDetector.predict`.

        Returns
        -------
        intervals : list of Interval
            Detected call intervals, sorted by onset time.
        """


class HysteresisDecoder(BaseDecoder):
    """Decode probabilities using hysteresis thresholding.

    Uses separate enter/exit thresholds to avoid rapid on/off switching when
    probabilities hover near a single threshold. After initial detection, nearby
    intervals are merged and short intervals are filtered out.

    Parameters
    ----------
    enter_threshold : float
        Probability threshold to START a call. When not in a call, a call begins
        when probability rises above this value.
    exit_threshold : float
        Probability threshold to END a call. When in a call, the call ends when
        probability falls below this value. Must be <= ``enter_threshold``.
    min_duration_s : float
        Minimum call duration in seconds. Calls shorter than this are discarded.
    min_gap_s : float
        Minimum gap between calls in seconds. Calls separated by less than this
        are merged into a single call.
    pad_s : float
        Padding to add around each detected call in seconds. The onset is moved
        earlier and the offset is moved later by this amount.

    Notes
    -----
    The decoding pipeline has four stages:

    1. **Hysteresis thresholding**: Mark frames as "in call" using enter/exit
       thresholds. This creates a binary mask.
    2. **Extract intervals**: Find contiguous regions in the mask and convert
       to time intervals.
    3. **Merge short gaps**: If two calls are separated by less than ``min_gap_s``,
       merge them into one.
    4. **Filter and pad**: Remove calls shorter than ``min_duration_s``, then
       add ``pad_s`` padding to remaining calls.

    Examples
    --------
    >>> import torch
    >>> decoder = HysteresisDecoder(
    ...     enter_threshold=0.6,
    ...     exit_threshold=0.4,
    ...     min_duration_s=0.05,
    ... )
    >>> times = torch.linspace(0, 1, 100)
    >>> probs = torch.zeros(100)
    >>> probs[20:40] = 0.8  # a clear call region
    >>> intervals = decoder.decode(times, probs)
    >>> len(intervals)
    1
    """

    def __init__(
        self,
        enter_threshold: float = 0.6,
        exit_threshold: float = 0.4,
        min_duration_s: float = 0.03,
        min_gap_s: float = 0.02,
        pad_s: float = 0.0,
    ) -> None:
        check_type(enter_threshold, ("numeric",), "enter_threshold")
        check_type(exit_threshold, ("numeric",), "exit_threshold")
        check_type(min_duration_s, ("numeric",), "min_duration_s")
        check_type(min_gap_s, ("numeric",), "min_gap_s")
        check_type(pad_s, ("numeric",), "pad_s")

        if not 0 <= exit_threshold <= enter_threshold <= 1:
            raise ValueError(
                f"Thresholds must satisfy 0 <= exit_threshold <= enter_threshold <= 1. "
                f"Got exit_threshold={exit_threshold}, "
                f"enter_threshold={enter_threshold}."
            )
        if min_duration_s < 0:
            raise ValueError(
                f"Argument 'min_duration_s' must be >= 0, got {min_duration_s}."
            )
        if min_gap_s < 0:
            raise ValueError(f"Argument 'min_gap_s' must be >= 0, got {min_gap_s}.")
        if pad_s < 0:
            raise ValueError(f"Argument 'pad_s' must be >= 0, got {pad_s}.")

        self._enter_threshold = float(enter_threshold)
        self._exit_threshold = float(exit_threshold)
        self._min_duration_s = float(min_duration_s)
        self._min_gap_s = float(min_gap_s)
        self._pad_s = float(pad_s)

    @property
    def enter_threshold(self) -> float:
        """Probability threshold to start a call.

        :type: :class:`float`
        """
        return self._enter_threshold

    @property
    def exit_threshold(self) -> float:
        """Probability threshold to end a call.

        :type: :class:`float`
        """
        return self._exit_threshold

    @property
    def min_duration_s(self) -> float:
        """Minimum call duration in seconds.

        :type: :class:`float`
        """
        return self._min_duration_s

    @property
    def min_gap_s(self) -> float:
        """Minimum gap between calls in seconds.

        :type: :class:`float`
        """
        return self._min_gap_s

    @property
    def pad_s(self) -> float:
        """Padding around calls in seconds.

        :type: :class:`float`
        """
        return self._pad_s

    def decode(self, times: Tensor, probabilities: Tensor) -> list[Interval]:
        """Convert frame probabilities to a list of time intervals.

        Parameters
        ----------
        times : Tensor of shape ``(n_frames,)``
            Time axis of shape ``(n_frames,)`` in seconds, from the feature extractor.
        probabilities : Tensor of shape ``(n_frames,)``
            Per-frame probabilities of shape ``(n_frames,)``, values in ``[0, 1]``,
            from :meth:`~callcut.nn.BaseDetector.predict`.

        Returns
        -------
        intervals : list of Interval
            Detected call intervals, sorted by onset time.
        """
        check_type(times, ("tensor",), "times")
        check_type(probabilities, ("tensor",), "probabilities")

        # Convert to numpy for processing
        times_np = times.detach().cpu().numpy().astype(np.float64)
        probs_np = probabilities.detach().cpu().numpy().astype(np.float64)

        if times_np.ndim != 1:
            raise ValueError(f"times must be 1D, got shape {times_np.shape}.")
        if probs_np.ndim != 1:
            raise ValueError(f"probabilities must be 1D, got shape {probs_np.shape}.")
        if times_np.size != probs_np.size:
            raise ValueError(
                f"times and probabilities must have same length. "
                f"Got {times_np.size} and {probs_np.size}."
            )

        if times_np.size == 0:
            return []

        # Stage 1: Hysteresis thresholding
        mask = self._hysteresis_threshold(probs_np)

        # Stage 2: Extract intervals from mask
        intervals = self._mask_to_intervals(mask, times_np)

        # Stage 3: Merge short gaps
        intervals = self._merge_gaps(intervals)

        # Stage 4: Filter short calls and add padding
        intervals = self._filter_and_pad(intervals, times_np)

        return intervals

    def _hysteresis_threshold(self, probabilities: NDArray) -> NDArray[np.bool_]:
        """Apply hysteresis thresholding to create a binary mask.

        Parameters
        ----------
        probabilities : NDArray
            Per-frame probabilities.

        Returns
        -------
        mask : NDArray[np.bool_]
            Boolean mask where True indicates "in call".
        """
        n_frames = len(probabilities)
        mask = np.zeros(n_frames, dtype=np.bool_)
        in_call = False

        for i in range(n_frames):
            p = probabilities[i]
            if not in_call and p >= self._enter_threshold:
                in_call = True
            if in_call:
                mask[i] = True
                if p <= self._exit_threshold:
                    in_call = False

        return mask

    def _mask_to_intervals(
        self, mask: NDArray[np.bool_], times: NDArray
    ) -> list[Interval]:
        """Convert a binary mask to a list of intervals.

        Parameters
        ----------
        mask : NDArray[np.bool_]
            Boolean mask where True indicates "in call".
        times : NDArray
            Time axis in seconds.

        Returns
        -------
        intervals : list of Interval
            Detected intervals.
        """
        if not np.any(mask):
            return []

        # Find transitions
        mask_int = mask.astype(np.int32)
        changes = np.diff(mask_int)
        starts = np.where(changes == 1)[0] + 1  # 0→1 transitions
        ends = np.where(changes == -1)[0] + 1  # 1→0 transitions

        # Handle edge cases
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])

        # Convert to intervals
        intervals = []
        for s, e in zip(starts, ends, strict=True):
            # Use time at start frame for onset, time at last frame (e-1) for offset
            onset = float(times[s])
            offset = float(times[e - 1])
            intervals.append(Interval(onset=onset, offset=offset))

        return intervals

    def _merge_gaps(self, intervals: list[Interval]) -> list[Interval]:
        """Merge intervals separated by gaps shorter than min_gap_s.

        Parameters
        ----------
        intervals : list of Interval
            Input intervals (must be sorted by onset).

        Returns
        -------
        merged : list of Interval
            Merged intervals.
        """
        if not intervals:
            return []

        merged: list[Interval] = []
        for interval in intervals:
            if len(merged) == 0:
                merged.append(interval)
                continue

            gap = interval.onset - merged[-1].offset
            if gap < self._min_gap_s:
                # Merge: extend previous interval
                merged[-1] = Interval(onset=merged[-1].onset, offset=interval.offset)
            else:
                merged.append(interval)

        return merged

    def _filter_and_pad(
        self, intervals: list[Interval], times: NDArray
    ) -> list[Interval]:
        """Filter short intervals and add padding.

        Parameters
        ----------
        intervals : list of Interval
            Input intervals.
        times : NDArray
            Time axis (used to determine recording bounds).

        Returns
        -------
        filtered : list of Interval
            Filtered and padded intervals.
        """
        if len(intervals) == 0:
            return []

        # Get recording time bounds
        t_min = float(times[0])
        t_max = float(times[-1])

        filtered: list[Interval] = []
        for interval in intervals:
            if interval.duration >= self._min_duration_s:
                # Add padding, but clamp to recording bounds
                onset = max(t_min, interval.onset - self._pad_s)
                offset = min(t_max, interval.offset + self._pad_s)
                filtered.append(Interval(onset=onset, offset=offset))

        return filtered

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"enter_threshold={self._enter_threshold}, "
            f"exit_threshold={self._exit_threshold}, "
            f"min_duration_s={self._min_duration_s}, "
            f"min_gap_s={self._min_gap_s}, "
            f"pad_s={self._pad_s})"
        )
