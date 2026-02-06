"""Data types for pipeline results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from callcut.evaluation._types import (
        BoundaryAccuracy,
        EventMetrics,
        FrameMetrics,
        Interval,
        Match,
    )
    from callcut.io import RecordingInfo


@dataclass(frozen=True)
class RecordingEvaluation:
    """Evaluation results for a single recording.

    Parameters
    ----------
    recording : RecordingInfo
        Metadata about the evaluated recording.
    ground_truth : tuple of Interval
        Ground truth call intervals.
    predictions : tuple of Interval
        Predicted call intervals.
    matches : tuple of Match
        Matches between ground truth and predictions.
    event_metrics : EventMetrics
        Event-level precision, recall, and F1.
    boundary_accuracy : BoundaryAccuracy
        Onset/offset timing errors.
    frame_metrics : FrameMetrics
        Frame-level precision, recall, and F1.
    """

    recording: RecordingInfo
    ground_truth: tuple[Interval, ...]
    predictions: tuple[Interval, ...]
    matches: tuple[Match, ...]
    event_metrics: EventMetrics
    boundary_accuracy: BoundaryAccuracy
    frame_metrics: FrameMetrics

    def __repr__(self) -> str:
        return (
            f"RecordingEvaluation({self.recording.audio_path.name!r}, "
            f"gt={len(self.ground_truth)}, pred={len(self.predictions)}, "
            f"event_f1={self.event_metrics.f1:.3f}, "
            f"frame_f1={self.frame_metrics.f1:.3f})"
        )


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate evaluation results across multiple recordings.

    Parameters
    ----------
    recordings : tuple of RecordingEvaluation
        Per-recording evaluation results.
    event_metrics : EventMetrics
        Aggregated event-level metrics across all recordings.
    boundary_accuracy : BoundaryAccuracy
        Aggregated boundary accuracy across all recordings.
    frame_metrics : FrameMetrics
        Aggregated frame-level metrics across all recordings.
    fp_per_minute : float
        False positives per minute of recording.
    total_duration_minutes : float
        Total duration of all evaluated recordings in minutes.
    """

    recordings: tuple[RecordingEvaluation, ...]
    event_metrics: EventMetrics
    boundary_accuracy: BoundaryAccuracy
    frame_metrics: FrameMetrics
    fp_per_minute: float
    total_duration_minutes: float

    def __repr__(self) -> str:
        return (
            f"EvaluationReport(\n"
            f"  n_recordings={len(self.recordings)}, "
            f"total_duration={self.total_duration_minutes:.1f} min\n"
            f"  Event: {self.event_metrics}\n"
            f"  Frame: {self.frame_metrics}\n"
            f"  Boundary: {self.boundary_accuracy}\n"
            f"  FP/min: {self.fp_per_minute:.3f}\n"
            f")"
        )


@dataclass(frozen=True)
class RecordingPrediction:
    """Predicted call intervals for a single recording.

    Parameters
    ----------
    audio_path : Path
        Path to the audio file.
    intervals : tuple of Interval
        Predicted call intervals.
    duration_s : float
        Duration of the recording in seconds.
    """

    audio_path: Path
    intervals: tuple[Interval, ...]
    duration_s: float

    def __repr__(self) -> str:
        return (
            f"RecordingPrediction({self.audio_path.name!r}, "
            f"n_intervals={len(self.intervals)}, "
            f"duration={self.duration_s:.1f}s)"
        )


