"""Tests for callcut.pipeline._types module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from callcut.evaluation._types import (
    BoundaryAccuracy,
    EventMetrics,
    FrameMetrics,
    Interval,
    Match,
)
from callcut.io import RecordingInfo
from callcut.pipeline._types import (
    EvaluationReport,
    RecordingEvaluation,
    RecordingPrediction,
)


def _make_event_metrics(**kwargs: object) -> EventMetrics:
    """Create EventMetrics with defaults."""
    defaults = {
        "n_ground_truth": 3,
        "n_predicted": 2,
        "tp": 2,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": 0.667,
        "f1": 0.8,
    }
    defaults.update(kwargs)
    return EventMetrics(**defaults)


def _make_frame_metrics(**kwargs: object) -> FrameMetrics:
    """Create FrameMetrics with defaults."""
    defaults = {
        "n_frames": 100,
        "tp": 40,
        "fp": 5,
        "fn": 10,
        "tn": 45,
        "precision": 0.889,
        "recall": 0.8,
        "f1": 0.842,
    }
    defaults.update(kwargs)
    return FrameMetrics(**defaults)


def _make_boundary_accuracy(**kwargs: object) -> BoundaryAccuracy:
    """Create BoundaryAccuracy with defaults."""
    defaults = {
        "n_matches": 2,
        "onset_errors_ms": np.array([10.0, -5.0]),
        "offset_errors_ms": np.array([20.0, -10.0]),
    }
    defaults.update(kwargs)
    return BoundaryAccuracy(**defaults)


def _make_recording_info(**kwargs: object) -> RecordingInfo:
    """Create RecordingInfo with defaults."""
    defaults = {
        "audio_path": Path("/fake/audio.wav"),
        "annotation_path": Path("/fake/annotations.csv"),
        "duration_s": 10.0,
        "n_annotations": 3,
    }
    defaults.update(kwargs)
    return RecordingInfo(**defaults)


class TestRecordingEvaluation:
    """Tests for RecordingEvaluation dataclass."""

    def test_frozen(self) -> None:
        """Test that RecordingEvaluation is immutable."""
        rec_eval = RecordingEvaluation(
            recording=_make_recording_info(),
            ground_truth=(Interval(0.0, 1.0),),
            predictions=(Interval(0.1, 0.9),),
            matches=(Match(gt_index=0, pred_index=0, iou=0.8),),
            event_metrics=_make_event_metrics(),
            boundary_accuracy=_make_boundary_accuracy(),
            frame_metrics=_make_frame_metrics(),
        )
        with __import__("pytest").raises(AttributeError):
            rec_eval.ground_truth = ()

    def test_repr(self) -> None:
        """Test string representation contains key info."""
        rec_eval = RecordingEvaluation(
            recording=_make_recording_info(),
            ground_truth=(Interval(0.0, 1.0), Interval(2.0, 3.0)),
            predictions=(Interval(0.1, 0.9),),
            matches=(Match(gt_index=0, pred_index=0, iou=0.8),),
            event_metrics=_make_event_metrics(),
            boundary_accuracy=_make_boundary_accuracy(),
            frame_metrics=_make_frame_metrics(),
        )
        repr_str = repr(rec_eval)
        assert "RecordingEvaluation" in repr_str
        assert "audio.wav" in repr_str
        assert "gt=2" in repr_str
        assert "pred=1" in repr_str


class TestEvaluationReport:
    """Tests for EvaluationReport dataclass."""

    def test_frozen(self) -> None:
        """Test that EvaluationReport is immutable."""
        report = EvaluationReport(
            recordings=(),
            event_metrics=_make_event_metrics(),
            boundary_accuracy=_make_boundary_accuracy(),
            frame_metrics=_make_frame_metrics(),
            fp_per_minute=0.5,
            total_duration_minutes=10.0,
        )
        with __import__("pytest").raises(AttributeError):
            report.fp_per_minute = 0.0

    def test_repr(self) -> None:
        """Test string representation contains key info."""
        report = EvaluationReport(
            recordings=(),
            event_metrics=_make_event_metrics(),
            boundary_accuracy=_make_boundary_accuracy(),
            frame_metrics=_make_frame_metrics(),
            fp_per_minute=0.5,
            total_duration_minutes=10.0,
        )
        repr_str = repr(report)
        assert "EvaluationReport" in repr_str
        assert "n_recordings=0" in repr_str
        assert "FP/min" in repr_str


class TestRecordingPrediction:
    """Tests for RecordingPrediction dataclass."""

    def test_frozen(self) -> None:
        """Test that RecordingPrediction is immutable."""
        pred = RecordingPrediction(
            audio_path=Path("/fake/audio.wav"),
            intervals=(Interval(0.0, 1.0),),
            duration_s=5.0,
        )
        with __import__("pytest").raises(AttributeError):
            pred.duration_s = 0.0

    def test_repr(self) -> None:
        """Test string representation contains key info."""
        pred = RecordingPrediction(
            audio_path=Path("/fake/audio.wav"),
            intervals=(Interval(0.0, 1.0), Interval(2.0, 3.0)),
            duration_s=5.0,
        )
        repr_str = repr(pred)
        assert "RecordingPrediction" in repr_str
        assert "audio.wav" in repr_str
        assert "n_intervals=2" in repr_str
        assert "5.0s" in repr_str
