"""Tests for callcut.pipeline._evaluate module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from callcut.evaluation import HysteresisDecoder, IoUMatcher
from callcut.evaluation._types import (
    BoundaryAccuracy,
    EventMetrics,
    FrameMetrics,
    Interval,
    Match,
)
from callcut.extractors import SNRExtractor
from callcut.io import RecordingInfo
from callcut.nn import TinySegCNN
from callcut.pipeline._evaluate import _aggregate_evaluations, evaluate_recordings
from callcut.pipeline._types import EvaluationReport, RecordingEvaluation


def _make_recording_eval(
    *,
    audio_name: str = "test.wav",
    duration_s: float = 60.0,
    n_gt: int = 3,
    n_pred: int = 2,
    tp: int = 2,
    frame_tp: int = 40,
    frame_fp: int = 5,
    frame_fn: int = 10,
    frame_tn: int = 45,
    onset_errors: np.ndarray | None = None,
    offset_errors: np.ndarray | None = None,
) -> RecordingEvaluation:
    """Create a RecordingEvaluation with controllable metrics."""
    fp = n_pred - tp
    fn = n_gt - tp
    gt = tuple(Interval(float(i), float(i) + 1.0) for i in range(n_gt))
    preds = tuple(Interval(float(i) + 0.1, float(i) + 0.9) for i in range(n_pred))
    matches = tuple(Match(gt_index=i, pred_index=i, iou=0.8) for i in range(tp))
    if onset_errors is None:
        onset_errors = np.array([10.0, -5.0])[:tp]
    if offset_errors is None:
        offset_errors = np.array([20.0, -10.0])[:tp]

    eps = 1e-12
    precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall + eps)
        if (precision + recall) > 0
        else 0.0
    )
    n_frames = frame_tp + frame_fp + frame_fn + frame_tn
    f_prec = (
        frame_tp / (frame_tp + frame_fp + eps) if (frame_tp + frame_fp) > 0 else 0.0
    )
    f_rec = frame_tp / (frame_tp + frame_fn + eps) if (frame_tp + frame_fn) > 0 else 0.0
    f_f1 = 2 * f_prec * f_rec / (f_prec + f_rec + eps) if (f_prec + f_rec) > 0 else 0.0

    return RecordingEvaluation(
        recording=RecordingInfo(
            audio_path=Path(f"/fake/{audio_name}"),
            annotation_path=Path(f"/fake/{audio_name}_annotations.csv"),
            duration_s=duration_s,
            n_annotations=n_gt,
        ),
        ground_truth=gt,
        predictions=preds,
        matches=matches,
        event_metrics=EventMetrics(
            n_ground_truth=n_gt,
            n_predicted=n_pred,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
        ),
        boundary_accuracy=BoundaryAccuracy(
            n_matches=tp,
            onset_errors_ms=onset_errors,
            offset_errors_ms=offset_errors,
        ),
        frame_metrics=FrameMetrics(
            n_frames=n_frames,
            tp=frame_tp,
            fp=frame_fp,
            fn=frame_fn,
            tn=frame_tn,
            precision=f_prec,
            recall=f_rec,
            f1=f_f1,
        ),
    )


class TestAggregateEvaluations:
    """Tests for _aggregate_evaluations."""

    def test_single_recording(self) -> None:
        """Test aggregation with a single recording."""
        rec = _make_recording_eval()
        report = _aggregate_evaluations([rec])
        assert isinstance(report, EvaluationReport)
        assert len(report.recordings) == 1
        assert report.event_metrics.tp == rec.event_metrics.tp
        assert report.frame_metrics.tp == rec.frame_metrics.tp

    def test_multiple_recordings_sums_tp_fp_fn(self) -> None:
        """Test that TP/FP/FN are summed across recordings."""
        r1 = _make_recording_eval(tp=2, n_gt=3, n_pred=3)
        r2 = _make_recording_eval(tp=1, n_gt=2, n_pred=2)
        report = _aggregate_evaluations([r1, r2])
        assert report.event_metrics.tp == 3
        assert report.event_metrics.fp == (3 - 2) + (2 - 1)
        assert report.event_metrics.fn == (3 - 2) + (2 - 1)

    def test_fp_per_minute(self) -> None:
        """Test FP/min is computed from total FP and total duration."""
        r1 = _make_recording_eval(tp=0, n_gt=0, n_pred=3, duration_s=120.0)
        report = _aggregate_evaluations([r1])
        # 3 FP in 2 minutes = 1.5 FP/min
        assert_allclose(report.fp_per_minute, 1.5, atol=1e-6)

    def test_boundary_errors_concatenated(self) -> None:
        """Test that boundary errors are concatenated across recordings."""
        r1 = _make_recording_eval(
            tp=1,
            n_gt=1,
            n_pred=1,
            onset_errors=np.array([10.0]),
            offset_errors=np.array([20.0]),
        )
        r2 = _make_recording_eval(
            tp=1,
            n_gt=1,
            n_pred=1,
            onset_errors=np.array([-5.0]),
            offset_errors=np.array([-10.0]),
        )
        report = _aggregate_evaluations([r1, r2])
        assert report.boundary_accuracy.n_matches == 2
        assert_allclose(report.boundary_accuracy.onset_errors_ms, [10.0, -5.0])
        assert_allclose(report.boundary_accuracy.offset_errors_ms, [20.0, -10.0])

    def test_empty_boundary_errors(self) -> None:
        """Test aggregation when no matches exist."""
        r1 = _make_recording_eval(
            tp=0,
            n_gt=2,
            n_pred=0,
            onset_errors=np.array([]),
            offset_errors=np.array([]),
        )
        report = _aggregate_evaluations([r1])
        assert report.boundary_accuracy.n_matches == 0
        assert report.boundary_accuracy.onset_errors_ms.size == 0

    def test_total_duration_minutes(self) -> None:
        """Test total duration is converted to minutes."""
        r1 = _make_recording_eval(duration_s=90.0)
        r2 = _make_recording_eval(duration_s=30.0)
        report = _aggregate_evaluations([r1, r2])
        assert_allclose(report.total_duration_minutes, 2.0, atol=1e-6)

    def test_frame_metrics_aggregation(self) -> None:
        """Test frame-level counts are summed."""
        r1 = _make_recording_eval(frame_tp=10, frame_fp=2, frame_fn=3, frame_tn=85)
        r2 = _make_recording_eval(frame_tp=20, frame_fp=4, frame_fn=6, frame_tn=70)
        report = _aggregate_evaluations([r1, r2])
        assert report.frame_metrics.tp == 30
        assert report.frame_metrics.fp == 6
        assert report.frame_metrics.fn == 9
        assert report.frame_metrics.tn == 155


class TestEvaluateRecordings:
    """Tests for evaluate_recordings."""

    def test_empty_recordings_raises(self) -> None:
        """Test that empty recordings list raises ValueError."""
        model = TinySegCNN(n_bands=8, window_frames=250)
        extractor = SNRExtractor(sample_rate=32000)
        decoder = HysteresisDecoder()
        matcher = IoUMatcher()
        with pytest.raises(ValueError, match="No recordings"):
            evaluate_recordings(model, extractor, [], decoder, matcher)

    def test_returns_evaluation_report(self, recordings: list[RecordingInfo]) -> None:
        """Test end-to-end evaluation returns an EvaluationReport."""
        extractor = SNRExtractor(sample_rate=32000)
        model = TinySegCNN(n_bands=extractor.n_features, window_frames=250, base=8)
        decoder = HysteresisDecoder()
        matcher = IoUMatcher()
        report = evaluate_recordings(model, extractor, recordings[:1], decoder, matcher)
        assert isinstance(report, EvaluationReport)
        assert len(report.recordings) == 1

    def test_per_recording_results(self, recordings: list[RecordingInfo]) -> None:
        """Test that per-recording results have expected fields."""
        extractor = SNRExtractor(sample_rate=32000)
        model = TinySegCNN(n_bands=extractor.n_features, window_frames=250, base=8)
        decoder = HysteresisDecoder()
        matcher = IoUMatcher()
        report = evaluate_recordings(model, extractor, recordings[:1], decoder, matcher)
        rec_eval = report.recordings[0]
        assert isinstance(rec_eval, RecordingEvaluation)
        assert len(rec_eval.ground_truth) > 0
        assert rec_eval.event_metrics.n_ground_truth == len(rec_eval.ground_truth)

    def test_multiple_recordings(self, recordings: list[RecordingInfo]) -> None:
        """Test evaluation on multiple recordings."""
        extractor = SNRExtractor(sample_rate=32000)
        model = TinySegCNN(n_bands=extractor.n_features, window_frames=250, base=8)
        decoder = HysteresisDecoder()
        matcher = IoUMatcher()
        n = min(2, len(recordings))
        report = evaluate_recordings(model, extractor, recordings[:n], decoder, matcher)
        assert len(report.recordings) == n
        assert report.total_duration_minutes > 0
