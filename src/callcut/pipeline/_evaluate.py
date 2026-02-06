"""Recording-level evaluation pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from callcut.evaluation import (
    BaseDecoder,
    BaseIntervalMatcher,
    compute_boundary_accuracy,
    compute_event_metrics,
    compute_frame_metrics,
)
from callcut.evaluation._types import (
    BoundaryAccuracy,
    EventMetrics,
    FrameMetrics,
    Interval,
)
from callcut.evaluation._utils import _precision_recall_f1
from callcut.io import intervals_to_frame_labels, load_annotations
from callcut.nn import BaseDetector
from callcut.pipeline._inference import _infer_recording
from callcut.pipeline._types import EvaluationReport, RecordingEvaluation
from callcut.utils._checks import check_type
from callcut.utils.logs import logger

if TYPE_CHECKING:
    from callcut.extractors import BaseExtractor
    from callcut.io import RecordingInfo


def evaluate_recordings(
    model: BaseDetector,
    extractor: BaseExtractor,
    recordings: list[RecordingInfo],
    decoder: BaseDecoder,
    matcher: BaseIntervalMatcher,
    *,
    hop_frames: int | None = None,
    boundary_tolerance_ms: float | None = None,
) -> EvaluationReport:
    """Evaluate a trained model on annotated recordings.

    For each recording: loads audio, extracts features, predicts frame-level
    probabilities, decodes to call intervals, matches against ground truth,
    and computes event-level, frame-level, and boundary metrics.

    Results are aggregated across all recordings.

    Parameters
    ----------
    model : BaseDetector
        Trained model for call detection. Should already be on the desired device.
    extractor : BaseExtractor
        Feature extractor matching the model's expected input.
    recordings : list of RecordingInfo
        Recordings to evaluate. Each must have a valid annotation file.
    decoder : BaseDecoder
        Decoder for converting probabilities to call intervals.
    matcher : BaseIntervalMatcher
        Matcher for pairing predicted and ground truth intervals.
    hop_frames : int | None
        Hop between inference windows in frames. If ``None``, uses the model's default
        (75%% overlap).
    boundary_tolerance_ms : float | None
        If set, discard matched events where either boundary error exceeds this
        tolerance when computing boundary accuracy statistics.

    Returns
    -------
    report : EvaluationReport
        Evaluation results with per-recording details and aggregate metrics.

    Examples
    --------
    >>> from callcut.extractors import SNRExtractor
    >>> from callcut.nn import TinySegCNN
    >>> from callcut.evaluation import HysteresisDecoder, IoUMatcher
    >>> from callcut.io import scan_recordings
    >>> from callcut.pipeline import evaluate_recordings
    >>>
    >>> extractor = SNRExtractor(sample_rate=32000)
    >>> model = TinySegCNN(n_bands=8, window_frames=250)
    >>> decoder = HysteresisDecoder()
    >>> matcher = IoUMatcher()
    >>>
    >>> recordings = scan_recordings(list(Path("data/").glob("*.wav")))
    >>> report = evaluate_recordings(model, extractor, recordings, decoder, matcher)
    >>> print(report)
    """
    check_type(model, (BaseDetector,), "model")
    check_type(decoder, (BaseDecoder,), "decoder")
    check_type(matcher, (BaseIntervalMatcher,), "matcher")
    if len(recordings) == 0:
        raise ValueError("No recordings to evaluate.")

    device = next(model.parameters()).device
    per_recording: list[RecordingEvaluation] = []

    for rec in recordings:
        logger.info("Evaluating %s", rec.audio_path.name)

        # Inference: load audio, extract features, predict, decode
        probabilities, times, predictions = _infer_recording(
            model, extractor, decoder, rec.audio_path, device, hop_frames
        )

        # Load ground truth annotations
        annotations = load_annotations(rec.annotation_path)
        ground_truth = [
            Interval(onset=float(row[0]), offset=float(row[1])) for row in annotations
        ]

        # Compute frame labels and frame metrics
        frame_labels = intervals_to_frame_labels(annotations, times)
        frame_metrics = compute_frame_metrics(probabilities, frame_labels)

        # Match and compute event metrics
        matches = matcher.match(ground_truth, predictions)
        event_metrics = compute_event_metrics(ground_truth, predictions, matches)
        boundary_accuracy = compute_boundary_accuracy(
            ground_truth,
            predictions,
            matches,
            boundary_tolerance_ms=boundary_tolerance_ms,
        )

        per_recording.append(
            RecordingEvaluation(
                recording=rec,
                ground_truth=tuple(ground_truth),
                predictions=tuple(predictions),
                matches=tuple(matches),
                event_metrics=event_metrics,
                boundary_accuracy=boundary_accuracy,
                frame_metrics=frame_metrics,
            )
        )

        logger.debug(
            "  %s: gt=%d, pred=%d, tp=%d, fp=%d, fn=%d, event_f1=%.3f",
            rec.audio_path.name,
            len(ground_truth),
            len(predictions),
            event_metrics.tp,
            event_metrics.fp,
            event_metrics.fn,
            event_metrics.f1,
        )

    report = _aggregate_evaluations(per_recording)

    logger.info(
        "Evaluation complete: %d recordings, event F1=%.3f, frame F1=%.3f, "
        "FP/min=%.3f",
        len(recordings),
        report.event_metrics.f1,
        report.frame_metrics.f1,
        report.fp_per_minute,
    )

    return report


def _aggregate_evaluations(
    evaluations: list[RecordingEvaluation],
) -> EvaluationReport:
    """Aggregate per-recording evaluations into a single report."""
    # Event-level aggregation
    total_tp = sum(r.event_metrics.tp for r in evaluations)
    total_fp = sum(r.event_metrics.fp for r in evaluations)
    total_fn = sum(r.event_metrics.fn for r in evaluations)
    precision, recall, f1 = _precision_recall_f1(total_tp, total_fp, total_fn)
    agg_event_metrics = EventMetrics(
        n_ground_truth=sum(len(r.ground_truth) for r in evaluations),
        n_predicted=sum(len(r.predictions) for r in evaluations),
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )

    # Frame-level aggregation
    frame_tp = sum(r.frame_metrics.tp for r in evaluations)
    frame_fp = sum(r.frame_metrics.fp for r in evaluations)
    frame_fn = sum(r.frame_metrics.fn for r in evaluations)
    frame_tn = sum(r.frame_metrics.tn for r in evaluations)
    frame_precision, frame_recall, frame_f1 = _precision_recall_f1(
        frame_tp, frame_fp, frame_fn
    )
    agg_frame_metrics = FrameMetrics(
        n_frames=sum(r.frame_metrics.n_frames for r in evaluations),
        tp=frame_tp,
        fp=frame_fp,
        fn=frame_fn,
        tn=frame_tn,
        precision=frame_precision,
        recall=frame_recall,
        f1=frame_f1,
    )

    # Boundary accuracy aggregation
    all_onset = [r.boundary_accuracy.onset_errors_ms for r in evaluations]
    all_offset = [r.boundary_accuracy.offset_errors_ms for r in evaluations]
    onset_errors = (
        np.concatenate(all_onset)
        if any(e.size > 0 for e in all_onset)
        else np.array([], dtype=np.float64)
    )
    offset_errors = (
        np.concatenate(all_offset)
        if any(e.size > 0 for e in all_offset)
        else np.array([], dtype=np.float64)
    )
    agg_boundary = BoundaryAccuracy(
        n_matches=int(onset_errors.size),
        onset_errors_ms=onset_errors,
        offset_errors_ms=offset_errors,
    )

    # FP per minute
    total_duration_s = sum(r.recording.duration_s for r in evaluations)
    total_duration_minutes = total_duration_s / 60.0
    fp_per_minute = (
        total_fp / total_duration_minutes if total_duration_minutes > 0 else 0.0
    )

    return EvaluationReport(
        recordings=tuple(evaluations),
        event_metrics=agg_event_metrics,
        boundary_accuracy=agg_boundary,
        frame_metrics=agg_frame_metrics,
        fp_per_minute=fp_per_minute,
        total_duration_minutes=total_duration_minutes,
    )
