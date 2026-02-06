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
from callcut.io import intervals_to_frame_labels, load_annotations, load_audio
from callcut.nn import BaseDetector
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

    per_recording: list[RecordingEvaluation] = []

    # Aggregation accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_frame_tp = 0
    total_frame_fp = 0
    total_frame_fn = 0
    total_frame_tn = 0
    total_n_frames = 0
    all_onset_errors: list[np.ndarray] = []
    all_offset_errors: list[np.ndarray] = []
    total_duration_s = 0.0

    for rec in recordings:
        logger.info("Evaluating %s", rec.audio_path.name)

        # Load audio and extract features
        waveform, _ = load_audio(
            rec.audio_path, sample_rate=extractor.sample_rate, mono=True
        )
        features, times = extractor.extract(waveform)

        # Move features to model's device
        device = next(model.parameters()).device
        features = features.to(device)

        # Predict frame probabilities
        probabilities = model.predict(features, hop_frames=hop_frames)

        # Decode to intervals
        predictions = decoder.decode(times, probabilities.cpu())

        # Load ground truth annotations
        annotations = load_annotations(rec.annotation_path)
        ground_truth = [
            Interval(onset=float(row[0]), offset=float(row[1])) for row in annotations
        ]

        # Compute frame labels and frame metrics
        frame_labels = intervals_to_frame_labels(annotations, times)
        frame_metrics = compute_frame_metrics(probabilities.cpu(), frame_labels)

        # Match and compute event metrics
        matches = matcher.match(ground_truth, predictions)
        event_metrics = compute_event_metrics(ground_truth, predictions, matches)
        boundary_accuracy = compute_boundary_accuracy(
            ground_truth,
            predictions,
            matches,
            boundary_tolerance_ms=boundary_tolerance_ms,
        )

        # Store per-recording result
        rec_eval = RecordingEvaluation(
            recording=rec,
            ground_truth=tuple(ground_truth),
            predictions=tuple(predictions),
            matches=tuple(matches),
            event_metrics=event_metrics,
            boundary_accuracy=boundary_accuracy,
            frame_metrics=frame_metrics,
        )
        per_recording.append(rec_eval)

        # Accumulate for aggregation
        total_tp += event_metrics.tp
        total_fp += event_metrics.fp
        total_fn += event_metrics.fn
        total_frame_tp += frame_metrics.tp
        total_frame_fp += frame_metrics.fp
        total_frame_fn += frame_metrics.fn
        total_frame_tn += frame_metrics.tn
        total_n_frames += frame_metrics.n_frames
        all_onset_errors.append(boundary_accuracy.onset_errors_ms)
        all_offset_errors.append(boundary_accuracy.offset_errors_ms)
        total_duration_s += rec.duration_s

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

    # Aggregate event metrics
    eps = 1e-12
    agg_precision = (
        total_tp / (total_tp + total_fp + eps) if (total_tp + total_fp) > 0 else 0.0
    )
    agg_recall = (
        total_tp / (total_tp + total_fn + eps) if (total_tp + total_fn) > 0 else 0.0
    )
    agg_f1 = (
        2 * agg_precision * agg_recall / (agg_precision + agg_recall + eps)
        if (agg_precision + agg_recall) > 0
        else 0.0
    )
    total_gt = sum(len(r.ground_truth) for r in per_recording)
    total_pred = sum(len(r.predictions) for r in per_recording)
    agg_event_metrics = EventMetrics(
        n_ground_truth=total_gt,
        n_predicted=total_pred,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        precision=agg_precision,
        recall=agg_recall,
        f1=agg_f1,
    )

    # Aggregate frame metrics
    frame_precision = (
        total_frame_tp / (total_frame_tp + total_frame_fp + eps)
        if (total_frame_tp + total_frame_fp) > 0
        else 0.0
    )
    frame_recall = (
        total_frame_tp / (total_frame_tp + total_frame_fn + eps)
        if (total_frame_tp + total_frame_fn) > 0
        else 0.0
    )
    frame_f1 = (
        2 * frame_precision * frame_recall / (frame_precision + frame_recall + eps)
        if (frame_precision + frame_recall) > 0
        else 0.0
    )
    agg_frame_metrics = FrameMetrics(
        n_frames=total_n_frames,
        tp=total_frame_tp,
        fp=total_frame_fp,
        fn=total_frame_fn,
        tn=total_frame_tn,
        precision=frame_precision,
        recall=frame_recall,
        f1=frame_f1,
    )

    # Aggregate boundary accuracy
    onset_errors = (
        np.concatenate(all_onset_errors)
        if any(e.size > 0 for e in all_onset_errors)
        else np.array([], dtype=np.float64)
    )
    offset_errors = (
        np.concatenate(all_offset_errors)
        if any(e.size > 0 for e in all_offset_errors)
        else np.array([], dtype=np.float64)
    )
    agg_boundary = BoundaryAccuracy(
        n_matches=int(onset_errors.size),
        onset_errors_ms=onset_errors,
        offset_errors_ms=offset_errors,
    )

    # FP per minute
    total_duration_minutes = total_duration_s / 60.0
    fp_per_minute = (
        total_fp / total_duration_minutes if total_duration_minutes > 0 else 0.0
    )

    report = EvaluationReport(
        recordings=tuple(per_recording),
        event_metrics=agg_event_metrics,
        boundary_accuracy=agg_boundary,
        frame_metrics=agg_frame_metrics,
        fp_per_minute=fp_per_minute,
        total_duration_minutes=total_duration_minutes,
    )

    logger.info(
        "Evaluation complete: %d recordings, event F1=%.3f, frame F1=%.3f, FP/min=%.3f",
        len(recordings),
        agg_f1,
        frame_f1,
        fp_per_minute,
    )

    return report
