"""Prediction pipeline for running inference on recordings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from callcut.evaluation import BaseDecoder
from callcut.nn import BaseDetector
from callcut.pipeline._inference import _infer_recording
from callcut.pipeline._types import RecordingPrediction
from callcut.utils._checks import check_type, ensure_path
from callcut.utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path

    from callcut.extractors import BaseExtractor


def predict_recordings(
    model: BaseDetector,
    extractor: BaseExtractor,
    audio_paths: list[Path | str],
    decoder: BaseDecoder,
    *,
    hop_frames: int | None = None,
) -> list[RecordingPrediction]:
    """Run inference on recordings and decode to call intervals.

    For each recording: loads audio, extracts features, predicts frame-level
    probabilities, and decodes to call intervals. No ground truth annotations
    are needed.

    Parameters
    ----------
    model : BaseDetector
        Trained model for call detection. Should already be on the desired device.
    extractor : BaseExtractor
        Feature extractor matching the model's expected input.
    audio_paths : list of Path | str
        Paths to audio files to process.
    decoder : BaseDecoder
        Decoder for converting probabilities to call intervals.
    hop_frames : int | None
        Hop between inference windows in frames. If ``None``, uses the model's default
        (75%% overlap).

    Returns
    -------
    predictions : list of RecordingPrediction
        Predicted call intervals for each recording.

    Examples
    --------
    >>> from callcut.pipeline import load_pipeline, predict_recordings
    >>>
    >>> model, extractor, decoder = load_pipeline("pipeline.pt")
    >>> predictions = predict_recordings(model, extractor, audio_files, decoder)
    >>> for pred in predictions:
    ...     print(f"{pred.audio_path.name}: {len(pred.intervals)} calls")
    """
    check_type(model, (BaseDetector,), "model")
    check_type(decoder, (BaseDecoder,), "decoder")
    if len(audio_paths) == 0:
        return []

    results: list[RecordingPrediction] = []
    device = next(model.parameters()).device

    for audio_path in audio_paths:
        audio_path = ensure_path(audio_path, must_exist=True)
        logger.info("Predicting on %s", audio_path.name)

        _, times, intervals = _infer_recording(
            model, extractor, decoder, audio_path, device, hop_frames
        )

        duration_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0

        results.append(
            RecordingPrediction(
                audio_path=audio_path,
                intervals=tuple(intervals),
                duration_s=duration_s,
            )
        )

        logger.debug(
            "  %s: %d intervals detected (%.1fs)",
            audio_path.name,
            len(intervals),
            duration_s,
        )

    logger.info(
        "Prediction complete: %d recordings, %d total intervals",
        len(results),
        sum(len(r.intervals) for r in results),
    )

    return results
