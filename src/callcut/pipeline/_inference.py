"""Shared inference logic for pipeline functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from callcut.evaluation import BaseDecoder
from callcut.evaluation._types import Interval
from callcut.io import load_audio
from callcut.nn import BaseDetector

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from torch import Tensor

    from callcut.extractors import BaseExtractor


def _infer_recording(
    model: BaseDetector,
    extractor: BaseExtractor,
    decoder: BaseDecoder,
    audio_path: Path,
    device: torch.device,
    hop_frames: int | None,
) -> tuple[Tensor, Tensor, list[Interval]]:
    """Load audio, extract features, predict, and decode for a single recording.

    Parameters
    ----------
    model : BaseDetector
        Trained model (already on ``device``).
    extractor : BaseExtractor
        Feature extractor.
    decoder : BaseDecoder
        Probability-to-interval decoder.
    audio_path : Path
        Path to audio file.
    device : torch.device
        Device the model is on.
    hop_frames : int | None
        Hop between inference windows, or ``None`` for default.

    Returns
    -------
    probabilities : Tensor
        Per-frame probabilities of shape ``(n_frames,)`` on CPU.
    times : Tensor
        Frame center times of shape ``(n_frames,)``.
    intervals : list of Interval
        Decoded call intervals.
    """
    waveform, _ = load_audio(
        audio_path, sample_rate=extractor.sample_rate, mono=True
    )
    features, times = extractor.extract(waveform)
    features = features.to(device)

    probabilities = model.predict(features, hop_frames=hop_frames)
    probabilities = probabilities.cpu()
    intervals = decoder.decode(times, probabilities)

    return probabilities, times, intervals
