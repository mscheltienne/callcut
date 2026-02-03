"""I/O utilities for audio files and annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torchaudio
from torch import Tensor

from callcut.utils._checks import check_type, ensure_int, ensure_path
from callcut.utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path


def load_audio(
    fname: str | Path,
    *,
    sample_rate: int | None = None,
    mono: bool = True,
) -> tuple[Tensor, int]:
    """Load an audio file.

    Supports common audio formats (wav, mp3, flac, ogg, etc.) and video files
    with audio streams (mp4, etc.) via torchcodec/FFmpeg.

    Parameters
    ----------
    fname : str | Path
        Path to the audio file.
    sample_rate : int | None
        Target sample rate in Hz. If ``None``, the original sample rate is preserved.
        If specified, the audio is resampled to this rate.
    mono : bool
        If ``True``, convert multi-channel audio to mono by averaging channels.

    Returns
    -------
    waveform : Tensor
        Audio waveform of shape ``(channels, samples)`` or ``(1, samples)`` if
        ``mono=True``. Values are normalized to ``[-1, 1]``.
    sample_rate : int
        Sample rate of the returned waveform in Hz.

    Examples
    --------
    Load an audio file at its original sample rate:

    >>> waveform, sr = load_audio("recording.wav")
    >>> waveform.shape
    torch.Size([1, 32000])

    Load and resample to 16 kHz:

    >>> waveform, sr = load_audio("recording.wav", sample_rate=16000)
    >>> sr
    16000
    """
    fname = ensure_path(fname, must_exist=True)
    check_type(sample_rate, ("int-like", None), "sample_rate")
    if sample_rate is not None:
        sample_rate = ensure_int(sample_rate, "sample_rate")
    check_type(mono, (bool,), "mono")
    if sample_rate is not None and sample_rate <= 0:
        raise ValueError(
            f"Argument 'sample_rate' must be a positive integer, got {sample_rate}."
        )

    logger.debug("Loading audio file: %s", fname)
    waveform, original_sr = torchaudio.load(fname)
    logger.debug(
        "Loaded waveform: shape=%s, sample_rate=%d, duration=%.2fs",
        waveform.shape,
        original_sr,
        waveform.shape[1] / original_sr,
    )

    # resample if needed
    if sample_rate is not None and sample_rate != original_sr:
        logger.debug("Resampling from %d Hz to %d Hz", original_sr, sample_rate)
        resampler = torchaudio.transforms.Resample(original_sr, sample_rate)
        waveform = resampler(waveform)
        original_sr = sample_rate

    # convert to mono if needed
    if mono and waveform.shape[0] > 1:
        logger.debug("Converting %d channels to mono", waveform.shape[0])
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, original_sr
