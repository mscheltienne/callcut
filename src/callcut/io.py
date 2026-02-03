"""I/O utilities for audio files and annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import Tensor

from callcut.utils._checks import check_type, ensure_device, ensure_int, ensure_path
from callcut.utils.logs import logger, warn

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def load_audio(
    fname: str | Path,
    *,
    sample_rate: int | None = None,
    mono: bool = True,
    device: str | torch.device | None = None,
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
    device : str | torch.device | None
        Device to place the loaded tensor on (e.g., ``"cpu"``, ``"cuda:0"``, ``"mps"``).
        If ``None``, uses the default torch device.

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

    Load directly to GPU:

    >>> waveform, sr = load_audio("recording.wav", device="cuda:0")
    >>> waveform.device
    device(type='cuda', index=0)
    """
    fname = ensure_path(fname, must_exist=True)
    check_type(sample_rate, ("int-like", None), "sample_rate")
    if sample_rate is not None:
        sample_rate = ensure_int(sample_rate, "sample_rate")
    check_type(mono, (bool,), "mono")
    device = ensure_device(device)
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

    # move to device
    waveform = waveform.to(device)
    logger.debug("Moved waveform to device: %s", device)

    return waveform, original_sr


def load_annotations(fname: str | Path) -> NDArray[np.floating]:
    """Load call annotations from a CSV file.

    The CSV file must contain columns ``start_seconds`` and ``stop_seconds``.
    Despite the column names, values in the CSV are expected to be in milliseconds
    and are converted to seconds.

    Parameters
    ----------
    fname : str | Path
        Path to the CSV annotation file.

    Returns
    -------
    intervals : NDArray of shape (n_intervals, 2)
        Array of ``(start, stop)`` times in seconds. Each row represents one
        annotated call interval. Intervals are sorted by start time.

    Examples
    --------
    >>> intervals = load_annotations("recording_annotations.csv")
    >>> intervals.shape
    (42, 2)
    >>> intervals[0]  # first interval in seconds
    array([0.1234, 0.4562])
    """
    fname = ensure_path(fname, must_exist=True)
    logger.debug("Loading annotations from: %s", fname)

    df = pd.read_csv(fname)

    # check required columns
    required = {"start_seconds", "stop_seconds"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV file must contain 'start_seconds' and 'stop_seconds' columns. "
            f"Found columns: {list(df.columns)}"
        )

    # drop rows with missing values and invalid intervals
    df = df.dropna(subset=["start_seconds", "stop_seconds"])
    df = df[df["stop_seconds"] > df["start_seconds"]]

    if len(df) == 0:
        warn(f"No valid intervals found in {fname}", ignore_namespaces=("callcut",))
        return np.empty((0, 2), dtype=np.float64)

    # extract intervals and convert from ms to seconds
    intervals = df[["start_seconds", "stop_seconds"]].values.astype(np.float64)
    intervals = intervals / 1000.0

    # sort by start time
    intervals = intervals[np.argsort(intervals[:, 0])]

    logger.debug("Loaded %d intervals from %s", len(intervals), fname)
    return intervals
