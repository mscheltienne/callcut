"""SNR-based feature extraction for audio signals."""

from __future__ import annotations

import torch
from torch import Tensor

from callcut.features._bands import compute_band_edges
from callcut.features._normalize import robust_normalize
from callcut.features._utils import median_filter_1d
from callcut.utils._checks import check_type, ensure_int
from callcut.utils.logs import logger


def compute_snr_features(
    waveform: Tensor,
    sample_rate: int,
    *,
    win_ms: float = 32.0,
    hop_ms: float = 8.0,
    baseline_s: float = 10.0,
    band_low: float = 300.0,
    band_high: float = 10000.0,
    n_bands: int = 8,
    eps: float = 1e-12,
    normalize: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""Compute multi-band SNR features from an audio waveform.

    Extracts signal-to-noise ratio (SNR) features across multiple frequency bands.
    The pipeline computes:

    1. Short-time Fourier transform (STFT)
    2. Power spectrum in logarithmically-spaced frequency bands
    3. Baseline estimation via median filtering
    4. SNR in decibels relative to baseline
    5. Optional robust normalization per band

    All computations run on the input tensor's device, enabling GPU acceleration.

    Parameters
    ----------
    waveform : Tensor
        Audio waveform of shape ``(1, samples)`` or ``(samples,)``. Should be
        mono (single channel). Values should be normalized to ``[-1, 1]``.
    sample_rate : int
        Sample rate of the waveform in Hz.
    win_ms : float
        STFT window length in milliseconds.
    hop_ms : float
        STFT hop length in milliseconds. Determines the time resolution of
        the output features.
    baseline_s : float
        Window length in seconds for baseline estimation. A longer window
        provides a more stable noise floor estimate but may smooth out
        slow signal variations.
    band_low : float
        Lower frequency bound in Hz for the lowest frequency band.
    band_high : float
        Upper frequency bound in Hz for the highest frequency band.
    n_bands : int
        Number of logarithmically-spaced frequency bands.
    eps : float
        Small constant for numerical stability in logarithm and division.
    normalize : bool
        If ``True``, apply robust z-score normalization to each band.

    Returns
    -------
    features : Tensor
        SNR features of shape ``(n_bands, n_frames)``. Each row contains the
        SNR time series for one frequency band. If ``normalize=True``, values
        are approximately zero-centered with unit scale per band.
    times : Tensor
        Time axis of shape ``(n_frames,)`` in seconds, indicating the center
        time of each frame.

    Notes
    -----
    The SNR for each frequency band is computed as:

    .. math::

        \\text{SNR}_{\\text{dB}} = 10 \\cdot \\log_{10}
        \\left( \\frac{E_{\\text{band}} + \\epsilon}
        {E_{\\text{baseline}} + \\epsilon} \\right)

    where :math:`E_{\\text{band}}` is the instantaneous band energy and
    :math:`E_{\\text{baseline}}` is the median-filtered baseline estimate.

    The FFT size is automatically chosen as the smallest power of 2 that is
    at least as large as the window size.

    Examples
    --------
    Extract features from an audio file:

    >>> from callcut.io import load_audio
    >>> waveform, sr = load_audio("recording.wav")
    >>> features, times = compute_snr_features(waveform, sr)
    >>> features.shape
    torch.Size([8, 635])  # 8 bands, 635 time frames

    Customize frequency range and bands:

    >>> features, times = compute_snr_features(
    ...     waveform,
    ...     sr,
    ...     band_low=500.0,
    ...     band_high=8000.0,
    ...     n_bands=4,
    ... )
    >>> features.shape
    torch.Size([4, 635])
    """
    check_type(waveform, (Tensor,), "waveform")
    sample_rate = ensure_int(sample_rate, "sample_rate")
    check_type(win_ms, ("numeric",), "win_ms")
    check_type(hop_ms, ("numeric",), "hop_ms")
    check_type(baseline_s, ("numeric",), "baseline_s")
    check_type(band_low, ("numeric",), "band_low")
    check_type(band_high, ("numeric",), "band_high")
    n_bands = ensure_int(n_bands, "n_bands")
    check_type(eps, ("numeric",), "eps")
    check_type(normalize, (bool,), "normalize")

    if sample_rate <= 0:
        raise ValueError(f"Argument 'sample_rate' must be positive, got {sample_rate}.")
    if win_ms <= 0:
        raise ValueError(f"Argument 'win_ms' must be positive, got {win_ms}.")
    if hop_ms <= 0:
        raise ValueError(f"Argument 'hop_ms' must be positive, got {hop_ms}.")
    if baseline_s <= 0:
        raise ValueError(f"Argument 'baseline_s' must be positive, got {baseline_s}.")

    device = waveform.device
    dtype = waveform.dtype

    # ensure 1D waveform
    if waveform.dim() == 2:
        if waveform.shape[0] != 1:
            raise ValueError(
                f"Argument 'waveform' must be mono (shape (1, samples) or (samples,)), "
                f"got shape {tuple(waveform.shape)}."
            )
        waveform = waveform.squeeze(0)
    elif waveform.dim() != 1:
        raise ValueError(
            f"Argument 'waveform' must be 1D or 2D, got {waveform.dim()}D."
        )

    logger.debug(
        "Computing SNR features: sample_rate=%d, win_ms=%.1f, hop_ms=%.1f, "
        "n_bands=%d, device=%s",
        sample_rate,
        win_ms,
        hop_ms,
        n_bands,
        device,
    )

    # ----------------------------------------------------------------------------------
    # STFT parameters
    # ----------------------------------------------------------------------------------
    win_samples = int(round(win_ms * 1e-3 * sample_rate))
    hop_samples = int(round(hop_ms * 1e-3 * sample_rate))

    # FFT size: smallest power of 2 >= win_samples
    n_fft = 1
    while n_fft < win_samples:
        n_fft <<= 1

    logger.debug(
        "STFT params: win_samples=%d, hop_samples=%d, n_fft=%d",
        win_samples,
        hop_samples,
        n_fft,
    )

    # ----------------------------------------------------------------------------------
    # Compute STFT and power spectrum
    # ----------------------------------------------------------------------------------
    window = torch.hann_window(win_samples, device=device, dtype=dtype)
    S = torch.stft(  # S shape: (n_fft // 2 + 1, n_frames)
        waveform,
        n_fft=n_fft,
        hop_length=hop_samples,
        win_length=win_samples,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )

    # power spectrum
    P = S.abs().pow(2)
    n_frames = P.shape[1]

    logger.debug("Power spectrum shape: %s", tuple(P.shape))

    # ----------------------------------------------------------------------------------
    # Frequency and time axes
    # ----------------------------------------------------------------------------------
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).to(device)
    times = (
        torch.arange(n_frames, device=device, dtype=dtype) * hop_samples / sample_rate
    )

    # ----------------------------------------------------------------------------------
    # Baseline filter size
    # ----------------------------------------------------------------------------------
    baseline_frames = max(3, int(round(baseline_s / (hop_ms * 1e-3))))
    if baseline_frames % 2 == 0:
        baseline_frames += 1

    logger.debug("Baseline filter kernel size: %d frames", baseline_frames)

    # ----------------------------------------------------------------------------------
    # Compute SNR for each frequency band
    # ----------------------------------------------------------------------------------
    bands = compute_band_edges(band_low, band_high, n_bands)

    snr_list = []
    for i in range(n_bands):
        lo, hi = bands[i]

        # mask frequencies in this band
        mask = (freqs >= lo) & (freqs < hi)

        # sum power across frequencies in band
        band_energy = P[mask, :].sum(dim=0)

        # baseline estimation via median filter
        baseline = median_filter_1d(band_energy, baseline_frames)

        # SNR in decibels
        snr_db = 10.0 * torch.log10((band_energy + eps) / (baseline + eps))
        snr_list.append(snr_db)

    features = torch.stack(snr_list, dim=0)  # (n_bands, n_frames)

    logger.debug("SNR features shape: %s", tuple(features.shape))

    # ----------------------------------------------------------------------------------
    # Optional normalization
    # ----------------------------------------------------------------------------------
    if normalize:
        features = robust_normalize(features, eps=eps)
        logger.debug("Applied robust normalization")

    return features, times
