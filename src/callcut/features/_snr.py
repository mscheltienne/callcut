"""SNR-based feature extraction for audio signals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from callcut.features._base import BaseExtractor
from callcut.utils._checks import check_type, ensure_int
from callcut.utils.logs import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SNRExtractor(BaseExtractor):
    r"""Multi-band SNR feature extractor.

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
    sample_rate : int
        Expected sample rate in Hz.
    hop_ms : float
        STFT hop length in milliseconds. Determines time resolution.
    n_bands : int
        Number of logarithmically-spaced frequency bands.
    win_ms : float
        STFT window length in milliseconds.
    band_low : float
        Lower frequency bound in Hz.
    band_high : float
        Upper frequency bound in Hz.
    baseline_s : float
        Baseline estimation window in seconds.
    normalize : bool
        If ``True``, apply robust z-score normalization per band.
    eps : float
        Small constant for numerical stability.

    Notes
    -----
    The SNR for each frequency band is computed as:

    .. math::

        \text{SNR}_{\text{dB}} = 10 \cdot \log_{10}
        \left( \frac{E_{\text{band}} + \epsilon}
        {E_{\text{baseline}} + \epsilon} \right)

    where :math:`E_{\text{band}}` is the instantaneous band energy and
    :math:`E_{\text{baseline}}` is the median-filtered baseline estimate.

    The FFT size is automatically chosen as the smallest power of 2 that is
    at least as large as the window size.

    Examples
    --------
    Extract features from an audio file:

    >>> extractor = SNRExtractor(sample_rate=32000, hop_ms=8.0, n_bands=8)
    >>> waveform, sr = load_audio("recording.wav", sample_rate=32000)
    >>> features, times = extractor(waveform)
    >>> features.shape
    torch.Size([8, 635])

    Customize frequency range and bands:

    >>> extractor = SNRExtractor(
    ...     sample_rate=32000,
    ...     band_low=500.0,
    ...     band_high=8000.0,
    ...     n_bands=4,
    ... )
    >>> features, times = extractor(waveform)
    >>> features.shape
    torch.Size([4, 635])
    """

    def __init__(
        self,
        sample_rate: int,
        hop_ms: float = 8.0,
        n_bands: int = 8,
        *,
        win_ms: float = 32.0,
        band_low: float = 300.0,
        band_high: float = 10000.0,
        baseline_s: float = 10.0,
        normalize: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(sample_rate)

        # Validate and store parameters
        check_type(hop_ms, ("numeric",), "hop_ms")
        check_type(win_ms, ("numeric",), "win_ms")
        check_type(band_low, ("numeric",), "band_low")
        check_type(band_high, ("numeric",), "band_high")
        check_type(baseline_s, ("numeric",), "baseline_s")
        check_type(normalize, (bool,), "normalize")
        check_type(eps, ("numeric",), "eps")
        n_bands = ensure_int(n_bands, "n_bands")

        if hop_ms <= 0:
            raise ValueError(f"Argument 'hop_ms' must be positive, got {hop_ms}.")
        if win_ms <= 0:
            raise ValueError(f"Argument 'win_ms' must be positive, got {win_ms}.")
        if n_bands <= 0:
            raise ValueError(f"Argument 'n_bands' must be positive, got {n_bands}.")
        if baseline_s <= 0:
            raise ValueError(
                f"Argument 'baseline_s' must be positive, got {baseline_s}."
            )
        if band_low <= 0:
            raise ValueError(f"Argument 'band_low' must be positive, got {band_low}.")
        if band_high <= band_low:
            raise ValueError(
                f"Argument 'band_high' ({band_high}) must be greater than "
                f"'band_low' ({band_low})."
            )
        nyquist = sample_rate / 2
        if band_high >= nyquist:
            raise ValueError(
                f"Argument 'band_high' ({band_high} Hz) must be below Nyquist "
                f"frequency ({nyquist} Hz)."
            )

        self._hop_ms = float(hop_ms)
        self._n_bands = n_bands
        self._win_ms = float(win_ms)
        self._band_low = float(band_low)
        self._band_high = float(band_high)
        self._baseline_s = float(baseline_s)
        self._normalize = normalize
        self._eps = float(eps)

    @property
    def n_features(self) -> int:
        """Number of frequency bands.

        :type: :class:`int`
        """
        return self._n_bands

    @property
    def hop_ms(self) -> float:
        """Hop length in milliseconds.

        :type: :class:`float`
        """
        return self._hop_ms

    @property
    def win_ms(self) -> float:
        """Window length in milliseconds.

        :type: :class:`float`
        """
        return self._win_ms

    @property
    def n_bands(self) -> int:
        """Number of frequency bands (alias for :attr:`n_features`).

        :type: :class:`int`
        """
        return self._n_bands

    @property
    def band_low(self) -> float:
        """Lower frequency bound in Hz.

        :type: :class:`float`
        """
        return self._band_low

    @property
    def band_high(self) -> float:
        """Upper frequency bound in Hz.

        :type: :class:`float`
        """
        return self._band_high

    @property
    def baseline_s(self) -> float:
        """Baseline estimation window in seconds.

        :type: :class:`float`
        """
        return self._baseline_s

    @property
    def normalize(self) -> bool:
        """Whether normalization is applied.

        :type: :class:`bool`
        """
        return self._normalize

    def extract(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        """Extract multi-band SNR features from a waveform.

        Parameters
        ----------
        waveform : Tensor
            Audio waveform of shape ``(1, samples)`` or ``(samples,)``. Should be
            mono (single channel). Values should be normalized to ``[-1, 1]``.

        Returns
        -------
        features : Tensor
            SNR features of shape ``(n_bands, n_frames)``. Each row contains the
            SNR time series for one frequency band. If ``normalize=True``, values
            are approximately zero-centered with unit scale per band.
        times : Tensor
            Time axis of shape ``(n_frames,)`` in seconds, indicating the center
            time of each frame.
        """
        check_type(waveform, (Tensor,), "waveform")

        device = waveform.device
        dtype = waveform.dtype

        # ensure 1D waveform
        if waveform.dim() == 2:
            if waveform.shape[0] != 1:
                raise ValueError(
                    f"Argument 'waveform' must be mono (shape (1, samples) or "
                    f"(samples,)), got shape {tuple(waveform.shape)}."
                )
            waveform = waveform.squeeze(0)
        elif waveform.dim() != 1:
            raise ValueError(
                f"Argument 'waveform' must be 1D or 2D, got {waveform.dim()}D."
            )

        logger.debug(
            "Computing SNR features: sample_rate=%d, win_ms=%.1f, hop_ms=%.1f, "
            "n_bands=%d, device=%s",
            self._sample_rate,
            self._win_ms,
            self._hop_ms,
            self._n_bands,
            device,
        )

        # ──────────────────────────────────────────────────────────────────────
        # STFT parameters
        # ──────────────────────────────────────────────────────────────────────
        win_samples = int(round(self._win_ms * 1e-3 * self._sample_rate))
        hop_samples = int(round(self._hop_ms * 1e-3 * self._sample_rate))

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

        # ──────────────────────────────────────────────────────────────────────
        # Compute STFT and power spectrum
        # ──────────────────────────────────────────────────────────────────────
        window = torch.hann_window(win_samples, device=device, dtype=dtype)
        S = torch.stft(
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

        # ──────────────────────────────────────────────────────────────────────
        # Frequency and time axes
        # ──────────────────────────────────────────────────────────────────────
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / self._sample_rate).to(device)
        times = (
            torch.arange(n_frames, device=device, dtype=dtype)
            * hop_samples
            / self._sample_rate
        )

        # ──────────────────────────────────────────────────────────────────────
        # Baseline filter size
        # ──────────────────────────────────────────────────────────────────────
        baseline_frames = max(3, int(round(self._baseline_s / (self._hop_ms * 1e-3))))
        if baseline_frames % 2 == 0:
            baseline_frames += 1

        logger.debug("Baseline filter kernel size: %d frames", baseline_frames)

        # ──────────────────────────────────────────────────────────────────────
        # Compute SNR for each frequency band
        # ──────────────────────────────────────────────────────────────────────
        bands = compute_band_edges(self._band_low, self._band_high, self._n_bands)

        snr_list = []
        for i in range(self._n_bands):
            lo, hi = bands[i]

            # mask frequencies in this band
            mask = (freqs >= lo) & (freqs < hi)

            # sum power across frequencies in band
            band_energy = P[mask, :].sum(dim=0)

            # baseline estimation via median filter
            baseline = median_filter_1d(band_energy, baseline_frames)

            # SNR in decibels
            snr_db = 10.0 * torch.log10(
                (band_energy + self._eps) / (baseline + self._eps)
            )
            snr_list.append(snr_db)

        features = torch.stack(snr_list, dim=0)  # (n_bands, n_frames)

        logger.debug("SNR features shape: %s", tuple(features.shape))

        # ──────────────────────────────────────────────────────────────────────
        # Optional normalization
        # ──────────────────────────────────────────────────────────────────────
        if self._normalize:
            features = robust_normalize(features, eps=self._eps)
            logger.debug("Applied robust normalization")

        return features, times

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sample_rate={self.sample_rate}, "
            f"hop_ms={self.hop_ms}, "
            f"n_bands={self.n_bands}, "
            f"band_low={self.band_low}, "
            f"band_high={self.band_high})"
        )


def robust_normalize(features: Tensor, *, eps: float = 1e-12) -> Tensor:
    r"""Apply robust z-score normalization per band.

    Normalizes each band (row) independently using the median and median absolute
    deviation (MAD), which are robust to outliers. This is preferred over mean/std
    normalization for audio features that may contain transient events.

    Parameters
    ----------
    features : Tensor
        Input features of shape ``(n_bands, time)``.
    eps : float
        Small constant added to MAD for numerical stability.

    Returns
    -------
    normalized : Tensor
        Normalized features of the same shape. Each band has approximately zero median
        and unit scale (when measured by MAD).

    Notes
    -----
    The normalization formula is:

    .. math::

        z = \\frac{x - \\text{median}(x)}{1.4826 \\cdot \\text{MAD}(x)}

    where MAD is the median absolute deviation. The constant 1.4826 is the consistency
    constant that makes MAD asymptotically equivalent to standard deviation for normally
    distributed data.

    Examples
    --------
    >>> features = torch.randn(8, 100) * 10 + 5  # shifted and scaled
    >>> normalized = robust_normalize(features)
    >>> normalized.median(dim=1).values  # approximately zero
    tensor([...])
    """
    check_type(features, (Tensor,), "features")
    check_type(eps, ("numeric",), "eps")

    if features.dim() != 2:
        raise ValueError(
            f"Argument 'features' must be 2D (n_bands, time), got {features.dim()}D."
        )

    # median per band: (n_bands, 1)
    med = features.median(dim=1, keepdim=True).values

    # MAD (median absolute deviation) per band
    mad = (features - med).abs().median(dim=1, keepdim=True).values + eps

    # robust z-score with consistency constant for normal distribution
    return (features - med) / (1.4826 * mad)


def compute_band_edges(
    low_hz: float, high_hz: float, n_bands: int
) -> NDArray[np.floating]:
    """Compute logarithmically-spaced frequency band edges.

    Generates frequency band boundaries that are evenly spaced on a logarithmic
    scale, which better matches human auditory perception and is commonly used
    in audio analysis.

    Parameters
    ----------
    low_hz : float
        Lower frequency bound in Hz. Must be positive.
    high_hz : float
        Upper frequency bound in Hz. Must be greater than ``low_hz``.
    n_bands : int
        Number of frequency bands to create. Must be at least 1.

    Returns
    -------
    bands : array of shape ``(n_bands, 2)``
        Array where each row contains ``[low_edge, high_edge]`` for a frequency
        band, in Hz. Bands are contiguous (each band's high edge equals the next
        band's low edge).

    Examples
    --------
    Create 4 bands spanning 300 Hz to 10 kHz:

    >>> bands = compute_band_edges(300, 10000, 4)
    >>> bands.shape
    (4, 2)
    >>> bands[0]  # first band
    array([ 300.        ,  547.72255751])

    The bands are logarithmically spaced:

    >>> ratios = bands[:, 1] / bands[:, 0]
    >>> np.allclose(ratios, ratios[0])  # all ratios are equal
    True
    """
    check_type(low_hz, ("numeric",), "low_hz")
    check_type(high_hz, ("numeric",), "high_hz")
    n_bands = ensure_int(n_bands, "n_bands")

    if low_hz <= 0:
        raise ValueError(f"Argument 'low_hz' must be positive, got {low_hz}.")
    if high_hz <= low_hz:
        raise ValueError(
            f"Argument 'high_hz' must be greater than 'low_hz'. "
            f"Got low_hz={low_hz}, high_hz={high_hz}."
        )
    if n_bands < 1:
        raise ValueError(f"Argument 'n_bands' must be at least 1, got {n_bands}.")

    # compute log-spaced edges (n_bands + 1 edges for n_bands bands)
    edges = np.geomspace(low_hz, high_hz, n_bands + 1)

    # create (n_bands, 2) array with [low, high] per band
    bands = np.column_stack([edges[:-1], edges[1:]])

    return bands


def median_filter_1d(x: Tensor, kernel_size: int) -> Tensor:
    """Apply 1D median filter with zero padding.

    This implementation uses zero (constant) padding at boundaries to match the
    behavior of :func:`scipy.signal.medfilt`.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(T,)`` or ``(B, T)``.
    kernel_size : int
        Size of the median filter kernel. Must be a positive odd integer.
        If even, it will be incremented by ``1``.

    Returns
    -------
    filtered : Tensor
        Filtered tensor of the same shape as input.
    """
    # ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    # handle 1D input
    squeeze = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True

    pad = kernel_size // 2

    # zero padding to match scipy.signal.medfilt behavior
    x_padded = torch.nn.functional.pad(x, (pad, pad), mode="constant", value=0)

    # unfold to get sliding windows: (B, T, kernel_size)
    windows = x_padded.unfold(dimension=1, size=kernel_size, step=1)

    # median along last dimension
    result = windows.median(dim=-1).values

    if squeeze:
        result = result.squeeze(0)

    return result
