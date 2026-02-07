from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from callcut.utils._checks import check_type, ensure_int

if TYPE_CHECKING:
    from torch import Tensor


class BaseExtractor(ABC):
    """Abstract base class for feature extractors.

    All feature extractors must implement:

    - :meth:`extract`: Extract features from a waveform
    - :attr:`n_features`: Number of output feature channels
    - :attr:`hop_ms`: Hop length in milliseconds (determines frame rate)

    The base class provides utility methods for time-frame conversion.

    Parameters
    ----------
    sample_rate : int
        Expected sample rate in Hz. Waveforms should be resampled to this
        rate before extraction.

    Examples
    --------
    Create a custom extractor by subclassing :class:`~callcut.extractors.BaseExtractor`:

    >>> class MyExtractor(BaseExtractor):
    ...     def __init__(self, sample_rate: int, hop_ms: float = 10.0):
    ...         super().__init__(sample_rate)
    ...         self._hop_ms = hop_ms
    ...
    ...     @property
    ...     def n_features(self) -> int:
    ...         return 64
    ...
    ...     @property
    ...     def hop_ms(self) -> float:
    ...         return self._hop_ms
    ...
    ...     def extract(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
    ...         # Implementation here
    ...         ...
    """

    def __init__(self, sample_rate: int) -> None:
        sample_rate = ensure_int(sample_rate, "sample_rate")
        if sample_rate <= 0:
            raise ValueError(
                f"Argument 'sample_rate' must be positive, got {sample_rate}."
            )
        self._sample_rate = sample_rate

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of output feature channels.

        For SNR features, this is the number of frequency bands.

        :type: :class:`int`
        """

    @property
    @abstractmethod
    def hop_ms(self) -> float:
        """Hop length in milliseconds.

        Determines the time resolution of extracted features.

        :type: :class:`float`
        """

    @property
    def sample_rate(self) -> int:
        """Expected sample rate in Hz.

        :type: :class:`int`
        """
        return self._sample_rate

    @property
    def hop_s(self) -> float:
        """Hop length in seconds.

        :type: :class:`float`
        """
        return self.hop_ms / 1000.0

    @property
    def frame_rate(self) -> float:
        """Frame rate in Hz (frames per second).

        :type: :class:`float`
        """
        return 1000.0 / self.hop_ms

    @abstractmethod
    def _save_config(self) -> dict:
        """Return constructor kwargs for serialization.

        Subclasses must implement this to enable saving and loading of pipeline
        configurations. The returned dictionary should contain all constructor
        arguments needed to recreate the extractor.

        Returns
        -------
        config : dict
            Dictionary of constructor keyword arguments.
        """

    @abstractmethod
    def __hash__(self) -> int:
        """Return hash based on extractor configuration.

        Subclasses must implement this to enable caching of extracted features.
        The hash should be based on all parameters that affect the output.
        """

    def __eq__(self, other: object) -> bool:
        """Check equality based on extractor configuration."""
        if not isinstance(other, BaseExtractor):
            return False
        if type(self) is not type(other):
            return False
        return self._save_config() == other._save_config()

    @abstractmethod
    def extract(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        """Extract features from a waveform.

        Parameters
        ----------
        waveform : Tensor
            Audio waveform of shape ``(1, samples)`` or ``(samples,)``.
            Should be mono and sampled at :attr:`sample_rate`.

        Returns
        -------
        features : Tensor
            Extracted features of shape ``(n_features, n_frames)``.
        times : Tensor
            Time axis of shape ``(n_frames,)`` in seconds.
        """

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert duration in seconds to number of frames.

        Parameters
        ----------
        seconds : float
            Duration in seconds.

        Returns
        -------
        frames : int
            Number of frames (rounded to nearest integer).

        Examples
        --------
        >>> extractor = SNRExtractor(sample_rate=32000, hop_ms=8.0)
        >>> extractor.seconds_to_frames(2.0)
        250
        """
        check_type(seconds, ("numeric",), "seconds")
        return int(round(seconds / self.hop_s))

    def frames_to_seconds(self, frames: int) -> float:
        """Convert number of frames to duration in seconds.

        Parameters
        ----------
        frames : int
            Number of frames.

        Returns
        -------
        seconds : float
            Duration in seconds.

        Examples
        --------
        >>> extractor = SNRExtractor(sample_rate=32000, hop_ms=8.0)
        >>> extractor.frames_to_seconds(250)
        2.0
        """
        frames = ensure_int(frames, "frames")
        return frames * self.hop_s

    def __call__(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        """Extract features (alias for :meth:`extract`).

        Parameters
        ----------
        waveform : Tensor
            Audio waveform of shape ``(1, samples)`` or ``(samples,)``.

        Returns
        -------
        features : Tensor
            Extracted features of shape ``(n_features, n_frames)``.
        times : Tensor
            Time axis of shape ``(n_frames,)`` in seconds.
        """
        return self.extract(waveform)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sample_rate={self.sample_rate}, "
            f"hop_ms={self.hop_ms}, "
            f"n_features={self.n_features})"
        )
