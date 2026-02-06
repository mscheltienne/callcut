"""Built-in model architectures for call detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from callcut.utils._checks import check_type, ensure_int

if TYPE_CHECKING:
    from torch import Tensor


class BaseDetector(ABC, nn.Module):
    """Abstract base class for call detection models.

    Subclasses must implement:

    - ``forward``: Process input features and return logits.
    - ``receptive_field``: Property returning the receptive field in frames.
    - ``_save_config``: Return additional constructor kwargs for serialization.

    Models accept input of shape ``(batch, n_bands, time)`` and return logits of shape
    ``(batch, time)``.

    Parameters
    ----------
    n_bands : int
        Number of input frequency bands.
    window_frames : int
        Number of frames per input window. This determines the temporal context the
        model sees during training and inference. The corresponding duration in seconds
        depends on the feature extractor's hop size:
        ``window_duration_s = window_frames * hop_ms / 1000``.

    Notes
    -----
    The **receptive field** is the number of input frames that influence a single output
    prediction. For a CNN, it is typically the sum of ``(kernel_size - 1)`` across all
    convolutional layers. This determines how much temporal context the model uses when
    making predictions.

    Examples
    --------
    Create a custom model by subclassing :class:`BaseDetector`:

    >>> class MyModel(BaseDetector):
    ...     def __init__(self, n_bands: int, window_frames: int):
    ...         super().__init__(n_bands, window_frames)
    ...         self._conv = nn.Conv1d(n_bands, 1, kernel_size=5, padding=2)
    ...
    ...     @property
    ...     def receptive_field(self) -> int:
    ...         return 4  # kernel_size - 1
    ...
    ...     def forward(self, x: Tensor) -> Tensor:
    ...         return self._conv(x).squeeze(1)
    ...
    ...     def _save_config(self) -> dict:
    ...         return {}  # no additional constructor args
    """

    def __init__(self, n_bands: int, window_frames: int) -> None:
        super().__init__()
        n_bands = ensure_int(n_bands, "n_bands")
        window_frames = ensure_int(window_frames, "window_frames")
        if n_bands <= 0:
            raise ValueError(
                f"Argument 'n_bands' must be a positive integer, got {n_bands}."
            )
        if window_frames <= 0:
            raise ValueError(
                f"Argument 'window_frames' must be a positive integer, "
                f"got {window_frames}."
            )
        self._n_bands = n_bands
        self._window_frames = window_frames

    @property
    def n_bands(self) -> int:
        """Number of input frequency bands.

        :type: :class:`int`
        """
        return self._n_bands

    @property
    def window_frames(self) -> int:
        """Number of frames per input window.

        The corresponding duration in seconds depends on the feature extractor's hop
        size: ``window_duration_s = window_frames * hop_ms / 1000``.

        :type: :class:`int`
        """
        return self._window_frames

    @property
    @abstractmethod
    def receptive_field(self) -> int:
        """Receptive field in frames.

        The number of input frames that influence a single output prediction.
        Used to determine padding requirements during inference.

        :type: :class:`int`
        """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input features of shape ``(batch, n_bands, time)``.

        Returns
        -------
        logits : Tensor
            Output logits of shape ``(batch, time)``.
        """

    @abstractmethod
    def _save_config(self) -> dict:
        """Return additional constructor kwargs needed for model reconstruction.

        Subclasses should override this to return a dictionary of any constructor
        arguments beyond ``n_bands`` and ``window_frames`` that are needed to recreate
        the model.

        Returns
        -------
        kwargs : dict
            Dictionary of additional constructor keyword arguments.

        Examples
        --------
        For a model with a ``base`` parameter:

        >>> def _save_config(self) -> dict:
        ...     return {"base": self._base}

        For a model with no additional parameters:

        >>> def _save_config(self) -> dict:
        ...     return {}
        """

    def predict(self, features: Tensor, *, hop_frames: int | None = None) -> Tensor:
        """Run sliding window inference on a full recording.

        The model is applied to overlapping windows across the recording. Where windows
        overlap, predictions are averaged to produce smoother, more robust per-frame
        probability estimates.

        Parameters
        ----------
        features : Tensor
            Input features of shape ``(n_bands, n_frames)``. Should be on the same
            device as the model.
        hop_frames : int | None
            Hop between consecutive windows in frames. Smaller values produce more
            overlap and smoother predictions but increase computation time. If ``None``,
            defaults to ``window_frames // 4`` (75% overlap).

        Returns
        -------
        probabilities : Tensor
            Per-frame call probabilities of shape ``(n_frames,)``. Values are in
            ``[0, 1]``, where higher values indicate higher confidence that a call
            is present.

        Notes
        -----
        The inference process:

        1. Slide a window of size :attr:`window_frames` across the recording with step
           ``hop_frames``.
        2. For each window, run the model to get logits, then apply sigmoid to get
           probabilities.
        3. Accumulate predictions for each frame. Frames covered by multiple windows
           receive multiple predictions.
        4. Average the accumulated predictions to get final per-frame probabilities.

        For frames near the end of the recording that don't fit a full window, the
        window is padded using edge values.

        Examples
        --------
        >>> from callcut.pipeline import load_pipeline
        >>> from callcut.io import load_audio
        >>>
        >>> model, extractor, decoder = load_pipeline("pipeline.pt", device="cpu")
        >>>
        >>> waveform, sr = load_audio("recording.wav", sample_rate=32000)
        >>> features, times = extractor(waveform)
        >>>
        >>> probs = model.predict(features)
        >>> probs.shape
        torch.Size([1234])
        """
        check_type(features, (torch.Tensor,), "features")
        hop_frames = (
            self._window_frames // 4
            if hop_frames is None
            else ensure_int(hop_frames, "hop_frames")
        )

        if hop_frames <= 0:
            raise ValueError(
                f"Argument 'hop_frames' must be positive, got {hop_frames}."
            )
        if hop_frames > self._window_frames:
            raise ValueError(
                f"Argument 'hop_frames' ({hop_frames}) cannot be greater "
                f"than 'window_frames' ({self._window_frames})."
            )

        if features.dim() != 2:
            raise ValueError(
                "Argument 'features' must be 2D (n_bands, n_frames), got "
                f"{features.dim()}D."
            )

        device = features.device
        dtype = features.dtype
        n_bands, n_frames = features.shape

        if n_bands != self._n_bands:
            raise ValueError(
                f"Feature bands ({n_bands}) do not match model input bands "
                f"({self._n_bands})."
            )

        # Initialize accumulators
        prob_sum = torch.zeros(n_frames, dtype=dtype, device=device)
        prob_count = torch.zeros(n_frames, dtype=dtype, device=device)

        # Compute window start positions
        if n_frames <= self._window_frames:
            starts = [0]
        else:
            starts = list(range(0, n_frames - self._window_frames + 1, hop_frames))
            if not starts:
                starts = [0]

        # Run inference
        self.eval()
        with torch.no_grad():
            for start in starts:
                end = start + self._window_frames

                # Extract window
                if end <= n_frames:
                    window = features[:, start:end]
                else:
                    # Pad with edge values if window extends beyond recording
                    available = features[:, start:n_frames]
                    pad_size = end - n_frames
                    padding = features[:, -1:].expand(-1, pad_size)
                    window = torch.cat([available, padding], dim=1)

                # Run model: (n_bands, window_frames) -> add batch -> model -> remove
                logits = self(window.unsqueeze(0)).squeeze(0)

                # Convert logits to probabilities
                probs = torch.sigmoid(logits)

                # Accumulate (only for valid frames, not padding)
                valid_end = min(n_frames, end)
                valid_len = valid_end - start
                prob_sum[start:valid_end] += probs[:valid_len]
                prob_count[start:valid_end] += 1.0

        # Average predictions
        prob_count = torch.clamp(prob_count, min=1e-12)
        probabilities = prob_sum / prob_count

        return probabilities


class TinySegCNN(BaseDetector):
    """Lightweight 1D CNN for call detection.

    A small convolutional neural network (~10K parameters) that processes multi-band SNR
    features to detect animal calls. The architecture uses four 1D convolutional layers
    to capture temporal patterns across frequency bands.

    Parameters
    ----------
    n_bands : int
        Number of input frequency bands.
    window_frames : int
        Number of frames per input window. The corresponding duration in seconds depends
        on the feature extractor's hop size:
        ``window_duration_s = window_frames * hop_ms / 1000``.
    base : int
        Base number of filters (channels in hidden layers).

    Notes
    -----
    Architecture::

        Input: (batch, n_bands, time)
          -> Conv1d(n_bands, base, kernel=9, padding=4) + ReLU
          -> Conv1d(base, base, kernel=9, padding=4) + ReLU
          -> Conv1d(base, base, kernel=5, padding=2) + ReLU
          -> Conv1d(base, 1, kernel=1)
        Output: (batch, time)

    The receptive field is 21 frames (sum of kernel_size - 1 for each layer).

    Examples
    --------
    >>> model = TinySegCNN(n_bands=8, window_frames=250)
    >>> x = torch.randn(4, 8, 250)  # batch=4, bands=8, time=250
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([4, 250])
    """

    def __init__(self, n_bands: int, window_frames: int, base: int = 32) -> None:
        super().__init__(n_bands, window_frames)
        check_type(base, ("int-like",), "base")
        base = ensure_int(base, "base")
        if base <= 0:
            raise ValueError(f"Argument 'base' must be a positive integer, got {base}.")

        self._base = base
        self._network = nn.Sequential(
            nn.Conv1d(n_bands, base, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(base, base, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(base, base, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(base, 1, kernel_size=1),
        )

    @property
    def base(self) -> int:
        """Base number of filters.

        :type: :class:`int`
        """
        return self._base

    @property
    def receptive_field(self) -> int:
        """Receptive field in frames."""
        # sum of (kernel_size - 1) for each convolutional layer
        return (9 - 1) + (9 - 1) + (5 - 1) + (1 - 1)  # = 21

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input features of shape ``(batch, n_bands, time)``.

        Returns
        -------
        logits : Tensor
            Output logits of shape ``(batch, time)``.
        """
        return self._network(x).squeeze(1)

    def _save_config(self) -> dict:
        """Return additional kwargs needed for reconstruction."""
        return {"base": self._base}
