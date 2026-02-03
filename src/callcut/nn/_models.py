"""Built-in model architectures for call detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from callcut.nn._base import BaseDetector
from callcut.nn._registry import register_model
from callcut.utils._checks import check_type, ensure_int

if TYPE_CHECKING:
    from torch import Tensor


@register_model
class TinySegCNN(BaseDetector):
    """Lightweight 1D CNN for call detection.

    A small convolutional neural network (~10K parameters) that processes multi-band SNR
    features to detect animal calls. The architecture uses four 1D convolutional layers
    to capture temporal patterns across frequency bands.

    Parameters
    ----------
    n_bands : int
        Number of input frequency bands.
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
    >>> model = TinySegCNN(n_bands=8, base=32)
    >>> x = torch.randn(4, 8, 250)  # batch=4, bands=8, time=250
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([4, 250])
    """

    def __init__(self, n_bands: int = 8, base: int = 32) -> None:
        super().__init__(n_bands)
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

    def _save_kwargs(self) -> dict:
        """Return additional kwargs needed for reconstruction."""
        return {"base": self._base}
