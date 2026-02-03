"""Built-in model architectures for call detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from callcut.nn._registry import get_model, register_model
from callcut.utils._checks import check_type, ensure_device, ensure_int, ensure_path
from callcut.utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


class BaseDetector(ABC, nn.Module):
    """Abstract base class for call detection models.

    Subclasses must implement:

    - :meth:`forward`: Process input features and return logits.
    - :attr:`receptive_field`: Property returning the receptive field in frames.
    - :meth:`_save_kwargs`: Return additional constructor kwargs for serialization.

    Models accept input of shape ``(batch, n_bands, time)`` and return logits of shape
    ``(batch, time)``.

    Parameters
    ----------
    n_bands : int
        Number of input frequency bands.

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
    ...     def __init__(self, n_bands: int = 8):
    ...         super().__init__(n_bands)
    ...         self._conv = nn.Conv1d(n_bands, 1, kernel_size=5, padding=2)
    ...
    ...     @property
    ...     def receptive_field(self) -> int:
    ...         return 4  # kernel_size - 1
    ...
    ...     def forward(self, x: Tensor) -> Tensor:
    ...         return self._conv(x).squeeze(1)
    ...
    ...     def _save_kwargs(self) -> dict:
    ...         return {}  # no additional constructor args
    """

    def __init__(self, n_bands: int) -> None:
        super().__init__()
        check_type(n_bands, ("int-like",), "n_bands")
        n_bands = ensure_int(n_bands, "n_bands")
        if n_bands <= 0:
            raise ValueError(
                f"Argument 'n_bands' must be a positive integer, got {n_bands}."
            )
        self._n_bands = n_bands

    @property
    def n_bands(self) -> int:
        """Number of input frequency bands.

        :type: :class:`int`
        """
        return self._n_bands

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
    def _save_kwargs(self) -> dict:
        """Return additional constructor kwargs needed for model reconstruction.

        Subclasses should override this to return a dictionary of any constructor
        arguments beyond ``n_bands`` that are needed to recreate the model.

        Returns
        -------
        kwargs : dict
            Dictionary of additional constructor keyword arguments.

        Examples
        --------
        For a model with a ``base`` parameter:

        >>> def _save_kwargs(self) -> dict:
        ...     return {"base": self._base}

        For a model with no additional parameters:

        >>> def _save_kwargs(self) -> dict:
        ...     return {}
        """


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


def save_model(
    model: BaseDetector,
    fname: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save a model to a file.

    Saves both the model state dict and metadata (class name, n_bands, receptive_field)
    needed for reconstruction.

    Parameters
    ----------
    model : BaseDetector
        The model to save.
    fname : str | Path
        Path to save the model. Conventionally use ``.pt`` extension.
    overwrite : bool
        If ``True``, overwrite the file if it exists. If ``False``, raises
        an error if the file already exists.

    See Also
    --------
    load_model : Load a model from a file.

    Examples
    --------
    >>> model = TinySegCNN(n_bands=8)
    >>> save_model(model, "my_model.pt")
    """
    check_type(model, (BaseDetector,), "model")
    fname = ensure_path(fname, must_exist=False)
    if not overwrite and fname.exists():
        raise FileExistsError(
            f"File {fname} already exists. Use overwrite=True to replace it."
        )
    logger.info("Saving model to %s", fname)
    checkpoint = {
        "class_name": model.__class__.__name__,
        "n_bands": model.n_bands,
        "receptive_field": model.receptive_field,
        "state_dict": model.state_dict(),
        "kwargs": model._save_kwargs(),
    }
    torch.save(checkpoint, fname)


def load_model(
    fname: str | Path,
    *,
    device: str | torch.device | None = None,
) -> BaseDetector:
    """Load a model from a file.

    Parameters
    ----------
    fname : str | Path
        Path to the saved model file.
    device : str | torch.device | None
        Device to load the model to (e.g., ``"cpu"``, ``"cuda:0"``, ``"mps"``).
        If ``None``, uses the default torch device.

    Returns
    -------
    model : BaseDetector
        The loaded model instance.

    See Also
    --------
    save_model : Save a model to a file.

    Examples
    --------
    >>> model = load_model("my_model.pt")
    >>> model = load_model("my_model.pt", device="cpu")
    """
    fname = ensure_path(fname, must_exist=True)
    device = ensure_device(device)
    logger.info("Loading model from %s", fname)
    checkpoint = torch.load(fname, map_location=device, weights_only=False)

    class_name = checkpoint["class_name"]
    n_bands = checkpoint["n_bands"]
    kwargs = checkpoint.get("kwargs", {})

    model = get_model(class_name, n_bands=n_bands, **kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    logger.debug(
        "Loaded %s with n_bands=%d, receptive_field=%d",
        class_name,
        n_bands,
        model.receptive_field,
    )
    return model
