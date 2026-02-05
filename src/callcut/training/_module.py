"""Lightning Module for call detection training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
import torch.nn as nn

from callcut.evaluation import compute_frame_metrics
from callcut.nn import BaseDetector
from callcut.utils._checks import check_type
from callcut.utils.logs import logger

if TYPE_CHECKING:
    from torch import Tensor


class CallDetectorModule(L.LightningModule):
    """Lightning Module wrapping a call detector for training.

    This module handles the training loop, validation, and optimizer configuration
    for any :class:`~callcut.nn.BaseDetector` model.

    Parameters
    ----------
    model : BaseDetector
        The detector model to train.
    lr : float
        Learning rate for the optimizer.
    pos_weight : float | None
        Positive class weight for imbalanced data. If ``None``, no weighting is
        applied. Can be computed using
        :meth:`~callcut.io.CallDataset.compute_pos_weight`.

    Examples
    --------
    >>> from callcut.nn import TinySegCNN
    >>> from callcut.training import CallDetectorModule, CallDataModule
    >>> import lightning as L
    >>>
    >>> # Create model and module
    >>> model = TinySegCNN(n_bands=8, window_frames=250)
    >>> module = CallDetectorModule(model, lr=1e-3)
    >>>
    >>> # Create data module
    >>> dm = CallDataModule(recordings=..., extractor=...)
    >>>
    >>> # Train
    >>> trainer = L.Trainer(max_epochs=10)
    >>> trainer.fit(module, datamodule=dm)
    """

    def __init__(
        self,
        model: BaseDetector,
        lr: float = 1e-3,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        check_type(model, (BaseDetector,), "model")
        check_type(lr, ("numeric",), "lr")
        check_type(pos_weight, ("numeric", None), "pos_weight")

        if lr <= 0:
            raise ValueError(f"Argument 'lr' must be positive, got {lr}.")
        if pos_weight is not None and pos_weight <= 0:
            raise ValueError(
                f"Argument 'pos_weight' must be positive, got {pos_weight}."
            )

        self._model = model
        self._lr = float(lr)
        self._pos_weight = float(pos_weight) if pos_weight is not None else None

        # Loss function (set up in setup() or on first forward if pos_weight is None)
        self._loss_fn: nn.BCEWithLogitsLoss | None = None

        # Save hyperparameters for checkpointing (excludes model)
        self.save_hyperparameters(ignore=["model"])

        logger.info(
            "CallDetectorModule initialized: lr=%.2e, pos_weight=%s", lr, pos_weight
        )

    @property
    def model(self) -> BaseDetector:
        """The underlying detector model.

        :type: :class:`~callcut.nn.BaseDetector`
        """
        return self._model

    def setup(self, stage: str) -> None:
        """Set up the module for a given stage.

        Parameters
        ----------
        stage : str
            One of ``"fit"``, ``"validate"``, ``"test"``, or ``"predict"``.
        """
        if stage == "fit" and self._loss_fn is None:
            self._setup_loss_fn()

    def _setup_loss_fn(self) -> None:
        """Initialize the loss function with optional pos_weight."""
        if self._pos_weight is not None:
            pos_weight = torch.tensor([self._pos_weight], dtype=torch.float32)
            self._loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.debug(
                "Loss function initialized with pos_weight=%.2f", self._pos_weight
            )
        else:
            self._loss_fn = nn.BCEWithLogitsLoss()
            logger.debug("Loss function initialized without pos_weight")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x : Tensor
            Input features of shape ``(batch, n_bands, time)``.

        Returns
        -------
        logits : Tensor
            Output logits of shape ``(batch, time)``.
        """
        return self._model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Perform a single training step.

        Parameters
        ----------
        batch : tuple of Tensor
            Tuple of (features, labels) tensors.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        loss : Tensor
            The training loss for this batch.
        """
        X, y = batch
        logits = self(X)

        if self._loss_fn is None:
            self._setup_loss_fn()
        assert self._loss_fn is not None  # sanity-check

        loss = self._loss_fn(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single validation step.

        Parameters
        ----------
        batch : tuple of Tensor
            Tuple of (features, labels) tensors.
        batch_idx : int
            Index of the current batch.
        """
        X, y = batch
        logits = self(X)

        if self._loss_fn is None:
            self._setup_loss_fn()
        assert self._loss_fn is not None  # sanity-check

        loss = self._loss_fn(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute frame-level metrics
        probs = torch.sigmoid(logits)

        # Flatten batch for metrics computation
        probs_flat = probs.reshape(-1)
        y_flat = y.reshape(-1)
        metrics = compute_frame_metrics(probs_flat, y_flat, threshold=0.5)

        self.log("val_precision", metrics.precision, on_step=False, on_epoch=True)
        self.log("val_recall", metrics.recall, on_step=False, on_epoch=True)
        self.log("val_f1", metrics.f1, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step.

        Parameters
        ----------
        batch : tuple of Tensor
            Tuple of (features, labels) tensors.
        batch_idx : int
            Index of the current batch.
        """
        X, y = batch
        logits = self(X)

        if self._loss_fn is None:
            self._setup_loss_fn()
        assert self._loss_fn is not None  # sanity-check

        loss = self._loss_fn(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # Compute frame-level metrics
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(-1)
        y_flat = y.reshape(-1)
        metrics = compute_frame_metrics(probs_flat, y_flat, threshold=0.5)

        self.log("test_precision", metrics.precision, on_step=False, on_epoch=True)
        self.log("test_recall", metrics.recall, on_step=False, on_epoch=True)
        self.log("test_f1", metrics.f1, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            The Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self._model.__class__.__name__}, "
            f"lr={self._lr}, "
            f"pos_weight={self._pos_weight})"
        )
