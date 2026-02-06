"""Lightning Module for call detection training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch

from callcut.evaluation import compute_frame_metrics
from callcut.nn import BaseDetector
from callcut.training._losses import BaseLoss
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
    loss : BaseLoss
        Loss function to use. Available loss functions:

        - :class:`~callcut.training.BCEWithLogitsLoss`: Standard binary cross-entropy
        - :class:`~callcut.training.FocalLoss`: Down-weights easy examples
        - :class:`~callcut.training.DiceLoss`: Optimizes overlap directly
        - :class:`~callcut.training.TverskyLoss`: Adjustable FP/FN penalties
    lr : float
        Learning rate for the optimizer.

    Examples
    --------
    >>> from callcut.nn import TinySegCNN
    >>> from callcut.training import (
    ...     CallDetectorModule,
    ...     CallDataModule,
    ...     BCEWithLogitsLoss,
    ... )
    >>> import lightning as L
    >>>
    >>> # Create model and module
    >>> model = TinySegCNN(n_bands=8, window_frames=250)
    >>> module = CallDetectorModule(model, loss=BCEWithLogitsLoss(), lr=1e-3)
    >>>
    >>> # Or with a different loss function
    >>> from callcut.training import FocalLoss
    >>> module = CallDetectorModule(model, loss=FocalLoss(gamma=2.0), lr=1e-3)
    >>>
    >>> # Create data module and train
    >>> dm = CallDataModule(recordings=..., extractor=...)
    >>> trainer = L.Trainer(max_epochs=10)
    >>> trainer.fit(module, datamodule=dm)
    """

    def __init__(self, model: BaseDetector, loss: BaseLoss, lr: float = 1e-3) -> None:
        super().__init__()
        check_type(model, (BaseDetector,), "model")
        check_type(loss, (BaseLoss,), "loss")
        check_type(lr, ("numeric",), "lr")

        if lr <= 0:
            raise ValueError(f"Argument 'lr' must be positive, got {lr}.")

        self._model = model
        self._lr = float(lr)
        self._loss_fn = loss

        # Save hyperparameters for checkpointing (excludes model and loss)
        self.save_hyperparameters(ignore=["model", "loss"])

        logger.info("CallDetectorModule initialized: lr=%.2e, loss=%s", lr, loss)

    @property
    def model(self) -> BaseDetector:
        """The underlying detector model.

        :type: :class:`~callcut.nn.BaseDetector`
        """
        return self._model

    @property
    def loss(self) -> BaseLoss:
        """The loss function.

        :type: :class:`~callcut.training.BaseLoss`
        """
        return self._loss_fn

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
            f"loss={self._loss_fn})"
        )
