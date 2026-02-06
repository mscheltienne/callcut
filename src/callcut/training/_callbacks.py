"""Custom Lightning callbacks for call detection training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning import Callback

from callcut.training._module import CallDetectorModule
from callcut.utils._checks import check_type, check_value, ensure_path
from callcut.utils.logs import logger, warn

if TYPE_CHECKING:
    from pathlib import Path

    from lightning import LightningModule, Trainer


class LoggingCallback(Callback):
    """Callback that logs training progress using the callcut logger.

    This callback provides formatted logging at the end of each epoch,
    complementing Lightning's built-in progress bar.

    Examples
    --------
    >>> from callcut.training import LoggingCallback
    >>> import lightning as L
    >>>
    >>> trainer = L.Trainer(
    ...     max_epochs=10,
    ...     callbacks=[LoggingCallback()],
    ... )
    """

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log training metrics at the end of each epoch."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        val_f1 = metrics.get("val_f1")

        msg_parts = [f"Epoch {epoch + 1:03d}"]
        if train_loss is not None:
            msg_parts.append(f"train_loss={train_loss:.4f}")
        if val_loss is not None:
            msg_parts.append(f"val_loss={val_loss:.4f}")
        if val_f1 is not None:
            msg_parts.append(f"val_f1={val_f1:.3f}")

        logger.info(" | ".join(msg_parts))


class MetricsHistoryCallback(Callback):
    """Callback that records training metrics history.

    Stores metrics at the end of each epoch for later analysis or plotting.

    Attributes
    ----------
    history : dict
        Dictionary mapping metric names to lists of values per epoch.

    Examples
    --------
    >>> from callcut.training import MetricsHistoryCallback
    >>> import lightning as L
    >>>
    >>> history_callback = MetricsHistoryCallback()
    >>> trainer = L.Trainer(
    ...     max_epochs=10,
    ...     callbacks=[history_callback],
    ... )
    >>> trainer.fit(module, datamodule=dm)
    >>> history_callback.history["val_f1"]
    [0.72, 0.78, 0.81, ...]
    """

    def __init__(self) -> None:
        super().__init__()
        self._history: dict[str, list[float]] = {}

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record metrics at the end of each training epoch."""
        metrics = trainer.callback_metrics

        for key, value in metrics.items():
            if key not in self._history:
                self._history[key] = []
            # Convert tensor to float if needed
            if hasattr(value, "item"):
                value = value.item()
            self._history[key].append(float(value))

    @property
    def history(self) -> dict[str, list[float]]:
        """Dictionary mapping metric names to lists of values per epoch.

        :type: :class:`dict`
        """
        return self._history


class SaveBestModelCallback(Callback):
    """Callback that saves the best model weights during training.

    Monitors a metric and saves the model's ``state_dict`` when it improves.
    After training, load the best weights with :func:`torch.load` and
    :meth:`model.load_state_dict`, then use :func:`~callcut.pipeline.save_pipeline`
    to save the full pipeline.

    Parameters
    ----------
    save_path : Path | str
        Path to save the best model weights.
    monitor : str
        Metric to monitor (default: ``"val_f1"``).
    mode : str
        One of ``"min"`` or ``"max"``. Whether to minimize or maximize the metric.

    Examples
    --------
    >>> from callcut.training import SaveBestModelCallback
    >>> import lightning as L
    >>>
    >>> trainer = L.Trainer(
    ...     max_epochs=10,
    ...     callbacks=[SaveBestModelCallback("best_weights.pt", monitor="val_f1")],
    ... )
    >>> trainer.fit(module, datamodule=dm)
    >>>
    >>> # After training, load best weights and save full pipeline
    >>> model.load_state_dict(torch.load("best_weights.pt", weights_only=True))
    >>> save_pipeline(model, extractor, "pipeline.pt", decoder=decoder)
    """

    def __init__(
        self, save_path: Path | str, monitor: str = "val_f1", mode: str = "max"
    ) -> None:
        super().__init__()

        self._save_path = ensure_path(save_path, must_exist=False)
        check_type(monitor, (str,), "monitor")
        self._monitor = monitor
        check_type(mode, (str,), "mode")
        check_value(mode, ("min", "max"), "mode")
        self._mode = mode

        if mode == "max":
            self._best_score = float("-inf")
            self._is_better = lambda new, old: new > old
        elif mode == "min":
            self._best_score = float("inf")
            self._is_better = lambda new, old: new < old

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Check if model improved and save weights if so."""
        metrics = trainer.callback_metrics
        current_score = metrics.get(self._monitor)

        if current_score is None:
            return

        if hasattr(current_score, "item"):
            current_score = current_score.item()

        if self._is_better(current_score, self._best_score):
            self._best_score = current_score
            logger.info(
                "New best %s=%.4f, saving weights to %s",
                self._monitor,
                current_score,
                self._save_path,
            )
            # Get the underlying model from the Lightning module
            if isinstance(pl_module, CallDetectorModule):
                torch.save(pl_module.model.state_dict(), self._save_path)
            else:
                warn(
                    "Cannot save model: pl_module is not a CallDetectorModule.",
                    module="callcut",
                    ignore_namespaces=("callcut",),
                )
