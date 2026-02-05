"""Tests for callcut.training._callbacks module."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from numpy.testing import assert_allclose

from callcut.nn import TinySegCNN, load_model
from callcut.training import (
    BCEWithLogitsLoss,
    CallDetectorModule,
    LoggingCallback,
    MetricsHistoryCallback,
    SaveBestModelCallback,
)


def _create_mock_trainer(metrics: dict[str, float], epoch: int = 0) -> MagicMock:
    """Create a mock trainer with given metrics."""
    trainer = MagicMock(spec=["callback_metrics", "current_epoch"])
    trainer.callback_metrics = {k: torch.tensor(v) for k, v in metrics.items()}
    trainer.current_epoch = epoch
    return trainer


def _create_mock_module() -> CallDetectorModule:
    """Create a CallDetectorModule for testing."""
    model = TinySegCNN(n_bands=8, window_frames=100)
    return CallDetectorModule(model, loss=BCEWithLogitsLoss(), lr=1e-3)


class TestLoggingCallback:
    """Tests for LoggingCallback."""

    def test_instantiation(self) -> None:
        """Test that LoggingCallback can be instantiated."""
        callback = LoggingCallback()
        assert callback is not None

    def test_on_train_epoch_end_with_all_metrics(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test logging when all metrics are present."""
        callback = LoggingCallback()
        trainer = _create_mock_trainer(
            {"train_loss": 0.5, "val_loss": 0.4, "val_f1": 0.85}, epoch=5
        )
        module = _create_mock_module()

        with caplog.at_level(logging.INFO, logger="callcut"):
            callback.on_train_epoch_end(trainer, module)

        # Check that epoch info was logged
        log_text = caplog.text
        assert "Epoch 006" in log_text
        assert "train_loss=0.5000" in log_text
        assert "val_loss=0.4000" in log_text
        assert "val_f1=0.850" in log_text

    def test_on_train_epoch_end_partial_metrics(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test logging when some metrics are missing."""
        callback = LoggingCallback()
        trainer = _create_mock_trainer({"train_loss": 0.5}, epoch=0)
        module = _create_mock_module()

        with caplog.at_level(logging.INFO, logger="callcut"):
            callback.on_train_epoch_end(trainer, module)

        # Should still log without errors
        log_text = caplog.text
        assert "Epoch 001" in log_text
        assert "train_loss=0.5000" in log_text

    def test_on_train_epoch_end_no_metrics(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test logging when no metrics are present."""
        callback = LoggingCallback()
        trainer = _create_mock_trainer({}, epoch=0)
        module = _create_mock_module()

        with caplog.at_level(logging.INFO, logger="callcut"):
            callback.on_train_epoch_end(trainer, module)

        # Should still log epoch number
        assert "Epoch 001" in caplog.text


class TestMetricsHistoryCallback:
    """Tests for MetricsHistoryCallback."""

    def test_instantiation(self) -> None:
        """Test that MetricsHistoryCallback can be instantiated."""
        callback = MetricsHistoryCallback()
        assert callback is not None
        assert callback.history == {}

    def test_history_property(self) -> None:
        """Test that history property returns dict."""
        callback = MetricsHistoryCallback()
        assert isinstance(callback.history, dict)

    def test_records_metrics(self) -> None:
        """Test that metrics are recorded each epoch."""
        callback = MetricsHistoryCallback()
        module = _create_mock_module()

        # Simulate multiple epochs
        for epoch, loss in enumerate([0.8, 0.6, 0.4]):
            trainer = _create_mock_trainer(
                {"train_loss": loss, "val_f1": 0.5 + epoch * 0.1}, epoch=epoch
            )
            callback.on_train_epoch_end(trainer, module)

        assert "train_loss" in callback.history
        assert "val_f1" in callback.history
        assert len(callback.history["train_loss"]) == 3
        assert len(callback.history["val_f1"]) == 3
        assert_allclose(callback.history["train_loss"], [0.8, 0.6, 0.4])

    def test_converts_tensors_to_float(self) -> None:
        """Test that tensor values are converted to float."""
        callback = MetricsHistoryCallback()
        trainer = _create_mock_trainer({"train_loss": 0.5}, epoch=0)
        module = _create_mock_module()

        callback.on_train_epoch_end(trainer, module)

        assert isinstance(callback.history["train_loss"][0], float)

    def test_handles_new_metrics(self) -> None:
        """Test handling of metrics that appear in later epochs."""
        callback = MetricsHistoryCallback()
        module = _create_mock_module()

        # First epoch: only train_loss
        trainer1 = _create_mock_trainer({"train_loss": 0.5}, epoch=0)
        callback.on_train_epoch_end(trainer1, module)

        # Second epoch: train_loss and val_f1
        trainer2 = _create_mock_trainer({"train_loss": 0.4, "val_f1": 0.8}, epoch=1)
        callback.on_train_epoch_end(trainer2, module)

        assert_allclose(callback.history["train_loss"], [0.5, 0.4])
        assert_allclose(callback.history["val_f1"], [0.8])


class TestSaveBestModelCallback:
    """Tests for SaveBestModelCallback."""

    def test_instantiation(self, tmp_path: Path) -> None:
        """Test that SaveBestModelCallback can be instantiated."""
        save_path = tmp_path / "best.pt"
        callback = SaveBestModelCallback(save_path, monitor="val_f1", mode="max")
        assert callback is not None

    def test_default_monitor(self, tmp_path: Path) -> None:
        """Test default monitor parameter."""
        callback = SaveBestModelCallback(tmp_path / "model.pt")
        assert callback._monitor == "val_f1"

    def test_default_mode(self, tmp_path: Path) -> None:
        """Test default mode parameter."""
        callback = SaveBestModelCallback(tmp_path / "model.pt")
        assert callback._mode == "max"

    def test_invalid_mode(self, tmp_path: Path) -> None:
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="Allowed values are"):
            SaveBestModelCallback(tmp_path / "model.pt", mode="invalid")

    def test_invalid_monitor_type(self, tmp_path: Path) -> None:
        """Test that invalid monitor type raises error."""
        with pytest.raises(TypeError):
            SaveBestModelCallback(tmp_path / "model.pt", monitor=123)  # type: ignore[arg-type]

    def test_saves_best_model_max_mode(self, tmp_path: Path) -> None:
        """Test that best model is saved in max mode."""
        save_path = tmp_path / "best.pt"
        callback = SaveBestModelCallback(save_path, monitor="val_f1", mode="max")
        module = _create_mock_module()

        # First validation: val_f1=0.7
        trainer1 = _create_mock_trainer({"val_f1": 0.7}, epoch=0)
        callback.on_validation_epoch_end(trainer1, module)
        assert save_path.exists()

        # Check saved model
        loaded = load_model(save_path)
        assert isinstance(loaded, TinySegCNN)

    def test_saves_best_model_min_mode(self, tmp_path: Path) -> None:
        """Test that best model is saved in min mode."""
        save_path = tmp_path / "best.pt"
        callback = SaveBestModelCallback(save_path, monitor="val_loss", mode="min")
        module = _create_mock_module()

        # First validation: val_loss=0.5
        trainer1 = _create_mock_trainer({"val_loss": 0.5}, epoch=0)
        callback.on_validation_epoch_end(trainer1, module)
        assert save_path.exists()

    def test_only_saves_on_improvement_max(self, tmp_path: Path) -> None:
        """Test that model is only saved when metric improves (max mode)."""
        save_path = tmp_path / "best.pt"
        callback = SaveBestModelCallback(save_path, monitor="val_f1", mode="max")
        module = _create_mock_module()

        # First validation: val_f1=0.8
        trainer1 = _create_mock_trainer({"val_f1": 0.8}, epoch=0)
        callback.on_validation_epoch_end(trainer1, module)
        mtime1 = save_path.stat().st_mtime

        # Second validation: val_f1=0.7 (worse, should not save)
        trainer2 = _create_mock_trainer({"val_f1": 0.7}, epoch=1)
        callback.on_validation_epoch_end(trainer2, module)
        mtime2 = save_path.stat().st_mtime
        assert mtime1 == mtime2  # File not modified

        # Third validation: val_f1=0.9 (better, should save)
        trainer3 = _create_mock_trainer({"val_f1": 0.9}, epoch=2)
        callback.on_validation_epoch_end(trainer3, module)
        mtime3 = save_path.stat().st_mtime
        assert mtime3 > mtime1  # File was modified

    def test_only_saves_on_improvement_min(self, tmp_path: Path) -> None:
        """Test that model is only saved when metric improves (min mode)."""
        save_path = tmp_path / "best.pt"
        callback = SaveBestModelCallback(save_path, monitor="val_loss", mode="min")
        module = _create_mock_module()

        # First validation: val_loss=0.5
        trainer1 = _create_mock_trainer({"val_loss": 0.5}, epoch=0)
        callback.on_validation_epoch_end(trainer1, module)
        mtime1 = save_path.stat().st_mtime

        # Second validation: val_loss=0.6 (worse, should not save)
        trainer2 = _create_mock_trainer({"val_loss": 0.6}, epoch=1)
        callback.on_validation_epoch_end(trainer2, module)
        mtime2 = save_path.stat().st_mtime
        assert mtime1 == mtime2

        # Third validation: val_loss=0.3 (better, should save)
        trainer3 = _create_mock_trainer({"val_loss": 0.3}, epoch=2)
        callback.on_validation_epoch_end(trainer3, module)
        mtime3 = save_path.stat().st_mtime
        assert mtime3 > mtime1

    def test_no_save_when_metric_missing(self, tmp_path: Path) -> None:
        """Test that nothing is saved when monitored metric is missing."""
        save_path = tmp_path / "best.pt"
        callback = SaveBestModelCallback(save_path, monitor="val_f1", mode="max")
        module = _create_mock_module()

        # Validation without val_f1
        trainer = _create_mock_trainer({"train_loss": 0.5}, epoch=0)
        callback.on_validation_epoch_end(trainer, module)

        assert not save_path.exists()

    def test_warns_for_non_calldetector_module(self, tmp_path: Path) -> None:
        """Test that warning is issued for non-CallDetectorModule."""
        save_path = tmp_path / "best.pt"
        callback = SaveBestModelCallback(save_path, monitor="val_f1", mode="max")

        # Use a mock that is not a CallDetectorModule
        mock_module = MagicMock()
        mock_module.model = TinySegCNN(n_bands=8, window_frames=100)

        trainer = _create_mock_trainer({"val_f1": 0.8}, epoch=0)

        # This should issue a warning but not crash
        with pytest.warns(RuntimeWarning, match="not a CallDetectorModule"):
            callback.on_validation_epoch_end(trainer, mock_module)

        # Model should not be saved (it's not a CallDetectorModule)
        assert not save_path.exists()

    def test_string_path(self, tmp_path: Path) -> None:
        """Test that string paths are accepted."""
        save_path = str(tmp_path / "best.pt")
        callback = SaveBestModelCallback(save_path, monitor="val_f1")
        module = _create_mock_module()

        trainer = _create_mock_trainer({"val_f1": 0.8}, epoch=0)
        callback.on_validation_epoch_end(trainer, module)

        assert Path(save_path).exists()
