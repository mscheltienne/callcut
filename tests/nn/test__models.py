"""Tests for callcut.nn._models module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

from callcut.nn import (
    BaseDetector,
    TinySegCNN,
    list_models,
    load_model,
    register_model,
    save_model,
    unregister_model,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


class _CustomModel(BaseDetector):
    """A custom model with extra kwargs for testing save/load."""

    def __init__(self, n_bands: int = 8, hidden: int = 16) -> None:
        super().__init__(n_bands)
        self._hidden = hidden
        self._layer = torch.nn.Linear(n_bands, 1)

    @property
    def receptive_field(self) -> int:
        return 1

    @property
    def hidden(self) -> int:
        return self._hidden

    def forward(self, x: Tensor) -> Tensor:
        return self._layer(x.transpose(1, 2)).squeeze(-1)

    def _save_kwargs(self) -> dict:
        return {"hidden": self._hidden}


@pytest.fixture
def registered_custom_model() -> Generator[type[_CustomModel]]:
    """Register a custom model for testing and unregister on teardown."""
    register_model(_CustomModel)
    yield _CustomModel
    unregister_model("_CustomModel")


class TestBaseDetector:
    """Tests for BaseDetector class."""

    def test_n_bands_validation(self) -> None:
        """Test n_bands parameter validation."""
        with pytest.raises(ValueError, match="positive integer"):
            TinySegCNN(n_bands=0)

        with pytest.raises(ValueError, match="positive integer"):
            TinySegCNN(n_bands=-1)

        with pytest.raises(TypeError):
            TinySegCNN(n_bands="invalid")

    def test_n_bands_property(self) -> None:
        """Test n_bands property returns correct value."""
        model = TinySegCNN(n_bands=4)
        assert model.n_bands == 4


class TestSaveLoad:
    """Tests for save and load methods."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading a model preserves attributes."""
        model = TinySegCNN(n_bands=8, base=16)
        save_path = tmp_path / "model.pt"

        assert not save_path.exists()
        save_model(model, save_path)
        assert save_path.exists()

        loaded = load_model(save_path)
        assert isinstance(loaded, TinySegCNN)
        assert loaded.n_bands == 8
        assert loaded.base == 16
        assert loaded.receptive_field == model.receptive_field

    def test_save_overwrite(self, tmp_path: Path) -> None:
        """Test overwrite behavior."""
        model = TinySegCNN(n_bands=8)
        save_path = tmp_path / "model.pt"
        save_model(model, save_path)

        with pytest.raises(FileExistsError, match="already exists"):
            save_model(model, save_path)

        save_model(model, save_path, overwrite=True)  # should succeed

    def test_load_preserves_weights(self, tmp_path: Path) -> None:
        """Test that loaded model produces identical outputs."""
        model = TinySegCNN(n_bands=8)
        x = torch.randn(1, 8, 100)

        with torch.no_grad():
            output_before = model(x)

        save_path = tmp_path / "model.pt"
        save_model(model, save_path)
        loaded = load_model(save_path)

        with torch.no_grad():
            output_after = loaded(x)

        torch.testing.assert_close(output_before, output_after)

    def test_load_nonexistent(self) -> None:
        """Test error when loading from nonexistent file."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_model("nonexistent_model.pt")

    def test_save_load_custom_kwargs(
        self, tmp_path: Path, registered_custom_model: type[_CustomModel]
    ) -> None:
        """Test save/load preserves custom model kwargs."""
        model = registered_custom_model(n_bands=4, hidden=32)
        save_path = tmp_path / "custom_model.pt"
        save_model(model, save_path)

        loaded = load_model(save_path)
        assert isinstance(loaded, registered_custom_model)
        assert loaded.n_bands == 4
        assert loaded.hidden == 32


class TestTinySegCNN:
    """Tests for TinySegCNN model."""

    def test_default_params(self) -> None:
        """Test construction with default parameters."""
        model = TinySegCNN()
        assert model.n_bands == 8
        assert model.base == 32

    def test_custom_params(self) -> None:
        """Test construction with custom parameters."""
        model = TinySegCNN(n_bands=4, base=16)
        assert model.n_bands == 4
        assert model.base == 16

    def test_base_validation(self) -> None:
        """Test base parameter validation."""
        with pytest.raises(ValueError, match="positive integer"):
            TinySegCNN(base=0)

        with pytest.raises(ValueError, match="positive integer"):
            TinySegCNN(base=-1)

    def test_receptive_field(self) -> None:
        """Test receptive field property."""
        model = TinySegCNN()
        assert model.receptive_field > 0, "Receptive field should be positive"

    def test_forward_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        model = TinySegCNN(n_bands=8, base=32)
        x = torch.randn(4, 8, 250)  # batch=4, bands=8, time=250
        output = model(x)
        assert output.shape == (4, 250)

    def test_forward_various_lengths(self) -> None:
        """Test that output time dimension equals input time dimension."""
        model = TinySegCNN(n_bands=8)
        for time_steps in [100, 250, 500]:
            x = torch.randn(2, 8, time_steps)
            output = model(x)
            assert output.shape == (2, time_steps)

    def test_is_registered(self) -> None:
        """Test that TinySegCNN is registered in the model registry."""
        assert "TinySegCNN" in list_models()
