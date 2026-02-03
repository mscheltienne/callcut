"""Tests for callcut.nn._models module."""

from __future__ import annotations

import pytest
import torch

from callcut.nn import TinySegCNN, list_models


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
