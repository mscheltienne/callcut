"""Tests for callcut.nn module."""

from __future__ import annotations

import pytest
import torch

from callcut.nn import TinySegCNN


class TestBaseDetector:
    """Tests for BaseDetector class."""

    def test_n_bands_validation(self) -> None:
        """Test n_bands parameter validation."""
        with pytest.raises(ValueError, match="positive integer"):
            TinySegCNN(n_bands=0, window_frames=100)

        with pytest.raises(ValueError, match="positive integer"):
            TinySegCNN(n_bands=-1, window_frames=100)

        with pytest.raises(TypeError):
            TinySegCNN(n_bands="invalid", window_frames=100)

    def test_window_frames_validation(self) -> None:
        """Test window_frames parameter validation."""
        with pytest.raises(ValueError, match="positive integer"):
            TinySegCNN(n_bands=8, window_frames=0)

        with pytest.raises(ValueError, match="positive integer"):
            TinySegCNN(n_bands=8, window_frames=-1)

        with pytest.raises(TypeError):
            TinySegCNN(n_bands=8, window_frames="invalid")

    def test_n_bands_property(self) -> None:
        """Test n_bands property returns correct value."""
        model = TinySegCNN(n_bands=4, window_frames=100)
        assert model.n_bands == 4

    def test_window_frames_property(self) -> None:
        """Test window_frames property returns correct value."""
        model = TinySegCNN(n_bands=8, window_frames=250)
        assert model.window_frames == 250


class TestTinySegCNN:
    """Tests for TinySegCNN model."""

    def test_default_base(self) -> None:
        """Test construction with default base parameter."""
        model = TinySegCNN(n_bands=8, window_frames=250)
        assert model.base == 32

    def test_custom_params(self) -> None:
        """Test construction with custom parameters."""
        model = TinySegCNN(n_bands=4, window_frames=100, base=16)
        assert model.n_bands == 4
        assert model.window_frames == 100
        assert model.base == 16

    def test_base_validation(self) -> None:
        """Test base parameter validation."""
        with pytest.raises(ValueError, match="positive integer"):
            TinySegCNN(n_bands=8, window_frames=100, base=0)

        with pytest.raises(ValueError, match="positive integer"):
            TinySegCNN(n_bands=8, window_frames=100, base=-1)

    def test_receptive_field(self) -> None:
        """Test receptive field property."""
        model = TinySegCNN(n_bands=8, window_frames=250)
        assert model.receptive_field > 0, "Receptive field should be positive"

    def test_forward_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        model = TinySegCNN(n_bands=8, window_frames=250, base=32)
        x = torch.randn(4, 8, 250)  # batch=4, bands=8, time=250
        output = model(x)
        assert output.shape == (4, 250)

    def test_forward_various_lengths(self) -> None:
        """Test that output time dimension equals input time dimension."""
        model = TinySegCNN(n_bands=8, window_frames=250)
        for time_steps in [100, 250, 500]:
            x = torch.randn(2, 8, time_steps)
            output = model(x)
            assert output.shape == (2, time_steps)

