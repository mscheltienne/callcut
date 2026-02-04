"""Tests for callcut.nn._registry module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from torch import Tensor

from callcut.nn import (
    BaseDetector,
    get_model,
    list_models,
    register_model,
    unregister_model,
)

if TYPE_CHECKING:
    from collections.abc import Generator


class _DummyModel(BaseDetector):
    """A minimal model for testing registration."""

    @property
    def receptive_field(self) -> int:
        return 1

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1)

    def _save_kwargs(self) -> dict:
        return {}


@pytest.fixture
def registered_dummy_model() -> Generator[type[_DummyModel]]:
    """Register a dummy model for testing and unregister on teardown."""
    register_model(_DummyModel)
    yield _DummyModel
    unregister_model("_DummyModel")


class TestListModels:
    """Tests for list_models function."""

    def test_contains_tinysegcnn(self) -> None:
        """Test that TinySegCNN is in the registry."""
        assert "TinySegCNN" in list_models()


class TestGetModel:
    """Tests for get_model function."""

    def test_basic(self) -> None:
        """Test getting a registered model."""
        model = get_model("TinySegCNN", n_bands=8, window_frames=250)
        assert isinstance(model, BaseDetector)
        assert model.n_bands == 8
        assert model.window_frames == 250

    def test_with_kwargs(self) -> None:
        """Test passing additional kwargs."""
        model = get_model("TinySegCNN", n_bands=4, window_frames=100, base=16)
        assert model.n_bands == 4
        assert model.window_frames == 100
        assert model.base == 16

    def test_invalid_name(self) -> None:
        """Test error on unknown model name."""
        with pytest.raises(ValueError, match="'name'"):
            get_model("NonExistentModel")

    def test_custom_model(self, registered_dummy_model: type[_DummyModel]) -> None:
        """Test getting a custom registered model."""
        model = get_model("_DummyModel", n_bands=4, window_frames=100)
        assert isinstance(model, registered_dummy_model)
        assert model.n_bands == 4
        assert model.window_frames == 100


class TestRegisterUnregister:
    """Tests for register_model and unregister_model functions."""

    def test_registered_in_list(
        self, registered_dummy_model: type[_DummyModel]
    ) -> None:
        """Test that registered model appears in list_models."""
        assert registered_dummy_model.__name__ in list_models()

    def test_duplicate_registration(
        self, registered_dummy_model: type[_DummyModel]
    ) -> None:
        """Test error on duplicate registration."""
        with pytest.raises(ValueError, match="already registered"):
            register_model(registered_dummy_model)

    def test_unregister_invalid(self) -> None:
        """Test error when unregistering unknown model."""
        with pytest.raises(ValueError, match="'name'"):
            unregister_model("NonExistentModel")
