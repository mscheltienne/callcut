"""Tests for callcut.training._module module."""

from __future__ import annotations

import pytest
import torch

from callcut.nn import TinySegCNN
from callcut.training import (
    BaseLoss,
    BCEWithLogitsLoss,
    CallDetectorModule,
    DiceLoss,
    FocalLoss,
    TverskyLoss,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:You are trying to `self.log\\(\\):UserWarning"
)


class TestCallDetectorModuleInit:
    """Tests for CallDetectorModule initialization."""

    def test_basic_instantiation(self) -> None:
        """Test basic instantiation with required parameters."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)
        assert module is not None

    def test_model_property(self) -> None:
        """Test model property returns the underlying model."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)
        assert module.model is model

    def test_loss_property(self) -> None:
        """Test loss property returns the loss function."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)
        assert module.loss is loss

    def test_invalid_model_type(self) -> None:
        """Test that invalid model type raises error."""
        loss = BCEWithLogitsLoss()
        with pytest.raises(TypeError):
            CallDetectorModule("not a model", loss=loss, lr=1e-3)  # type: ignore[arg-type]

    def test_invalid_loss_type(self) -> None:
        """Test that invalid loss type raises error."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        with pytest.raises(TypeError):
            CallDetectorModule(model, loss="not a loss", lr=1e-3)  # type: ignore[arg-type]

    def test_invalid_lr_type(self) -> None:
        """Test that invalid learning rate type raises error."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        with pytest.raises(TypeError):
            CallDetectorModule(model, loss=loss, lr="not a number")  # type: ignore[arg-type]

    def test_invalid_lr_negative(self) -> None:
        """Test that negative learning rate raises error."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        with pytest.raises(ValueError, match="must be positive"):
            CallDetectorModule(model, loss=loss, lr=-1e-3)

    def test_invalid_lr_zero(self) -> None:
        """Test that zero learning rate raises error."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        with pytest.raises(ValueError, match="must be positive"):
            CallDetectorModule(model, loss=loss, lr=0.0)

    def test_default_lr(self) -> None:
        """Test that default learning rate is 1e-3."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss)
        assert module._lr == 1e-3

    def test_repr(self) -> None:
        """Test string representation."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)
        repr_str = repr(module)
        assert "CallDetectorModule" in repr_str
        assert "TinySegCNN" in repr_str
        assert "lr=" in repr_str


class TestCallDetectorModuleLossFunctions:
    """Tests for CallDetectorModule with different loss functions."""

    @pytest.mark.parametrize(
        ("loss_cls", "kwargs"),
        [
            (BCEWithLogitsLoss, {}),
            (BCEWithLogitsLoss, {"pos_weight": 2.0}),
            (FocalLoss, {}),
            (FocalLoss, {"alpha": 0.5, "gamma": 1.0}),
            (DiceLoss, {}),
            (DiceLoss, {"smooth": 0.5}),
            (TverskyLoss, {}),
            (TverskyLoss, {"alpha": 0.3, "beta": 0.7}),
        ],
    )
    def test_with_various_losses(self, loss_cls: type[BaseLoss], kwargs: dict) -> None:
        """Test instantiation with various loss functions."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = loss_cls(**kwargs)
        module = CallDetectorModule(model, loss=loss, lr=1e-3)
        assert module.loss is loss


class TestCallDetectorModuleForward:
    """Tests for CallDetectorModule forward pass."""

    def test_forward_shape(self) -> None:
        """Test that forward produces correct output shape."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)

        x = torch.randn(4, 8, 100)
        output = module(x)

        assert output.shape == (4, 100)

    def test_forward_returns_logits(self) -> None:
        """Test that forward returns logits (same as underlying model)."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)

        x = torch.randn(4, 8, 100)
        module_output = module(x)
        model_output = model(x)

        # Module forward should return raw logits, same as model
        torch.testing.assert_close(module_output, model_output)


class TestCallDetectorModuleTrainingStep:
    """Tests for CallDetectorModule training_step."""

    def test_training_step_returns_loss(self) -> None:
        """Test that training_step returns a scalar loss."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)

        X = torch.randn(4, 8, 100)
        y = torch.randint(0, 2, (4, 100)).float()

        result = module.training_step((X, y), 0)

        assert isinstance(result, torch.Tensor)
        assert result.shape == ()  # Scalar
        assert result.item() >= 0

    def test_training_step_gradient_flow(self) -> None:
        """Test that gradients flow through training_step."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)

        X = torch.randn(4, 8, 100)
        y = torch.randint(0, 2, (4, 100)).float()

        result = module.training_step((X, y), 0)
        result.backward()

        # Check that at least one parameter has gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad


class TestCallDetectorModuleValidationStep:
    """Tests for CallDetectorModule validation_step."""

    def test_validation_step_returns_none(self) -> None:
        """Test that validation_step returns None."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)

        X = torch.randn(4, 8, 100)
        y = torch.randint(0, 2, (4, 100)).float()

        result = module.validation_step((X, y), 0)

        assert result is None

    def test_validation_step_no_gradient(self) -> None:
        """Test that validation_step does not compute gradients."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)

        X = torch.randn(4, 8, 100)
        y = torch.randint(0, 2, (4, 100)).float()

        # Clear any existing gradients
        module.zero_grad()

        with torch.no_grad():
            module.validation_step((X, y), 0)

        # No gradients should have been computed
        for param in model.parameters():
            assert param.grad is None or (param.grad == 0).all()


class TestCallDetectorModuleOptimizer:
    """Tests for CallDetectorModule optimizer configuration."""

    def test_configure_optimizers_returns_adam(self) -> None:
        """Test that configure_optimizers returns Adam optimizer."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)

        optimizer = module.configure_optimizers()

        assert isinstance(optimizer, torch.optim.Adam)

    def test_optimizer_lr(self) -> None:
        """Test that optimizer has correct learning rate."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        lr = 5e-4
        module = CallDetectorModule(model, loss=loss, lr=lr)

        optimizer = module.configure_optimizers()

        # Check learning rate in first param group
        assert optimizer.param_groups[0]["lr"] == lr

    def test_optimizer_parameters(self) -> None:
        """Test that optimizer has correct parameters."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)

        optimizer = module.configure_optimizers()

        # Get parameter ids from optimizer
        opt_param_ids = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }

        # Get parameter ids from module
        module_param_ids = {id(p) for p in module.parameters()}

        assert opt_param_ids == module_param_ids


class TestCallDetectorModuleHyperparameters:
    """Tests for CallDetectorModule hyperparameter saving."""

    def test_hparams_saved(self) -> None:
        """Test that hyperparameters are saved."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)

        # Lightning saves hparams in self.hparams
        assert hasattr(module, "hparams")

    def test_model_not_in_hparams(self) -> None:
        """Test that model is excluded from hparams."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)

        assert "model" not in module.hparams

    def test_loss_not_in_hparams(self) -> None:
        """Test that loss is excluded from hparams."""
        model = TinySegCNN(n_bands=8, window_frames=100)
        loss = BCEWithLogitsLoss()
        module = CallDetectorModule(model, loss=loss, lr=1e-3)

        assert "loss" not in module.hparams
