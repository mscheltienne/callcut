"""Tests for callcut.training._losses module."""

from __future__ import annotations

import pytest
import torch

from callcut.training import (
    BaseLoss,
    BCEWithLogitsLoss,
    DiceLoss,
    FocalLoss,
    TverskyLoss,
)


class TestBaseLoss:
    """Tests for BaseLoss abstract class."""

    def test_cannot_instantiate(self) -> None:
        """Test that BaseLoss cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseLoss()

    def test_subclass_must_implement_forward(self) -> None:
        """Test that subclasses must implement forward."""

        class IncompleteLoss(BaseLoss):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteLoss()


class TestBCEWithLogitsLoss:
    """Tests for BCEWithLogitsLoss."""

    def test_default_init(self) -> None:
        """Test construction with default parameters."""
        loss = BCEWithLogitsLoss()
        assert loss.pos_weight is None

    def test_with_pos_weight(self) -> None:
        """Test construction with pos_weight."""
        loss = BCEWithLogitsLoss(pos_weight=2.0)
        assert loss.pos_weight == 2.0

    def test_invalid_pos_weight_negative(self) -> None:
        """Test that negative pos_weight raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            BCEWithLogitsLoss(pos_weight=-1.0)

    def test_invalid_pos_weight_zero(self) -> None:
        """Test that zero pos_weight raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            BCEWithLogitsLoss(pos_weight=0.0)

    def test_invalid_pos_weight_type(self) -> None:
        """Test that invalid pos_weight type raises error."""
        with pytest.raises(TypeError):
            BCEWithLogitsLoss(pos_weight="invalid")

    def test_forward_shape(self) -> None:
        """Test forward produces scalar output."""
        loss = BCEWithLogitsLoss()
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 2, (4, 100)).float()
        result = loss(logits, targets)
        assert result.shape == ()
        assert result.item() > 0

    def test_forward_with_pos_weight(self) -> None:
        """Test forward with pos_weight produces valid output."""
        loss = BCEWithLogitsLoss(pos_weight=2.0)
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 2, (4, 100)).float()
        result = loss(logits, targets)
        assert result.shape == ()
        assert result.item() > 0

    def test_repr(self) -> None:
        """Test string representation."""
        loss = BCEWithLogitsLoss(pos_weight=2.0)
        assert "BCEWithLogitsLoss" in repr(loss)
        assert "pos_weight=2.0" in repr(loss)


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_default_init(self) -> None:
        """Test construction with default parameters."""
        loss = FocalLoss()
        assert loss.alpha == 0.25
        assert loss.gamma == 2.0

    def test_custom_params(self) -> None:
        """Test construction with custom parameters."""
        loss = FocalLoss(alpha=0.75, gamma=1.5)
        assert loss.alpha == 0.75
        assert loss.gamma == 1.5

    def test_invalid_alpha_below_zero(self) -> None:
        """Test that alpha below 0 raises error."""
        with pytest.raises(ValueError, match="must be in"):
            FocalLoss(alpha=-0.1)

    def test_invalid_alpha_above_one(self) -> None:
        """Test that alpha above 1 raises error."""
        with pytest.raises(ValueError, match="must be in"):
            FocalLoss(alpha=1.1)

    def test_invalid_gamma_negative(self) -> None:
        """Test that negative gamma raises error."""
        with pytest.raises(ValueError, match="must be >= 0"):
            FocalLoss(gamma=-1.0)

    def test_forward_shape(self) -> None:
        """Test forward produces scalar output."""
        loss = FocalLoss()
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 2, (4, 100)).float()
        result = loss(logits, targets)
        assert result.shape == ()
        assert result.item() >= 0

    def test_gamma_zero_similar_to_bce(self) -> None:
        """Test that gamma=0 produces similar results to BCE (up to alpha weighting)."""
        focal = FocalLoss(alpha=0.5, gamma=0.0)
        bce = BCEWithLogitsLoss()

        torch.manual_seed(42)
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 2, (4, 100)).float()

        focal_result = focal(logits, targets)
        bce_result = bce(logits, targets)

        # With alpha=0.5 and gamma=0, focal loss is 0.5 * BCE
        assert torch.allclose(focal_result, bce_result * 0.5, atol=0.01)

    def test_repr(self) -> None:
        """Test string representation."""
        loss = FocalLoss(alpha=0.75, gamma=2.0)
        assert "FocalLoss" in repr(loss)
        assert "alpha=0.75" in repr(loss)
        assert "gamma=2.0" in repr(loss)


class TestDiceLoss:
    """Tests for DiceLoss."""

    def test_default_init(self) -> None:
        """Test construction with default parameters."""
        loss = DiceLoss()
        assert loss.smooth == 1.0

    def test_custom_smooth(self) -> None:
        """Test construction with custom smooth parameter."""
        loss = DiceLoss(smooth=0.5)
        assert loss.smooth == 0.5

    def test_invalid_smooth_negative(self) -> None:
        """Test that negative smooth raises error."""
        with pytest.raises(ValueError, match="must be >= 0"):
            DiceLoss(smooth=-1.0)

    def test_forward_shape(self) -> None:
        """Test forward produces scalar output."""
        loss = DiceLoss()
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 2, (4, 100)).float()
        result = loss(logits, targets)
        assert result.shape == ()
        assert 0 <= result.item() <= 1

    def test_perfect_prediction_low_loss(self) -> None:
        """Test that perfect prediction produces low loss."""
        loss = DiceLoss()
        # High logits -> probability near 1
        logits = torch.full((4, 100), 10.0)
        targets = torch.ones(4, 100)
        result = loss(logits, targets)
        assert result.item() < 0.1

    def test_worst_prediction_high_loss(self) -> None:
        """Test that worst prediction produces high loss."""
        loss = DiceLoss(smooth=0.0)
        # Low logits -> probability near 0
        logits = torch.full((4, 100), -10.0)
        targets = torch.ones(4, 100)
        result = loss(logits, targets)
        assert result.item() > 0.9

    def test_repr(self) -> None:
        """Test string representation."""
        loss = DiceLoss(smooth=1.0)
        assert "DiceLoss" in repr(loss)
        assert "smooth=1.0" in repr(loss)


class TestTverskyLoss:
    """Tests for TverskyLoss."""

    def test_default_init(self) -> None:
        """Test construction with default parameters."""
        loss = TverskyLoss()
        assert loss.alpha == 0.5
        assert loss.beta == 0.5
        assert loss.smooth == 1.0

    def test_custom_params(self) -> None:
        """Test construction with custom parameters."""
        loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=0.5)
        assert loss.alpha == 0.3
        assert loss.beta == 0.7
        assert loss.smooth == 0.5

    def test_invalid_alpha_negative(self) -> None:
        """Test that negative alpha raises error."""
        with pytest.raises(ValueError, match="must be >= 0"):
            TverskyLoss(alpha=-0.1)

    def test_invalid_beta_negative(self) -> None:
        """Test that negative beta raises error."""
        with pytest.raises(ValueError, match="must be >= 0"):
            TverskyLoss(beta=-0.1)

    def test_invalid_smooth_negative(self) -> None:
        """Test that negative smooth raises error."""
        with pytest.raises(ValueError, match="must be >= 0"):
            TverskyLoss(smooth=-0.1)

    def test_forward_shape(self) -> None:
        """Test forward produces scalar output."""
        loss = TverskyLoss()
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 2, (4, 100)).float()
        result = loss(logits, targets)
        assert result.shape == ()
        assert 0 <= result.item() <= 1

    def test_alpha_beta_half_similar_to_dice(self) -> None:
        """Test that alpha=beta=0.5 produces similar result to Dice loss.

        Note: The formulas differ slightly in how smooth is applied, so we use
        smooth=0 for an exact match or a loose tolerance with smooth>0.
        """
        # With smooth=0, should be mathematically equivalent
        tversky = TverskyLoss(alpha=0.5, beta=0.5, smooth=0.0)
        dice = DiceLoss(smooth=0.0)

        torch.manual_seed(42)
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 2, (4, 100)).float()

        tversky_result = tversky(logits, targets)
        dice_result = dice(logits, targets)

        assert torch.allclose(tversky_result, dice_result, atol=1e-5)

    def test_repr(self) -> None:
        """Test string representation."""
        loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=1.0)
        assert "TverskyLoss" in repr(loss)
        assert "alpha=0.3" in repr(loss)
        assert "beta=0.7" in repr(loss)
        assert "smooth=1.0" in repr(loss)


class TestGradientFlow:
    """Tests for gradient flow through loss functions."""

    @pytest.mark.parametrize(
        ("loss_cls", "kwargs"),
        [
            (BCEWithLogitsLoss, {}),
            (BCEWithLogitsLoss, {"pos_weight": 2.0}),
            (FocalLoss, {}),
            (FocalLoss, {"alpha": 0.75, "gamma": 1.0}),
            (DiceLoss, {}),
            (DiceLoss, {"smooth": 0.5}),
            (TverskyLoss, {}),
            (TverskyLoss, {"alpha": 0.3, "beta": 0.7}),
        ],
    )
    def test_gradient_flow(self, loss_cls: type[BaseLoss], kwargs: dict) -> None:
        """Test that gradients flow through the loss function."""
        loss = loss_cls(**kwargs)
        logits = torch.randn(4, 100, requires_grad=True)
        targets = torch.randint(0, 2, (4, 100)).float()

        result = loss(logits, targets)
        result.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert not torch.isinf(logits.grad).any()
