"""Loss functions for call detection training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from callcut.utils._checks import check_type

if TYPE_CHECKING:
    from torch import Tensor


class BaseLoss(ABC, nn.Module):
    """Abstract base class for training loss functions.

    Subclasses must implement :meth:`forward` to compute the loss between
    model logits and ground truth labels.

    All loss functions expect:

    - logits: Raw model output of shape ``(batch, time)`` or ``(batch,)``
    - targets: Binary labels of shape ``(batch, time)`` or ``(batch,)``,
      values in ``[0, 1]``

    Examples
    --------
    Create a custom loss by subclassing:

    >>> class MyLoss(BaseLoss):
    ...     def __init__(self, weight: float = 1.0):
    ...         super().__init__()
    ...         self._weight = weight
    ...
    ...     def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
    ...         # Custom loss computation
    ...         ...
    """

    @abstractmethod
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the loss.

        Parameters
        ----------
        logits : Tensor
            Raw model output (before sigmoid).
        targets : Tensor
            Ground truth binary labels.

        Returns
        -------
        loss : Tensor
            Scalar loss value.
        """


class BCEWithLogitsLoss(BaseLoss):
    """Binary cross-entropy loss with logits.

    Wraps :class:`torch.nn.BCEWithLogitsLoss` with the BaseLoss interface.

    Parameters
    ----------
    pos_weight : float | None
        Weight for the positive class. If ``> 1``, increases recall; if ``< 1``,
        increases precision. Can be computed using
        :meth:`~callcut.io.CallDataset.compute_pos_weight`.

    Examples
    --------
    >>> loss_fn = BCEWithLogitsLoss(pos_weight=2.0)
    >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        check_type(pos_weight, ("numeric", None), "pos_weight")
        if pos_weight is not None and pos_weight <= 0:
            raise ValueError(
                f"Argument 'pos_weight' must be positive, got {pos_weight}."
            )
        self._pos_weight = float(pos_weight) if pos_weight is not None else None
        # Internal PyTorch loss (created lazily for device compatibility)
        self._loss: nn.BCEWithLogitsLoss | None = None

    @property
    def pos_weight(self) -> float | None:
        """Positive class weight.

        :type: :class:`float` | None
        """
        return self._pos_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute binary cross-entropy loss."""
        if self._loss is None:
            if self._pos_weight is not None:
                pw = torch.tensor(
                    [self._pos_weight], dtype=logits.dtype, device=logits.device
                )
                self._loss = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                self._loss = nn.BCEWithLogitsLoss()
        return self._loss(logits, targets)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pos_weight={self._pos_weight})"


class FocalLoss(BaseLoss):
    r"""Focal loss for handling class imbalance.

    Down-weights easy examples to focus training on hard negatives. Particularly
    useful when positive (call) frames are rare.

    The focal loss is defined as:

    .. math::

        FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)

    where :math:`p_t` is the model's estimated probability for the correct class.

    Parameters
    ----------
    alpha : float
        Weighting factor for the positive class, in ``[0, 1]``.
        Use ``alpha > 0.5`` to weight positives more heavily.
    gamma : float
        Focusing parameter. ``gamma = 0`` recovers standard cross-entropy.
        ``gamma > 0`` reduces the loss for well-classified examples, focusing
        on hard examples. Typical values are 1.0 to 5.0.

    Examples
    --------
    >>> loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
    >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        check_type(alpha, ("numeric",), "alpha")
        check_type(gamma, ("numeric",), "gamma")
        if not 0 <= alpha <= 1:
            raise ValueError(f"Argument 'alpha' must be in [0, 1], got {alpha}.")
        if gamma < 0:
            raise ValueError(f"Argument 'gamma' must be >= 0, got {gamma}.")
        self._alpha = float(alpha)
        self._gamma = float(gamma)

    @property
    def alpha(self) -> float:
        """Weighting factor for positive class.

        :type: :class:`float`
        """
        return self._alpha

    @property
    def gamma(self) -> float:
        """Focusing parameter.

        :type: :class:`float`
        """
        return self._gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss."""
        probs = torch.sigmoid(logits)

        # p_t = p if y=1, else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self._gamma

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self._alpha * targets + (1 - self._alpha) * (1 - targets)

        # BCE loss (computed manually for numerical stability)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Apply focal weighting
        loss = alpha_t * focal_weight * bce
        return loss.mean()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self._alpha}, gamma={self._gamma})"


class DiceLoss(BaseLoss):
    r"""Dice loss for optimizing overlap directly.

    Minimizes 1 - Dice coefficient, where Dice measures the overlap between
    predictions and targets. Effective for segmentation tasks with imbalanced
    classes.

    The Dice coefficient is:

    .. math::

        Dice = \\frac{2 |P \\cap T|}{|P| + |T|}

    Parameters
    ----------
    smooth : float
        Smoothing factor for numerical stability. Prevents division by zero
        when both prediction and target are empty.

    Examples
    --------
    >>> loss_fn = DiceLoss(smooth=1.0)
    >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        check_type(smooth, ("numeric",), "smooth")
        if smooth < 0:
            raise ValueError(f"Argument 'smooth' must be >= 0, got {smooth}.")
        self._smooth = float(smooth)

    @property
    def smooth(self) -> float:
        """Smoothing factor.

        :type: :class:`float`
        """
        return self._smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute Dice loss."""
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)

        # Compute Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self._smooth) / (union + self._smooth)
        return 1.0 - dice

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(smooth={self._smooth})"


class TverskyLoss(BaseLoss):
    r"""Tversky loss with adjustable false positive/negative penalties.

    Generalization of Dice loss that allows separate weighting of false
    positives and false negatives. Useful when recall is more important
    than precision (or vice versa).

    The Tversky index is:

    .. math::

        TI = \\frac{TP}{TP + \\alpha \\cdot FP + \\beta \\cdot FN}

    Parameters
    ----------
    alpha : float
        Weight for false positives. Higher values penalize FP more.
    beta : float
        Weight for false negatives. Higher values penalize FN more.
        Use ``beta > alpha`` to favor recall over precision.
    smooth : float
        Smoothing factor for numerical stability.

    Notes
    -----
    - ``alpha = beta = 0.5`` recovers Dice loss
    - ``alpha = beta = 1.0`` recovers Tanimoto coefficient
    - ``beta > alpha`` favors recall (fewer missed calls)
    - ``alpha > beta`` favors precision (fewer false alarms)

    Examples
    --------
    >>> # Favor recall (fewer missed calls)
    >>> loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
    >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0
    ) -> None:
        super().__init__()
        check_type(alpha, ("numeric",), "alpha")
        check_type(beta, ("numeric",), "beta")
        check_type(smooth, ("numeric",), "smooth")
        if alpha < 0:
            raise ValueError(f"Argument 'alpha' must be >= 0, got {alpha}.")
        if beta < 0:
            raise ValueError(f"Argument 'beta' must be >= 0, got {beta}.")
        if smooth < 0:
            raise ValueError(f"Argument 'smooth' must be >= 0, got {smooth}.")
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._smooth = float(smooth)

    @property
    def alpha(self) -> float:
        """False positive weight.

        :type: :class:`float`
        """
        return self._alpha

    @property
    def beta(self) -> float:
        """False negative weight.

        :type: :class:`float`
        """
        return self._beta

    @property
    def smooth(self) -> float:
        """Smoothing factor.

        :type: :class:`float`
        """
        return self._smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute Tversky loss."""
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)

        # Compute components
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1 - targets_flat)).sum()
        fn = ((1 - probs_flat) * targets_flat).sum()

        # Tversky index
        tversky = (tp + self._smooth) / (
            tp + self._alpha * fp + self._beta * fn + self._smooth
        )
        return 1.0 - tversky

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"alpha={self._alpha}, beta={self._beta}, smooth={self._smooth})"
        )
