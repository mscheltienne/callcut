"""Normalization functions for feature tensors."""

from __future__ import annotations

from torch import Tensor

from callcut.utils._checks import check_type


def robust_normalize(features: Tensor, *, eps: float = 1e-12) -> Tensor:
    r"""Apply robust z-score normalization per band.

    Normalizes each band (row) independently using the median and median absolute
    deviation (MAD), which are robust to outliers. This is preferred over mean/std
    normalization for audio features that may contain transient events.

    Parameters
    ----------
    features : Tensor
        Input features of shape ``(n_bands, time)``.
    eps : float
        Small constant added to MAD for numerical stability.

    Returns
    -------
    normalized : Tensor
        Normalized features of the same shape. Each band has approximately zero median
        and unit scale (when measured by MAD).

    Notes
    -----
    The normalization formula is:

    .. math::

        z = \\frac{x - \\text{median}(x)}{1.4826 \\cdot \\text{MAD}(x)}

    where MAD is the median absolute deviation. The constant 1.4826 is the consistency
    constant that makes MAD asymptotically equivalent to standard deviation for normally
    distributed data.

    Examples
    --------
    >>> features = torch.randn(8, 100) * 10 + 5  # shifted and scaled
    >>> normalized = robust_normalize(features)
    >>> normalized.median(dim=1).values  # approximately zero
    tensor([...])
    """
    check_type(features, (Tensor,), "features")
    check_type(eps, ("numeric",), "eps")

    if features.dim() != 2:
        raise ValueError(
            f"Argument 'features' must be 2D (n_bands, time), got {features.dim()}D."
        )

    # median per band: (n_bands, 1)
    med = features.median(dim=1, keepdim=True).values

    # MAD (median absolute deviation) per band
    mad = (features - med).abs().median(dim=1, keepdim=True).values + eps

    # robust z-score with consistency constant for normal distribution
    return (features - med) / (1.4826 * mad)
