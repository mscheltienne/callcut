"""Frequency band computation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from callcut.utils._checks import check_type, ensure_int

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_band_edges(
    low_hz: float, high_hz: float, n_bands: int
) -> NDArray[np.floating]:
    """Compute logarithmically-spaced frequency band edges.

    Generates frequency band boundaries that are evenly spaced on a logarithmic
    scale, which better matches human auditory perception and is commonly used
    in audio analysis.

    Parameters
    ----------
    low_hz : float
        Lower frequency bound in Hz. Must be positive.
    high_hz : float
        Upper frequency bound in Hz. Must be greater than ``low_hz``.
    n_bands : int
        Number of frequency bands to create. Must be at least 1.

    Returns
    -------
    bands : array of shape ``(n_bands, 2)``
        Array where each row contains ``[low_edge, high_edge]`` for a frequency
        band, in Hz. Bands are contiguous (each band's high edge equals the next
        band's low edge).

    Examples
    --------
    Create 4 bands spanning 300 Hz to 10 kHz:

    >>> bands = compute_band_edges(300, 10000, 4)
    >>> bands.shape
    (4, 2)
    >>> bands[0]  # first band
    array([ 300.        ,  547.72255751])

    The bands are logarithmically spaced:

    >>> ratios = bands[:, 1] / bands[:, 0]
    >>> np.allclose(ratios, ratios[0])  # all ratios are equal
    True
    """
    check_type(low_hz, ("numeric",), "low_hz")
    check_type(high_hz, ("numeric",), "high_hz")
    n_bands = ensure_int(n_bands, "n_bands")

    if low_hz <= 0:
        raise ValueError(f"Argument 'low_hz' must be positive, got {low_hz}.")
    if high_hz <= low_hz:
        raise ValueError(
            f"Argument 'high_hz' must be greater than 'low_hz'. "
            f"Got low_hz={low_hz}, high_hz={high_hz}."
        )
    if n_bands < 1:
        raise ValueError(f"Argument 'n_bands' must be at least 1, got {n_bands}.")

    # compute log-spaced edges (n_bands + 1 edges for n_bands bands)
    edges = np.geomspace(low_hz, high_hz, n_bands + 1)

    # create (n_bands, 2) array with [low, high] per band
    bands = np.column_stack([edges[:-1], edges[1:]])

    return bands
