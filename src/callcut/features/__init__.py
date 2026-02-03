"""Feature extraction module for audio signals.

This module provides functions for extracting multi-band SNR (signal-to-noise ratio)
features from audio waveforms. The features are designed for animal call detection
and are computed using PyTorch for GPU acceleration.
"""

from callcut.features._bands import compute_band_edges
from callcut.features._normalize import robust_normalize
from callcut.features._snr import compute_snr_features
