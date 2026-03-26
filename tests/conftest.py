from __future__ import annotations

from pathlib import Path

import pytest
import torch

from callcut.extractors import BaseExtractor
from callcut.utils._fixes import _preload_nvidia_npp
from callcut.utils.logs import logger


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest options."""
    logger.propagate = True  # setup logging
    _preload_nvidia_npp()


class DummyExtractor(BaseExtractor):
    """Minimal extractor for testing."""

    def __init__(
        self, sample_rate: int = 32000, hop_ms: float = 8.0, n_features: int = 8
    ) -> None:
        super().__init__(sample_rate)
        self._hop_ms = hop_ms
        self._n_features = n_features

    @property
    def n_features(self) -> int:
        """Number of output features."""
        return self._n_features

    @property
    def hop_ms(self) -> float:
        """Hop length in milliseconds."""
        return self._hop_ms

    def __hash__(self) -> int:
        """Hash based on configuration."""
        return hash((self._sample_rate, self._hop_ms, self._n_features))

    def extract(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract random features for testing."""
        n_samples = waveform.shape[-1]
        hop_samples = int(self._hop_ms * self._sample_rate / 1000)
        n_frames = n_samples // hop_samples
        features = torch.randn(self._n_features, n_frames)
        times = torch.arange(n_frames) * (self._hop_ms / 1000.0)
        return features, times

    def _save_config(self) -> dict:
        return {
            "sample_rate": self._sample_rate,
            "hop_ms": self._hop_ms,
            "n_features": self._n_features,
        }


@pytest.fixture(scope="session")
def assets_path() -> Path:
    """Path to test assets directory."""
    return Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def all_audio_files(assets_path: Path) -> list[Path]:
    """All audio files in assets directory."""
    return sorted(assets_path.glob("*.wav"))
