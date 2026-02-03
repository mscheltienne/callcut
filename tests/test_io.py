"""Tests for callcut.io module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from callcut.io import load_annotations, load_audio

_SAMPLES = [
    "or60yw70_280912_1616.1941",
    "gy6or6_baseline_250312_1456.1276",
]


@pytest.fixture(scope="module")
def assets_path() -> Path:
    """Path to test assets directory."""
    return Path(__file__).parent / "assets"


@pytest.fixture(scope="module", params=_SAMPLES)
def audio_file(request: pytest.FixtureRequest, assets_path: Path) -> Path:
    """Test audio file, parametrized over all samples."""
    return assets_path / f"{request.param}.wav"


@pytest.fixture(scope="module", params=_SAMPLES)
def annotations_file(request: pytest.FixtureRequest, assets_path: Path) -> Path:
    """Test annotations file, parametrized over all samples."""
    return assets_path / f"{request.param}_annotations.csv"


@pytest.fixture(scope="module", params=_SAMPLES)
def sample_pair(request: pytest.FixtureRequest, assets_path: Path) -> tuple[Path, Path]:
    """Paired audio and annotations files, parametrized over all samples."""
    name = request.param
    return assets_path / f"{name}.wav", assets_path / f"{name}_annotations.csv"


class TestLoadAudio:
    """Tests for load_audio function."""

    def test_basic(self, audio_file: Path) -> None:
        """Test basic audio loading."""
        waveform, sr = load_audio(audio_file)
        assert isinstance(waveform, torch.Tensor)
        assert isinstance(sr, int)
        assert sr > 0
        assert waveform.ndim == 2
        assert waveform.shape[0] == 1  # mono by default

    def test_preserves_sample_rate(self, audio_file: Path) -> None:
        """Test that sample rate is preserved when not specified."""
        _, sr = load_audio(audio_file)
        assert sr == 32000  # expected sample rate for this dataset

    def test_resampling(self, audio_file: Path) -> None:
        """Test audio resampling."""
        waveform_original, sr_original = load_audio(audio_file)
        waveform_resampled, sr_resampled = load_audio(audio_file, sample_rate=16000)

        assert sr_resampled == 16000
        assert sr_original == 32000
        # resampled should have roughly half the samples
        ratio = waveform_original.shape[1] / waveform_resampled.shape[1]
        assert 1.9 < ratio < 2.1

    def test_device(self, audio_file: Path) -> None:
        """Test loading to specific device."""
        waveform, _ = load_audio(audio_file, device="cpu")
        assert waveform.device == torch.device("cpu")

    def test_file_not_found(self) -> None:
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_audio("nonexistent_file.wav")

    def test_invalid_sample_rate(self, audio_file: Path) -> None:
        """Test error on invalid sample rate."""
        with pytest.raises(ValueError, match="positive integer"):
            load_audio(audio_file, sample_rate=-1)

        with pytest.raises(ValueError, match="positive integer"):
            load_audio(audio_file, sample_rate=0)

    def test_normalized(self, audio_file: Path) -> None:
        """Test that audio values are normalized."""
        waveform, _ = load_audio(audio_file)
        assert waveform.min() >= -1.0
        assert waveform.max() <= 1.0


class TestLoadAnnotations:
    """Tests for load_annotations function."""

    def test_basic(self, annotations_file: Path) -> None:
        """Test basic annotation loading."""
        intervals = load_annotations(annotations_file)
        assert isinstance(intervals, np.ndarray)
        assert intervals.ndim == 2
        assert intervals.shape[1] == 2
        assert len(intervals) > 0

    def test_values_in_seconds(self, annotations_file: Path) -> None:
        """Test that values are converted to seconds."""
        intervals = load_annotations(annotations_file)
        # values should be in seconds (small numbers), not ms (large numbers)
        assert intervals.max() < 20  # should be seconds, not ms

    def test_within_audio_duration(self, sample_pair: tuple[Path, Path]) -> None:
        """Test that annotations are within audio duration."""
        audio_file, annotations_file = sample_pair
        waveform, sr = load_audio(audio_file)
        duration_s = waveform.shape[1] / sr
        intervals = load_annotations(annotations_file)
        assert intervals.max() <= duration_s

    def test_sorted_by_start(self, annotations_file: Path) -> None:
        """Test that intervals are sorted by start time."""
        intervals = load_annotations(annotations_file)
        starts = intervals[:, 0]
        assert np.all(starts[:-1] <= starts[1:])

    def test_valid_intervals(self, annotations_file: Path) -> None:
        """Test that all intervals have stop > start."""
        intervals = load_annotations(annotations_file)
        assert np.all(intervals[:, 1] > intervals[:, 0])

    def test_file_not_found(self) -> None:
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_annotations("nonexistent_file.csv")
