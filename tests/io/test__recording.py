"""Tests for callcut.io._recording module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from callcut.features import BaseExtractor
from callcut.io import RecordingInfo, scan_recordings


class _DummyExtractor(BaseExtractor):
    """Minimal extractor for testing."""

    def __init__(
        self, sample_rate: int = 32000, hop_ms: float = 8.0, n_features: int = 8
    ) -> None:
        super().__init__(sample_rate)
        self._hop_ms = hop_ms
        self._n_features = n_features

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def hop_ms(self) -> float:
        return self._hop_ms

    def __hash__(self) -> int:
        return hash((self._sample_rate, self._hop_ms, self._n_features))

    def extract(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_samples = waveform.shape[-1]
        hop_samples = int(self._hop_ms * self._sample_rate / 1000)
        n_frames = n_samples // hop_samples
        features = torch.randn(self._n_features, n_frames)
        times = torch.arange(n_frames) * (self._hop_ms / 1000.0)
        return features, times


@pytest.fixture(scope="module")
def assets_path() -> Path:
    """Path to test assets directory."""
    return Path(__file__).parents[1] / "assets"


@pytest.fixture(scope="module")
def all_audio_files(assets_path: Path) -> list[Path]:
    """All audio files in assets directory."""
    return sorted(assets_path.glob("*.wav"))


@pytest.fixture(scope="module")
def extractor() -> _DummyExtractor:
    """Dummy extractor for testing."""  # noqa: D401
    return _DummyExtractor(sample_rate=32000, hop_ms=8.0, n_features=8)


@pytest.fixture(scope="module")
def valid_recordings(all_audio_files: list[Path]) -> list[RecordingInfo]:
    """Valid recordings from scan_recordings."""  # noqa: D401
    return scan_recordings(all_audio_files)


class TestRecordingInfo:
    """Tests for RecordingInfo dataclass."""

    def test_frozen(self, valid_recordings: list[RecordingInfo]) -> None:
        """Test that RecordingInfo is immutable (frozen dataclass)."""
        recording = valid_recordings[0]
        with pytest.raises(AttributeError):
            recording.duration_s = 100.0

    def test_attributes(self, valid_recordings: list[RecordingInfo]) -> None:
        """Test that RecordingInfo has expected attributes."""
        recording = valid_recordings[0]
        assert isinstance(recording.audio_path, Path)
        assert isinstance(recording.annotation_path, Path)
        assert isinstance(recording.duration_s, float)
        assert isinstance(recording.n_annotations, int)
        assert recording.duration_s > 0
        assert recording.n_annotations > 0

    def test_repr(self, valid_recordings: list[RecordingInfo]) -> None:
        """Test string representation."""
        recording = valid_recordings[0]
        repr_str = repr(recording)
        assert "RecordingInfo" in repr_str
        assert recording.audio_path.name in repr_str
        assert "duration=" in repr_str
        assert "n_annotations=" in repr_str

    def test_estimate_frames(
        self, valid_recordings: list[RecordingInfo], extractor: _DummyExtractor
    ) -> None:
        """Test estimate_frames method."""
        recording = valid_recordings[0]
        n_frames = recording.estimate_frames(extractor)
        assert isinstance(n_frames, int)
        assert n_frames > 0
        # Should be roughly duration / hop_s
        expected = int(round(recording.duration_s / extractor.hop_s))
        assert abs(n_frames - expected) <= 1

    def test_estimate_windows(
        self, valid_recordings: list[RecordingInfo], extractor: _DummyExtractor
    ) -> None:
        """Test estimate_windows method."""
        recording = valid_recordings[0]
        n_windows = recording.estimate_windows(
            extractor, window_s=2.0, window_hop_s=0.5
        )
        assert isinstance(n_windows, int)
        assert n_windows > 0

    def test_estimate_windows_too_short(self, assets_path: Path) -> None:
        """Test estimate_windows returns 0 for recordings shorter than window."""
        # Create a mock RecordingInfo with very short duration
        recording = RecordingInfo(
            audio_path=assets_path / "test.wav",
            annotation_path=assets_path / "test_annotations.csv",
            duration_s=0.5,  # 500ms
            n_annotations=1,
        )
        extractor = _DummyExtractor(sample_rate=32000, hop_ms=8.0)
        n_windows = recording.estimate_windows(
            extractor, window_s=2.0, window_hop_s=0.5
        )
        assert n_windows == 0


class TestScanRecordings:
    """Tests for scan_recordings function."""

    def test_returns_list(self, all_audio_files: list[Path]) -> None:
        """Test that scan_recordings returns a list."""
        result = scan_recordings(all_audio_files)
        assert isinstance(result, list)

    def test_returns_recording_info(self, all_audio_files: list[Path]) -> None:
        """Test that all items are RecordingInfo instances."""
        result = scan_recordings(all_audio_files)
        for item in result:
            assert isinstance(item, RecordingInfo)

    def test_expected_count(self, all_audio_files: list[Path]) -> None:
        """Test that all valid recordings are found."""
        result = scan_recordings(all_audio_files)
        assert len(result) == len(all_audio_files)

    def test_sorted_by_path(self, all_audio_files: list[Path]) -> None:
        """Test that results are sorted by audio path."""
        result = scan_recordings(all_audio_files)
        paths = [r.audio_path for r in result]
        assert paths == sorted(paths)

    def test_annotations_match(self, all_audio_files: list[Path]) -> None:
        """Test that annotation paths follow naming convention."""
        result = scan_recordings(all_audio_files)
        for recording in result:
            expected_name = recording.audio_path.stem + "_annotations.csv"
            assert recording.annotation_path.name == expected_name

    def test_string_paths(self, assets_path: Path) -> None:
        """Test that string paths are accepted."""
        files = [str(p) for p in assets_path.glob("*.wav")]
        result = scan_recordings(files)
        assert len(result) > 0

    def test_missing_annotations_skipped(
        self, assets_path: Path, tmp_path: Path
    ) -> None:
        """Test that files without annotations are skipped."""
        # Create a temporary wav file without annotations in tmp_path
        temp_wav = tmp_path / "no_annotations.wav"
        temp_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        # Combine real files with the temp file that has no annotations
        all_files = list(assets_path.glob("*.wav")) + [temp_wav]
        result = scan_recordings(all_files)

        # The temp file should be skipped (no annotations)
        assert all(r.audio_path != temp_wav for r in result), (
            "File without annotations should be skipped"
        )

    def test_empty_list(self) -> None:
        """Test that empty list returns empty result."""
        result = scan_recordings([])
        assert result == []

    def test_nonexistent_path(self) -> None:
        """Test that nonexistent paths raise error."""
        with pytest.raises(FileNotFoundError):
            scan_recordings([Path("/nonexistent/path.wav")])

    def test_positive_durations(self, valid_recordings: list[RecordingInfo]) -> None:
        """Test that all durations are positive."""
        for recording in valid_recordings:
            assert recording.duration_s > 0

    def test_positive_annotations(self, valid_recordings: list[RecordingInfo]) -> None:
        """Test that all recordings have at least one annotation."""
        for recording in valid_recordings:
            assert recording.n_annotations > 0

    def test_total_duration(self, valid_recordings: list[RecordingInfo]) -> None:
        """Test that total duration is reasonable."""
        total_duration = sum(r.duration_s for r in valid_recordings)
        # Our test files are small, total should be < 100 seconds
        assert 0 < total_duration < 100
