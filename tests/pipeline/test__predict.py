"""Tests for callcut.pipeline._predict module."""

from __future__ import annotations

from pathlib import Path

import pytest

from callcut.evaluation import HysteresisDecoder
from callcut.evaluation._types import Interval
from callcut.extractors import SNRExtractor
from callcut.nn import TinySegCNN
from callcut.pipeline._predict import predict_recordings
from callcut.pipeline._types import RecordingPrediction


class TestPredictRecordings:
    """Tests for predict_recordings."""

    def test_empty_paths_returns_empty(self) -> None:
        """Test that empty audio_paths returns an empty list."""
        model = TinySegCNN(n_bands=8, window_frames=250)
        extractor = SNRExtractor(sample_rate=32000)
        decoder = HysteresisDecoder()
        result = predict_recordings(model, extractor, [], decoder)
        assert result == []

    def test_returns_list_of_predictions(self, audio_files: list[Path]) -> None:
        """Test end-to-end prediction returns RecordingPrediction list."""
        extractor = SNRExtractor(sample_rate=32000)
        model = TinySegCNN(n_bands=extractor.n_features, window_frames=250, base=8)
        decoder = HysteresisDecoder()
        results = predict_recordings(model, extractor, audio_files[:1], decoder)
        assert len(results) == 1
        assert isinstance(results[0], RecordingPrediction)

    def test_prediction_fields(self, audio_files: list[Path]) -> None:
        """Test that prediction results have expected fields."""
        extractor = SNRExtractor(sample_rate=32000)
        model = TinySegCNN(n_bands=extractor.n_features, window_frames=250, base=8)
        decoder = HysteresisDecoder()
        results = predict_recordings(model, extractor, audio_files[:1], decoder)
        pred = results[0]
        assert pred.audio_path.exists()
        assert isinstance(pred.intervals, tuple)
        for iv in pred.intervals:
            assert isinstance(iv, Interval)
        assert pred.duration_s > 0

    def test_multiple_files(self, audio_files: list[Path]) -> None:
        """Test prediction on multiple files."""
        extractor = SNRExtractor(sample_rate=32000)
        model = TinySegCNN(n_bands=extractor.n_features, window_frames=250, base=8)
        decoder = HysteresisDecoder()
        n = min(2, len(audio_files))
        results = predict_recordings(model, extractor, audio_files[:n], decoder)
        assert len(results) == n

    def test_nonexistent_path_raises(self) -> None:
        """Test that a nonexistent audio path raises."""
        model = TinySegCNN(n_bands=8, window_frames=250)
        extractor = SNRExtractor(sample_rate=32000)
        decoder = HysteresisDecoder()
        with pytest.raises((FileNotFoundError, RuntimeError)):
            predict_recordings(
                model, extractor, [Path("/nonexistent/audio.wav")], decoder
            )

    def test_accepts_string_paths(self, audio_files: list[Path]) -> None:
        """Test that string paths are accepted."""
        extractor = SNRExtractor(sample_rate=32000)
        model = TinySegCNN(n_bands=extractor.n_features, window_frames=250, base=8)
        decoder = HysteresisDecoder()
        results = predict_recordings(model, extractor, [str(audio_files[0])], decoder)
        assert len(results) == 1
