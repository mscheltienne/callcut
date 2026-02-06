"""Tests for callcut.pipeline._inference module."""

from __future__ import annotations

from pathlib import Path

import torch

from callcut.evaluation import HysteresisDecoder
from callcut.evaluation._types import Interval
from callcut.extractors import SNRExtractor
from callcut.nn import TinySegCNN
from callcut.pipeline._inference import _infer_recording


class TestInferRecording:
    """Tests for _infer_recording."""

    def test_returns_tuple_of_three(self, assets_path: Path) -> None:
        """Test that _infer_recording returns (probabilities, times, intervals)."""
        audio_path = sorted(assets_path.glob("*.wav"))[0]
        extractor = SNRExtractor(sample_rate=32000)
        model = TinySegCNN(n_bands=extractor.n_features, window_frames=250, base=8)
        decoder = HysteresisDecoder()
        device = torch.device("cpu")

        result = _infer_recording(
            model, extractor, decoder, audio_path, device, hop_frames=None
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_probabilities_shape_and_device(self, assets_path: Path) -> None:
        """Test probabilities are 1D CPU tensors in [0, 1]."""
        audio_path = sorted(assets_path.glob("*.wav"))[0]
        extractor = SNRExtractor(sample_rate=32000)
        model = TinySegCNN(n_bands=extractor.n_features, window_frames=250, base=8)
        decoder = HysteresisDecoder()
        device = torch.device("cpu")

        probs, times, _ = _infer_recording(
            model, extractor, decoder, audio_path, device, hop_frames=None
        )
        assert probs.dim() == 1
        assert probs.device == torch.device("cpu")
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_times_shape(self, assets_path: Path) -> None:
        """Test times tensor matches probabilities length."""
        audio_path = sorted(assets_path.glob("*.wav"))[0]
        extractor = SNRExtractor(sample_rate=32000)
        model = TinySegCNN(n_bands=extractor.n_features, window_frames=250, base=8)
        decoder = HysteresisDecoder()
        device = torch.device("cpu")

        probs, times, _ = _infer_recording(
            model, extractor, decoder, audio_path, device, hop_frames=None
        )
        assert times.dim() == 1
        assert times.shape == probs.shape

    def test_intervals_are_list(self, assets_path: Path) -> None:
        """Test that decoded intervals are a list of Interval."""
        audio_path = sorted(assets_path.glob("*.wav"))[0]
        extractor = SNRExtractor(sample_rate=32000)
        model = TinySegCNN(n_bands=extractor.n_features, window_frames=250, base=8)
        decoder = HysteresisDecoder()
        device = torch.device("cpu")

        _, _, intervals = _infer_recording(
            model, extractor, decoder, audio_path, device, hop_frames=None
        )
        assert isinstance(intervals, list)
        for iv in intervals:
            assert isinstance(iv, Interval)
