"""Tests for callcut.pipeline._save_load module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from numpy.testing import assert_allclose

from callcut.evaluation import BaseDecoder, HysteresisDecoder
from callcut.extractors import BaseExtractor, SNRExtractor
from callcut.nn import BaseDetector, TinySegCNN
from callcut.pipeline._save_load import load_pipeline, save_pipeline


@pytest.fixture()
def pipeline_components() -> tuple[TinySegCNN, SNRExtractor, HysteresisDecoder]:
    """Create a model, extractor, and decoder for testing."""
    extractor = SNRExtractor(sample_rate=32000, hop_ms=8.0, n_bands=8)
    model = TinySegCNN(n_bands=8, window_frames=250, base=16)
    decoder = HysteresisDecoder(
        enter_threshold=0.7, exit_threshold=0.3, min_duration_s=0.05
    )
    return model, extractor, decoder


class TestSavePipeline:
    """Tests for save_pipeline."""

    def test_creates_file(
        self,
        pipeline_components: tuple[TinySegCNN, SNRExtractor, HysteresisDecoder],
        tmp_path: Path,
    ) -> None:
        """Test that save_pipeline creates a file."""
        model, extractor, decoder = pipeline_components
        fname = tmp_path / "pipeline.pt"
        save_pipeline(model, extractor, decoder, fname)
        assert fname.exists()

    def test_overwrite_false_raises(
        self,
        pipeline_components: tuple[TinySegCNN, SNRExtractor, HysteresisDecoder],
        tmp_path: Path,
    ) -> None:
        """Test that saving to existing file raises without overwrite."""
        model, extractor, decoder = pipeline_components
        fname = tmp_path / "pipeline.pt"
        save_pipeline(model, extractor, decoder, fname)
        with pytest.raises(FileExistsError):
            save_pipeline(model, extractor, decoder, fname)

    def test_overwrite_true(
        self,
        pipeline_components: tuple[TinySegCNN, SNRExtractor, HysteresisDecoder],
        tmp_path: Path,
    ) -> None:
        """Test that overwrite=True replaces the file."""
        model, extractor, decoder = pipeline_components
        fname = tmp_path / "pipeline.pt"
        save_pipeline(model, extractor, decoder, fname)
        save_pipeline(model, extractor, decoder, fname, overwrite=True)
        assert fname.exists()

    def test_checkpoint_structure(
        self,
        pipeline_components: tuple[TinySegCNN, SNRExtractor, HysteresisDecoder],
        tmp_path: Path,
    ) -> None:
        """Test that saved checkpoint has expected structure."""
        model, extractor, decoder = pipeline_components
        fname = tmp_path / "pipeline.pt"
        save_pipeline(model, extractor, decoder, fname)
        checkpoint = torch.load(fname, weights_only=False)
        assert "model" in checkpoint
        assert "extractor" in checkpoint
        assert "decoder" in checkpoint
        assert checkpoint["model"]["class_name"] == "TinySegCNN"
        assert checkpoint["extractor"]["class_name"] == "SNRExtractor"
        assert checkpoint["decoder"]["class_name"] == "HysteresisDecoder"
        assert "state_dict" in checkpoint["model"]
        assert "config" in checkpoint["model"]
        assert "config" in checkpoint["extractor"]
        assert "config" in checkpoint["decoder"]

    def test_accepts_string_path(
        self,
        pipeline_components: tuple[TinySegCNN, SNRExtractor, HysteresisDecoder],
        tmp_path: Path,
    ) -> None:
        """Test that string paths are accepted."""
        model, extractor, decoder = pipeline_components
        fname = str(tmp_path / "pipeline.pt")
        save_pipeline(model, extractor, decoder, fname)
        assert Path(fname).exists()


class TestLoadPipeline:
    """Tests for load_pipeline."""

    def test_roundtrip(
        self,
        pipeline_components: tuple[TinySegCNN, SNRExtractor, HysteresisDecoder],
        tmp_path: Path,
    ) -> None:
        """Test save then load reconstructs components."""
        model, extractor, decoder = pipeline_components
        fname = tmp_path / "pipeline.pt"
        save_pipeline(model, extractor, decoder, fname)

        loaded_model, loaded_ext, loaded_dec = load_pipeline(fname, device="cpu")
        assert isinstance(loaded_model, BaseDetector)
        assert isinstance(loaded_ext, BaseExtractor)
        assert isinstance(loaded_dec, BaseDecoder)

    def test_model_type_preserved(
        self,
        pipeline_components: tuple[TinySegCNN, SNRExtractor, HysteresisDecoder],
        tmp_path: Path,
    ) -> None:
        """Test that model class is preserved."""
        model, extractor, decoder = pipeline_components
        fname = tmp_path / "pipeline.pt"
        save_pipeline(model, extractor, decoder, fname)

        loaded_model, _, _ = load_pipeline(fname, device="cpu")
        assert isinstance(loaded_model, TinySegCNN)
        assert loaded_model.n_bands == model.n_bands
        assert loaded_model.window_frames == model.window_frames
        assert loaded_model.base == model.base

    def test_extractor_config_preserved(
        self,
        pipeline_components: tuple[TinySegCNN, SNRExtractor, HysteresisDecoder],
        tmp_path: Path,
    ) -> None:
        """Test that extractor config is preserved."""
        model, extractor, decoder = pipeline_components
        fname = tmp_path / "pipeline.pt"
        save_pipeline(model, extractor, decoder, fname)

        _, loaded_ext, _ = load_pipeline(fname, device="cpu")
        assert isinstance(loaded_ext, SNRExtractor)
        assert loaded_ext.sample_rate == extractor.sample_rate

    def test_decoder_config_preserved(
        self,
        pipeline_components: tuple[TinySegCNN, SNRExtractor, HysteresisDecoder],
        tmp_path: Path,
    ) -> None:
        """Test that decoder config is preserved."""
        model, extractor, decoder = pipeline_components
        fname = tmp_path / "pipeline.pt"
        save_pipeline(model, extractor, decoder, fname)

        _, _, loaded_dec = load_pipeline(fname, device="cpu")
        assert isinstance(loaded_dec, HysteresisDecoder)
        assert_allclose(loaded_dec.enter_threshold, decoder.enter_threshold)
        assert_allclose(loaded_dec.exit_threshold, decoder.exit_threshold)
        assert_allclose(loaded_dec.min_duration_s, decoder.min_duration_s)

    def test_weights_preserved(
        self,
        pipeline_components: tuple[TinySegCNN, SNRExtractor, HysteresisDecoder],
        tmp_path: Path,
    ) -> None:
        """Test that model weights are preserved through save/load."""
        model, extractor, decoder = pipeline_components
        fname = tmp_path / "pipeline.pt"
        save_pipeline(model, extractor, decoder, fname)

        loaded_model, _, _ = load_pipeline(fname, device="cpu")
        for (name, param), (_, loaded_param) in zip(
            model.named_parameters(), loaded_model.named_parameters(), strict=True
        ):
            assert torch.equal(param, loaded_param), f"Mismatch in {name}"

    def test_nonexistent_file_raises(self) -> None:
        """Test that loading a nonexistent file raises."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_pipeline("/nonexistent/pipeline.pt", device="cpu")

    def test_returns_three_tuple(
        self,
        pipeline_components: tuple[TinySegCNN, SNRExtractor, HysteresisDecoder],
        tmp_path: Path,
    ) -> None:
        """Test that load_pipeline returns a 3-tuple."""
        model, extractor, decoder = pipeline_components
        fname = tmp_path / "pipeline.pt"
        save_pipeline(model, extractor, decoder, fname)

        result = load_pipeline(fname, device="cpu")
        assert isinstance(result, tuple)
        assert len(result) == 3
