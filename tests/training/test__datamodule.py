"""Tests for callcut.training._datamodule module."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from callcut.extractors import SNRExtractor
from callcut.io import RecordingInfo
from callcut.training._datamodule import CallDataModule, _split_by_windows

from tests.conftest import DummyExtractor

pytestmark = pytest.mark.filterwarnings("ignore:.*pin_memory.*:UserWarning")


@pytest.fixture(scope="module")
def extractor() -> SNRExtractor:
    """Return an SNR extractor for testing."""
    return SNRExtractor(sample_rate=32000, hop_ms=8.0, n_bands=8)


@pytest.fixture(scope="module")
def dummy_extractor() -> DummyExtractor:
    """Return a dummy extractor for unit tests."""
    return DummyExtractor(sample_rate=32000, hop_ms=8.0, n_features=8)


class TestSplitByWindows:
    """Tests for _split_by_windows helper function."""

    def test_returns_three_lists(
        self, all_audio_files: list[Path], dummy_extractor: DummyExtractor
    ) -> None:
        """Test that _split_by_windows returns three lists."""
        from callcut.io import scan_recordings

        recordings = scan_recordings(all_audio_files)
        train, val, test = _split_by_windows(
            recordings,
            dummy_extractor,
            window_s=2.0,
            window_hop_s=0.5,
            train_frac=0.7,
            val_frac=0.15,
        )
        assert isinstance(train, list)
        assert isinstance(val, list)
        assert isinstance(test, list)

    def test_all_recordings_assigned(
        self, all_audio_files: list[Path], dummy_extractor: DummyExtractor
    ) -> None:
        """Test that all recordings are assigned to exactly one split."""
        from callcut.io import scan_recordings

        recordings = scan_recordings(all_audio_files)
        train, val, test = _split_by_windows(
            recordings,
            dummy_extractor,
            window_s=2.0,
            window_hop_s=0.5,
            train_frac=0.7,
            val_frac=0.15,
        )
        total = len(train) + len(val) + len(test)
        assert total == len(recordings)

    def test_no_duplicates(
        self, all_audio_files: list[Path], dummy_extractor: DummyExtractor
    ) -> None:
        """Test that no recording appears in multiple splits."""
        from callcut.io import scan_recordings

        recordings = scan_recordings(all_audio_files)
        train, val, test = _split_by_windows(
            recordings,
            dummy_extractor,
            window_s=2.0,
            window_hop_s=0.5,
            train_frac=0.7,
            val_frac=0.15,
        )
        train_paths = {r.audio_path for r in train}
        val_paths = {r.audio_path for r in val}
        test_paths = {r.audio_path for r in test}

        assert train_paths.isdisjoint(val_paths)
        assert train_paths.isdisjoint(test_paths)
        assert val_paths.isdisjoint(test_paths)

    def test_train_not_empty(
        self, all_audio_files: list[Path], dummy_extractor: DummyExtractor
    ) -> None:
        """Test that train split is not empty."""
        from callcut.io import scan_recordings

        recordings = scan_recordings(all_audio_files)
        train, _, _ = _split_by_windows(
            recordings,
            dummy_extractor,
            window_s=2.0,
            window_hop_s=0.5,
            train_frac=0.7,
            val_frac=0.15,
        )
        assert len(train) > 0

    def test_raises_on_no_valid_recordings(
        self, dummy_extractor: DummyExtractor
    ) -> None:
        """Test that error is raised when no recordings have valid windows."""
        # Create recordings that are too short
        recordings = [
            RecordingInfo(
                audio_path=Path("/fake/short.wav"),
                annotation_path=Path("/fake/short_annotations.csv"),
                duration_s=0.5,  # Too short for 2s window
                n_annotations=1,
            )
        ]
        with pytest.raises(ValueError, match="No recordings have enough frames"):
            _split_by_windows(
                recordings,
                dummy_extractor,
                window_s=2.0,
                window_hop_s=0.5,
                train_frac=0.7,
                val_frac=0.15,
            )


class TestCallDataModule:
    """Tests for CallDataModule class."""

    def test_instantiation(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test that CallDataModule can be instantiated."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            batch_size=2,
            num_workers=0,
        )
        assert dm is not None

    def test_n_recordings(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test n_recordings property."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            batch_size=2,
            num_workers=0,
        )
        assert dm.n_recordings == len(all_audio_files)

    def test_extractor_property(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test extractor property."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            batch_size=2,
            num_workers=0,
        )
        assert dm.extractor is extractor

    def test_repr(self, all_audio_files: list[Path], extractor: SNRExtractor) -> None:
        """Test string representation."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            batch_size=2,
            num_workers=0,
        )
        repr_str = repr(dm)
        assert "CallDataModule" in repr_str
        assert "n_recordings=" in repr_str
        assert "batch_size=2" in repr_str

    def test_invalid_fractions_sum(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test that fractions must sum to 1."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            CallDataModule(
                recordings=all_audio_files,
                extractor=extractor,
                train_frac=0.5,
                val_frac=0.3,
                test_frac=0.3,  # Sum = 1.1
                batch_size=2,
                num_workers=0,
            )

    def test_invalid_train_frac_zero(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test that train_frac must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            CallDataModule(
                recordings=all_audio_files,
                extractor=extractor,
                train_frac=0.0,
                val_frac=0.5,
                test_frac=0.5,
                batch_size=2,
                num_workers=0,
            )

    def test_invalid_batch_size(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test that batch_size must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            CallDataModule(
                recordings=all_audio_files,
                extractor=extractor,
                batch_size=0,
                num_workers=0,
            )

    def test_invalid_num_workers_negative(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test that num_workers must be non-negative."""
        with pytest.raises(ValueError, match="must be non-negative"):
            CallDataModule(
                recordings=all_audio_files,
                extractor=extractor,
                batch_size=2,
                num_workers=-1,
            )

    def test_no_valid_recordings(self, tmp_path: Path, extractor: SNRExtractor) -> None:
        """Test error when no valid recordings found."""
        # Create a file that exists but has no annotations
        fake_wav = tmp_path / "no_annotations.wav"
        fake_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        with pytest.raises(ValueError, match="No valid recordings"):
            CallDataModule(
                recordings=[fake_wav],
                extractor=extractor,
                batch_size=2,
                num_workers=0,
            )

    def test_setup_fit(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test setup for fit stage."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            batch_size=2,
            num_workers=0,
        )
        dm.setup("fit")

        assert dm.train_dataset is not None
        assert len(dm.train_dataset) > 0

    def test_setup_test_raises(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test that setup('test') raises ValueError."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            batch_size=2,
            num_workers=0,
        )
        with pytest.raises(ValueError, match="Only stage='fit' is supported"):
            dm.setup("test")

    def test_train_dataloader(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test train_dataloader method."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            batch_size=2,
            num_workers=0,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 2

    def test_train_dataloader_without_setup(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test that train_dataloader raises error without setup."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            batch_size=2,
            num_workers=0,
        )
        with pytest.raises(RuntimeError, match="not initialized"):
            dm.train_dataloader()

    def test_val_dataloader(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test val_dataloader method."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            batch_size=2,
            num_workers=0,
        )
        dm.setup("fit")
        loader = dm.val_dataloader()

        # May be None if no val recordings
        if loader is not None:
            assert isinstance(loader, DataLoader)

    def test_datasets_properties(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test dataset properties before and after setup."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            batch_size=2,
            num_workers=0,
        )

        # Before setup
        assert dm.train_dataset is None
        assert dm.val_dataset is None

        # After setup
        dm.setup("fit")
        assert dm.train_dataset is not None

    def test_batch_content(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test that batches contain expected data."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            window_s=2.0,
            window_hop_s=0.5,
            batch_size=2,
            num_workers=0,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()

        batch = next(iter(loader))
        X, y = batch

        # Check shapes
        assert X.ndim == 3  # (batch, n_bands, time)
        assert y.ndim == 2  # (batch, time)
        assert X.shape[0] == 2  # batch_size
        assert X.shape[1] == extractor.n_features
        assert y.shape[0] == 2
        assert X.shape[2] == y.shape[1]  # same time dimension

    def test_custom_window_params(
        self, all_audio_files: list[Path], extractor: SNRExtractor
    ) -> None:
        """Test with custom window parameters."""
        dm = CallDataModule(
            recordings=all_audio_files,
            extractor=extractor,
            window_s=1.0,
            window_hop_s=0.25,
            batch_size=2,
            num_workers=0,
        )
        dm.setup("fit")

        assert dm.train_dataset is not None
        assert dm.train_dataset.window_s == 1.0
        assert dm.train_dataset.window_hop_s == 0.25
