"""Lightning DataModule for call detection training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from callcut.io import CallDataset, RecordingInfo, scan_recordings
from callcut.utils._checks import check_type, ensure_int
from callcut.utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path

    from callcut.extractors import BaseExtractor


def _split_by_windows(
    recordings: list[RecordingInfo],
    extractor: BaseExtractor,
    window_s: float,
    window_hop_s: float,
    train_frac: float,
    val_frac: float,
) -> tuple[list[RecordingInfo], list[RecordingInfo], list[RecordingInfo]]:
    """Split recordings into train/val/test sets balanced by window count.

    This ensures each split has approximately the target fraction of total
    training windows, rather than the target fraction of files.

    Parameters
    ----------
    recordings : list of RecordingInfo
        Recordings to split.
    extractor : BaseExtractor
        Feature extractor (for window estimation).
    window_s : float
        Window length in seconds.
    window_hop_s : float
        Window hop in seconds.
    train_frac : float
        Target fraction of windows for training.
    val_frac : float
        Target fraction of windows for validation.

    Returns
    -------
    train : list of RecordingInfo
        Training recordings.
    val : list of RecordingInfo
        Validation recordings.
    test : list of RecordingInfo
        Test recordings.

    Notes
    -----
    Shuffling uses the legacy NumPy random API, which is seeded by
    ``lightning.seed_everything()``. Call it before this function for
    reproducible splits.
    """
    # Compute window counts for each recording
    window_counts = [
        (rec, rec.estimate_windows(extractor, window_s, window_hop_s))
        for rec in recordings
    ]

    # Filter out recordings with no windows
    window_counts = [(rec, n) for rec, n in window_counts if n > 0]

    if len(window_counts) == 0:
        raise ValueError("No recordings have enough frames for the window size.")

    total_windows = sum(n for _, n in window_counts)

    # Shuffle recordings (uses legacy np.random, seeded by L.seed_everything)
    np.random.shuffle(window_counts)

    # Greedily assign recordings to splits
    target_train = train_frac * total_windows
    target_val = val_frac * total_windows

    train_recs: list[RecordingInfo] = []
    val_recs: list[RecordingInfo] = []
    test_recs: list[RecordingInfo] = []

    train_windows = 0
    val_windows = 0

    for rec, n_windows in window_counts:
        if train_windows < target_train:
            train_recs.append(rec)
            train_windows += n_windows
        elif val_windows < target_val:
            val_recs.append(rec)
            val_windows += n_windows
        else:
            test_recs.append(rec)

    # Log the split statistics
    test_windows = total_windows - train_windows - val_windows
    logger.info(
        "Split by windows: train=%d recs (%d windows, %.1f%%), "
        "val=%d recs (%d windows, %.1f%%), "
        "test=%d recs (%d windows, %.1f%%)",
        len(train_recs),
        train_windows,
        100 * train_windows / total_windows,
        len(val_recs),
        val_windows,
        100 * val_windows / total_windows if total_windows > 0 else 0,
        len(test_recs),
        test_windows,
        100 * test_windows / total_windows if total_windows > 0 else 0,
    )

    return train_recs, val_recs, test_recs


class CallDataModule(LightningDataModule):
    """Lightning DataModule for call detection.

    Handles data loading, train/val/test splitting (balanced by window count),
    and DataLoader creation for training and validation. The test split is
    exposed via :attr:`test_recordings` for use with
    :func:`~callcut.pipeline.evaluate_recordings`.

    Parameters
    ----------
    recordings : list of Path | str
        Paths to audio files. Each audio file should have a corresponding
        annotation CSV file with the same stem and ``_annotations.csv`` suffix.
    extractor : BaseExtractor
        Feature extractor instance.
    train_frac : float
        Target fraction of windows for training (default: 0.7).
    val_frac : float
        Target fraction of windows for validation (default: 0.1).
    test_frac : float
        Target fraction of windows for testing (default: 0.2).
    window_s : float
        Window length in seconds for each sample.
    window_hop_s : float
        Hop between consecutive windows in seconds.
    batch_size : int
        Batch size for DataLoaders.
    num_workers : int
        Number of workers for DataLoaders.

    Notes
    -----
    The splitting is done at the **recording level**, balanced by **window
    count** rather than file count. This ensures each split contains
    approximately the target fraction of training samples, even when
    recordings have different durations.

    For reproducible splits, call ``lightning.seed_everything(seed)`` before
    instantiating and calling :meth:`setup`.

    Only the ``"fit"`` stage is supported for :meth:`setup`. For evaluation on
    the held-out test split, use :func:`~callcut.pipeline.evaluate_recordings`
    with :attr:`test_recordings`.

    Examples
    --------
    >>> from pathlib import Path
    >>> from callcut.extractors import SNRExtractor
    >>> from callcut.training import CallDataModule
    >>> from callcut.pipeline import evaluate_recordings
    >>> import lightning as L
    >>>
    >>> extractor = SNRExtractor(sample_rate=32000, hop_ms=8.0, n_bands=8)
    >>> recordings = list(Path("data/").glob("*.wav"))
    >>> dm = CallDataModule(
    ...     recordings=recordings,
    ...     extractor=extractor,
    ...     batch_size=32,
    ... )
    >>> trainer = L.Trainer(max_epochs=10)
    >>> trainer.fit(model, datamodule=dm)
    >>>
    >>> # Evaluate on held-out test recordings
    >>> report = evaluate_recordings(
    ...     model, extractor, dm.test_recordings, decoder, matcher
    ... )
    """

    def __init__(
        self,
        recordings: list[Path | str],
        extractor: BaseExtractor,
        *,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        test_frac: float = 0.2,
        window_s: float = 2.0,
        window_hop_s: float = 0.5,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()

        # Validate inputs
        check_type(train_frac, ("numeric",), "train_frac")
        check_type(val_frac, ("numeric",), "val_frac")
        check_type(test_frac, ("numeric",), "test_frac")
        check_type(window_s, ("numeric",), "window_s")
        check_type(window_hop_s, ("numeric",), "window_hop_s")
        batch_size = ensure_int(batch_size, "batch_size")
        num_workers = ensure_int(num_workers, "num_workers")

        total_frac = train_frac + val_frac + test_frac
        if abs(total_frac - 1.0) > 0.01:
            raise ValueError(
                f"Split fractions must sum to 1.0, got {total_frac:.3f} "
                f"(train={train_frac}, val={val_frac}, test={test_frac})."
            )
        if train_frac <= 0:
            raise ValueError(
                f"Argument 'train_frac' must be positive, got {train_frac}."
            )
        if batch_size <= 0:
            raise ValueError(
                f"Argument 'batch_size' must be positive, got {batch_size}."
            )
        if num_workers < 0:
            raise ValueError(
                f"Argument 'num_workers' must be non-negative, got {num_workers}."
            )

        # Scan recordings once
        all_recordings = scan_recordings(recordings)
        if len(all_recordings) == 0:
            raise ValueError(
                "No valid recordings found. Ensure audio files have matching "
                "*_annotations.csv files with valid annotations."
            )

        self._all_recordings = all_recordings
        self._extractor = extractor
        self._train_frac = float(train_frac)
        self._val_frac = float(val_frac)
        self._test_frac = float(test_frac)
        self._window_s = float(window_s)
        self._window_hop_s = float(window_hop_s)
        self._batch_size = batch_size
        self._num_workers = num_workers

        # Will be set in setup()
        self._train_recordings: list[RecordingInfo] = []
        self._val_recordings: list[RecordingInfo] = []
        self._test_recordings: list[RecordingInfo] = []
        self._train_dataset: CallDataset | None = None
        self._val_dataset: CallDataset | None = None

        logger.info(
            "CallDataModule initialized: %d recordings, batch_size=%d",
            len(all_recordings),
            batch_size,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets for the fit stage.

        Parameters
        ----------
        stage : str
            Must be ``"fit"``. Other stages are not supported; use
            :func:`~callcut.pipeline.evaluate_recordings` for evaluation on
            held-out test recordings (available via :attr:`test_recordings`).
        """
        if stage != "fit":
            raise ValueError(
                f"Only stage='fit' is supported, got {stage!r}. "
                "Use evaluate_recordings() with test_recordings for evaluation."
            )

        # Split recordings if not already done
        if len(self._train_recordings) == 0:
            self._train_recordings, self._val_recordings, self._test_recordings = (
                _split_by_windows(
                    self._all_recordings,
                    self._extractor,
                    self._window_s,
                    self._window_hop_s,
                    self._train_frac,
                    self._val_frac,
                )
            )

        self._train_dataset = CallDataset(
            recordings=self._train_recordings,
            extractor=self._extractor,
            window_s=self._window_s,
            window_hop_s=self._window_hop_s,
        )
        if len(self._val_recordings) > 0:
            self._val_dataset = CallDataset(
                recordings=self._val_recordings,
                extractor=self._extractor,
                window_s=self._window_s,
                window_hop_s=self._window_hop_s,
            )
        logger.info(
            "Setup fit: train=%d windows, val=%d windows",
            len(self._train_dataset) if self._train_dataset else 0,
            len(self._val_dataset) if self._val_dataset else 0,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        if self._train_dataset is None:
            raise RuntimeError(
                "train_dataset not initialized. Call setup('fit') first."
            )
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Return the validation DataLoader."""
        if self._val_dataset is None:
            return None
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
        )

    @property
    def train_dataset(self) -> CallDataset | None:
        """Training dataset (available after setup).

        :type: :class:`~callcut.io.CallDataset` | None
        """
        return self._train_dataset

    @property
    def val_dataset(self) -> CallDataset | None:
        """Validation dataset (available after setup).

        :type: :class:`~callcut.io.CallDataset` | None
        """
        return self._val_dataset

    @property
    def train_recordings(self) -> list[RecordingInfo]:
        """Training recordings (available after setup or split).

        :type: :class:`list` of :class:`~callcut.io.RecordingInfo`
        """
        return self._train_recordings

    @property
    def val_recordings(self) -> list[RecordingInfo]:
        """Validation recordings (available after setup or split).

        :type: :class:`list` of :class:`~callcut.io.RecordingInfo`
        """
        return self._val_recordings

    @property
    def test_recordings(self) -> list[RecordingInfo]:
        """Test recordings (available after setup or split).

        :type: :class:`list` of :class:`~callcut.io.RecordingInfo`
        """
        return self._test_recordings

    @property
    def n_recordings(self) -> int:
        """Total number of valid recordings.

        :type: :class:`int`
        """
        return len(self._all_recordings)

    @property
    def extractor(self) -> BaseExtractor:
        """Feature extractor used by this data module.

        :type: :class:`~callcut.extractors.BaseExtractor`
        """
        return self._extractor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_recordings={self.n_recordings}, "
            f"batch_size={self._batch_size}, "
            f"train_frac={self._train_frac}, "
            f"val_frac={self._val_frac}, "
            f"test_frac={self._test_frac})"
        )
