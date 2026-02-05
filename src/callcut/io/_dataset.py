"""PyTorch Dataset for call detection training."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset

from callcut.io._labels import intervals_to_frame_labels
from callcut.io._loader import load_annotations, load_audio
from callcut.io._recording import RecordingInfo
from callcut.utils._checks import check_type, ensure_int
from callcut.utils.logs import logger, warn

if TYPE_CHECKING:
    from callcut.features import BaseExtractor


@lru_cache(maxsize=400)
def _load_recording(
    audio_path: Path,
    annotation_path: Path,
    extractor: BaseExtractor,
) -> tuple[Tensor, Tensor]:
    """Load audio and extract features with LRU caching.

    Parameters
    ----------
    audio_path : Path
        Path to audio file.
    annotation_path : Path
        Path to annotation CSV file.
    extractor : BaseExtractor
        Feature extractor (must be hashable).

    Returns
    -------
    features : Tensor
        Extracted features of shape ``(n_features, n_frames)``.
    labels : Tensor
        Per-frame binary labels of shape ``(n_frames,)``.
    """
    waveform, _ = load_audio(audio_path, sample_rate=extractor.sample_rate, mono=True)
    features, times = extractor(waveform)
    intervals = load_annotations(annotation_path)
    labels = intervals_to_frame_labels(intervals, times)
    return features, labels


class CallDataset(Dataset):
    """PyTorch Dataset for frame-level call detection.

    Provides windowed samples of features with per-frame binary labels for training
    call detection models.

    Parameters
    ----------
    recordings : list of RecordingInfo
        Recording metadata from :func:`~callcut.io.scan_recordings`.
    extractor : BaseExtractor
        Feature extractor instance.
    window_s : float
        Window length in seconds for each sample.
    window_hop_s : float
        Hop between consecutive windows in seconds.

    Attributes
    ----------
    n_recordings : int
        Number of recordings in the dataset.
    n_windows : int
        Total number of windows (samples) across all recordings.
    window_frames : int
        Window length in frames.
    window_hop_frames : int
        Window hop in frames.

    Notes
    -----
    **Lazy loading**: Features are computed on-demand in ``__getitem__`` rather than
    pre-loaded into memory. Results are cached via LRU cache.

    Examples
    --------
    >>> from pathlib import Path
    >>> from callcut.features import SNRExtractor
    >>> from callcut.io import CallDataset, scan_recordings
    >>> from torch.utils.data import DataLoader
    >>>
    >>> extractor = SNRExtractor(sample_rate=32000, hop_ms=8.0, n_bands=8)
    >>> recordings = scan_recordings(list(Path("data/").glob("*.wav")))
    >>> dataset = CallDataset(
    ...     recordings=recordings,
    ...     extractor=extractor,
    ...     window_s=2.0,
    ...     window_hop_s=0.5,
    ... )
    >>> len(dataset)
    1250
    >>> X, y = dataset[0]
    >>> X.shape  # (n_bands, window_frames)
    torch.Size([8, 250])
    """

    def __init__(
        self,
        recordings: list[RecordingInfo],
        extractor: BaseExtractor,
        window_s: float = 2.0,
        window_hop_s: float = 0.5,
    ) -> None:
        check_type(recordings, (list,), "recordings")
        check_type(extractor, (BaseExtractor,), "extractor")
        check_type(window_s, ("numeric",), "window_s")
        check_type(window_hop_s, ("numeric",), "window_hop_s")

        if len(recordings) == 0:
            raise ValueError("Argument 'recordings' cannot be empty.")
        if window_s <= 0:
            raise ValueError(f"Argument 'window_s' must be positive, got {window_s}.")
        if window_hop_s <= 0:
            raise ValueError(
                f"Argument 'window_hop_s' must be positive, got {window_hop_s}."
            )
        if window_hop_s > window_s:
            raise ValueError(
                f"Argument 'window_hop_s' ({window_hop_s}) cannot be greater than "
                f"'window_s' ({window_s})."
            )

        self._extractor = extractor
        self._window_s = float(window_s)
        self._window_hop_s = float(window_hop_s)
        self._window_frames = extractor.seconds_to_frames(window_s)
        self._window_hop_frames = extractor.seconds_to_frames(window_hop_s)

        # Store recordings and build window index
        self._recordings: list[RecordingInfo] = []
        self._index: list[tuple[int, int]] = []  # (recording_idx, start_frame)

        for recording in recordings:
            check_type(recording, (RecordingInfo,), "recording")
            n_windows = recording.estimate_windows(extractor, window_s, window_hop_s)

            if n_windows == 0:
                logger.debug(
                    "Skipping %s: too short for window size",
                    recording.audio_path.name,
                )
                continue

            # Compute window start positions
            n_frames = recording.estimate_frames(extractor)
            starts = list(
                range(0, n_frames - self._window_frames + 1, self._window_hop_frames)
            )
            if len(starts) == 0:
                starts = [0]

            recording_idx = len(self._recordings)
            self._recordings.append(recording)

            for start in starts:
                self._index.append((recording_idx, start))

        if len(self._index) == 0:
            raise RuntimeError(
                "No training windows were generated. "
                "Check that recordings are long enough for the window size."
            )

        logger.info(
            "CallDataset ready: %d recordings, %d windows",
            len(self._recordings),
            len(self._index),
        )

    def __len__(self) -> int:
        """Return the number of windows in the dataset."""
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a windowed sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        features : Tensor of shape ``(n_features, window_frames)``
            Feature window.
        labels : Tensor of shape ``(window_frames,)``
            Per-frame binary labels.
        """
        idx = ensure_int(idx, "idx")
        if idx < 0 or idx >= len(self._index):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        recording_idx, start = self._index[idx]
        recording = self._recordings[recording_idx]

        # Load features and labels (cached)
        features, labels = _load_recording(
            recording.audio_path, recording.annotation_path, self._extractor
        )

        # Handle edge case: actual frames may differ slightly from estimate
        n_frames = features.shape[1]
        end = start + self._window_frames

        if end > n_frames:
            start = max(0, n_frames - self._window_frames)
            end = start + self._window_frames

        return features[:, start:end], labels[start:end]

    @property
    def n_recordings(self) -> int:
        """Number of recordings in the dataset.

        :type: :class:`int`
        """
        return len(self._recordings)

    @property
    def n_windows(self) -> int:
        """Total number of windows (samples).

        :type: :class:`int`
        """
        return len(self._index)

    @property
    def window_frames(self) -> int:
        """Window length in frames.

        :type: :class:`int`
        """
        return self._window_frames

    @property
    def window_hop_frames(self) -> int:
        """Window hop in frames.

        :type: :class:`int`
        """
        return self._window_hop_frames

    @property
    def window_s(self) -> float:
        """Window length in seconds.

        :type: :class:`float`
        """
        return self._window_s

    @property
    def window_hop_s(self) -> float:
        """Window hop in seconds.

        :type: :class:`float`
        """
        return self._window_hop_s

    @property
    def extractor(self) -> BaseExtractor:
        """Feature extractor used by this dataset.

        :type: :class:`~callcut.features.BaseExtractor`
        """
        return self._extractor

    @property
    def recordings(self) -> list[RecordingInfo]:
        """Recordings in this dataset.

        :type: list of :class:`~callcut.io.RecordingInfo`
        """
        return list(self._recordings)

    def compute_pos_weight(self) -> Tensor:
        """Compute positive class weight for imbalanced data.

        Useful for ``torch.nn.BCEWithLogitsLoss(pos_weight=...)``.

        Returns
        -------
        pos_weight : Tensor of shape ``(1,)``
            Weight for positive class: ``n_negative / n_positive``.

        Notes
        -----
        This method loads all recordings to compute exact label statistics.
        For large datasets, this may take some time.
        """
        logger.info("Computing pos_weight (loading all recordings)...")

        total_positive = 0.0
        total_frames = 0.0

        for recording in self._recordings:
            _, labels = _load_recording(
                recording.audio_path, recording.annotation_path, self._extractor
            )
            total_positive += labels.sum().item()
            total_frames += labels.numel()

        total_negative = total_frames - total_positive

        if total_positive == 0:
            warn(
                "No positive samples found, returning pos_weight=1.0",
                ignore_namespaces=("callcut",),
            )
            return torch.tensor([1.0], dtype=torch.float32)

        pos_weight = total_negative / total_positive
        logger.info(
            "Computed pos_weight: %.2f (%.1f%% positive frames)",
            pos_weight,
            100 * total_positive / total_frames,
        )
        return torch.tensor([pos_weight], dtype=torch.float32)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_recordings={self.n_recordings}, "
            f"n_windows={self.n_windows}, "
            f"window_s={self.window_s}, "
            f"window_hop_s={self.window_hop_s})"
        )
