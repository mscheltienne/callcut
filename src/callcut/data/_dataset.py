"""PyTorch Dataset for call detection training."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from callcut.data._labels import intervals_to_frame_labels
from callcut.features import BaseExtractor
from callcut.io import load_annotations, load_audio
from callcut.utils._checks import check_type, ensure_int, ensure_path
from callcut.utils.logs import logger, warn


class CallDataset(Dataset):
    """PyTorch Dataset for frame-level call detection.

    Loads audio recordings and their annotations, extracts features, and provides
    windowed samples for training. Each sample is a fixed-length window of features
    with corresponding per-frame binary labels.

    The dataset performs **frame-level segmentation**: each frame within a window
    has its own label (``1 = call``, ``0 = silence``), rather than a single label per
    window.

    Parameters
    ----------
    recordings : list of Path
        Paths to audio files. Each audio file should have a corresponding
        annotation CSV file with the same stem and ``_annotations.csv`` suffix.
    extractor : BaseExtractor
        Feature extractor instance. Must be already configured with the desired
        parameters (``sample_rate``, ``hop_ms``, ``n_bands``, etc.).
    window_s : float
        Window length in seconds for each sample.
    window_hop_s : float
        Hop between consecutive windows in seconds. Controls overlap between
        training samples.

    Attributes
    ----------
    n_recordings : int
        Number of recordings with valid annotations.
    n_windows : int
        Total number of windows (samples) across all recordings.
    window_frames : int
        Window length in frames.
    window_hop_frames : int
        Window hop in frames.

    Notes
    -----
    **Lazy loading**: Features are computed on-demand in ``__getitem__`` rather than
    pre-loaded into memory. This reduces memory usage but increases I/O during
    training. For faster training with sufficient RAM, consider using a
    ``DataLoader`` with ``num_workers > 0`` to parallelize loading.

    **Annotation format**: Annotation files must be CSV with columns
    ``start_seconds`` and ``stop_seconds`` (values in milliseconds, converted
    to seconds internally).

    **Missing annotations**: Recordings without a corresponding annotation file
    are silently skipped.

    Examples
    --------
    >>> from pathlib import Path
    >>> from callcut.features import SNRExtractor
    >>> from callcut.data import CallDataset
    >>> from torch.utils.data import DataLoader
    >>>
    >>> extractor = SNRExtractor(sample_rate=32000, hop_ms=8.0, n_bands=8)
    >>> recordings = list(Path("data/").glob("*.wav"))
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
    >>> y.shape  # (window_frames,)
    torch.Size([250])
    >>>
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    >>> for X_batch, y_batch in loader:
    ...     # X_batch: (32, 8, 250), y_batch: (32, 250)
    ...     pass
    """

    def __init__(
        self,
        recordings: list[Path | str],
        extractor: BaseExtractor,
        window_s: float = 2.0,
        window_hop_s: float = 0.5,
    ) -> None:
        check_type(extractor, (BaseExtractor,), "extractor")
        check_type(window_s, ("numeric",), "window_s")
        check_type(window_hop_s, ("numeric",), "window_hop_s")

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

        # Convert to frames
        self._window_frames = extractor.seconds_to_frames(window_s)
        self._window_hop_frames = extractor.seconds_to_frames(window_hop_s)

        logger.info(
            "Creating CallDataset: window_s=%.2f (=%d frames), "
            "window_hop_s=%.2f (=%d frames)",
            window_s,
            self._window_frames,
            window_hop_s,
            self._window_hop_frames,
        )

        # Storage for recording paths (lazy loading)
        self._recordings: list[tuple[Path, Path]] = []  # (audio_path, annotation_path)
        self._index: list[tuple[int, int]] = []  # (recording_idx, start_frame)

        # Scan recordings and build index
        for recording in recordings:
            recording = ensure_path(recording, must_exist=True)
            annotation_path = recording.with_name(recording.stem + "_annotations.csv")

            if not annotation_path.exists():
                logger.debug("Skipping %s: no annotation file found", recording.name)
                continue

            self._index_recording(recording, annotation_path)

        if len(self._index) == 0:
            raise RuntimeError(
                "No training windows were generated. Check that recordings have "
                "matching annotation files and are long enough for the window size."
            )

        logger.info(
            "CallDataset ready: %d recordings, %d windows",
            len(self._recordings),
            len(self._index),
        )

    def _index_recording(self, audio_path: Path, annotation_path: Path) -> None:
        """Index a recording without loading features into memory."""
        # Get audio duration using torchaudio.info (lightweight, no full load)
        try:
            info = torchaudio.info(audio_path)
        except Exception as exc:
            logger.error("Failed to read info for %s: %s", audio_path.name, exc)
            return

        duration_s = info.num_frames / info.sample_rate

        # Estimate number of feature frames
        n_frames = self._extractor.seconds_to_frames(duration_s)

        # Check if recording is long enough
        if n_frames < self._window_frames:
            warn(
                f"Recording {audio_path.name} has ~{n_frames} frames "
                f"(need {self._window_frames} for window). Skipping.",
                ignore_namespaces=("callcut",),
            )
            return

        # Compute window start positions
        starts = list(
            range(0, n_frames - self._window_frames + 1, self._window_hop_frames)
        )

        if len(starts) == 0:
            starts = [0]

        # Store recording path and index windows
        recording_idx = len(self._recordings)
        self._recordings.append((audio_path, annotation_path))

        for start in starts:
            self._index.append((recording_idx, start))

        logger.debug(
            "Indexed %s: ~%.1fs (~%d frames), %d windows",
            audio_path.name,
            duration_s,
            n_frames,
            len(starts),
        )

    def _load_recording(self, recording_idx: int) -> tuple[Tensor, Tensor]:
        """Load and extract features for a recording (called on-demand)."""
        audio_path, annotation_path = self._recordings[recording_idx]

        # Load audio
        waveform, _ = load_audio(
            audio_path, sample_rate=self._extractor.sample_rate, mono=True
        )

        # Extract features
        features, times = self._extractor(waveform)

        # Load annotations and generate labels
        intervals = load_annotations(annotation_path)
        labels = intervals_to_frame_labels(intervals, times)

        return features, labels

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

        # Load features and labels on-demand
        features, labels = self._load_recording(recording_idx)

        # Handle edge case: actual frames may differ slightly from estimate
        n_frames = features.shape[1]
        end = start + self._window_frames

        if end > n_frames:
            # Adjust start to fit window at end of recording
            start = max(0, n_frames - self._window_frames)
            end = start + self._window_frames

        X = features[:, start:end]
        y = labels[start:end]

        return X, y

    @property
    def n_recordings(self) -> int:
        """Number of recordings with valid annotations.

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

        Examples
        --------
        >>> dataset = CallDataset(...)
        >>> pos_weight = dataset.compute_pos_weight()
        >>> loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        """
        logger.info("Computing pos_weight (loading all recordings)...")

        total_positive = 0.0
        total_frames = 0.0

        for recording_idx in range(len(self._recordings)):
            _, labels = self._load_recording(recording_idx)
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
