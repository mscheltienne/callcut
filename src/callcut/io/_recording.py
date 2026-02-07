"""Recording metadata for dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchcodec.decoders import AudioDecoder

from callcut.io._loader import _read_annotation_csv
from callcut.utils._checks import ensure_path
from callcut.utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path

    from callcut.extractors import BaseExtractor


@dataclass(frozen=True)
class RecordingInfo:
    """Metadata about a recording for dataset construction.

    This dataclass holds pre-computed information about a recording, avoiding
    repeated I/O operations when building datasets.

    Parameters
    ----------
    audio_path : Path
        Path to the audio file.
    annotation_path : Path
        Path to the annotation CSV file.
    duration_s : float
        Duration of the recording in seconds.
    n_annotations : int
        Number of annotated call intervals.

    Examples
    --------
    >>> from callcut.io import scan_recordings
    >>> from pathlib import Path
    >>>
    >>> recordings = scan_recordings(list(Path("data/").glob("*.wav")))
    >>> recordings[0].duration_s
    45.2
    >>> recordings[0].n_annotations
    23
    """

    audio_path: Path
    annotation_path: Path
    duration_s: float
    n_annotations: int

    def estimate_frames(self, extractor: BaseExtractor) -> int:
        """Estimate the number of feature frames for this recording.

        Parameters
        ----------
        extractor : BaseExtractor
            Feature extractor (used to get hop size).

        Returns
        -------
        n_frames : int
            Estimated number of frames.
        """
        return extractor.seconds_to_frames(self.duration_s)

    def estimate_windows(
        self, extractor: BaseExtractor, window_s: float, window_hop_s: float
    ) -> int:
        """Estimate the number of training windows for this recording.

        Parameters
        ----------
        extractor : BaseExtractor
            Feature extractor (used to convert seconds to frames).
        window_s : float
            Window length in seconds.
        window_hop_s : float
            Window hop in seconds.

        Returns
        -------
        n_windows : int
            Estimated number of training windows. Returns 0 if the recording
            is too short for even one window.
        """
        n_frames = self.estimate_frames(extractor)
        window_frames = extractor.seconds_to_frames(window_s)
        window_hop_frames = extractor.seconds_to_frames(window_hop_s)

        if n_frames < window_frames:
            return 0

        starts = range(0, n_frames - window_frames + 1, window_hop_frames)
        return len(starts) if starts else 1

    def __repr__(self) -> str:
        return (
            f"RecordingInfo({self.audio_path.name!r}, "
            f"duration={self.duration_s:.1f}s, "
            f"n_annotations={self.n_annotations})"
        )


def scan_recordings(recordings: list[Path | str]) -> list[RecordingInfo]:
    """Scan recordings and return metadata for valid ones.

    Filters recordings to only include those with:

    - An existing annotation file (``*_annotations.csv``)
    - Valid audio metadata (readable duration)
    - At least one valid annotation

    Parameters
    ----------
    recordings : list of Path | str
        Paths to audio files.

    Returns
    -------
    recording_infos : list of RecordingInfo
        Metadata for valid recordings, sorted by audio path.

    Examples
    --------
    >>> from pathlib import Path
    >>> from callcut.io import scan_recordings
    >>>
    >>> recordings = scan_recordings(list(Path("data/").glob("*.wav")))
    >>> len(recordings)
    42
    >>> total_duration = sum(r.duration_s for r in recordings)
    >>> total_annotations = sum(r.n_annotations for r in recordings)
    """
    results: list[RecordingInfo] = []

    for recording in recordings:
        recording = ensure_path(recording, must_exist=True)
        annotation_path = recording.with_name(recording.stem + "_annotations.csv")

        # Check annotation file exists
        if not annotation_path.exists():
            logger.debug("Skipping %s: no annotation file found", recording.name)
            continue

        # Get audio duration
        try:
            decoder = AudioDecoder(str(recording))
            duration_s = decoder.metadata.duration_seconds
        except Exception as exc:
            logger.warning(
                "Skipping %s: failed to read audio metadata: %s", recording.name, exc
            )
            continue

        # Count valid annotations
        try:
            df = _read_annotation_csv(annotation_path)
            n_annotations = len(df)
        except Exception as exc:
            logger.warning(
                "Skipping %s: failed to read annotations: %s", recording.name, exc
            )
            continue

        if n_annotations == 0:
            logger.debug("Skipping %s: no valid annotations", recording.name)
            continue

        results.append(
            RecordingInfo(
                audio_path=recording,
                annotation_path=annotation_path,
                duration_s=duration_s,
                n_annotations=n_annotations,
            )
        )
        logger.debug(
            "Scanned %s: duration=%.1fs, n_annotations=%d",
            recording.name,
            duration_s,
            n_annotations,
        )

    # Sort by path for reproducibility
    results.sort(key=lambda r: r.audio_path)

    logger.info(
        "Scanned %d recordings: %d valid (%.1f hours, %d annotations)",
        len(recordings),
        len(results),
        sum(r.duration_s for r in results) / 3600,
        sum(r.n_annotations for r in results),
    )

    return results
