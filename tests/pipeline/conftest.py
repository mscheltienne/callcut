"""Shared fixtures for pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from callcut.io import RecordingInfo, scan_recordings


@pytest.fixture(scope="module")
def assets_path() -> Path:
    """Path to test assets directory."""
    return Path(__file__).parents[1] / "assets"


@pytest.fixture(scope="module")
def audio_files(assets_path: Path) -> list[Path]:
    """All audio files in assets directory."""
    return sorted(assets_path.glob("*.wav"))


@pytest.fixture(scope="module")
def recordings(audio_files: list[Path]) -> list[RecordingInfo]:
    """Scanned recordings from test assets."""  # noqa: D401
    return scan_recordings(audio_files)
