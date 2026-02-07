"""Shared fixtures for pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from callcut.io import RecordingInfo, scan_recordings


@pytest.fixture(scope="module")
def recordings(all_audio_files: list[Path]) -> list[RecordingInfo]:
    """Scanned recordings from test assets."""  # noqa: D401
    return scan_recordings(all_audio_files)
