"""Shared fixtures for Phase 3 dataset tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``$BLOWFISH_CACHE_DIR`` to a tmp path for the duration of the test."""
    monkeypatch.setenv("BLOWFISH_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def sample_text() -> str:
    return (
        "This is the first sentence. And here is the second.\n\n"
        "A new paragraph begins. With another sentence. And one more.\n\n"
        "Last paragraph here."
    )
