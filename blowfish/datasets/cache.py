"""Content-addressed cache layout for the dataset sweep artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from blowfish.utils.constants import CACHE_DIR_ENV


_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "blowfish"


def cache_root() -> Path:
    """Return the cache root, honoring ``$BLOWFISH_CACHE_DIR`` if set."""
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_CACHE_DIR


def cache_path_for(key: str) -> Path:
    """Return the cache directory for a content-addressed key (first 16 hex chars)."""
    return cache_root() / key[:16]


def ensure_cache_dir(key: str) -> Path:
    p = cache_path_for(key)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
