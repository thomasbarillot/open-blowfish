"""SHA-256 verified downloader with mirror fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import requests


class DownloadError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_mirrors(
    urls: Sequence[str],
    expected_sha256: str,
    destination: Path,
    *,
    timeout: float = 30.0,
) -> Path:
    """Try ``urls`` in order; the first whose payload matches ``expected_sha256``
    is saved to ``destination``. Raises ``DownloadError`` if all mirrors fail.

    Returns ``destination`` early if it already exists with the correct SHA.
    """
    destination = Path(destination)
    if destination.exists() and sha256_file(destination) == expected_sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            actual_sha = hashlib.sha256(response.content).hexdigest()
            if actual_sha != expected_sha256:
                last_error = DownloadError(
                    f"{url}: SHA mismatch (expected {expected_sha256[:12]}…, got {actual_sha[:12]}…)"
                )
                continue
            destination.write_bytes(response.content)
            return destination
        except Exception as e:  # noqa: BLE001  — any network / parsing failure
            last_error = e
            continue
    raise DownloadError(
        f"All {len(urls)} mirrors failed for {destination.name}. "
        f"Last error: {last_error}. To remediate, add a new mirror URL to the corpus manifest."
    )
