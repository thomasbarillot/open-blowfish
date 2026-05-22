"""Phase 3 — download_with_mirrors SHA + fallback semantics."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from blowfish.datasets.download import (
    DownloadError,
    download_with_mirrors,
    sha256_file,
)


def _mock_response(payload: bytes) -> MagicMock:
    resp = MagicMock()
    resp.content = payload
    resp.raise_for_status = lambda: None
    return resp


def test_download_success_with_correct_sha(tmp_path):
    payload = b"hello open-blowfish"
    expected = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "out.txt"
    with patch("blowfish.datasets.download.requests.get", return_value=_mock_response(payload)):
        path = download_with_mirrors(["http://nope/x"], expected, dest)
    assert path == dest
    assert dest.read_bytes() == payload
    assert sha256_file(dest) == expected


def test_download_fails_when_all_mirrors_have_bad_sha(tmp_path):
    dest = tmp_path / "out.txt"
    with patch(
        "blowfish.datasets.download.requests.get",
        return_value=_mock_response(b"wrong payload"),
    ):
        with pytest.raises(DownloadError, match="remediate"):
            download_with_mirrors(["http://a", "http://b"], "0" * 64, dest)


def test_download_falls_back_to_second_mirror(tmp_path):
    payload = b"the right payload"
    expected = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "out.txt"
    responses = [
        _mock_response(b"wrong"),
        _mock_response(payload),
    ]
    with patch("blowfish.datasets.download.requests.get", side_effect=responses):
        path = download_with_mirrors(["http://broken/x", "http://good/x"], expected, dest)
    assert path == dest
    assert dest.read_bytes() == payload


def test_download_skipped_when_destination_already_correct(tmp_path):
    payload = b"already there"
    dest = tmp_path / "out.txt"
    dest.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    mock_get = MagicMock()
    with patch("blowfish.datasets.download.requests.get", side_effect=mock_get):
        download_with_mirrors(["http://nope"], sha, dest)
    mock_get.assert_not_called()
