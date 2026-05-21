"""Phase 3 — runtime per-source fetchers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from blowfish.datasets.fetchers import (
    FETCHERS,
    ChroniclingAmericaFetcher,
    WikinewsFetcher,
    WikipediaFetcher,
    WikisourceFetcher,
    get_fetcher,
)


def _mock_json(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json = lambda: payload
    resp.raise_for_status = lambda: None
    return resp


def test_get_fetcher_returns_known_source():
    assert isinstance(get_fetcher("wikipedia"), WikipediaFetcher)
    assert isinstance(get_fetcher("wikinews"), WikinewsFetcher)
    assert isinstance(get_fetcher("wikisource"), WikisourceFetcher)
    assert isinstance(get_fetcher("chronicling_america"), ChroniclingAmericaFetcher)


def test_get_fetcher_raises_for_unknown_source():
    with pytest.raises(KeyError, match="No runtime fetcher"):
        get_fetcher("does_not_exist")


def test_wikipedia_fetcher_extracts_text():
    payload = {"query": {"pages": {"123": {"pageid": 123, "extract": "Einstein was born in 1879."}}}}
    entry = {"doc_id": "x", "mirror_urls": ["http://nope"]}
    with patch("blowfish.datasets.fetchers.requests.get", return_value=_mock_json(payload)):
        text = WikipediaFetcher().fetch(entry)
    assert text == "Einstein was born in 1879."


def test_wikipedia_fetcher_falls_back_across_mirrors():
    bad_payload = {"query": {"pages": {"0": {"missing": ""}}}}
    good_payload = {"query": {"pages": {"123": {"extract": "OK"}}}}
    responses = [_mock_json(bad_payload), _mock_json(good_payload)]
    entry = {"doc_id": "x", "mirror_urls": ["http://broken", "http://good"]}
    with patch("blowfish.datasets.fetchers.requests.get", side_effect=responses):
        text = WikipediaFetcher().fetch(entry)
    assert text == "OK"


def test_wikipedia_fetcher_raises_when_all_mirrors_empty():
    bad_payload = {"query": {"pages": {"0": {"missing": ""}}}}
    entry = {"doc_id": "x", "mirror_urls": ["http://a", "http://b"]}
    with patch("blowfish.datasets.fetchers.requests.get", return_value=_mock_json(bad_payload)):
        with pytest.raises(RuntimeError, match="all mirrors failed"):
            WikipediaFetcher().fetch(entry)


def test_chronicling_america_fetcher_extracts_ocr_text():
    long_text = "Stocks rose on the New York exchange today. " * 30
    payload = {"ocr_eng": long_text, "title": "Daily Tribune"}
    entry = {"doc_id": "x", "mirror_urls": ["http://nope/lccn/123/1900-01-01/ed-1/seq-1"]}
    with patch("blowfish.datasets.fetchers.requests.get", return_value=_mock_json(payload)):
        text = ChroniclingAmericaFetcher().fetch(entry)
    assert "Stocks" in text


def test_wikisource_fetcher_strips_wikitext():
    wikitext = (
        "'''Algebra''' is a branch of mathematics. "
        "It deals with [[symbols|symbolic]] manipulation.\n\n"
        "{{citation needed}}It originated in ancient times. " * 5
    )
    payload = {
        "query": {
            "pages": {
                "1": {
                    "revisions": [
                        {"revid": 1, "slots": {"main": {"*": wikitext}}}
                    ]
                }
            }
        }
    }
    entry = {
        "doc_id": "x",
        "mirror_urls": [
            "http://en.wikisource.org/w/api.php?action=query&format=json&prop=extracts&explaintext=1&titles=X&oldid=1"
        ],
    }
    with patch("blowfish.datasets.fetchers.requests.get", return_value=_mock_json(payload)):
        text = WikisourceFetcher().fetch(entry)
    assert "Algebra" in text
    assert "[[" not in text
    assert "'''" not in text
    assert "{{" not in text


def test_fetchers_dict_covers_all_implementations():
    """Every concrete Fetcher in this module appears in FETCHERS by its source name."""
    for source in ("direct", "wikipedia", "wikinews", "wikisource", "chronicling_america"):
        assert source in FETCHERS
        assert FETCHERS[source].source == source
