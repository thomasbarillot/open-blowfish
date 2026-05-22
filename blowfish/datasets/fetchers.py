"""Per-source document fetchers used by ``Corpus.iter_documents()``.

Each manifest entry declares a ``source`` (e.g. ``"wikipedia"``, ``"wikisource"``,
``"chronicling_america"``); the corpus loader dispatches to the matching
``Fetcher`` to re-materialize the document text from its archive URL when the
local cache is empty or stale. This is what makes end-user runtime download
work without first running ``scripts/bootstrap_corpora.py``.

Adding a new source:

1. Subclass :class:`Fetcher`, implement ``fetch(entry) -> str``.
2. Register it in :data:`FETCHERS` keyed by the ``source`` string.
3. Bootstrap script entries with the same ``source`` will then be re-fetchable
   at runtime.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import requests


USER_AGENT = "open-blowfish runtime (https://github.com/thomasbarillot/open-blowfish)"


class Fetcher(ABC):
    """Resolves one manifest entry into raw document text."""

    source: ClassVar[str]

    @abstractmethod
    def fetch(self, entry: dict[str, Any]) -> str: ...


class DirectFetcher(Fetcher):
    """Direct URL → UTF-8 text. For mirrors that serve raw text files."""

    source: ClassVar[str] = "direct"

    def fetch(self, entry: dict[str, Any]) -> str:
        for url in entry.get("mirror_urls", []):
            try:
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
                resp.raise_for_status()
                return resp.text
            except Exception:
                continue
        raise RuntimeError(
            f"DirectFetcher: all mirrors failed for {entry.get('doc_id')!r}"
        )


class _MediaWikiExtractFetcher(Fetcher):
    """Shared logic for Wikipedia / Wikinews — same MediaWiki ``prop=extracts`` API.

    Mirror URLs are full API endpoints (``…?action=query&prop=extracts&…&titles=…&oldid=…``);
    response is JSON; the plain-text article body is in
    ``query.pages.<id>.extract``.
    """

    def fetch(self, entry: dict[str, Any]) -> str:
        last_exc: Exception | None = None
        for url in entry.get("mirror_urls", []):
            try:
                resp = requests.get(
                    url, headers={"User-Agent": USER_AGENT}, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                pages = data["query"]["pages"]
                page = next(iter(pages.values()))
                extract = page.get("extract")
                if extract:
                    return extract
                last_exc = RuntimeError(f"empty extract for {url}")
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            f"{self.source} fetcher: all mirrors failed for "
            f"{entry.get('doc_id')!r}: {last_exc}"
        )


class WikipediaFetcher(_MediaWikiExtractFetcher):
    source: ClassVar[str] = "wikipedia"


class WikinewsFetcher(_MediaWikiExtractFetcher):
    source: ClassVar[str] = "wikinews"


_WIKITEXT_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_WIKITABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_REF_BLOCK_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL)
_FILE_LINK_RE = re.compile(r"\[\[(?:Category|File|Image):[^\[\]]*\]\]")
_LINK_RE = re.compile(r"\[\[(?:[^\|\]]*\|)?([^\]]*)\]\]")
_BOLD_RE = re.compile(r"'''([^']*)'''")
_ITALIC_RE = re.compile(r"''([^']*)''")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _strip_wikitext(text: str) -> str:
    for _ in range(2):
        text = _WIKITEXT_TEMPLATE_RE.sub("", text)
    text = _WIKITABLE_RE.sub("", text)
    text = _REF_BLOCK_RE.sub("", text)
    text = re.sub(r"<ref[^/]*/>", "", text)
    text = re.sub(r"<noinclude>.*?</noinclude>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?pages[^>]*/?>", "", text)
    text = _FILE_LINK_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


class WikisourceFetcher(Fetcher):
    """Wikisource pages: ``prop=revisions`` returns raw wikitext; we strip markup."""

    source: ClassVar[str] = "wikisource"

    def fetch(self, entry: dict[str, Any]) -> str:
        last_exc: Exception | None = None
        for url in entry.get("mirror_urls", []):
            try:
                # The bootstrap URL is the extracts API; we need the revisions API.
                # Rewrite ``prop=extracts&explaintext=1`` → ``prop=revisions&rvprop=content&rvslots=main``.
                revisions_url = re.sub(
                    r"prop=extracts&explaintext=1",
                    "prop=revisions&rvprop=content&rvslots=main",
                    url,
                )
                resp = requests.get(
                    revisions_url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                pages = data["query"]["pages"]
                page = next(iter(pages.values()))
                rev = page["revisions"][0]
                wikitext = rev["slots"]["main"]["*"]
                cleaned = _strip_wikitext(wikitext)
                if len(cleaned) >= 200:
                    return cleaned
                last_exc = RuntimeError(f"cleaned wikitext too short for {url}")
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            f"wikisource fetcher: all mirrors failed for "
            f"{entry.get('doc_id')!r}: {last_exc}"
        )


class ChroniclingAmericaFetcher(Fetcher):
    """LoC Chronicling America: item-detail JSON has the OCR text in ``ocr_eng``."""

    source: ClassVar[str] = "chronicling_america"

    def fetch(self, entry: dict[str, Any]) -> str:
        last_exc: Exception | None = None
        for url in entry.get("mirror_urls", []):
            try:
                json_url = url if url.endswith(".json") else url.rstrip("/") + ".json"
                resp = requests.get(
                    json_url, headers={"User-Agent": USER_AGENT}, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                text = (data.get("ocr_eng") or "").strip()
                if len(text) >= 500:
                    return text
                last_exc = RuntimeError(
                    f"chronicling_america: OCR too short for {url} ({len(text)} chars)"
                )
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            f"chronicling_america fetcher: all mirrors failed for "
            f"{entry.get('doc_id')!r}: {last_exc}"
        )


FETCHERS: dict[str, Fetcher] = {
    "direct": DirectFetcher(),
    "wikipedia": WikipediaFetcher(),
    "wikinews": WikinewsFetcher(),
    "wikisource": WikisourceFetcher(),
    "chronicling_america": ChroniclingAmericaFetcher(),
}


def get_fetcher(source: str) -> Fetcher:
    """Look up a fetcher by ``source`` string. Raises ``KeyError`` if unknown."""
    fetcher = FETCHERS.get(source)
    if fetcher is None:
        raise KeyError(
            f"No runtime fetcher registered for source {source!r}. "
            f"Known sources: {sorted(FETCHERS)}. "
            f"For sources without a runtime fetcher (e.g. 'cc_news'), populate the "
            f"cache via scripts/bootstrap_corpora.py before iterating the corpus."
        )
    return fetcher
