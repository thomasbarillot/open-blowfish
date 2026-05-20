from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

from eyecite import get_citations
from eyecite.models import CaseCitation


@dataclass
class ParsedCitation:
    raw: str
    normalized: str
    reporter: str
    volume: str
    page: str
    span: tuple[int, int]
    adjacent_quote: Optional[str]


_QUOTE_RE = re.compile(r'["“]([^"“”]{6,1500})["”]')

# Case-name patterns rendered inside quotes ("X v. Y", "In re X", "Ex parte X").
# These are titles, not verbatim opinion text, so they should never count as a
# grounding quote even when positionally nearest the citation.
_CASE_NAME_PATTERNS = (
    re.compile(r"\bv[.,]?\s+\w", re.IGNORECASE),
    re.compile(r"\bIn\s+re\b", re.IGNORECASE),
    re.compile(r"\bEx\s+parte\b", re.IGNORECASE),
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _looks_like_case_name(quote: str) -> bool:
    if any(p.search(quote) for p in _CASE_NAME_PATTERNS):
        # Allow if it's a long passage that merely *mentions* a case in passing
        # (>=12 words) — only treat as a case name when the entire quote is
        # essentially the title.
        return len(_WORD_RE.findall(quote)) < 12
    return False


def _looks_like_mid_prose_capture(quote: str) -> bool:
    """The regex captures `... " X " ...` greedily — when X starts with a
    space + lowercase letter and ends with whitespace + lowercase + comma/period,
    it's almost always sentence prose between two real quotes, not a quote.
    """
    if not quote:
        return False
    if quote[0].isspace() and len(quote) > 1 and quote[1].islower():
        return True
    return False


def _is_acceptable_quote(quote: str) -> bool:
    """Reject case-name renderings and regex-greedy mid-prose captures.

    No word-count floor: short doctrinal phrases like 'charitable purposes'
    can be legitimate verbatim quotes, and the eval-time _MIN_FULL_MATCH_LEN
    of 12 chars already guards against pathologically short matches at the
    chunk-substring stage.
    """
    if _looks_like_case_name(quote):
        return False
    if _looks_like_mid_prose_capture(quote):
        return False
    return True


def _nearest_quote(text: str, span: tuple[int, int], window: int = 300) -> Optional[str]:
    """Return the closest quoted passage that passes sanity filters.

    Filters reject case names, regex-greedy mid-prose captures, and jargon
    shorter than _MIN_QUOTE_WORDS. Walks candidates in order of distance to
    the cite and returns the first acceptable one — falling back to None
    rather than the unconditional nearest, since a wrong quote pollutes
    chunk-grounding more than a missing one (the paraphrase fallback can
    recover missing-quote cases).
    """
    start = max(0, span[0] - window)
    end = min(len(text), span[1] + window)
    region = text[start:end]
    matches = list(_QUOTE_RE.finditer(region))
    if not matches:
        return None
    cite_rel = span[0] - start
    matches.sort(key=lambda m: min(abs(m.start() - cite_rel), abs(m.end() - cite_rel)))
    for m in matches:
        candidate = m.group(1)
        if _is_acceptable_quote(candidate):
            return candidate
    return None


def extract_citations(text: str) -> list[ParsedCitation]:
    if not text or not text.strip():
        return []
    out: list[ParsedCitation] = []
    for c in get_citations(text):
        if not isinstance(c, CaseCitation):
            continue
        groups = c.groups
        volume = groups.get("volume", "") or ""
        page = groups.get("page", "") or ""
        reporter = (groups.get("reporter") or c.corrected_reporter() or "").strip()
        if not (volume and reporter and page):
            continue
        normalized = f"{volume} {reporter} {page}".strip()
        span = (c.span()[0], c.span()[1])
        out.append(ParsedCitation(
            raw=text[span[0]:span[1]],
            normalized=normalized,
            reporter=reporter,
            volume=str(volume),
            page=str(page),
            span=span,
            adjacent_quote=_nearest_quote(text, span),
        ))
    return out
