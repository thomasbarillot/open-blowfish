from __future__ import annotations

import html
import re
from functools import lru_cache

from transformers import AutoTokenizer

from .. import config
from ..evaluation.citation_parser import extract_citations

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")


def _strip_html(raw: str) -> str:
    text = _TAG_RE.sub("", raw)
    text = html.unescape(text)
    return _WS_RE.sub("\n\n", text).strip()


@lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)


def chunk_text(
    text: str,
    opinion_id: str,
    normalized_citation: str,
    chunk_tokens: int = config.CHUNK_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """Split text into overlapping token-indexed chunks.

    Returns a list of dicts with keys chunk_id, opinion_id, text,
    token_offset_start, token_offset_end (end is exclusive — matches
    Python slice semantics). The final chunk's `text` extends to
    len(source) to preserve trailing whitespace; its `token_offset_end`
    still equals the exclusive token index.
    """
    text = _strip_html(text)
    tok = _tokenizer()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    n = len(token_ids)
    if n == 0:
        return []

    step = chunk_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("chunk_tokens must exceed overlap_tokens")

    out = []
    idx = 0
    i = 0
    while i < n:
        j = min(i + chunk_tokens, n)
        char_start = offsets[i][0]
        char_end = offsets[j - 1][1]
        if j == n:
            char_end = len(text)
        chunk_text_str = text[char_start:char_end]
        secondary = sorted({
            c.normalized for c in extract_citations(chunk_text_str)
            if c.normalized and c.normalized != normalized_citation
        })
        out.append({
            "chunk_id": f"{opinion_id}::{idx}",
            "opinion_id": opinion_id,
            "normalized_citation": normalized_citation,
            "secondary_citations": secondary,
            "text": chunk_text_str,
            "token_offset_start": i,
            "token_offset_end": j,
        })
        if j == n:
            break
        i += step
        idx += 1
    return out
