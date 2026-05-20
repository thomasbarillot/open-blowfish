"""Audit cites where adjacent_quote is non-empty but didn't match any chunk
of the cited opinion.

For each such cite, exports the quote, the surrounding response paragraph,
the cited opinion's text snippets that look closest to the quote, and basic
chunk-count context — enough to manually label the failure as either:
  - "wrong_quote"     : the heuristic attached the wrong nearby quote
  - "chunking_loss"   : the quote is in the opinion but split by a chunk boundary
  - "ocr_drift"       : minor textual differences (curly quotes, dashes, etc.)
  - "not_in_opinion"  : the released answer's quote isn't in this opinion at all

Output: data/dataset/quote_miss_audit.csv with one row per (qid, cite).
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from rapidfuzz import fuzz

from . import config
from .run_evaluate import (
    _MIN_FULL_MATCH_LEN,
    _PREFIX_MATCH_LEN,
    _normalize_for_quote_match,
    _quote_in_chunk,
)

logger = logging.getLogger(__name__)


_WORD_RE = __import__("re").compile(r"[A-Za-z][A-Za-z'-]+")


def _word_tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _ngram_overlap(needle_words: list[str], hay_words_set: set[str], n: int = 3) -> float:
    """Fraction of needle word-n-grams that appear in haystack's n-gram set.

    Word-level n-grams beat char-level partial_ratio for short legal phrases
    because they reject random character overlap (e.g. 'mirror-image' vs
    'Pennsylvania' — high char overlap, zero meaningful word overlap).
    """
    if len(needle_words) < n:
        # Fall back to single-word containment for very short quotes
        if not needle_words:
            return 0.0
        return sum(1 for w in needle_words if w in hay_words_set) / len(needle_words)
    needle_ngrams = {
        " ".join(needle_words[i : i + n])
        for i in range(len(needle_words) - n + 1)
    }
    if not needle_ngrams:
        return 0.0
    return len(needle_ngrams & hay_words_set) / len(needle_ngrams)


def _best_chunk_match(needle: str, chunks: list[tuple[str, str]]) -> tuple[str, int, float, str]:
    """Return (chunk_id, partial_ratio, ngram_overlap, snippet) for the chunk
    whose word-level 3-gram overlap with `needle` is highest. Ties broken by
    rapidfuzz.partial_ratio.

    The ngram score is the trustworthy signal; partial_ratio is reported for
    backward compat with the prior column layout.
    """
    needle_words = _word_tokens(needle)
    best = ("", 0, 0.0, "")
    n = 3
    for cid, normalized_text in chunks:
        if not normalized_text:
            continue
        hay_words = _word_tokens(normalized_text)
        if len(hay_words) < n and len(needle_words) >= n:
            continue
        if len(needle_words) < n:
            hay_ngrams = set(hay_words)
        else:
            hay_ngrams = {
                " ".join(hay_words[i : i + n])
                for i in range(len(hay_words) - n + 1)
            }
        overlap = _ngram_overlap(needle_words, hay_ngrams, n=n)
        if overlap > best[2] or (overlap == best[2] and overlap > 0):
            score = int(fuzz.partial_ratio(needle, normalized_text))
            if overlap > best[2] or score > best[1]:
                head = normalized_text[:160].replace("\n", " ")
                best = (cid, score, overlap, head)
    return best


def _surrounding_paragraph(response_text: str, quote: str, window: int = 600) -> str:
    """Return ~window chars of context around the quote in the response."""
    if not response_text:
        return ""
    idx = response_text.find(quote)
    if idx < 0:
        return ""
    start = max(0, idx - window // 2)
    end = min(len(response_text), idx + len(quote) + window // 2)
    return response_text[start:end].replace("\n", " ").strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        default=str(config.DATASET_DIR / "quote_miss_audit.csv"),
        help="Output CSV path (default: data/dataset/quote_miss_audit.csv)",
    )
    args = p.parse_args()

    gold = pd.read_parquet(config.DATASET_DIR / "gold_citations.parquet")
    chunks = pd.read_parquet(config.CORPUS_DIR / "chunks.parquet")

    chunks_by_opinion: dict[str, list[tuple[str, str]]] = {}
    for cid, op_id, text in zip(
        chunks["chunk_id"].values,
        chunks["opinion_id"].values,
        chunks["text"].fillna("").values,
    ):
        chunks_by_opinion.setdefault(op_id, []).append(
            (cid, _normalize_for_quote_match(text))
        )

    rows = []
    for _, row in gold.iterrows():
        qid = row["source_question_id"]
        op_id = row["opinion_id"]
        quote = row["adjacent_quote"] or ""
        if not op_id or not quote.strip():
            continue
        needle = _normalize_for_quote_match(quote)
        if len(needle) < _MIN_FULL_MATCH_LEN:
            continue
        op_chunks = chunks_by_opinion.get(op_id, [])
        if not op_chunks:
            continue

        if any(_quote_in_chunk(needle, h) for _, h in op_chunks):
            continue

        # Quote present, opinion present, no chunk matched. Diagnose.
        best_cid, best_score, best_ngram, snippet = _best_chunk_match(needle, op_chunks)
        n_quote_words = len(_word_tokens(needle))
        prefix_in_any = any(
            len(needle) > _PREFIX_MATCH_LEN and needle[:_PREFIX_MATCH_LEN] in h
            for _, h in op_chunks
        )
        rows.append({
            "source_question_id": qid,
            "normalized_citation": row["normalized_citation"],
            "cite_class": row.get("cite_class", ""),
            "opinion_id": op_id,
            "n_chunks_for_opinion": len(op_chunks),
            "quote_len": len(needle),
            "n_quote_words": n_quote_words,
            "adjacent_quote": quote,
            "best_chunk_id": best_cid,
            "best_ngram_overlap": round(best_ngram, 3),
            "best_partial_ratio": best_score,
            "best_chunk_snippet": snippet,
            "prefix60_in_any_chunk": prefix_in_any,
            "response_context": _surrounding_paragraph(row["response_text"], quote),
        })

    df = pd.DataFrame(rows).sort_values(
        ["best_ngram_overlap", "best_partial_ratio", "source_question_id"],
        ascending=[False, False, True],
    )
    df.to_csv(args.out, index=False)
    logger.info("Wrote %d rows to %s", len(df), args.out)
    print(f"Wrote {len(df)} rows to {args.out}")
    if len(df) == 0:
        return

    print()
    print("Score distribution (best_ngram_overlap, word 3-grams):")
    print(df["best_ngram_overlap"].describe().to_string())
    print()
    print("Likely categories (heuristic, ngram-based — confirm manually):")
    chunking = (
        (df["best_ngram_overlap"] >= 0.5)
        & (df["n_quote_words"] >= 3)
        & ~df["prefix60_in_any_chunk"]
    ).sum()
    boundary_short = (
        (df["best_ngram_overlap"] >= 0.5) & (df["n_quote_words"] < 3)
    ).sum()
    moderate = (
        (df["best_ngram_overlap"] >= 0.25) & (df["best_ngram_overlap"] < 0.5)
    ).sum()
    wrong = (df["best_ngram_overlap"] < 0.25).sum()
    print(f"  ngram_overlap >= 0.50 (3+ words)  -> probable chunking_loss/ocr: {chunking}")
    print(f"  ngram_overlap >= 0.50 (1-2 words) -> probable boilerplate match: {boundary_short}")
    print(f"  ngram_overlap 0.25-0.49           -> ambiguous:                  {moderate}")
    print(f"  ngram_overlap <  0.25             -> probable wrong_quote:       {wrong}")


if __name__ == "__main__":
    main()
