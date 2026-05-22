"""Phase 3 — chunkers (Sentence, Paragraph, Recursive, TokenSlidingWindow)."""

from __future__ import annotations

import json

import pytest

from blowfish.datasets.chunkers import (
    ParagraphChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
    TokenSlidingWindowChunker,
)
from blowfish.datasets.types import Document


def _doc(text: str) -> Document:
    return Document(doc_id="d1", title="t", text=text)


def test_sentence_chunker_splits_on_sentence_boundaries(sample_text):
    chunks = SentenceChunker().chunk(_doc(sample_text))
    texts = [c.text for c in chunks]
    assert any("first sentence" in t for t in texts)
    assert any("second" in t.lower() for t in texts)
    assert all(c.unit == "sentence" for c in chunks)


def test_paragraph_chunker_splits_on_blank_lines(sample_text):
    chunks = ParagraphChunker().chunk(_doc(sample_text))
    assert len(chunks) == 3
    assert "first sentence" in chunks[0].text
    assert "Last paragraph here" in chunks[-1].text
    assert all(c.unit == "paragraph" for c in chunks)


def test_paragraph_chunker_empty_doc():
    chunks = ParagraphChunker().chunk(_doc("   \n\n  \n  "))
    assert chunks == []


def test_recursive_chunker_respects_chunk_size(sample_text):
    chunks = RecursiveCharacterChunker(chunk_size=40, chunk_overlap=10).chunk(
        _doc(sample_text * 5)
    )
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 41  # 1-char boundary slack
    assert all(c.unit == "recursive" for c in chunks)


def test_token_chunker_produces_chunks_with_overlap_aware_count(sample_text):
    chunks = TokenSlidingWindowChunker(size=10, stride=5).chunk(_doc(sample_text * 5))
    assert len(chunks) >= 2
    assert all(c.unit == "token" for c in chunks)


def test_token_chunker_default_stride_no_overlap(sample_text):
    chunks = TokenSlidingWindowChunker(size=20).chunk(_doc(sample_text * 3))
    assert len(chunks) >= 1
    assert all(c.unit == "token" for c in chunks)


@pytest.mark.parametrize(
    "chunker",
    [
        SentenceChunker(),
        ParagraphChunker(),
        RecursiveCharacterChunker(chunk_size=128, chunk_overlap=16),
        TokenSlidingWindowChunker(size=50),
    ],
)
def test_chunker_params_are_json_serializable(chunker):
    """Params land in DatasetManifest.chunker_params and must JSON-serialize."""
    params = chunker.params()
    assert isinstance(params, dict)
    json.dumps(params)
