import pytest

from scripts.legal_rag_homology.corpus.chunking import chunk_text
from scripts.legal_rag_homology import config


def test_chunk_text_single_short_opinion_yields_one_chunk():
    text = "Short opinion text. " * 10
    chunks = chunk_text(text, opinion_id="op1", normalized_citation="1 U.S. 1", chunk_tokens=512, overlap_tokens=100)
    assert len(chunks) == 1
    assert chunks[0]["opinion_id"] == "op1"
    assert chunks[0]["text"] == text
    assert chunks[0]["chunk_id"] == "op1::0"


def test_chunk_text_long_opinion_yields_overlapping_chunks():
    text = " ".join([f"word{i}" for i in range(2000)])
    chunks = chunk_text(text, opinion_id="op2", normalized_citation="2 U.S. 2", chunk_tokens=512, overlap_tokens=100)
    assert len(chunks) >= 3
    assert all(c["opinion_id"] == "op2" for c in chunks)
    assert chunks[0]["token_offset_start"] == 0
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt["token_offset_start"] < prev["token_offset_end"]


def test_chunk_ids_are_unique_and_ordered():
    text = " ".join([f"w{i}" for i in range(1500)])
    chunks = chunk_text(text, opinion_id="opX", normalized_citation="3 U.S. 3", chunk_tokens=512, overlap_tokens=100)
    ids = [c["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids, key=lambda s: int(s.split("::")[1]))


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("", opinion_id="op_empty", normalized_citation="") == []


def test_chunk_text_raises_when_overlap_geq_chunk():
    text = " ".join([f"w{i}" for i in range(100)])
    with pytest.raises(ValueError):
        chunk_text(text, opinion_id="op", normalized_citation="4 U.S. 4", chunk_tokens=100, overlap_tokens=100)
    with pytest.raises(ValueError):
        chunk_text(text, opinion_id="op", normalized_citation="4 U.S. 4", chunk_tokens=100, overlap_tokens=200)


def test_chunk_text_stride_equals_chunk_minus_overlap():
    text = " ".join([f"tok{i}" for i in range(3000)])
    chunks = chunk_text(text, opinion_id="op_step", normalized_citation="5 U.S. 5", chunk_tokens=512, overlap_tokens=100)
    stride = 512 - 100
    for prev, nxt in zip(chunks[:-1], chunks[1:]):
        assert nxt["token_offset_start"] == prev["token_offset_start"] + stride
        assert nxt["token_offset_start"] == prev["token_offset_end"] - 100
