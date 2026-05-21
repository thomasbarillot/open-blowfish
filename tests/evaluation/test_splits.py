"""Phase 1 — evaluation/splits.py."""

import numpy as np
import pytest

from blowfish.evaluation.splits import document_level_split
from blowfish.evaluation.types import RetrievalRecord, RetrievedChunk


def _mk_record(query_id: str, docname: str) -> RetrievalRecord:
    return RetrievalRecord(
        query_id=query_id,
        query_embedding=np.zeros(4),
        top_k=[
            RetrievedChunk(
                hash_key=f"h_{query_id}_{i}",
                docname=docname,
                chunk_embedding=np.zeros(4),
                score=float(i),
                rank=i,
            )
            for i in range(3)
        ],
    )


def test_split_no_docname_leakage_across_partitions():
    records = [_mk_record(f"q{i}", f"doc{i % 20}") for i in range(500)]
    train, val, test = document_level_split(records, ratios=(0.6, 0.2, 0.2), seed=0)
    train_docs = {r.top_k[0].docname for r in train}
    val_docs = {r.top_k[0].docname for r in val}
    test_docs = {r.top_k[0].docname for r in test}
    assert not train_docs & val_docs
    assert not train_docs & test_docs
    assert not val_docs & test_docs


def test_split_ratios_within_tolerance():
    records = [_mk_record(f"q{i}", f"doc{i % 50}") for i in range(1000)]
    train, val, test = document_level_split(records, ratios=(0.6, 0.2, 0.2), seed=0)
    n = len(records)
    assert abs(len(train) / n - 0.6) < 0.1
    assert abs(len(val) / n - 0.2) < 0.1
    assert abs(len(test) / n - 0.2) < 0.1


def test_split_seed_deterministic():
    records = [_mk_record(f"q{i}", f"doc{i % 30}") for i in range(300)]
    a = document_level_split(records, seed=42)
    b = document_level_split(records, seed=42)
    assert [r.query_id for r in a[0]] == [r.query_id for r in b[0]]


def test_split_rejects_ratios_that_dont_sum_to_one():
    with pytest.raises(ValueError):
        document_level_split([], ratios=(0.5, 0.3, 0.3))


def test_split_handles_empty_input():
    train, val, test = document_level_split([])
    assert train == [] and val == [] and test == []
