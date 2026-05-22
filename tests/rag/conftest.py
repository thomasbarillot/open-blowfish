"""Shared fixtures for Phase 4 RAG tests."""

from __future__ import annotations

import numpy as np
import pytest

from blowfish.evaluation.types import RetrievalRecord, RetrievedChunk


@pytest.fixture(scope="module")
def rag_records() -> list[RetrievalRecord]:
    """20 records (10 correct, 10 incorrect) × 10 chunks each, with text fields populated.

    Correct queries: rank-0 chunk is the gold answer (close to query, text = gold).
    Incorrect queries: chunks are random unit vectors with placeholder text;
    no chunk's text matches the (different) gold answer.

    With ``EchoGenerator`` (returns chunks[0].text), the EM/F1 judge will mark
    correct queries correct and incorrect queries wrong — giving the harness a
    clean signal to score.
    """
    rng = np.random.default_rng(0)
    records: list[RetrievalRecord] = []
    for i in range(20):
        is_correct = i % 2 == 0
        q = rng.normal(size=16)
        q /= np.linalg.norm(q)
        gold_text = f"the answer to query {i} is forty two"
        chunks_data: list[tuple[np.ndarray, str]] = []
        if is_correct:
            gold_emb = q + rng.normal(0, 0.02, size=16)
            chunks_data.append((gold_emb, gold_text))
            for j in range(9):
                emb = q + rng.normal(0, 0.15, size=16)
                chunks_data.append((emb, f"distractor {i}-{j}"))
        else:
            for j in range(10):
                emb = rng.normal(size=16)
                emb /= np.linalg.norm(emb)
                chunks_data.append((emb, f"random_chunk_{i}_{j}"))
        # Sort by distance to query
        chunks_data.sort(key=lambda kv: float(np.linalg.norm(kv[0] - q)))
        chunks = []
        for rank, (emb, text) in enumerate(chunks_data):
            chunks.append(
                RetrievedChunk(
                    hash_key=f"q{i}_c{rank}",
                    docname=f"doc{i % 4}",
                    chunk_embedding=emb,
                    score=float(np.linalg.norm(emb - q)),
                    rank=rank,
                    text=text,
                    topic_label=f"doc{i % 4}_0",
                    silhouette_score=0.5,
                )
            )
        records.append(
            RetrievalRecord(
                query_id=f"q{i}",
                query_embedding=q,
                top_k=chunks,
                query_text=f"What is the answer to query {i}?",
                gold_text=gold_text if is_correct else "an answer that no chunk contains",
                gold_chunk_hash=chunks[0].hash_key if is_correct else None,
                correct_prediction=int(is_correct),
            )
        )
    return records


@pytest.fixture(scope="module")
def rag_labels(rag_records: list[RetrievalRecord]) -> np.ndarray:
    return np.asarray([r.correct_prediction for r in rag_records], dtype=int)
