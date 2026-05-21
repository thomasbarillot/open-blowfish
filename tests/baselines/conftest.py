"""Synthetic ``RetrievalRecord`` fixtures shared across baseline tests."""

from __future__ import annotations

import numpy as np
import pytest

from blowfish.evaluation.types import RetrievalRecord, RetrievedChunk


@pytest.fixture(scope="module")
def synthetic_records() -> list[RetrievalRecord]:
    """24 records (12 correct, 12 incorrect) × 20 top-k chunks of dim 16.

    Correct queries: one **gold** chunk very close to the query embedding plus
    19 moderately-close distractors. Incorrect queries: 20 chunks uniformly
    scattered on the unit sphere. This shape gives all geometric baselines a
    real signal (clear top-1 gap, peaky softmax, small mean kNN distance, etc.)
    while staying small enough that B7/B8 fit + score is fast.

    Module-scoped so the per-record VR / feature computation that B7/B8 do is
    amortized via ``id(records)`` cache hits inside the baseline.
    """
    rng = np.random.default_rng(0)
    records = []
    for q_id in range(24):
        is_correct = q_id % 2 == 0
        q = rng.normal(size=16)
        q /= np.linalg.norm(q)
        chunks_data: list[tuple[np.ndarray, float]] = []
        if is_correct:
            gold = q + rng.normal(0, 0.02, size=16)
            chunks_data.append((gold, float(np.linalg.norm(gold - q))))
            for _ in range(19):
                # σ=0.15 → distractor norms ~0.6, well below the ~1.41 mean of
                # random unit vectors used for incorrect queries below. Keeps
                # B4/B5 (mean kNN distance) above chance.
                c = q + rng.normal(0, 0.15, size=16)
                chunks_data.append((c, float(np.linalg.norm(c - q))))
        else:
            for _ in range(20):
                c = rng.normal(size=16)
                c /= np.linalg.norm(c)
                chunks_data.append((c, float(np.linalg.norm(c - q))))
        chunks_data.sort(key=lambda kv: kv[1])
        chunks = []
        for rank, (c, dist) in enumerate(chunks_data):
            chunks.append(
                RetrievedChunk(
                    hash_key=f"q{q_id}_c{rank}",
                    docname=f"doc{q_id % 6}",
                    chunk_embedding=c,
                    score=dist,
                    rank=rank,
                    topic_label=f"doc{q_id % 6}_{rank // 5}",
                    silhouette_score=0.5,
                )
            )
        gold_hash = chunks[0].hash_key if is_correct else None
        records.append(
            RetrievalRecord(
                query_id=f"q{q_id}",
                query_embedding=q,
                top_k=chunks,
                gold_chunk_hash=gold_hash,
                correct_prediction=int(is_correct),
            )
        )
    return records


@pytest.fixture(scope="module")
def synthetic_labels(synthetic_records: list[RetrievalRecord]) -> np.ndarray:
    return np.asarray([r.correct_prediction for r in synthetic_records], dtype=int)
