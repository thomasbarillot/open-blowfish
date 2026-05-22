"""Shared fixtures for Phase 5 experiment tests."""

from __future__ import annotations

import numpy as np
import pytest

from blowfish.evaluation.types import RetrievalRecord, RetrievedChunk


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    """Redirect ``$BLOWFISH_CACHE_DIR`` so prereg locks don't touch the real cache."""
    monkeypatch.setenv("BLOWFISH_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="module")
def experiment_records() -> list[RetrievalRecord]:
    """Same shape as the baselines/RAG fixtures but smaller — 12 records, k=8."""
    rng = np.random.default_rng(0)
    records: list[RetrievalRecord] = []
    for i in range(12):
        is_correct = i % 2 == 0
        q = rng.normal(size=8)
        q /= np.linalg.norm(q)
        items: list[tuple[np.ndarray, float]] = []
        if is_correct:
            gold = q + rng.normal(0, 0.02, size=8)
            items.append((gold, float(np.linalg.norm(gold - q))))
            for _ in range(7):
                c = q + rng.normal(0, 0.15, size=8)
                items.append((c, float(np.linalg.norm(c - q))))
        else:
            for _ in range(8):
                c = rng.normal(size=8)
                c /= np.linalg.norm(c)
                items.append((c, float(np.linalg.norm(c - q))))
        items.sort(key=lambda kv: kv[1])
        chunks = []
        for rank, (emb, dist) in enumerate(items):
            chunks.append(
                RetrievedChunk(
                    hash_key=f"q{i}_c{rank}",
                    docname=f"doc{i % 3}",
                    chunk_embedding=emb,
                    score=dist,
                    rank=rank,
                    topic_label=f"doc{i % 3}_0",
                    silhouette_score=0.5,
                )
            )
        records.append(
            RetrievalRecord(
                query_id=f"q{i}",
                query_embedding=q,
                top_k=chunks,
                gold_chunk_hash=chunks[0].hash_key if is_correct else None,
                correct_prediction=int(is_correct),
            )
        )
    return records
