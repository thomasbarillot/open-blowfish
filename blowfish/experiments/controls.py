"""Randomized topology controls R0/R1/R2 from ``ABLATION_PLAN.md``.

Three falsifiability checks anyone publishing a Blowfish result should run:

- **R0** ``permute_neighborhoods`` — swap each query's top-k for k random
  chunks pooled across the corpus. If the topology pipeline still shows a
  signal under R0, the signal is contamination, not topology.
- **R1** ``rotate_embeddings`` — apply a random orthogonal rotation to every
  embedding (query + chunks). VR persistence is rotation-invariant, so
  W₁(H₀) / LT_max(H₁) must change by at most numerical tolerance. Failure
  reveals an implementation bug.
- **R2** ``shuffle_feature_block`` — within an already-computed feature
  matrix, shuffle a chosen block of columns across rows. If discrimination
  AUROC survives shuffling the topology block, topology isn't contributing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from blowfish.evaluation.types import RetrievalRecord, RetrievedChunk


def permute_neighborhoods(
    records: Sequence[RetrievalRecord], *, seed: int = 0
) -> list[RetrievalRecord]:
    """R0 — pool every chunk across every query, shuffle, deal out random
    top-k bags. Each query keeps its embedding but loses its retrieved
    neighborhood. Discriminative signal that survives R0 is corpus-wide,
    not query-specific."""
    if not records:
        return []
    rng = np.random.default_rng(seed)
    pool: list[RetrievedChunk] = [c for r in records for c in r.top_k]
    rng.shuffle(pool)
    k = len(records[0].top_k)
    new_records: list[RetrievalRecord] = []
    for i, r in enumerate(records):
        slice_start = i * k
        bag = pool[slice_start : slice_start + k]
        if len(bag) < k:
            extra = list(rng.choice(pool, size=k - len(bag), replace=True))
            bag = bag + extra
        q = np.asarray(r.query_embedding, dtype=np.float64)
        rescored = []
        for new_rank, c in enumerate(bag):
            chunk_emb = np.asarray(c.chunk_embedding, dtype=np.float64)
            new_score = float(np.linalg.norm(chunk_emb - q))
            rescored.append(
                c.model_copy(update={"rank": new_rank, "score": new_score, "gold_match": False})
            )
        rescored.sort(key=lambda c: c.score)
        for new_rank, c in enumerate(rescored):
            rescored[new_rank] = c.model_copy(update={"rank": new_rank})
        new_records.append(
            r.model_copy(update={"top_k": rescored, "gold_chunk_hash": None})
        )
    return new_records


def rotate_embeddings(
    records: Sequence[RetrievalRecord], *, seed: int = 0
) -> list[RetrievalRecord]:
    """R1 — apply one random orthogonal rotation Q to every embedding.

    Q is generated via QR on a random Gaussian, so it's uniformly distributed
    on the orthogonal group. All Euclidean distances are preserved, so VR
    persistence diagrams (and W₁(H₀) / LT_max(H₁)) are invariant.
    """
    if not records:
        return []
    rng = np.random.default_rng(seed)
    d = np.asarray(records[0].query_embedding).shape[0]
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    new_records: list[RetrievalRecord] = []
    for r in records:
        new_q = Q @ np.asarray(r.query_embedding, dtype=np.float64)
        new_chunks = []
        for c in r.top_k:
            new_emb = Q @ np.asarray(c.chunk_embedding, dtype=np.float64)
            new_chunks.append(c.model_copy(update={"chunk_embedding": new_emb}))
        new_records.append(
            r.model_copy(update={"query_embedding": new_q, "top_k": new_chunks})
        )
    return new_records


def shuffle_feature_block(
    features_df: pd.DataFrame,
    block: list[str],
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """R2 — shuffle ``block`` columns of ``features_df`` across rows.

    Permutes only the listed columns; other columns + the label stay aligned.
    Used to test "if I scramble just the topology features, does my AUROC
    drop?" — if not, the topology block wasn't doing work.
    """
    missing = [c for c in block if c not in features_df.columns]
    if missing:
        raise KeyError(f"Columns missing from features_df: {missing}")
    rng = np.random.default_rng(seed)
    n = len(features_df)
    indices = np.arange(n)
    rng.shuffle(indices)
    shuffled = features_df.copy()
    shuffled[block] = features_df[block].iloc[indices].to_numpy()
    return shuffled
