"""B6 — Mahalanobis distance from query to centroid of top-k chunk embeddings.

Uses the per-query empirical covariance of the k chunk embeddings, regularized
with ``regularization * I`` and pseudo-inverted to handle the rank-deficient
case (n_chunks ≤ embedding_dim).
"""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from blowfish.baselines.base import Baseline
from blowfish.evaluation.types import RetrievalRecord, ScoreVector


class MahalanobisCentroidBaseline(Baseline):
    name: ClassVar[str] = "B6"

    def __init__(self, *, regularization: float = 1e-6) -> None:
        self.regularization = float(regularization)

    def score(self, records: Sequence[RetrievalRecord]) -> ScoreVector:
        out = np.empty(len(records), dtype=np.float64)
        for i, r in enumerate(records):
            if not r.top_k:
                out[i] = 0.0
                continue
            chunks = np.stack(
                [np.asarray(c.chunk_embedding, dtype=np.float64) for c in r.top_k], axis=0
            )
            centroid = chunks.mean(axis=0)
            diffs = chunks - centroid
            n_eff = max(1, chunks.shape[0] - 1)
            cov = (diffs.T @ diffs) / n_eff
            cov_reg = cov + self.regularization * np.eye(cov.shape[0])
            inv = np.linalg.pinv(cov_reg)
            delta = np.asarray(r.query_embedding, dtype=np.float64) - centroid
            mahal_sq = float(delta @ inv @ delta)
            out[i] = -np.sqrt(max(0.0, mahal_sq))
        return out
