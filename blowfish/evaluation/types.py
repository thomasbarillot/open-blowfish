"""Canonical per-query intermediate types for the evaluation, baseline, and RAG layers.

Introduced in the evaluation PR to decouple downstream code from the legacy
query-DataFrame shape used by ``training/queries_evaluation.py``. Every
baseline (B0..B9) and every gating policy (G0..G6) consumes ``RetrievalRecord``.
"""

from __future__ import annotations

from typing import Optional, TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict


ScoreVector: TypeAlias = np.ndarray
LabelVector: TypeAlias = np.ndarray


class RetrievedChunk(BaseModel):
    """One retrieved chunk's worth of information for a single query.

    ``text`` is optional; baselines and gates that only need geometric signals
    (B0–B6, G0–G6 except G4/G5) operate on ``chunk_embedding`` alone. The RAG
    harness (Phase 4) requires ``text`` to populate LLM context.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    hash_key: str
    docname: str
    chunk_embedding: np.ndarray
    score: float
    rank: int
    text: Optional[str] = None
    topic_label: Optional[str] = None
    silhouette_score: Optional[float] = None
    gold_match: Optional[bool] = None


class RetrievalRecord(BaseModel):
    """Per-query record: query embedding + top-k retrieved chunks + optional label.

    ``query_text`` is optional but required by the RAG harness to construct the
    LLM prompt; baselines and gates work without it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    query_id: str
    query_embedding: np.ndarray
    top_k: list[RetrievedChunk]
    query_text: Optional[str] = None
    gold_chunk_hash: Optional[str] = None
    gold_text: Optional[str] = None
    stratum: Optional[str] = None
    correct_prediction: Optional[int] = None
