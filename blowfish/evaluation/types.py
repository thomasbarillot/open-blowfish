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
    """One retrieved chunk's worth of information for a single query."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    hash_key: str
    docname: str
    chunk_embedding: np.ndarray
    score: float
    rank: int
    topic_label: Optional[str] = None
    silhouette_score: Optional[float] = None
    gold_match: Optional[bool] = None


class RetrievalRecord(BaseModel):
    """Per-query record: query embedding + top-k retrieved chunks + optional label."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    query_id: str
    query_embedding: np.ndarray
    top_k: list[RetrievedChunk]
    gold_chunk_hash: Optional[str] = None
    gold_text: Optional[str] = None
    stratum: Optional[str] = None
    correct_prediction: Optional[int] = None
