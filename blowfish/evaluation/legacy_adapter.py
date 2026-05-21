"""Adapter from the new ``RetrievalRecord`` shape to the legacy DataFrame
format expected by ``blowfish.calculations.calculate_relevant_features`` and
``AmbiguityScorer.run_scoring``. Lets baselines like B7/B8 reuse the existing
feature pipeline without forcing a breaking change on the legacy API.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from blowfish.evaluation.types import RetrievalRecord


def to_legacy_query_df(record: RetrievalRecord) -> pd.DataFrame:
    """Convert one ``RetrievalRecord`` to the per-query DataFrame the legacy
    feature pipeline consumes (one row per retrieved chunk)."""
    rows = []
    for c in record.top_k:
        rows.append(
            {
                "score": c.score,
                "chunk_embeddings": list(c.chunk_embedding),
                "query_embedding": list(record.query_embedding),
                "docname": c.docname,
                "topic_label": c.topic_label if c.topic_label is not None else f"{c.docname}_0",
                "silhouette_score": c.silhouette_score if c.silhouette_score is not None else 0.0,
                "hash_key": c.hash_key,
                "rank": c.rank,
            }
        )
    return pd.DataFrame(rows)
