"""RAG-benchmark metrics computed over the harness output DataFrame.

Inputs (DataFrame columns produced by :class:`RAGHarness`):
``query_id``, ``action`` ("answer" or "abstain"), ``em``, ``f1``, ``correct``,
``utility``.

Outputs are dicts of scalar metrics suitable for ``reports.py`` tables.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from blowfish.evaluation.types import RetrievalRecord


def answered_subset_em_f1(df: pd.DataFrame) -> dict[str, float]:
    """EM/F1 averaged only over the rows where the system answered."""
    answered = df[df["action"] == "answer"]
    if answered.empty:
        return {"em": float("nan"), "f1": float("nan"), "n_answered": 0}
    return {
        "em": float(answered["em"].mean()),
        "f1": float(answered["f1"].mean()),
        "n_answered": int(len(answered)),
    }


def abstain_rate(df: pd.DataFrame) -> float:
    """Fraction of queries the gate fired on."""
    if df.empty:
        return 0.0
    return float((df["action"] == "abstain").mean())


def abstain_precision_recall(
    df: pd.DataFrame, records: list[RetrievalRecord]
) -> dict[str, float]:
    """Precision/recall of the abstain decision against the answerability label.

    A query is "unanswerable" iff its gold chunk is not in the retrieved top-k.
    Abstain-precision: of the queries we abstained on, what fraction were
    truly unanswerable. Abstain-recall: of the truly unanswerable queries,
    what fraction did we abstain on.
    """
    if df.empty or not records:
        return {"abstain_precision": float("nan"), "abstain_recall": float("nan")}
    record_by_id = {r.query_id: r for r in records}

    def _answerable(qid: str) -> bool:
        r = record_by_id.get(qid)
        if r is None or r.gold_chunk_hash is None:
            return True
        return any(c.hash_key == r.gold_chunk_hash for c in r.top_k)

    abstained_mask = df["action"] == "abstain"
    unanswerable_mask = ~df["query_id"].map(_answerable).astype(bool)

    n_abstained = int(abstained_mask.sum())
    n_unanswerable = int(unanswerable_mask.sum())
    n_abstained_and_unanswerable = int((abstained_mask & unanswerable_mask).sum())

    precision = (
        n_abstained_and_unanswerable / n_abstained if n_abstained else float("nan")
    )
    recall = (
        n_abstained_and_unanswerable / n_unanswerable if n_unanswerable else float("nan")
    )
    return {
        "abstain_precision": float(precision),
        "abstain_recall": float(recall),
        "n_abstained": n_abstained,
        "n_unanswerable": n_unanswerable,
    }


def expected_utility(df: pd.DataFrame) -> float:
    """Mean per-query utility (already populated by the harness)."""
    if df.empty:
        return 0.0
    return float(df["utility"].mean())


def summarize(
    df: pd.DataFrame,
    *,
    records: Optional[list[RetrievalRecord]] = None,
) -> dict[str, float]:
    """One-shot helper returning every reported scalar in §5.5."""
    out: dict[str, float] = {
        "n_queries": float(len(df)),
        "abstain_rate": abstain_rate(df),
        "expected_utility": expected_utility(df),
    }
    out.update(answered_subset_em_f1(df))
    if records is not None:
        out.update(abstain_precision_recall(df, records))
    return out
