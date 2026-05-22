"""Phase 4 — RAG-benchmark metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blowfish.evaluation.types import RetrievalRecord, RetrievedChunk
from blowfish.rag.metrics import (
    abstain_precision_recall,
    abstain_rate,
    answered_subset_em_f1,
    expected_utility,
    summarize,
)


def _mk_df(actions, em_vals, f1_vals, correct_vals, utility_vals, query_ids=None):
    n = len(actions)
    return pd.DataFrame(
        {
            "query_id": query_ids or [f"q{i}" for i in range(n)],
            "action": actions,
            "em": em_vals,
            "f1": f1_vals,
            "correct": correct_vals,
            "utility": utility_vals,
        }
    )


def test_answered_subset_excludes_abstained():
    df = _mk_df(
        actions=["answer", "abstain", "answer"],
        em_vals=[1, float("nan"), 0],
        f1_vals=[1.0, float("nan"), 0.5],
        correct_vals=[True, None, False],
        utility_vals=[1.0, 0.0, -3.0],
    )
    out = answered_subset_em_f1(df)
    assert out["n_answered"] == 2
    assert out["em"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(0.75)


def test_answered_subset_all_abstain_returns_nan():
    df = _mk_df(
        actions=["abstain", "abstain"],
        em_vals=[float("nan"), float("nan")],
        f1_vals=[float("nan"), float("nan")],
        correct_vals=[None, None],
        utility_vals=[0.0, 0.0],
    )
    out = answered_subset_em_f1(df)
    assert np.isnan(out["em"])
    assert np.isnan(out["f1"])
    assert out["n_answered"] == 0


def test_abstain_rate_fraction():
    df = _mk_df(
        actions=["answer", "abstain", "abstain", "answer", "answer"],
        em_vals=[1, float("nan"), float("nan"), 0, 1],
        f1_vals=[1.0, float("nan"), float("nan"), 0.0, 1.0],
        correct_vals=[True, None, None, False, True],
        utility_vals=[1.0, 0.0, 0.0, -3.0, 1.0],
    )
    assert abstain_rate(df) == pytest.approx(0.4)


def test_expected_utility_is_mean_of_utility_column():
    df = _mk_df(
        actions=["answer", "answer", "abstain"],
        em_vals=[1, 0, float("nan")],
        f1_vals=[1.0, 0.0, float("nan")],
        correct_vals=[True, False, None],
        utility_vals=[1.0, -3.0, 0.0],
    )
    assert expected_utility(df) == pytest.approx(-2.0 / 3.0)


def test_summarize_emits_all_scalars():
    df = _mk_df(
        actions=["answer", "abstain"],
        em_vals=[1, float("nan")],
        f1_vals=[1.0, float("nan")],
        correct_vals=[True, None],
        utility_vals=[1.0, 0.0],
    )
    out = summarize(df)
    for key in ("n_queries", "abstain_rate", "expected_utility", "em", "f1", "n_answered"):
        assert key in out


def _stub_record(query_id: str, gold_in_topk: bool) -> RetrievalRecord:
    return RetrievalRecord(
        query_id=query_id,
        query_embedding=np.zeros(4),
        top_k=[
            RetrievedChunk(
                hash_key="h_gold" if gold_in_topk else "h_other",
                docname="d",
                chunk_embedding=np.zeros(4),
                score=0.0,
                rank=0,
            )
        ],
        gold_chunk_hash="h_gold",
    )


def test_abstain_precision_recall_perfect_when_gate_matches_answerability():
    # Records: 2 answerable, 2 not.
    records = [
        _stub_record("q0", gold_in_topk=True),
        _stub_record("q1", gold_in_topk=True),
        _stub_record("q2", gold_in_topk=False),
        _stub_record("q3", gold_in_topk=False),
    ]
    # Gate abstains on q2, q3 (the unanswerable ones).
    df = _mk_df(
        actions=["answer", "answer", "abstain", "abstain"],
        em_vals=[1, 1, float("nan"), float("nan")],
        f1_vals=[1.0, 1.0, float("nan"), float("nan")],
        correct_vals=[True, True, None, None],
        utility_vals=[1.0, 1.0, 0.0, 0.0],
        query_ids=["q0", "q1", "q2", "q3"],
    )
    out = abstain_precision_recall(df, records)
    assert out["abstain_precision"] == pytest.approx(1.0)
    assert out["abstain_recall"] == pytest.approx(1.0)
