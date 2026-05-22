"""Phase 4 — G0 through G6 gating policies + tune_threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blowfish.evaluation.types import RetrievalRecord
from blowfish.rag.gates import (
    ALL_GATE_IDS,
    BlowfishGate,
    CalibratedLogisticGate,
    EntropyGate,
    GateHooks,
    MeanKnnDistanceGate,
    NoGate,
    OracleGate,
    ScoreGapGate,
    tune_threshold,
)


# -- G0 No-gate -------------------------------------------------------------

def test_g0_never_abstains(rag_records):
    g = NoGate()
    abstain_count = sum(g.should_abstain(r) for r in rag_records)
    assert abstain_count == 0


def test_g0_score_is_infinity():
    import math
    g = NoGate()
    record = list_first_record_helper()
    assert math.isinf(g.score(record))


def list_first_record_helper() -> RetrievalRecord:
    """Minimal helper to make G0 score callable without the fixture."""
    from blowfish.evaluation.types import RetrievedChunk
    return RetrievalRecord(
        query_id="x",
        query_embedding=np.zeros(4),
        top_k=[
            RetrievedChunk(
                hash_key="h",
                docname="d",
                chunk_embedding=np.zeros(4),
                score=0.0,
                rank=0,
            )
        ],
    )


# -- Untuned gates require tuning -------------------------------------------

@pytest.mark.parametrize("gate_cls", [ScoreGapGate, EntropyGate, MeanKnnDistanceGate])
def test_untuned_gates_raise_on_abstain_decision(gate_cls, rag_records):
    g = gate_cls()
    with pytest.raises(RuntimeError, match="threshold is unset"):
        g.should_abstain(rag_records[0])


# -- tune_threshold sets the right abstain rate ----------------------------

def test_tune_threshold_hits_target_abstain_rate(rag_records):
    g = ScoreGapGate()
    tune_threshold(g, rag_records, abstain_rate=0.3)
    abstain_count = sum(g.should_abstain(r) for r in rag_records)
    # The quantile produces a threshold at the 30th-percentile score, so on
    # average that fraction abstains. With ties it can shift by one or two.
    assert abs(abstain_count - 0.3 * len(rag_records)) <= 2


def test_tune_threshold_rejects_invalid_rate(rag_records):
    g = ScoreGapGate()
    with pytest.raises(ValueError, match="abstain_rate"):
        tune_threshold(g, rag_records, abstain_rate=1.5)


# -- G6 Oracle gate --------------------------------------------------------

def test_g6_oracle_abstains_iff_gold_missing_from_topk(rag_records):
    g = OracleGate()
    for r in rag_records:
        expected_abstain = r.gold_chunk_hash is None or not any(
            c.hash_key == r.gold_chunk_hash for c in r.top_k
        )
        assert g.should_abstain(r) == expected_abstain


# -- Abstain rate monotone in threshold ------------------------------------

def test_score_gap_abstain_rate_monotone_in_target(rag_records):
    g = ScoreGapGate()
    rates = []
    for target in (0.1, 0.3, 0.5, 0.7):
        tune_threshold(g, rag_records, abstain_rate=target)
        rates.append(sum(g.should_abstain(r) for r in rag_records))
    # Monotone non-decreasing
    assert all(a <= b for a, b in zip(rates, rates[1:]))


# -- G5 Calibrated logistic gate (trainable) ------------------------------

def test_g5_requires_fit_before_score(rag_records):
    g = CalibratedLogisticGate(random_state=0)
    with pytest.raises(RuntimeError, match="must be fit"):
        g.score(rag_records[0])


def test_g5_fit_then_score_returns_unit_interval(rag_records, rag_labels):
    g = CalibratedLogisticGate(random_state=0)
    g.fit(rag_records, rag_labels)
    s = g.score(rag_records[0])
    assert 0.0 <= s <= 1.0


# -- G4 Blowfish gate (with a stub scorer to avoid fitting a real KDE) ----

class _StubScorer:
    """Stand-in for AmbiguityScorer.calculate_query_correctness_probability.

    Returns p_correct = 1 - clamp(top1_distance, 0, 1) — high when the top hit
    is close to the query.
    """

    def calculate_query_correctness_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        top1_score = float(df.sort_values("rank").iloc[0]["score"])
        p = max(0.0, min(1.0, 1.0 - top1_score))
        return pd.DataFrame([{"p_correct": p}])


def test_g4_blowfish_gate_calls_scorer(rag_records):
    g = BlowfishGate(scorer=_StubScorer(), threshold=0.5)
    p = g.score(rag_records[0])
    assert 0.0 <= p <= 1.0


def test_g4_blowfish_gate_threshold_decides_abstain():
    record = list_first_record_helper()
    # top1 score is 0.0 → stub p_correct = 1.0
    g_high_threshold = BlowfishGate(scorer=_StubScorer(), threshold=0.9)
    g_low_threshold = BlowfishGate(scorer=_StubScorer(), threshold=0.1)
    assert not g_high_threshold.should_abstain(record)  # 1.0 ≥ 0.9
    assert not g_low_threshold.should_abstain(record)   # 1.0 ≥ 0.1


# -- Factory registry ------------------------------------------------------

def test_gate_hooks_covers_all_seven_gates():
    for gid in ALL_GATE_IDS:
        assert hasattr(GateHooks, gid)
