"""Phase 4 — RAGHarness end-to-end pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blowfish.rag.cost import CostModel
from blowfish.rag.gates import NoGate, OracleGate, ScoreGapGate, tune_threshold
from blowfish.rag.generator import EchoGenerator
from blowfish.rag.harness import RAGHarness
from blowfish.rag.judge import ExactMatchJudge, F1Judge


def test_harness_with_no_gate_and_echo_generator_produces_one_row_per_query(rag_records):
    harness = RAGHarness(generator=EchoGenerator(), gate=NoGate(), judge=F1Judge(threshold=0.5))
    df = harness.run(rag_records)
    assert len(df) == len(rag_records)
    for col in ("query_id", "action", "gate_score", "predicted", "em", "f1", "correct", "utility"):
        assert col in df.columns


def test_harness_no_gate_never_abstains(rag_records):
    harness = RAGHarness(generator=EchoGenerator(), gate=NoGate(), judge=F1Judge())
    df = harness.run(rag_records)
    assert (df["action"] == "answer").all()


def test_harness_oracle_gate_abstains_on_incorrect_queries(rag_records):
    harness = RAGHarness(generator=EchoGenerator(), gate=OracleGate(), judge=F1Judge())
    df = harness.run(rag_records)
    # Half the fixture records have no gold chunk in top-k → all abstained.
    expected_abstain = sum(
        1 for r in rag_records if r.gold_chunk_hash is None
    )
    assert (df["action"] == "abstain").sum() == expected_abstain


def test_harness_echo_with_f1_judge_scores_fixture_correctly(rag_records):
    """For the synthetic fixture, correct queries have gold text as the
    rank-0 chunk; EchoGenerator returns that text; F1 should be 1.0."""
    harness = RAGHarness(generator=EchoGenerator(), gate=NoGate(), judge=F1Judge(threshold=0.5))
    df = harness.run(rag_records)
    correct_query_ids = {f"q{i}" for i in range(20) if i % 2 == 0}
    answered_correct = df[df["query_id"].isin(correct_query_ids)]
    # Every "correct" query should have F1 ≥ 0.5 since the gold text IS the chunk text.
    assert (answered_correct["correct"]).all()


def test_harness_utility_uses_cost_model(rag_records):
    cost = CostModel(correct=2.0, wrong=-5.0, abstain=0.0)
    harness = RAGHarness(generator=EchoGenerator(), gate=NoGate(), judge=F1Judge(), cost_model=cost)
    df = harness.run(rag_records)
    correct_count = int(df["correct"].sum())
    wrong_count = len(df) - correct_count - int((df["action"] == "abstain").sum())
    expected_total_utility = correct_count * 2.0 + wrong_count * -5.0
    assert df["utility"].sum() == pytest.approx(expected_total_utility)


def test_harness_records_zero_latency_on_abstain(rag_records):
    g = ScoreGapGate()
    tune_threshold(g, rag_records, abstain_rate=0.5)
    harness = RAGHarness(generator=EchoGenerator(), gate=g, judge=F1Judge())
    df = harness.run(rag_records)
    abstain_rows = df[df["action"] == "abstain"]
    assert (abstain_rows["latency_ms"] == 0.0).all()


def test_harness_uses_query_text_in_prompt(rag_records, monkeypatch):
    """Verify the prompt template received by the generator contains the question text."""

    captured = {}

    class _CapturingGen:
        name = "capturing"

        def generate(self, prompt, context, **kw):
            captured["prompt"] = prompt
            captured["context"] = list(context)
            from blowfish.rag.generator import GenerationResult
            return GenerationResult(text="x", model="capturing")

    harness = RAGHarness(generator=_CapturingGen(), gate=NoGate(), judge=ExactMatchJudge())
    harness.run([rag_records[0]])
    assert rag_records[0].query_text in captured["prompt"]
    assert len(captured["context"]) > 0
