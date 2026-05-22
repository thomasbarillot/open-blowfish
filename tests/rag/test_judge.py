"""Phase 4 — ExactMatch, F1, OracleAnswerability judges."""

from __future__ import annotations

import numpy as np
import pytest

from blowfish.evaluation.types import RetrievalRecord, RetrievedChunk
from blowfish.rag.judge import (
    ExactMatchJudge,
    F1Judge,
    JudgeResult,
    OracleAnswerabilityJudge,
)


def _mk_record(gold_in_topk: bool) -> RetrievalRecord:
    chunks = [
        RetrievedChunk(
            hash_key="h_gold" if gold_in_topk else "h_not_gold",
            docname="d",
            chunk_embedding=np.zeros(4),
            score=0.0,
            rank=0,
            text="some chunk text",
        )
    ]
    return RetrievalRecord(
        query_id="q",
        query_embedding=np.zeros(4),
        top_k=chunks,
        gold_chunk_hash="h_gold",
        gold_text="forty two",
    )


def test_exact_match_judge_normalizes_whitespace_and_punctuation():
    j = ExactMatchJudge()
    r = j.judge(predicted="Forty-two.", gold="forty two")
    assert r.correct is True
    assert r.em == 1


def test_exact_match_judge_rejects_different_strings():
    j = ExactMatchJudge()
    r = j.judge(predicted="something else", gold="forty two")
    assert r.correct is False
    assert r.em == 0


def test_f1_judge_perfect_match():
    j = F1Judge()
    r = j.judge(predicted="the answer is forty two", gold="forty two")
    # All gold tokens in predicted, plus extras → high recall, lower precision
    assert r.f1 > 0.5
    assert r.correct is True


def test_f1_judge_no_overlap():
    j = F1Judge()
    r = j.judge(predicted="something entirely different", gold="forty two")
    assert r.f1 == 0.0
    assert r.correct is False


def test_f1_judge_threshold_configurable():
    strict = F1Judge(threshold=0.9)
    loose = F1Judge(threshold=0.1)
    # Partial overlap
    strict_r = strict.judge(predicted="forty other", gold="forty two")
    loose_r = loose.judge(predicted="forty other", gold="forty two")
    assert loose_r.correct is True
    assert strict_r.correct is False


def test_oracle_answerability_correct_when_gold_in_topk():
    j = OracleAnswerabilityJudge()
    record = _mk_record(gold_in_topk=True)
    r = j.judge(predicted="anything", gold="any gold", record=record)
    assert r.correct is True


def test_oracle_answerability_wrong_when_gold_missing():
    j = OracleAnswerabilityJudge()
    record = _mk_record(gold_in_topk=False)
    r = j.judge(predicted="anything", gold="any gold", record=record)
    assert r.correct is False


def test_oracle_answerability_ignores_predicted_text():
    """The oracle judges by retrieval, not generator output."""
    j = OracleAnswerabilityJudge()
    record = _mk_record(gold_in_topk=True)
    r1 = j.judge(predicted="totally wrong answer", gold="anything", record=record)
    r2 = j.judge(predicted="the right answer", gold="anything", record=record)
    assert r1.correct == r2.correct  # both true; predicted irrelevant


def test_judge_result_pydantic_round_trip():
    r = JudgeResult(em=1, f1=0.75, correct=True)
    rebuilt = JudgeResult.model_validate_json(r.model_dump_json())
    assert rebuilt == r
