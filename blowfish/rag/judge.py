"""Answer judges: predicted text + gold → correctness verdict.

Three concrete judges:

- :class:`ExactMatchJudge` — string-equality after whitespace normalization.
- :class:`F1Judge` — token-level F1 (SQuAD-style); ``correct`` if F1 ≥ 0.5.
- :class:`OracleAnswerabilityJudge` — ignores predicted text; marks correct
  iff the gold chunk was retrieved into the top-k. Used as an upper-bound
  reference in RAG benchmarks.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from blowfish.evaluation.types import RetrievalRecord


class JudgeResult(BaseModel):
    """Per-query judge verdict."""

    model_config = ConfigDict(extra="ignore")

    em: int  # 0 or 1
    f1: float
    correct: bool


@runtime_checkable
class AnswerJudge(Protocol):
    """Common judge interface."""

    name: ClassVar[str]

    def judge(
        self,
        *,
        predicted: str,
        gold: str,
        record: Optional[RetrievalRecord] = None,
    ) -> JudgeResult: ...


_WS_RE = re.compile(r"\s+")
# Hyphen included so "forty-two" normalizes to "forty two" — same convention
# as SQuAD-style EM scoring.
_PUNCT_RE = re.compile(r"[\.,;:!\?\(\)\[\]\"\'`\-]+")


def _normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


class ExactMatchJudge:
    """String-equality after lowercase + whitespace + punctuation normalization."""

    name: ClassVar[str] = "exact_match"

    def judge(
        self,
        *,
        predicted: str,
        gold: str,
        record: Optional[RetrievalRecord] = None,
    ) -> JudgeResult:
        match = _normalize(predicted) == _normalize(gold)
        return JudgeResult(em=int(match), f1=float(match), correct=match)


class F1Judge:
    """Token-level F1. ``correct`` when F1 ≥ threshold (default 0.5)."""

    name: ClassVar[str] = "f1"

    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def judge(
        self,
        *,
        predicted: str,
        gold: str,
        record: Optional[RetrievalRecord] = None,
    ) -> JudgeResult:
        p_toks = _normalize(predicted).split()
        g_toks = _normalize(gold).split()
        if not g_toks:
            f1 = 0.0
        elif not p_toks:
            f1 = 0.0
        else:
            p_set, g_set = set(p_toks), set(g_toks)
            tp = len(p_set & g_set)
            precision = tp / len(p_set)
            recall = tp / len(g_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        em = int(_normalize(predicted) == _normalize(gold))
        return JudgeResult(em=em, f1=float(f1), correct=f1 >= self.threshold)


class OracleAnswerabilityJudge:
    """Judges whether the gold chunk was retrieved — independent of generated text.

    Useful as an upper-bound reference: this judge marks the same records
    "correct" that the G6 oracle gate would let through. Comparing a real
    generator + EM/F1 against this judge surfaces the generator's contribution
    separately from the retriever's.
    """

    name: ClassVar[str] = "oracle_answerability"

    def judge(
        self,
        *,
        predicted: str,
        gold: str,
        record: Optional[RetrievalRecord] = None,
    ) -> JudgeResult:
        if record is None or record.gold_chunk_hash is None:
            return JudgeResult(em=0, f1=0.0, correct=False)
        had_gold = any(c.hash_key == record.gold_chunk_hash for c in record.top_k)
        return JudgeResult(em=int(had_gold), f1=float(had_gold), correct=had_gold)


class JudgeHooks:
    """Class-attribute registry of judges."""

    exact_match: ClassVar = ExactMatchJudge
    f1: ClassVar = F1Judge
    oracle_answerability: ClassVar = OracleAnswerabilityJudge


ALL_JUDGE_NAMES = ("exact_match", "f1", "oracle_answerability")
