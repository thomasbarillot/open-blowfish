"""End-to-end RAG harness: retrieve → gate → generate → judge → score."""

from __future__ import annotations

from typing import Any, ClassVar, Optional, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from blowfish.evaluation.types import RetrievalRecord
from blowfish.rag.cost import CostModel
from blowfish.rag.gates import Gate
from blowfish.rag.generator import Generator
from blowfish.rag.judge import AnswerJudge, ExactMatchJudge


DEFAULT_PROMPT_TEMPLATE = (
    "Answer the question using only the provided context. If the context "
    "is insufficient, say so.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)


class RAGHarness(BaseModel):
    """Per-query pipeline: ``retrieve → gate → generate → judge → score``.

    ``records`` come in already-retrieved (the harness does **not** call
    FAISS itself — that's the upstream retriever's job). For each record:

    1. ``gate.score(record)`` and ``gate.should_abstain(record)`` decide
       whether to call the generator.
    2. If not abstaining, the prompt template is formatted with the query
       text and concatenated chunk text; ``generator.generate(prompt, context)``
       returns a :class:`GenerationResult`.
    3. ``judge.judge(predicted, gold, record)`` returns EM/F1/correct.
    4. The cost model converts (correct / wrong / abstain) into a scalar
       utility.

    Output: a pandas DataFrame with one row per query, columns
    ``query_id, action, gate_score, predicted, latency_ms, em, f1, correct,
    utility``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    generator: Any  # Generator Protocol
    gate: Any  # Gate Protocol
    judge: Any = Field(default_factory=ExactMatchJudge)
    cost_model: CostModel = Field(default_factory=CostModel)
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    context_size: int = 5

    def _format_prompt(self, record: RetrievalRecord) -> tuple[str, list[str]]:
        question = record.query_text or ""
        chunks = [c.text for c in record.top_k[: self.context_size] if c.text is not None]
        context_block = "\n---\n".join(chunks) if chunks else "[no context available]"
        prompt = self.prompt_template.format(question=question, context=context_block)
        return prompt, chunks

    def run(self, records: Sequence[RetrievalRecord]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for record in records:
            row: dict[str, Any] = {"query_id": record.query_id}
            gate_score = float(self.gate.score(record))
            row["gate_score"] = gate_score
            if self.gate.should_abstain(record):
                row["action"] = "abstain"
                row["predicted"] = None
                row["latency_ms"] = 0.0
                row["em"] = float("nan")
                row["f1"] = float("nan")
                row["correct"] = None
                row["utility"] = self.cost_model.abstain
                rows.append(row)
                continue
            prompt, _chunks = self._format_prompt(record)
            context = [c.text for c in record.top_k[: self.context_size] if c.text is not None]
            gen_result = self.generator.generate(prompt, context)
            row["action"] = "answer"
            row["predicted"] = gen_result.text
            row["latency_ms"] = float(gen_result.latency_ms)
            judge_result = self.judge.judge(
                predicted=gen_result.text,
                gold=record.gold_text or "",
                record=record,
            )
            row["em"] = float(judge_result.em)
            row["f1"] = float(judge_result.f1)
            row["correct"] = bool(judge_result.correct)
            row["utility"] = (
                self.cost_model.correct if judge_result.correct else self.cost_model.wrong
            )
            rows.append(row)
        return pd.DataFrame(rows)
