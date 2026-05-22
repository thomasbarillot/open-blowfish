"""CLI: ``python -m blowfish.experiments.bench_rag``.

Runs G0/G1/G2/G3/G5/G6 against a synthetic RAG fixture (G4 is omitted from
the CLI smoke surface because it requires a fitted AmbiguityScorer — wire it
up directly via :class:`ExperimentRunner` for real runs).

The generator defaults to ``echo`` (deterministic, no LLM needed).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import numpy as np

from blowfish.evaluation.types import RetrievalRecord, RetrievedChunk
from blowfish.experiments.harness import ExperimentRunner
from blowfish.experiments.reports import rag_table, write_csv
from blowfish.rag.cost import CostModel
from blowfish.rag.gates import GateHooks
from blowfish.rag.generator import EchoGenerator


_DEFAULT_GATES = "G0,G1,G2,G3,G5,G6"


def _synthetic_rag_records(n: int = 20, k: int = 10, dim: int = 16, *, seed: int = 0) -> list[RetrievalRecord]:
    """Same shape as ``tests/rag/conftest.py``: each correct query's rank-0
    chunk text is the gold answer, so EchoGenerator + F1 judge produces a
    meaningful signal without an LLM dependency."""
    rng = np.random.default_rng(seed)
    records: list[RetrievalRecord] = []
    for i in range(n):
        is_correct = i % 2 == 0
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        gold_text = f"the answer to query {i} is forty two"
        items: list[tuple[np.ndarray, str]] = []
        if is_correct:
            items.append((q + rng.normal(0, 0.02, size=dim), gold_text))
            for j in range(k - 1):
                items.append((q + rng.normal(0, 0.15, size=dim), f"distractor {i}-{j}"))
        else:
            for j in range(k):
                emb = rng.normal(size=dim)
                emb /= np.linalg.norm(emb)
                items.append((emb, f"random_chunk_{i}_{j}"))
        items.sort(key=lambda kv: float(np.linalg.norm(kv[0] - q)))
        chunks = []
        for rank, (emb, text) in enumerate(items):
            chunks.append(
                RetrievedChunk(
                    hash_key=f"q{i}_c{rank}",
                    docname=f"doc{i % 4}",
                    chunk_embedding=emb,
                    score=float(np.linalg.norm(emb - q)),
                    rank=rank,
                    text=text,
                    topic_label=f"doc{i % 4}_0",
                    silhouette_score=0.5,
                )
            )
        records.append(
            RetrievalRecord(
                query_id=f"q{i}",
                query_embedding=q,
                top_k=chunks,
                query_text=f"What is the answer to query {i}?",
                gold_text=gold_text if is_correct else "an answer that no chunk contains",
                gold_chunk_hash=chunks[0].hash_key if is_correct else None,
                correct_prediction=int(is_correct),
            )
        )
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="blowfish.experiments.bench_rag",
        description="Run G0–G6 (minus G4) against a synthetic RAG fixture.",
    )
    parser.add_argument(
        "--dummy", action="store_true", help="Use a synthetic 20-query fixture (required)."
    )
    parser.add_argument(
        "--gates",
        default=_DEFAULT_GATES,
        help="Comma-separated gate IDs (default: %(default)s).",
    )
    parser.add_argument("--generator", default="echo")
    parser.add_argument("--cost-wrong", type=float, default=-3.0)
    parser.add_argument("--abstain-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", help="Optional path to also write a CSV.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.dummy:
        parser.error("Only --dummy is supported in this CLI surface. "
                     "Real-corpus runs go through ExperimentRunner directly.")

    records = _synthetic_rag_records(seed=args.seed)
    gate_ids = [g.strip() for g in args.gates.split(",") if g.strip()]
    gates = []
    for gid in gate_ids:
        if gid == "G4":
            print(f"# skipping {gid} — requires a fitted AmbiguityScorer (use ExperimentRunner directly)",
                  file=sys.stderr)
            continue
        gates.append(getattr(GateHooks, gid)())
    if args.generator != "echo":
        parser.error("Only --generator echo is supported in the smoke CLI. "
                     "Wire other generators via ExperimentRunner directly.")
    generator = EchoGenerator()
    cost = CostModel(wrong=args.cost_wrong)
    runner = ExperimentRunner(records=records, seed=args.seed)
    df = runner.run_rag(
        gates=gates,
        generator=generator,
        cost_model=cost,
        abstain_rate=args.abstain_rate,
    )
    print(rag_table(df.to_dict("records")))
    if args.csv:
        write_csv(df.to_dict("records"), args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
