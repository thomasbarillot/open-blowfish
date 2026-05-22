"""CLI: ``python -m blowfish.experiments.bench_baselines``.

Runs B0–B9 against a record set and prints an AUROC ± 95% bootstrap CI
table. ``--dummy`` generates a synthetic fixture for smoke testing without
needing a real corpus.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import numpy as np

from blowfish.baselines.factory import ALL_BASELINE_IDS, BaselineHooks
from blowfish.evaluation.types import RetrievalRecord, RetrievedChunk
from blowfish.experiments.harness import ExperimentRunner
from blowfish.experiments.reports import baseline_table, write_csv


def _synthetic_records(n: int = 24, k: int = 20, dim: int = 16, *, seed: int = 0) -> list[RetrievalRecord]:
    """Same shape as ``tests/baselines/conftest.py``: half correct, half
    incorrect, with realistic top-k geometry."""
    rng = np.random.default_rng(seed)
    records: list[RetrievalRecord] = []
    for i in range(n):
        is_correct = i % 2 == 0
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        chunks_data: list[tuple[np.ndarray, float]] = []
        if is_correct:
            gold = q + rng.normal(0, 0.02, size=dim)
            chunks_data.append((gold, float(np.linalg.norm(gold - q))))
            for _ in range(k - 1):
                c = q + rng.normal(0, 0.15, size=dim)
                chunks_data.append((c, float(np.linalg.norm(c - q))))
        else:
            for _ in range(k):
                c = rng.normal(size=dim)
                c /= np.linalg.norm(c)
                chunks_data.append((c, float(np.linalg.norm(c - q))))
        chunks_data.sort(key=lambda kv: kv[1])
        chunks = []
        for rank, (emb, dist) in enumerate(chunks_data):
            chunks.append(
                RetrievedChunk(
                    hash_key=f"q{i}_c{rank}",
                    docname=f"doc{i % 6}",
                    chunk_embedding=emb,
                    score=dist,
                    rank=rank,
                    topic_label=f"doc{i % 6}_0",
                    silhouette_score=0.5,
                )
            )
        records.append(
            RetrievalRecord(
                query_id=f"q{i}",
                query_embedding=q,
                top_k=chunks,
                gold_chunk_hash=chunks[0].hash_key if is_correct else None,
                correct_prediction=int(is_correct),
            )
        )
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="blowfish.experiments.bench_baselines",
        description="Run B0–B9 against a record set; print AUROC ± 95% CI.",
    )
    parser.add_argument(
        "--dummy", action="store_true", help="Generate a synthetic 24-query fixture."
    )
    parser.add_argument(
        "--baselines",
        default=",".join(ALL_BASELINE_IDS),
        help="Comma-separated baseline IDs (default: B0..B9).",
    )
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", help="Optional path to also write a CSV.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.dummy:
        parser.error("Only --dummy is supported in this CLI surface. "
                     "Real-corpus runs go through ExperimentRunner directly.")

    records = _synthetic_records(seed=args.seed)
    baseline_ids = [b.strip() for b in args.baselines.split(",") if b.strip()]
    baselines = [getattr(BaselineHooks, bid)() for bid in baseline_ids]

    runner = ExperimentRunner(records=records, bootstrap_n=args.bootstrap, seed=args.seed)
    df = runner.run_baselines(baselines)
    print(baseline_table(df.to_dict("records")))
    if args.csv:
        write_csv(df.to_dict("records"), args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
