"""Profile the eval-PR critical paths against pre-declared budgets.

Run from the repo root after installing the relevant extras::

    pip install -e ".[evaluation,datasets,rag]"
    python scripts/profile_eval_pr.py

Output: a markdown table {case → seconds → budget → PASS/FAIL}. Optional
``--csv <path>`` writes the same rows to disk for later comparison.

Cases mirror the six budgets pre-declared in the pre-push sanity-sweep plan
(``docs/review/PRE_PUSH_SANITY_REVIEW.md`` once the review lands).
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional

import numpy as np


def _time_block(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def case_bench_baselines() -> float:
    cmd = [
        sys.executable,
        "-m",
        "blowfish.experiments.bench_baselines",
        "--dummy",
        "--bootstrap",
        "500",
        "--seed",
        "0",
    ]
    start = time.perf_counter()
    subprocess.run(cmd, capture_output=True, check=True, timeout=300)
    return time.perf_counter() - start


def case_bench_rag() -> float:
    cmd = [
        sys.executable,
        "-m",
        "blowfish.experiments.bench_rag",
        "--dummy",
        "--gates",
        "G0,G1,G2,G3,G5,G6",
        "--seed",
        "0",
    ]
    start = time.perf_counter()
    subprocess.run(cmd, capture_output=True, check=True, timeout=300)
    return time.perf_counter() - start


def case_b7_fit_and_score() -> float:
    from blowfish.baselines.b7_calibrated_logistic import CalibratedLogisticBaseline
    from blowfish.experiments.bench_baselines import _synthetic_records

    records = _synthetic_records(n=24, k=20, dim=16, seed=0)
    labels = np.asarray([r.correct_prediction for r in records], dtype=int)
    baseline = CalibratedLogisticBaseline(random_state=0)
    start = time.perf_counter()
    baseline.fit(records, labels)
    baseline.score(records)
    return time.perf_counter() - start


def case_auroc_bca_bootstrap_10k() -> float:
    from blowfish.evaluation.bootstrap import bootstrap_metric
    from blowfish.evaluation.metrics import auroc

    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 2, size=n)
    scores = rng.random(size=n) * 0.5 + labels * 0.3
    start = time.perf_counter()
    bootstrap_metric(auroc, scores, labels, n=10_000, method="bca", seed=0)
    return time.perf_counter() - start


def case_vr_persistence_per_query() -> float:
    import pandas as pd

    from blowfish.calculations.calculations import calculate_vr_persistence_features

    rng = np.random.default_rng(0)
    n_iter = 20
    chunks_embed = rng.normal(size=(50, 16))
    chunks_embed /= np.linalg.norm(chunks_embed, axis=1, keepdims=True)
    query_embed = rng.normal(size=16)
    query_embed /= np.linalg.norm(query_embed)
    sub_df = pd.DataFrame(
        {
            "chunk_embeddings": [list(row) for row in chunks_embed],
            "query_embedding": [list(query_embed)] * 50,
        }
    )
    start = time.perf_counter()
    for _ in range(n_iter):
        calculate_vr_persistence_features(sub_df)
    elapsed = time.perf_counter() - start
    return elapsed / n_iter


def case_corpus_cache_hit() -> float:
    """Hit the cached path for NobelPhysics — no network calls expected."""
    from blowfish.datasets.corpora import NobelPhysics

    corpus = NobelPhysics()
    start = time.perf_counter()
    docs = list(corpus.iter_documents())
    elapsed = time.perf_counter() - start
    if not docs:
        # Skip — bootstrap script wasn't run. Return -1 to mark "skipped".
        return -1.0
    return elapsed


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Optional path to write a CSV.")
    args = parser.parse_args(argv)

    cases = [
        ("bench_baselines (24 queries, bootstrap=500)", case_bench_baselines, 30.0),
        ("bench_rag (20 queries, 6 gates)", case_bench_rag, 30.0),
        ("B7 fit + score (24 records, VR per query)", case_b7_fit_and_score, 60.0),
        ("AUROC BCa bootstrap n=10_000 on 200 records", case_auroc_bca_bootstrap_10k, 60.0),
        ("VR persistence per query (50 pts, 16-dim, avg over 20)", case_vr_persistence_per_query, 0.1),
        ("Corpus.iter_documents() (Nobel cache hit, 8 docs)", case_corpus_cache_hit, 1.0),
    ]

    rows: list[tuple[str, float, float, str]] = []
    for name, fn, budget in cases:
        try:
            elapsed = fn()
        except Exception as exc:  # noqa: BLE001
            rows.append((name, float("nan"), budget, f"ERROR: {exc}"))
            continue
        if elapsed < 0:
            verdict = "SKIPPED (cache empty)"
        else:
            verdict = "PASS" if elapsed <= budget else "FAIL"
        rows.append((name, elapsed, budget, verdict))

    print("\n| case | seconds | budget | verdict |")
    print("| --- | --- | --- | --- |")
    for name, elapsed, budget, verdict in rows:
        elapsed_s = "—" if elapsed != elapsed else f"{elapsed:.3f}"  # nan check
        if isinstance(elapsed, float) and not (elapsed != elapsed):
            elapsed_s = f"{elapsed:.3f}"
        print(f"| {name} | {elapsed_s} | {budget:.2f} | {verdict} |")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "seconds", "budget_s", "verdict"])
            for row in rows:
                w.writerow(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
