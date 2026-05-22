# Baseline results

**Purpose:** record head-to-head performance of the B0–B9 baselines defined in
`PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md` §4.1, against Blowfish KDE+topology, on a
held-out test set. The B-table numbering below is the **canonical** spec
(updated from an earlier draft in this file).

**Dataset:** _(name, train / val / test split, embargo rules)_

**Label:** `correct_prediction` (or operational ambiguity label) — define explicitly.

**Metrics:** AUROC, AUPRC, Brier, ECE, NLL, FPR@TPR, latency/query, peak RSS.

## How to run

```bash
pip install -e ".[evaluation,datasets,rag]"
python -m blowfish.experiments.bench_baselines --dummy --bootstrap 500 --seed 0
```

For a real corpus, build a list of `blowfish.evaluation.RetrievalRecord`
objects and call `blowfish.experiments.ExperimentRunner.run_baselines(...)`
directly — the CLI surface is intentionally minimal (synthetic-fixture smoke
only).

## Baselines (B0–B9)

| ID | Definition | Implementation | AUROC ± 95% CI (val) | AUPRC | Notes |
|---|---|---|---|---|---|
| B0 | **Random** scoring | [`blowfish/baselines/b0_random.py`](../../blowfish/baselines/b0_random.py) `RandomBaseline` | _to fill_ | _to fill_ | Sanity floor; AUROC ≈ 0.5 |
| B1 | **Top-1 retrieval score** | [`b1_top1.py`](../../blowfish/baselines/b1_top1.py) `Top1ScoreBaseline` | | | Cheapest baseline |
| B2 | **Score gap** ``score_2 − score_1`` (distance-like) | [`b2_score_gap.py`](../../blowfish/baselines/b2_score_gap.py) `ScoreGapBaseline` | | | Common heuristic, ~free |
| B3 | **Top-k score entropy** (softmax) | [`b3_score_entropy.py`](../../blowfish/baselines/b3_score_entropy.py) `ScoreEntropyBaseline` | | | Distribution-aware |
| B4 | **Mean kNN distance** | [`b4_knn_distance.py`](../../blowfish/baselines/b4_knn_distance.py) `MeanKnnDistanceBaseline` | | | Geometric analogue without topology |
| B5 | **kNN density** = 1 / mean kNN distance | [`b5_knn_density.py`](../../blowfish/baselines/b5_knn_density.py) `KnnDensityBaseline` | | | Density baseline for KDE comparison |
| B6 | **Mahalanobis** from query to top-k centroid (regularized + pinv) | [`b6_mahalanobis.py`](../../blowfish/baselines/b6_mahalanobis.py) `MahalanobisCentroidBaseline` | | | Covariance-aware |
| B7 | **Calibrated logistic** on the **same** Blowfish features | [`b7_calibrated_logistic.py`](../../blowfish/baselines/b7_calibrated_logistic.py) `CalibratedLogisticBaseline` | | | Isolates KDE vs discriminative on same inputs |
| B8 | **GBM** on the same features (default `sklearn` params) | [`b8_gbm.py`](../../blowfish/baselines/b8_gbm.py) `GBMBaseline` | | | Stronger discriminative baseline |
| B9 | **Oracle** — gold rank in top-k | [`b9_oracle.py`](../../blowfish/baselines/b9_oracle.py) `OracleBaseline` | | | Upper bound for gating use case |

Reporting each row with AUROC ± 95% paired-bootstrap CI is the minimum bar.
Blowfish's claim of being "topological" is only meaningful if it beats B7/B8
on the same features.

## Proposed Blowfish variants

| Variant ID | Features | AUROC | Notes |
|---|---|---|---|
| V0 | KDE full default | | Reference |
| V1 | Topology features zeroed / ablated | | Feeds §G.1 of the ABLATION_PLAN |
| V2 | KDE without topology columns | | Direct apples-to-apples vs B7 |

## Calibration table

| Method | ECE | Brier | Comment |
|---|---|---|---|
| Blowfish KDE | | | |
| Blowfish + isotonic | | | Via `blowfish.evaluation.CalibratedScorer(method="isotonic")` |
| B7 (already-calibrated) | | | |

## Runtime

| Method | Fit (s) | Infer / query (ms) | Peak RSS (MB) |
|---|---|---|---|
| | | | |

## Verdict (`METHODOLOGY_REVIEW.md` §5)

**Do topology + KDE beat the best simple baseline on held-out data?** _(Yes / No / Not run)_

**If No:** topology/KDE are not justified by the empirical evidence; report
as exploratory or remove.

**Evidence path:** link to the experiment run (commit hash, manifest hash,
PR), the CSV output of `bench_baselines --csv`, and the prereg lock file
under `~/.cache/blowfish/prereg/`.

## Cross-references

- [`ABLATION_PLAN.md`](./ABLATION_PLAN.md) — feature ablations + R0/R1/R2 randomized controls.
- [`METHODOLOGY_REVIEW.md`](./METHODOLOGY_REVIEW.md) — paper-vs-implementation traceability.
- [`PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md`](../../PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md) §4.1 — canonical B0–B9 spec.
