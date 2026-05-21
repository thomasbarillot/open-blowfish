# Testing and profiling plan

This document defines **unit**, **statistical**, **topology**, and **performance regression** coverage required before treating remediation as complete (review spec §12–13).

---

## 1. Unit tests (required categories)

| Category | Example | Target modules |
|----------|---------|----------------|
| Shape handling | Empty `features_df` → defined error | [`scorer.py`](../../blowfish/inference/scorer.py) |
| Empty input | No rows after merge | [`scorer.py`](../../blowfish/inference/scorer.py) |
| Single sample | Query with 1 retrieved chunk — training skips; inference path explicit | [`disambiguator_training.py`](../../blowfish/training/disambiguator_training.py) |
| Duplicate embeddings | Identical chunk vectors | [`calculations.py`](../../blowfish/calculations/calculations.py) |
| High-dimensional `d` | 768-d with k=5 | VR numerical stability |
| Norm. vs unnorm. | Cosine path vs L2 path once TASK-006 fixed | ingestion + calculations |
| Deterministic seed | UMAP/HDBSCAN optional seed where supported | [`topic_clustering.py`](../../blowfish/ingestion/topic_clustering.py) |
| Serialization roundtrip | KDE pickle + scorer expected columns | training + inference |
| Batch inference equiv. | `score(X)` vs loop | [`scorer.py`](../../blowfish/inference/scorer.py) |
| Unique `hash_key` | Assert failure or groupby policy | [`queries_evaluation.py`](../../blowfish/training/queries_evaluation.py) |

**Existing tests:** [`tests/inference/test_scorer.py`](../../tests/inference/test_scorer.py), [`tests/training/test_disambiguator_training.py`](../../tests/training/test_disambiguator_training.py) — extend; remove module-level shared `AmbiguityScorer` fixture (TASK-204).

---

## 2. Statistical tests

| Test | Assertion | Notes |
|------|-------------|-------|
| KDE log-density vs sklearn | On 2D toy Gaussian mixture, `score_samples` matches within atol | Fit same bandwidth |
| Monotone distance | For isotropic Gaussian cloud, density( center ) > density(far tail) | Label-conditional if joint model |
| Bandwidth direction | Increasing h smooths log-density curve on grid | Smoke |
| Calibration curve | On labeled validation, reliability plot stored as artifact | After TASK-104 |

---

## 3. Topology tests (tiny deterministic fixtures)

| Scenario | Expectation |
|----------|-------------|
| 3 well-separated clusters, k=3 | At small VR threshold / seeded diagram, component count behavior documented — **golden-file** persistence diagram hash or scalar tolerance |
| Merge at large scale | Collapse toward single H₀ death ordering |
| Known graph | If switching to explicit distance matrix mode, assert **known** β₀ |
| Paper metrics (post TASK-001) | `W₁(H₀)` on hand-computed 3-point cloud |
| H₁ lifetime | Toy square/circle point sets with known loop (where k permits) |

Use **fixed random seed** and **atol** generous to library version drift; pin `giotto-tda` minor version in CI.

---

## 4. Performance regression tests

**Matrix (required):**

| Corpus size `n` | Embedding dim `d` | Measures |
|-----------------|-------------------|----------|
| 100 | 32, 128, 768 | fit clustering (optional), VR time, KDE `score_samples` |
| 1_000 | 32, 128, 768 | + bulk FAISS search |
| 10_000 | 32, 128, 768 | + memory cap check (soft: abort if RSS > budget) |

**Record:** wall time, peak RSS, **optional** `pytest-benchmark` median.

**Gate:** thresholds stored in YAML; CI runs `--benchmark-compare` against main on weekly schedule or manual.

---

## 5. Profiling commands

See [`PERFORMANCE_REVIEW.md`](./PERFORMANCE_REVIEW.md) §2.

**Deliverables to add in remediation:**

- `scripts/benchmark_inference.py` — synthetic `features_df` generator + `AmbiguityScorer.run_scoring` loop
- `scripts/benchmark_training.py` — subset `DisambiguationModelGenerator.__call__`
- `tests/perf/test_regression.py` — wraps benchmarks with size limits for CI

---

## 6. Acceptance standard (review spec §13)

Remediation complete when:

- [ ] Ambiguity definition documented and tested
- [ ] Baselines implemented + table in [`BASELINE_RESULTS.md`](./BASELINE_RESULTS.md)
- [ ] Assumptions doc linked from README
- [ ] KDE + topology **golden** tests
- [ ] Profiling scripts committed
- [ ] Batch inference API + equiv. tests
- [ ] Stable log-density path
- [ ] `experiments/` vs package split (if applicable)

---

## 7. Cross-references

- [`AGENTIC_REMEDIATION_SPEC.md`](./AGENTIC_REMEDIATION_SPEC.md)
- [`ABLATION_PLAN.md`](./ABLATION_PLAN.md)
