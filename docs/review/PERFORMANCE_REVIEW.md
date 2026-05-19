# Performance review

**Method:** Identify bottlenecks by **code structure + complexity** first; **measurement** is required before merging optimizations (see plan below). No benchmarks were executed in the review pass.

---

## 1. Hotspot inventory

| Rank | Location | Pattern | Estimated complexity / risk |
|------|----------|---------|----------------------------|
| H1 | [`topic_clustering.py`](../../blowfish/ingestion/topic_clustering.py) `detect_umap_nneighbours_optimum` | For each trial `u`, full 2D pairwise distance matrix `(n,n)` for histogram | **O(U · n²)** with U≈19 UMAP fits + O(n²) each |
| H2 | [`topic_clustering.py`](../../blowfish/ingestion/topic_clustering.py) `run_hdbscan` | Up to ~98 HDBSCAN fits on n points | **O(H · (n log n or worse))** depending on HDBSCAN path |
| H3 | [`queries_evaluation.py`](../../blowfish/training/queries_evaluation.py) `__call__` | `iterrows` + **single-query** `faiss.Index.search` | **O(Q · (log N_emb + k))** vs batched **O(1)** round-trips |
| H4 | [`calculations.py`](../../blowfish/calculations/calculations.py) VR persistence | giotto-tda Vietoris–Rips per query training example | **O(k³)**–scale or worse in k (k small today but repeated Q times) |
| H5 | [`scorer.py`](../../blowfish/inference/scorer.py) `get_correctness_probability` | **2×** `KernelDensity.score_samples` on 1×1 arrays after log-domain remediation | Constant overhead dominated by Python/sklearn call overhead at scale |
| H6 | [`decider.py`](../../blowfish/inference/decider.py) `FeedbackDecider.__init__` | Rejection sampling loop + SHAP `KernelExplainer` construction | Unbounded worst-case until 50 samples; heavy cold start |
| H7 | KDE at high k | If k raised toward paper's 50, VR cost and feature vector construction grow | Memory for storing k embeddings per query row |

---

## 2. Required profiling commands (review spec §10)

Add thin driver scripts if missing (remediation tasks), then:

```bash
# CPU profile (example entrypoint name — create scripts/benchmark_inference.py in remediation)
python -m cProfile -o profile.out scripts/benchmark_inference.py
python -m pstats profile.out
# interactively: sort cumulative, stats 30
```

```bash
# Optional dependencies
pip install pytest-benchmark scalene memory-profiler
pytest tests/perf/ --benchmark-only
python -m scalene scripts/benchmark_inference.py
python -m memory_profiler scripts/benchmark_inference.py
```

**Lightweight fallback (no extra deps):**

```python
import time, tracemalloc, resource
t0 = time.perf_counter()
tracemalloc.start()
# ... run workload ...
print("wall_s", time.perf_counter() - t0)
print("peak_kb", tracemalloc.get_traced_memory()[1] / 1024)
print("ru_maxrss", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
```

---

## 3. Likely improvements (review spec §11) — evaluation order

| § | Improvement | Repo relevance | Acceptance sketch |
|---|-------------|----------------|-------------------|
| 11.1 | Cache embeddings / norms / NN indices | Training repeats topic table reads; FAISS already persistent | Same scores within tolerance; wall time down |
| 11.2 | Vectorize distance loops | `topic_clustering` pairwise distances already numpy but **avoid full matrix** via sampling or scipy sparse approximations | Benchmark + unit tests |
| 11.3 | Optional ANN | Currently exact flat L2; for huge N consider IVF/HNSW **behind interface** | Exact path default in CI |
| 11.4 | Reduce dim before KDE **if** KDE moves to raw embeddings | Not current path; document if features expand | Validation AUROC maintained |
| 11.5 | Replace / compare KDE to kNN density | Strongly recommended for baseline comparison | See [`BASELINE_RESULTS.md`](./BASELINE_RESULTS.md) |
| 11.6 | Log-domain KDE ratio | **Implemented** — [`scorer.py`](../../blowfish/inference/scorer.py), [`decider.py`](../../blowfish/inference/decider.py) | Regression test covers naive-equivalent finite case; add extreme tail fixture |
| 11.7 | Batch inference | **`score(X_query)`** matrix API | Batch == loop within tol, faster wall time |

---

## 4. Measurement plan (pre-merge gate)

1. **Baseline:** tag/commit + record `wall_time`, peak RSS on **Q ∈ {100, 1000}**, **n_chunks ∈ {1k, 10k}**, **d ∈ {32, 128, 768}** (synthetic or subsampled real).
2. **Regression ceiling:** any change that touches scoring must not move **>ε** on golden fixture logits without documented acceptance.
3. **Profiling artifact:** attach `profile.out` summary + 5-line takeaway to PR description.

---

## 5. Cross-references

- Tasks: TASK-102, TASK-105, TASK-106, TASK-107, [`AGENTIC_REMEDIATION_SPEC.md`](./AGENTIC_REMEDIATION_SPEC.md)
- Tests: [`TESTING_AND_PROFILING_PLAN.md`](./TESTING_AND_PROFILING_PLAN.md)
