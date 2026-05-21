# Methodology review (statistics + topology + experiments)

**Companion:** Barillot & De Castro, *Topological quantification of ambiguity in semantic search*, arXiv:2406.07990.

This document evaluates **mathematical correctness**, **statistical validity**, **experimental support**, and **engineering value** (per review rubric) for KDE, persistent homology, and paper experiments — and maps them to **this repository's code**.

---

## 1. Code ↔ paper traceability

| Paper symbol / claim | Paper definition / location | Code / repo | Match? |
|----------------------|----------------------------|-------------|--------|
| `W₁(H₀)` | Eq. (2), (5): normalized sum of vertical distances of H₀ diagram points to diagonal | `w1_h0`; helper `paper_w1_h0_from_birth_death` computes finite-bar half-distance mean | **YES for formula helper; needs end-to-end replication** |
| `LT_max(H₁)` | Eq. (3): sup of loop lifetimes | `lt_max_h1`; helper `paper_lt_max_h1` computes maximum finite H₁ half-lifetime | **YES for formula helper; needs end-to-end replication** |
| ε scaling / neighborhood | §3.2: scale relative to NN | `paper_eq1_scaled_neighbor_distances` + `epsilon=` kwarg on `calculate_vr_persistence_features` (approach A: subsample neighbors with `d(i,q) ≤ ε`, fall back to full neighborhood below `n_min`); plumbed through `AmbiguityScorer` and `DisambiguationModelGenerator` | **YES for code path; ε sweep on real data still open** |
| top-k retrieval | §3.6: k = **50** | README example now uses `top_k_results: 50`; evaluator warns below 15 | **Partial** |
| Similarity for FAISS | §3.6: **cosine** | `vdb_metric="cosine"` uses unit-normalized vectors + `IndexFlatIP`; `l2` remains legacy default | **Partial** |
| Nobel corpus + 4 models | §3.6, Table 2 | Not in repo; embedders exist but no experiment harness | **Partial** |
| Simulation appendix B | Synthetic orthogonal topics | Not shipped in package | **N/A (paper-only)** |

---

## 2. Statistical methodology: Gaussian KDE

### 2.1 KDE in the repo

- **Fitter:** `sklearn.neighbors.KernelDensity(bandwidth=0.2, kernel="gaussian")` ([`disambiguator_training.py`](../../blowfish/training/disambiguator_training.py)).
- **Training matrix `X`:** `balanced_queries_features.drop(columns=["query_id"])`, i.e. first column **`correct_prediction` ∈ {0,1}** then 11 continuous features (spread, silhouette, VR summaries, score ratios).
- **Inference:** `p_correct ∝ exp(log f(1, x)) / (exp(log f(1,x)) + exp(log f(0,x)))` ([`scorer.py`](../../blowfish/inference/scorer.py)) — a **likelihood ratio** from the **joint** KDE over `(y, x)`; **not** a properly trained binary classifier unless bandwidth and feature scaling are validated.

### 2.2 Required KDE questions (review spec §7.1)

| Question | Answer (this repo) |
|----------|-------------------|
| Embedding dimensionality `d` | Model-dependent (e.g. 768); **KDE does not operate on raw embeddings** — only on **hand-engineered scalars** (dimension ~11 continuous + label). |
| Sample size for KDE | Number of training queries after balancing min(class counts); **not** documented or asserted in code. |
| Bandwidth rule | **Fixed 0.2** — no Scott/Silverman, no cross-validation, no per-feature scaling. |
| Covariance | Isotropic Gaussian kernel in **12-D** joint space (default sklearn behavior: scalar bandwidth). |
| Normalized embeddings? | Chunk/query embeddings may be L2-normalized by model; **KDE inputs are not** normalized together (mixed units: spread fractions, homology scalars, silhouette). |
| Cosine vs Euclidean | Cosine FAISS path exists, but legacy default remains L2; experiments must record `vdb_metric`. |
| KDE space vs reduced space | Features mix **2D UMAP-derived** cluster identities and **high-D** retrieval geometry summaries — interpretability is mixed. |
| Density = uncertainty? | **Not calibrated** to error rate; must not be called "probability" without calibration study. |

### 2.3 Required KDE checks (review spec §7.2)

These belong in the test/benchmark layer (see [`TESTING_AND_PROFILING_PLAN.md`](./TESTING_AND_PROFILING_PLAN.md)); **status: not implemented.**

| Check | Purpose |
|-------|---------|
| Bandwidth sensitivity | Monotone or stable behavior vs `h` on known toy mixture |
| Sample size sensitivity | Learning curve for score quality |
| Embedding-dimension sensitivity | If KDE ever moved to raw embeddings |
| Log-density stability | `logsumexp`, no naked `exp(score_samples)` for large negative values |
| vs kNN density / Mahalanobis / score-gap | Baselines per review spec §3.3 |
| Calibration vs empirical error | Reliability diagrams, Brier, ECE on held-out Q |

### 2.4 Required warning (review spec §7.3)

**Gaussian KDE on high-dimensional spaces is fragile.** Here the KDE is only 12-D, so the curse of dimensionality is **moderated** — but **feature heterogeneity** (binary + bounded fractions + unbounded diagram stats) with **single bandwidth** is still statistically brittle. Joint KDE `(y,x)` for classification is convenient but **not** a substitute for discriminative validation (logistic, gradient boosting) unless shown competitive on held-out data.

**Verdict:** Statistical validity is **unproven** in repo; implementation is a **heuristic smoother**. The log-domain likelihood-ratio calculation is now numerically safer, but calibration and baseline superiority are still unmeasured.

---

## 3. Topological methodology

### 3.1 Methods in repo

| Name | Claimed purpose (README / paper) | Actual computation |
|------|----------------------------------|---------------------|
| **Vietoris–Rips H₀/H₁** | "Topological features" / ambiguity fingerprint | VR on **unit-normalized** `(chunk_i - query)` vectors for top-k neighbors; exports paper-aligned `w1_h0` and `lt_max_h1` plus legacy aliases ([`calculations.py`](../../blowfish/calculations/calculations.py)). |
| **UMAP + HDBSCAN** | Topic discovery | 2D embedding; sweep HDBSCAN `min_cluster_size`; silhouette on 2D ([`topic_clustering.py`](../../blowfish/ingestion/topic_clustering.py)). |

### 3.2 Implementation vs paper definitions

- Paper emphasizes **W₁(H₀)** and **LT_max(H₁)** on diagrams built after **ε-scaled** neighbor geometry (paper §3.2).
- Code now exports paper-aligned `w1_h0` / `lt_max_h1` helpers and default KDE keys. ε scaling is wired via `epsilon=` on `calculate_vr_persistence_features` (TASK-004, approach A); legacy `max_homology_birth` aliases still exist for compatibility. The ε **sweep over real corpora** is still open. See [`MATH_HONESTY_CHECK.md`](./MATH_HONESTY_CHECK.md).

### 3.3 Statistical interpretation on finite noisy embeddings

With small **k** (the evaluator now warns below 15; the paper uses 50), H₁ bars are **high-variance**; paper §5.1 notes difficulty of higher homologies with few neighbors — same applies to **H₁** unless k is large enough and noise controlled.

### 3.4 Simpler equivalents / ablations

Required comparisons (review spec §8.2): connected components at distance threshold, largest component size, mean kNN distance, local clustering coefficient, spectral gap, DBSCAN/HDBSCAN labels, retrieval entropy. **Plan:** [`ABLATION_PLAN.md`](./ABLATION_PLAN.md).

### 3.5 Verdict (topology block)

| Component | Verdict |
|-----------|---------|
| VR pipeline in code | **validate further** — paper metric helpers implemented, but no full replication yet |
| ε mechanism | **implemented in code (approach A); ε sweep on real data still required** |
| k=5 default for production README | **fixed in README example; warning added for small k** |
| UMAP/HDBSCAN topics | **keep** as engineering clustering but document as **2D heuristic** |

---

## 4. Experimental methodology (paper + repo)

### 4.1 Paper experiment (summary)

- **Corpus:** Nobel Physics lectures; chunks 250 vs 750 tokens; **cross-chunking** retrieval (fine corpus vs coarse queries and vice versa).
- **Design intent:** Proxy polysemic vs multi-factual via **topic count proxy** from chunk length.
- **Metrics shown:** KDE plots and curves vs ε for **W₁(H₀)** and **LT_max(H₁)** — not AUROC/CIs.
- **Confounder:** **Chunk length** differs between conditions; **topic count** is not independently manipulated at **fixed length**. **TASK-008** addresses this for paper v2.

### 4.2 Repo experiment surface

- Training evaluates **retrieval chunk_match** and feeds KDE.
- **No** bundled dataset; **no** replication of Nobel experiment; **no** baseline comparisons in code.

### 4.3 Required experiments (review spec §9.1)

Minimum grid (implementation backlog):

| Experiment | Status in repo |
|------------|----------------|
| Retrieval score only | **Not as first-class feature set** |
| KDE only | **Partial** (KDE uses multi-feature vector) |
| Topology only | **Not isolated** |
| KDE + topology | **Default** |
| kNN-density baseline | **Missing** |
| Score-gap / entropy baselines | **Missing** |
| Randomized topology controls | **Missing** |
| Bootstrap stability | **Missing** |
| Runtime / memory | **Missing** |

### 4.4 Required metrics (review spec §9.2)

| Metric | Paper | Repo |
|--------|-------|------|
| AUROC / AUPRC | No | No |
| Brier / ECE | No | No |
| Latency / memory | No | No |

---

## 5. Non-negotiable question (review spec §15)

> Are the topological and KDE components genuinely improving ambiguity estimation, or are they expensive proxies for simpler distance, density, or retrieval-score heuristics?

**Evidence in repo:** **Not demonstrated.**

**Evidence in paper:** Simulation and **visual** separation under specific chunking regimes; **no** reported head-to-head vs kNN-distance, score-gap, or calibrated logistic on **matched** labels; **confounder** (length vs topic count) not fully eliminated.

**Experiment needed:** Same as [`ABLATION_PLAN.md`](./ABLATION_PLAN.md) — held-out evaluation with (a) frozen embedding model, (b) matched ambiguous/clear labels, (c) identical k and metric, (d) bootstrap CIs, (e) topology vs ablated vs simple baselines.

---

## 6. Cross-references

- Honest terminology mapping: [`MATH_HONESTY_CHECK.md`](./MATH_HONESTY_CHECK.md)
- Per-file issues: [`TECHNICAL_REVIEW.md`](./TECHNICAL_REVIEW.md)
- Tasks: [`AGENTIC_REMEDIATION_SPEC.md`](./AGENTIC_REMEDIATION_SPEC.md)
