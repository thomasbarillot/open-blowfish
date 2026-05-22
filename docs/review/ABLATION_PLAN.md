# Ablation plan

**Goal:** Determine whether **topological (VR) features** and **joint KDE** add value beyond simpler signals (review spec §8–9, §15).

---

## 1. Feature ablations

Implemented as a sweep over `blowfish.experiments.ExperimentRunner.run_baselines`
with `feature_order` restricted per row (drive via the
`blowfish.calculations.calculate_relevant_features(kde_features_order=…)` knob).

| Run ID | Topology (VR) | KDE features | Notes |
|---|---|---|---|
| A0 | Off (omit `w1_h0`, `lt_max_h1`) | Full non-topology only | Retrieval + spread + silhouette |
| A1 | On | Full | Current default |
| A2 | On | Topology only + label | **Ill-conditioned** — expect poor |
| A3 | Off | Score-only baseline vector | Strong comparator |

## 2. Randomized / placebo controls

Implemented in `blowfish.experiments.controls`. Each function takes a list of
`RetrievalRecord` (or a feature DataFrame for R2) and returns a permuted
version on which the AUROC test should collapse if the topology signal is
real.

| Run ID | Procedure | Implementation | Pass criterion |
|---|---|---|---|
| R0 | Pool every chunk across queries, deal out random top-k bags | [`controls.py:permute_neighborhoods`](../../blowfish/experiments/controls.py) | AUROC ≈ 0.5; non-collapse indicates corpus-wide contamination |
| R1 | Apply one random orthogonal rotation Q to every embedding | [`controls.py:rotate_embeddings`](../../blowfish/experiments/controls.py) | Topology features invariant within numerical tolerance — failure is an implementation bug |
| R2 | Within a feature DataFrame, permute a specified block of columns across rows | [`controls.py:shuffle_feature_block`](../../blowfish/experiments/controls.py) | AUROC drop ≥ effect of the block; if not, the block isn't contributing |

---

## 3. Dimensionality reduction (if KDE on raw embeddings attempted)

| Step | Action | Metrics |
|------|--------|---------|
| D0 | PCA / random projection to d′ ∈ {32, 64, 128} before any density | Rank correlation of score vs full-dim |
| D1 | Whiten vs not | Condition number of covariance |

---

## 4. k and metric ablations

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| `top_k_results` | 5, 15, 50 | Match paper; stress H₁ |
| FAISS metric | L2 vs cosine-IP | TASK-006 |
| ε sweep | Once implemented — grid on paper values | TASK-004 |

---

## 5. Paper follow-up (TASK-108 / TASK-008)

| Experiment | Description |
|------------|-------------|
| P0 | **Length control:** fix token budget; vary only curated multi-topic vs single-topic queries |
| P1 | Report **CIs** (bootstrap) on `W₁(H₀)`, `LT_max(H₁)` differences between regimes |
| P2 | Plot effect sizes **vs** simplest baseline (B4 score-gap) on same queries |

---

## 6. Success criteria

- Topology+KDE **beats** best baseline by **margin > CI** on primary metric **or** is dropped.
- Randomized controls **destroy** signal — confirms not leakage.

---

## 7. Cross-references

- Results log: [`BASELINE_RESULTS.md`](./BASELINE_RESULTS.md)
- Tasks: [`AGENTIC_REMEDIATION_SPEC.md`](./AGENTIC_REMEDIATION_SPEC.md)
