# Ablation plan

**Goal:** Determine whether **topological (VR) features** and **joint KDE** add value beyond simpler signals (review spec §8–9, §15).

---

## 1. Feature ablations

| Run ID | Topology (VR) | KDE features | Notes |
|--------|---------------|--------------|-------|
| A0 | Off (zeros / omit columns) | Full non-topology only | Retrieval + spread + silhouette |
| A1 | On | Full | Current target after TASK-001 fixes |
| A2 | On | Topology only + label | **Ill-conditioned** — expect poor |
| A3 | Off | **Score-only** baseline vector | Strong comparator |

---

## 2. Randomized / placebo controls

| Run ID | Procedure | Pass criterion |
|--------|-----------|----------------|
| R0 | Shuffle VR features across queries within split | AUROC ≈ 0.5 — if real signal exists, shuffled should collapse |
| R1 | Random Gaussian noise with matched variance replaces VR block | Same |
| R2 | Permute neighbor order within each query | Invariant check — outputs should not depend on order unless bug |

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
