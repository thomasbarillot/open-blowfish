# Baseline results (template)

**Purpose:** Record head-to-head performance of **simple baselines** vs KDE+topology (review spec §3.3, §9). Fill after TASK-103 / TASK-104 execute.

**Dataset:** _(name, train/val/test split, embargo rules)_

**Label:** `correct_prediction` / operational ambiguity label — define explicitly.

**Metrics:** AUROC, AUPRC, Brier, ECE, NLL, FPR@R, latency/query, peak RSS _(as applicable)_.

---

## Required baselines (minimum set)

| Baseline ID | Definition | Implementation sketch | AUROC (val) | AUPRC | Notes |
|-------------|------------|----------------------|-------------|-------|-------|
| B0 | **Nearest-neighbor distance** — score = −min distance or max similarity | Single FAISS distance to top-1 | | | |
| B1 | **Mean k-NN distance** — mean of distances to top-k | Vector over k neighbors | | | |
| B2 | **kNN density** — local neighbor count within radius r or −mean log distance | `sklearn.neighbors` | | | |
| B3 | **KDE log-density** on raw similarity or distance features only | 1D KDE on top-1 score | | | |
| B4 | **Retrieval score gap** — s₁ − s₂ (similarity) or inverse for L2 | Sort top-k scores | | | |
| B5 | **Retrieval entropy** — normalized Shannon entropy of softmax(scores) | | | | |
| B6 | **Cluster assignment confidence** — e.g. max soft assignment if using HDBSCAN | | | | |
| B7 | **Mahalanobis** distance to class-conditional mean in feature space | Needs enough positives | | | |
| B8 | **Bootstrap instability** — variance of score under resample of k neighbors | | | | |
| B9 | **Simple calibrated classifier** — logistic / Isotonic on `[B0..B5]` only | `sklearn` Pipeline + CalibratedClassifierCV | | | |

---

## Proposed Blowfish variants

| Variant ID | Features | AUROC | Notes |
|----------|----------|-------|-------|
| V0 | KDE full default (after fixes) | | |
| V1 | Topology features zeroed / ablated | | |
| V2 | KDE without topology columns | | |

---

## Calibration table

| Method | ECE | Brier | Comment |
|--------|-----|-------|---------|
| | | | |

---

## Runtime

| Method | Fit (s) | Infer/query (ms) | Peak RSS (MB) |
|--------|---------|------------------|---------------|
| | | | |

---

## Verdict (for review spec §15)

**Do topology + KDE beat the best simple baseline on held-out data?** _(Yes / No / Not run)_

**If No:** topology/KDE are **not** justified; document as experimental or remove.

**Evidence path:** link to PR, notebooks, commit hash.

---

## Cross-references

- [`ABLATION_PLAN.md`](./ABLATION_PLAN.md)
- [`METHODOLOGY_REVIEW.md`](./METHODOLOGY_REVIEW.md)
