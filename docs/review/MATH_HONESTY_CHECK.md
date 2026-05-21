# Math honesty check (no terminology laundering)

This document maps **claimed** language in the repo and companion paper (arXiv:2406.07990) to **concrete numerical operations** actually performed (or missing) in code.

## How to read this

For each term: **Claim** → **What the code/paper actually does** → **Verdict**.

---

## Topology / persistent homology

| Claimed term | Actual numerical operation |
|--------------|---------------------------|
| **Vietoris–Rips persistence** | For each query, take the `k` retrieved chunk embeddings, subtract the query embedding, L2-normalize each difference to unit length, run `giotto-tda` `VietorisRipsPersistence` on that point cloud (see [`blowfish/calculations/calculations.py`](../../blowfish/calculations/calculations.py)). |
| **H₀ (components)** | In VR filtration, each point is born at radius 0. **Merges** of components occur at pairwise distance radii. Current code filters H0 rows, keeps finite `(birth, death)` intervals, and exports `w1_h0`; legacy outputs named `max_homology_birth`, `mean_homology_birth`, `std_homology_birth` remain for compatibility and are still H0 **death-time** summaries, not births. |
| **W₁(H₀)** (paper Eq. 2 / 5) | Current helper `paper_w1_h0_from_birth_death` computes the finite-bar mean half-distance to the diagonal and exposes it as `w1_h0`. This is formula-aligned for the diagram rows supplied by giotto-tda; it is not yet validated against the paper's full epsilon-sweep experiment. |
| **LT_max(H₁)** (paper Eq. 3) | Current helper `paper_lt_max_h1` computes the maximum finite H1 half-lifetime and exposes it as `lt_max_h1`. Legacy `mean_homology1st_lifetime` remains available when requested but is not the paper metric. |
| **ε neighborhood / scaling** (paper §3.2) | Paper: relative scaling of neighbor offsets vs nearest neighbor to probe local→global geometry. **Code:** no explicit ε sweep; a **single** VR complex is fit on the fixed normalized offsets. **Mechanism absent.** |
| **Topological ambiguity signal** | In finite sampled clouds with **small k** (`BulkQueriesEvaluator` now warns below 15; paper uses 50), H₁ can be absent or dominated by sampling noise. Meaning: "loops" are **short bars from few points on a sphere-slice**, not validated manifold holes. |

**Plain summary:** The repo runs a standard VR persistence pipeline on a tiny, query-centered point cloud. It now exports formula-aligned `w1_h0` and `lt_max_h1`, but still lacks the paper's epsilon sweep and full replication harness.

---

## Ambiguity (paper working definition)

| Claimed term | Actual numerical operation |
|--------------|---------------------------|
| **Polysemic / multi-factual regime** (paper) | **Operational proxy** in experiments: chunk **token length** (250 vs 750) as a stand-in for "fewer vs more topics" + cross-retrieval direction. **Not** the same as explicitly counting topics per chunk from an external topic model on real text. |
| **`correct_prediction` in training KDE** | Binary flag from exact chunk text match (Levenshtein with insertion/deletion weight 0) after retrieval — see [`blowfish/training/queries_evaluation.py`](../../blowfish/training/queries_evaluation.py). This is **retrieval correctness**, not linguistic ambiguity. |
| **Clarity / ambiguity score at inference** | `p_correct = exp(log p(c=1,x)) / (exp(...) + exp(...))` from a **KernelDensity** fit over `[correct_prediction, features...]` — see [`blowfish/inference/scorer.py`](../../blowfish/inference/scorer.py). This is a **kernel-smoothed relative density ratio** in feature space, not a calibrated **probability of ambiguity** unless validated (ECE, etc.). |

---

## Semantic manifold

| Claimed term | Actual numerical operation |
|--------------|---------------------------|
| **"Manifold" / "discontinuities"** (paper prose) | Empirically: **finite set of embedding vectors** under a chosen metric. Cosine FAISS is now available via `vdb_metric="cosine"`, while legacy L2 remains available; experiments must record which metric was used. No proof of manifold structure; TDA features are **filtration statistics on finite samples**. |

---

## Density / KDE

| Claimed term | Actual numerical operation |
|--------------|---------------------------|
| **Gaussian KDE** | `sklearn.neighbors.KernelDensity(bandwidth=0.2, kernel="gaussian")` fit on **12 columns**: `correct_prediction` + 11 engineered scalars ([`blowfish/training/disambiguator_training.py`](../../blowfish/training/disambiguator_training.py)). |
| **Log-density** | `KernelDensity.score_samples` returns **log** density; scorer and decider now compute the binary ratio in log space. Remaining KDE concerns are calibration, bandwidth, and feature scaling rather than immediate underflow in the main ratio. |
| **Density as uncertainty** | **Interpretation requires calibration** (reliability diagrams, proper scoring). Not present in repo. Treating KDE output as "probability of correctness" without calibration is **method laundering**. |

---

## Uncertainty

| Claimed term | Actual numerical operation |
|--------------|---------------------------|
| **Uncertainty / confidence** | Not measured as calibration error, entropy of prediction sets, or bootstrap variance. Closest: **KDE ratio** and **SHAP** attributions over KDE-based surrogate — see [`blowfish/inference/decider.py`](../../blowfish/inference/decider.py). |

---

## Clustering / topic spread

| Claimed term | Actual numerical operation |
|--------------|---------------------------|
| **Topic label** | HDBSCAN on **2D UMAP** of chunk embeddings, then `docname_label` string key; **spread** = `|unique topic_label| / k` ([`calculations.py`](../../blowfish/calculations/calculations.py)). This is **cluster-id diversity in 2D projection**, not BERTopic or human topics. |

---

## Retrieval scores (`scale_*` features)

| Claimed term | Actual numerical operation |
|--------------|---------------------------|
| **scale_mean, scale_min, iq25-75_scale** | Sort distance-like retrieval scores ascending; drop the first (closest); remaining values divided by a nonzero closest-distance denominator. For the cosine FAISS path, evaluator converts inner product similarity to cosine distance (`1 - similarity`) before this calculation. Caller-provided inference rows must obey the same distance-like convention. |

---

## Checklist for reviewers

- [x] Default KDE features use formula-aligned `w1_h0` and `lt_max_h1`.
- [ ] Legacy "homology birth" aliases in [`calculations.py`](../../blowfish/calculations/calculations.py) are either renamed in a breaking release or clearly documented as compatibility aliases.
- [ ] Paper `W₁(H₀)` and `LT_max(H₁)` are validated in the full epsilon-sweep replication harness.
- [ ] "Ambiguity" in product copy tied to **operational label** (retrieval polysemy proxy, chunking regime, etc.).
- [ ] KDE outputs validated as probabilities or renamed "score."

---

## Cross-references

- Implementation gaps: [`TECHNICAL_REVIEW.md`](./TECHNICAL_REVIEW.md), [`AGENTIC_REMEDIATION_SPEC.md`](./AGENTIC_REMEDIATION_SPEC.md) TASK-001, TASK-003, TASK-004, TASK-006.
- Statistical framing: [`METHODOLOGY_REVIEW.md`](./METHODOLOGY_REVIEW.md).
