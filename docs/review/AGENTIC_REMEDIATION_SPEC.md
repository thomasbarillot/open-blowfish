# Agentic remediation specification

Executable tasks for autonomous implementation. Format per review charter §4. Cross-docs: [`TECHNICAL_REVIEW.md`](./TECHNICAL_REVIEW.md), [`METHODOLOGY_REVIEW.md`](./METHODOLOGY_REVIEW.md), [`PERFORMANCE_REVIEW.md`](./PERFORMANCE_REVIEW.md), [`TESTING_AND_PROFILING_PLAN.md`](./TESTING_AND_PROFILING_PLAN.md).

---

## TASK-001: Implement paper-aligned VR metrics or rename outputs
### Priority
P0
### Status
Implemented in code; keep open for end-to-end replication and paper figure validation.
### Category
topology / implementation / documentation
### Target files
- [`blowfish/calculations/calculations.py`](../../blowfish/calculations/calculations.py)
- `tests/calculations/test_persistence_metrics.py` (create)
### Problem
Exported homology scalars do not match arXiv:2406.07990 `W₁(H₀)` (Eq. 2/5) or `LT_max(H₁)` (Eq. 3). H₀ statistics use diagram column 1 while naming them `*_homology_birth` — in Vietoris–Rips, H₀ births are 0 for distinct points; column 1 is **death** (merge scale).
### Evidence
[`calculations.py`](../../blowfish/calculations/calculations.py) lines 51–62: `neighbour_0th_homology = diagrams[..., 1]`; `mean_homology1st_lifetime = np.mean(holes_lifetimes)` vs paper **sup**.
### Required change
Either (A) implement `W1_H0` and `LT_max_H1` per paper/giotto conventions with tests on toy diagrams, or (B) rename keys to neutral `h0_death_max` etc. and update README to **not** cite paper equations. Prefer (A) if paper alignment is a product goal.
### Non-goals
Do not change UMAP/HDBSCAN in this task.
### Acceptance criteria
- Gold tests assert `W1_H0` / `LT_max_H1` within tolerance on ≥2 fixed point clouds.
- `DEFAULT_KDE_KEYS` updated consistently with backward-compat alias map if needed.
- [`METHODOLOGY_REVIEW.md`](./METHODOLOGY_REVIEW.md) traceability row updated to YES/NO accurately.
### Tests required
- Unit: toy VR diagrams; regression: stored numpy fixtures.
### Profiling required
None.
### Risks
giotto-tda version drift — pin minor version in CI.
### Rollback plan
Feature flag `use_paper_metrics: bool` default False until validated.

---

## TASK-002: Fix FeedbackDecider docspread branch
### Priority
P0
### Status
Implemented.
### Category
implementation
### Target files
- [`blowfish/inference/decider.py`](../../blowfish/inference/decider.py)
- [`tests/inference/test_decider.py`](../../tests/inference/test_decider.py) (create)
### Problem
Branch checks `"docspread"` but feature name is `"top_k_doc_spread"` ([`constants.py`](../../blowfish/utils/constants.py)). Docspread path never runs.
### Evidence
[`decider.py`](../../blowfish/inference/decider.py) lines 88–91 vs `top_k_doc_spread` key.
### Required change
Compare against `"top_k_doc_spread"` or normalize feature aliases in one module.
### Non-goals
Do not change SHAP algorithm in this task.
### Acceptance criteria
- Unit test forces negative SHAP on `top_k_doc_spread` only → output indicates doc spread (per intended label).
### Tests required
- Unit with mocked `explainer`.
### Profiling required
None.
### Risks
None.
### Rollback plan
Revert single conditional.

---

## TASK-003: Align default top_k with paper and H₁ validity
### Priority
P0
### Status
Implemented for README example and runtime warning; no forced config migration.
### Category
experiment / documentation
### Target files
- [`README.md`](../../README.md)
- [`blowfish/training/queries_evaluation.py`](../../blowfish/training/queries_evaluation.py) (defaults via config only — document)
### Problem
Paper §3.6 uses k=50; README example `top_k_results: 5` contradicts paper §3.2 "sufficiently high k" for stable homology.
### Evidence
README configuration vs paper text.
### Required change
Set documented default to **50** (or explain deviation prominently). Add runtime warning if `k < k_min` for topology features.
### Non-goals
Do not silently change user configs without semver note.
### Acceptance criteria
- Tests verify a warning fires when k < 15 (threshold tunable).
### Profiling required
None.
### Risks
Higher k increases VR cost — document in [`PERFORMANCE_REVIEW.md`](./PERFORMANCE_REVIEW.md).
### Rollback plan
Config-only revert.

---

## TASK-004: Add epsilon neighborhood scaling for VR features
### Priority
P0
### Status
Implemented (approach A: ε-threshold subsample of neighbors before VR, with `n_min` fallback). `epsilon=` kwarg threaded through `calculate_vr_persistence_features` → `calculate_relevant_features` → `AmbiguityScorer` and `DisambiguationModelGenerator`. ε sweep on real corpora is still open (gated on baseline/ablation work).
### Category
topology / experiment
### Target files
- [`blowfish/calculations/calculations.py`](../../blowfish/calculations/calculations.py)
- [`blowfish/training/disambiguator_training.py`](../../blowfish/training/disambiguator_training.py)
- [`blowfish/inference/scorer.py`](../../blowfish/inference/scorer.py)
### Problem
Paper figures vary ε; code uses a single implicit VR scale (no relative neighbor scaling pipeline).
### Evidence
[`METHODOLOGY_REVIEW.md`](./METHODOLOGY_REVIEW.md) traceability NO.
### Required change
Implement paper Eq. (1) scaling: `d(i,q) = ||Δv_iq|| / ||Δv_0q||` (with safe handling of zero), then **either** subsample neighbors by ε threshold **or** reweight distances before VR — document chosen approximation. Expose `epsilon` in feature API.
### Non-goals
Full replication of all paper figures inside package.
### Acceptance criteria
- Curve of metrics vs ε on synthetic data is smooth / monotone in toy cases.
- At least one golden ε value matches paper order-of-magnitude on bundled tiny fixture (optional).
### Tests required
- Unit on 10-point synthetic neighbor lists.
### Profiling required
Document multi-ε cost in [`PERFORMANCE_REVIEW.md`](./PERFORMANCE_REVIEW.md).
### Risks
API surface expansion — version semver minor.
### Rollback plan
Feature flag off → single-scale path.

---

## TASK-005: Declare Python dependencies in packaging
### Priority
P0
### Status
Implemented through `install_requires` and optional `explain` extra.
### Category
implementation
### Target files
- [`setup.py`](../../setup.py) **or** migrate to `pyproject.toml`
- [`blowfish/requirements.txt`](../../blowfish/requirements.txt)
### Problem
`install_requires=[]` — `pip install -e .` omits runtime deps.
### Evidence
[`setup.py`](../../setup.py) line 13–15.
### Required change
Mirror pins or compatible ranges from `requirements.txt` into `install_requires` / `project.dependencies`.
### Non-goals
Do not upgrade major versions without audit.
### Acceptance criteria
- Fresh venv: `pip install -e .` then `import blowfish` + `import gtda` succeeds.
### Tests required
- CI job: minimal install smoke.
### Profiling required
None.
### Risks
Version conflicts with user env — document extras.
### Rollback plan
Restore empty requires (undesired).

---

## TASK-006: Unify retrieval metric (cosine vs L2)
### Priority
P0
### Status
Partially implemented: cosine FAISS path, metric sidecar validation, and cosine-to-distance conversion exist; baseline experiments and full metric documentation remain.
### Category
implementation / statistics
### Target files
- [`blowfish/ingestion/vdb_indexing.py`](../../blowfish/ingestion/vdb_indexing.py)
- [`blowfish/utils/vdb_factory.py`](../../blowfish/utils/vdb_factory.py)
- [`blowfish/calculations/calculations.py`](../../blowfish/calculations/calculations.py)
- [`README.md`](../../README.md)
### Problem
Paper/README: cosine; FAISS: `IndexFlatL2`. `scale_*` assumes distance-like ordering (divide by smallest score).
### Evidence
[`vdb_indexing.py`](../../blowfish/ingestion/vdb_indexing.py) uses `IndexFlatL2`; README says cosine.
### Required change
**Either** L2 on raw embeddings with README fix **or** normalize vectors + `IndexFlatIP` for cosine similarity. Plumb `metric` into config. Branch `calculate_scaled_distance_distribution` for similarity vs distance semantics.
### Non-goals
Do not change embedding model outputs.
### Acceptance criteria
- Integration test: same ANN rankings as brute-force cosine on toy vectors within tolerance.
- `scale_*` sign convention documented per metric.
### Tests required
- Unit for scale stats under cosine + L2.
### Profiling required
Compare index build time IP vs L2.
### Risks
Breaking change for saved indexes — migration note.
### Rollback plan
Keep legacy `faiss_index_l2` hook name.

---

## TASK-007: Enforce unique hash_key / safe joins
### Priority
P0
### Status
Implemented for indexer, evaluator, and scorer.
### Category
implementation
### Target files
- [`blowfish/training/queries_evaluation.py`](../../blowfish/training/queries_evaluation.py)
- [`blowfish/inference/scorer.py`](../../blowfish/inference/scorer.py)
### Problem
`topics_df.loc[hash_key]` returns DataFrame if duplicate keys — downstream concat/scalar logic breaks silently.
### Evidence
[`queries_evaluation.py`](../../blowfish/training/queries_evaluation.py) line 92; [`scorer.py`](../../blowfish/inference/scorer.py) merge on `hash_key`.
### Required change
Assert unique index at load; or `groupby` + deterministic aggregate; fail fast with actionable error.
### Non-goals
Automatic dedupe without user policy.
### Acceptance criteria
- Test raises `ValueError` on duplicate `hash_key` in topics table.
### Tests required
- Unit negative test.
### Profiling required
None.
### Risks
Strict mode rejects dirty user data — document cleaning step.
### Rollback plan
Optional `strict_unique_keys=False` for migration.

---

## TASK-008: Paper — control chunk length confound
### Priority
P0
### Status
Open; requires paper/data workflow outside package code.
### Category
experiment / documentation
### Target files
- Paper source (outside this repo) + supplemental materials
- [`docs/review/METHODOLOGY_REVIEW.md`](./METHODOLOGY_REVIEW.md) (link to updated v2)
### Problem
Comparing C_250 vs C_750 confounds **topic count proxy** with **token length / syntactic density**.
### Evidence
Paper §3.6–4.2 design.
### Required change
Design **length-matched** strata: same token budget, varying number of curated topics; pre-register metrics and CIs; update figures.
### Non-goals
Re-run all embeddings if infeasible; at minimum acknowledge the limitation in the abstract.
### Acceptance criteria
- Revised figure/table shows effect persists under length control **or** claim downgraded.
### Tests required
n/a (paper)
### Profiling required
n/a
### Risks
May reduce effect size — scientific honesty tradeoff.
### Rollback mark
v1 arXiv note linking addendum.

---

## TASK-101: Bandwidth and feature scaling for KDE
### Priority
P1
### Category
statistics / implementation
### Target files
- [`blowfish/training/disambiguator_training.py`](../../blowfish/training/disambiguator_training.py)
### Problem
Single `bandwidth=0.2` for mixed-scale 12-D joint; no CV.
### Evidence
Line 144 `KernelDensity(bandwidth=0.2, ...)`.
### Required change
`StandardScaler` + `GridSearchCV` or rule-of-thumb per dimension; **or** separate bandwidths via `sklearn` limitations → consider `KDE` from statsmodels / custom for multivariate if needed.
### Non-goals
Neural density models.
### Acceptance criteria
- Held-out log-likelihood or Brier improves vs fixed-0.2 on validation — record in [`BASELINE_RESULTS.md`](./BASELINE_RESULTS.md).
### Tests required
- Statistical tests ([`TESTING_AND_PROFILING_PLAN.md`](./TESTING_AND_PROFILING_PLAN.md)).
### Profiling required
Fit time in benchmark script.
### Risks
Longer training — acceptable.
### Rollback plan
Fallback `bandwidth` config.

---

## TASK-102: Stabilize KDE ratio with log-domain math
### Priority
P1
### Category
implementation / statistics
### Target files
- [`blowfish/inference/scorer.py`](../../blowfish/inference/scorer.py)
- [`blowfish/inference/decider.py`](../../blowfish/inference/decider.py)
### Problem
Repeated `exp(score_samples)`; risk of underflow; redundant calls.
### Evidence
[`scorer.py`](../../blowfish/inference/scorer.py) lines 93–94; [`decider.py`](../../blowfish/inference/decider.py) `KDE_prediction`.
### Required change
Use `scipy.special.logsumexp` pattern: `log p = log f1 - logaddexp(log f1, log f0)`; batch `score_samples` on stacked inputs.
### Non-goals
Change statistical meaning beyond numerical stabilization.
### Acceptance criteria
- Extreme-distance synthetic test: no NaN; matches naive formula within `rtol` where naive is computable.
### Tests required
- Numerical stability tests.
### Profiling required
Compare call counts before/after.
### Risks
scipy dependency — already common stack; else numpy stable recipe.
### Rollback plan
Single-file revert.

---

## TASK-103: Implement baseline scoring methods
### Priority
P1
### Category
statistics / experiment
### Target files
- `blowfish/baselines/` (create) or `blowfish/experiments/`
- [`docs/review/BASELINE_RESULTS.md`](./BASELINE_RESULTS.md)
### Problem
No nearest-neighbor / kNN-density / score-gap / entropy / Mahalanobis / calibrated logistic baselines in repo.
### Required change
Implement B0–B9 per [`BASELINE_RESULTS.md`](./BASELINE_RESULTS.md); uniform eval harness.
### Non-goals
Tune baselines to beat Blowfish unfairly.
### Acceptance criteria
- Table filled on one public or internal snapshot.
### Tests required
- Unit tests on synthetic scores for entropy/gap.
### Profiling required
Document in benchmarks.
### Risks
Scope creep — ship minimal B0/B4/B5/B9 first.
### Rollback plan
Feature behind `extras`.

---

## TASK-104: Held-out metrics and uncertainty quantification
### Priority
P1
### Category
statistics / experiment
### Target files
- `blowfish/evaluation/` (create)
### Problem
No AUROC, AUPRC, Brier, or ECE reporting path in the repository.
### Required change
Split train/val/test; metrics + bootstrap CIs; export CSV/JSON artifact.
### Non-goals
Full MLflow integration.
### Acceptance criteria
- CI publishes metrics artifact on sample run.
### Tests required
- Golden metrics on fixed tiny dataset.
### Profiling required
None.
### Risks
None.
### Rollback plan
Remove CLI entrypoint.

---

## TASK-105: Replace O(n²) UMAP sweep heuristic
### Priority
P1
### Category
performance / implementation
### Target files
- [`blowfish/ingestion/topic_clustering.py`](../../blowfish/ingestion/topic_clustering.py)
### Problem
Full pairwise distance matrix per UMAP `n_neighbors` trial for histogram.
### Evidence
Lines 47–48 nested in loop over `u`.
### Required change
Subsample distances, use `sklearn.neighbors` k-distance curve, or fixed `n_neighbors` from data-size rule; vectorize once per projection if histogram still needed.
### Non-goals
Perfect reproduction of paper figure inside ingestion.
### Acceptance criteria
- Benchmark: ≥2× faster on n=5k with ≤1% difference in chosen `n_neighbors` vs legacy (or document accepted drift).
### Tests required
- Smoke clustering output shape.
### Profiling required
[`PERFORMANCE_REVIEW.md`](./PERFORMANCE_REVIEW.md) commands.
### Risks
Different cluster labels — semver note.
### Rollback plan
`use_legacy_umap_search=True` flag.

---

## TASK-106: Batch FAISS search and batch KDE scoring
### Priority
P1
### Category
performance / implementation
### Target files
- [`blowfish/training/queries_evaluation.py`](../../blowfish/training/queries_evaluation.py)
- [`blowfish/inference/scorer.py`](../../blowfish/inference/scorer.py)
### Problem
`iterrows` + single-query `search`; no `score(X_batch)`.
### Evidence
[`queries_evaluation.py`](../../blowfish/training/queries_evaluation.py) line 84+; scorer API.
### Required change
`index.search` on matrix of shape `(Q, d)`; `score_batch` for queries grouped with same k blocks.
### Non-goals
GPU FAISS in v1.
### Acceptance criteria
- Batch path bitwise or rtol matches loop path on toy data; wall time lower for Q≥64.
### Tests required
- Equiv tests per [`TESTING_AND_PROFILING_PLAN.md`](./TESTING_AND_PROFILING_PLAN.md).
### Profiling required
`scripts/benchmark_inference.py`.
### Risks
Memory for large Q*d matrices.
### Rollback plan
Expose `batch_size` default 1.

---

## TASK-107: Lazy / cached SHAP explainer initialization
### Priority
P1
### Category
performance / implementation
### Target files
- [`blowfish/inference/decider.py`](../../blowfish/inference/decider.py)
### Problem
Constructor runs rejection sampling + KernelExplainer setup eagerly.
### Evidence
[`decider.py`](../../blowfish/inference/decider.py) `__init__` lines 29–33, `get_high_accuracy_samples`.
### Required change
`lazy_init_explain=True`; cache background file; cap attempts with error.
### Non-goals
Replace SHAP with exact Shapley.
### Acceptance criteria
- Import + construct `FeedbackDecider` completes under time budget without sampling; first `explain_query` pays cost once.
### Tests required
- Unit timing smoke (loose).
### Profiling required
Cold vs warm start numbers in PR.
### Risks
Thread safety if caching global — document single-threaded use.
### Rollback plan
`eager_init=True` option.

---

## TASK-108: Rename and document H₀ diagram columns
### Priority
P1
### Category
topology / documentation
### Target files
- [`blowfish/calculations/calculations.py`](../../blowfish/calculations/calculations.py)
- [`blowfish/utils/constants.py`](../../blowfish/utils/constants.py)
### Problem
Misleading `*_birth` names for H₀ statistics.
### Evidence
See TASK-001.
### Required change
After TASK-001, align names; interim: rename to `h0_death_*` if not implementing W₁ yet.
### Non-goals
Change sklearn column order.
### Acceptance criteria
- README + API migration notes.
### Tests required
- Snapshot feature dict keys test.
### Profiling required
None.
### Rollback plan
Deprecation aliases 1 release.

---

## TASK-109: Paper figures — bandwidth, n, bootstrap CIs
### Priority
P1
### Category
experiment / documentation
### Target files
- Paper plotting scripts (external) + revision text
### Problem
KDE plots without stated bandwidth, sample size, or uncertainty.
### Evidence
Paper §4 figures narrative.
### Required change
Add methods text: KDE bandwidth rule, **n**, seed, bootstrap 95% CI bands or effect sizes with CIs.
### Non-goals
Redesign entire color scheme.
### Acceptance criteria
- Peer-review checklist satisfied.
### Tests required
n/a
### Profiling required
n/a
### Risks
Visuals change — v2 figure set.
### Rollback plan
Supplementary PDF retains v1.

---

## TASK-201: Validate embedding dimension against index
### Priority
P2
### Category
implementation
### Target files
- [`blowfish/ingestion/chunk_embeddings.py`](../../blowfish/ingestion/chunk_embeddings.py)
- [`blowfish/ingestion/vdb_indexing.py`](../../blowfish/ingestion/vdb_indexing.py)
### Problem
Silent mismatch between encoder output dim and `vdb_vector_size`.
### Required change
Assert `vectors.shape[1] == vdb_vector_size` on first add.
### Non-goals
Auto-resize index.
### Acceptance criteria
- Unit test raises on mismatch.
### Tests required
- Unit.
### Profiling required
None.
### Risks
Strict failure for misconfig — intended.
### Rollback plan
Warn-only mode.

---

## TASK-202: Replace prints with structured logging
### Priority
P2
### Category
implementation / documentation
### Target files
- [`blowfish/ingestion/topic_clustering.py`](../../blowfish/ingestion/topic_clustering.py)
- [`blowfish/training/disambiguator_training.py`](../../blowfish/training/disambiguator_training.py)
- [`blowfish/training/queries_evaluation.py`](../../blowfish/training/queries_evaluation.py)
### Problem
`print` statements hinder production ops.
### Required change
`logging.getLogger(__name__)` with levels.
### Non-goals
Centralized logging infra.
### Acceptance criteria
- Default log level INFO quieter than current prints.
### Tests required
- Smoke capturing caplog optional.
### Profiling required
None.
### Risks
None.
### Rollback plan
Revert commits.

---

## TASK-203: Remove module-level warnings filters
### Priority
P2
### Category
implementation
### Target files
- [`blowfish/inference/scorer.py`](../../blowfish/inference/scorer.py)
- [`blowfish/inference/decider.py`](../../blowfish/inference/decider.py)
### Problem
`warnings.filterwarnings('ignore')` hides real issues globally.
### Required change
Context managers around known noisy deps or fix root warnings.
### Non-goals
Silence security warnings.
### Acceptance criteria
- pytest shows no spurious spam — any expected warning caught with `pytest.warns` locally.
### Tests required
- CI warning policy (treat as errors optional).
### Profiling required
None.
### Risks
Noisy CI until upstream fixes.
### Rollback plan
Narrow filter to specific message.

---

## TASK-204: Fix test isolation and negative kde tests
### Priority
P2
### Category
testing
### Target files
- [`tests/inference/test_scorer.py`](../../tests/inference/test_scorer.py)
### Problem
Module-level `scorer` with `kde=None`; shared state; does not test real scoring.
### Required change
Pytest fixtures; add golden test with tiny fitted `KernelDensity`.
### Non-goals
Full integration in this task.
### Acceptance criteria
- No global AmbiguityScorer; kde path smoke test passes.
### Tests required
- Unit.
### Profiling required
None.
### Risks
None.
### Rollback plan
N/A

---

## TASK-205: Export public API in `__init__.py` files
### Priority
P3
### Category
documentation / implementation
### Target files
- [`blowfish/__init__.py`](../../blowfish/__init__.py)
- [`blowfish/ingestion/__init__.py`](../../blowfish/ingestion/__init__.py)
- [`blowfish/training/__init__.py`](../../blowfish/training/__init__.py)
- [`blowfish/inference/__init__.py`](../../blowfish/inference/__init__.py)
### Problem
Empty packages; discoverability poor.
### Required change
`__all__` listing stable names from README.
### Non-goals
Reorganize package layout.
### Acceptance criteria
- `from blowfish.ingestion import NaiveChunksEmbedding` still works; optional shorter imports documented.
### Tests required
Import smoke.
### Profiling required
None.
### Risks
Name clashes.
### Rollback plan
Remove exports.

---

## TASK-206: Rename homology helper for clarity
### Priority
P3
### Category
documentation / implementation
### Target files
- [`blowfish/calculations/calculations.py`](../../blowfish/calculations/calculations.py)
### Problem
`calculate_first_order_homology_distribution` computes both H₀ and H₁.
### Required change
Rename to `calculate_vr_persistence_features` with backward alias wrapping deprecated name.
### Non-goals
Change numerical outputs.
### Acceptance criteria
- DeprecationWarning on old name for one release.
### Tests required
- Import both names.
### Profiling required
None.
### Risks
External callers — semver note.
### Rollback plan
Keep alias forever if needed.

---

# Remediation Summary (template — fill after implementation)

## Changes made
- …

## Performance before/after
| Benchmark | Before | After | Speedup |
|-----------|-------:|------:|--------:|
| … | | | |

## Statistical behavior before/after
| Metric | Before | After | Delta |
|--------|-------:|------:|------:|
| … | | | |

## Tests added
- …

## Remaining risks
- …

## Recommended next work
- …
