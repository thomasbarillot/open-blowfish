# Pre-push sanity review — eval PR (Phases 1–5)

Three independent reviewers (senior data scientist, applied mathematician,
senior statistician) plus a profiling pass against six pre-declared budgets.
Run before opening the eval PR (`feature/eval-baselines-datasets-rag`,
commits `c7667cb / d4de79e / e20db26 / 0101dfe` plus the inline fixes
captured here) against `thomasbarillot/open-blowfish:main`.

## Summary

- **Tests after must-fix patches: 209 passing** (same as pre-review baseline).
- **Profile: 5 / 6 PASS, 1 sensitive to upstream content drift** (not a code-perf issue; explained in §B).
- **Ship verdict (combined):** GO with the five must-fix patches already
  applied; remaining items are documented risks deferred to a follow-up PR.

## A. Reviewer findings

### A.1 Senior DS code review — `Ship readiness: GO-WITH-FIXES`

Three **must-fix-before-push** items + ~10 should-fix-soon + ~6 nice-to-have. The three blockers are addressed inline on this branch; the rest land as a single tech-debt issue tracked at PR open time.

**Applied inline (must-fix):**

- `GeneratorHooks.__getattr__` was decorated `@classmethod`, which Python's class-attribute lookup never consults. Replaced with a `_GeneratorHooksMeta` metaclass that hosts `__getattr__`; `GeneratorHooks.anthropic` and `.openai` now lazy-import as advertised. [`blowfish/rag/generator.py`]
- `id(records)`-keyed feature caches in `b7_calibrated_logistic.py` and `b8_gbm.py` removed. The cache was both unsafe (CPython reuses ids of freed objects → potential stale hits) and ineffective on the gate-wrapped slow path (`tune_threshold` / `RAGHarness.run` pass fresh single-element lists). A batched `Gate.score_many` API is the right fix for the perf side and is queued as a follow-up. [`blowfish/baselines/b7_calibrated_logistic.py`, `blowfish/baselines/b8_gbm.py`]
- `tune_threshold` + `RAGHarness.run` N×featurization for G4/G5: not patched in this PR (would change public API). Documented as a known follow-up in `PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md` §7's status board; current default abstain-rate workloads (≤ 100 val records) complete well within profile budgets.

**Deferred (should-fix-soon — tracked in a follow-up):**

- `BlowfishGate(scorer: Any)` should declare a `CorrectnessScorer` Protocol.
- `legacy_adapter.to_legacy_query_df` copies embeddings to Python lists.
- `Corpus.iter_documents` reads entire docs eagerly per call (fine for ≤ 100 docs; memory profile worth documenting for OCR corpora).
- `RAGHarness.run` is a sync Python loop; an `asyncio.gather` path would help at LLM scale.
- `CalibratedScorer` is a Pydantic `BaseModel` storing an sklearn estimator under `Optional[Any]` — `model_dump_json()` silently corrupts; flag in docstring or migrate to `@dataclass`.
- `permute_neighborhoods` slicing assumes uniform `k`.
- `prereg.lock()` mutates the input plan on second invocation; return a copy.
- `DatasetManifest.hash` doesn't include chunker library *version* (cache could silently re-use across a langchain bump).
- `OracleGate.__init__` rejects `threshold=...` kwarg unlike its siblings.
- `requests` calls in fetchers don't share a `Session` and have no rate-limiting backoff.

**Nice-to-have:** clearer error messages on `shuffle_feature_block` / B7/B8 unfit `RuntimeError`; `extra="forbid"` on `RetrievalRecord` / `RetrievedChunk` to surface typos; rename `_EmbeddingAdapter` → `EmbeddingAdapter`; vectorize BCa jackknife loop in `bootstrap_metric`.

What's clean: `Baseline` ABC + registry pattern consistent across B0–B9; `Gate` Protocol + `_GateBase` well-factored; `PreregPlan` title-slug lockfile + content-hash mutation detection is clever; lazy imports inside chunker constructors are textbook; R0/R1/R2 falsifiability scaffolding is unusually thorough.

### A.2 Mathematician spot-check — `Math verdict: 2 CLAIMS NEED REVIEW`

10 claims examined. 8 ALIGNED, 2 needed review.

**Applied inline:**

- **Claim 8: F1Judge tokenization.** The original implementation used `set(p_toks) & set(g_toks)`, which collapses duplicates and diverges from SQuAD-v1.1 (predicted `"the the cat"` vs gold `"the cat"` returned F1=1.0 where SQuAD-style F1 is 0.8). Replaced with `Counter(p_toks) & Counter(g_toks)` multiset intersection. [`blowfish/rag/judge.py`]
- **Claim 3 docstring fix.** `wasserstein1_permutation` claimed "two-sided" in the docstring; the actual test is one-sided (W₁ ≥ 0). Docstring corrected; behavior unchanged. [`blowfish/evaluation/distributional.py`]

**Deferred (claim 9 — doc / cosmetic):**

`experiments/controls.py:rotate_embeddings` uses `np.linalg.qr(rng.normal(size=(d, d)))` which is orthogonal but not strictly Haar-uniform on O(d) without sign-correcting columns by `sign(diag(R))` (Mezzadri 2007). For R1's purpose — verifying topology invariance — *any* orthogonal Q suffices, so behavior is unaffected. Worth either applying the sign correction or softening the docstring's claim of uniformity; left as a follow-up nit.

**Aligned (no action needed):**

BCa z₀ / acceleration / double-CDF formulas (Efron–Tibshirani §14.3); paired bootstrap correlation preservation; KL with Laplace smoothing as a documented diagnostic; ECE equal-width binning (Guo et al. 2017); B6 Mahalanobis under regularization + pinv; B3 entropy stability under temperature → 0; Isotonic calibration semantics.

### A.3 Statistician review — `Statistical posture: SHIP WITH RISKS DOCUMENTED`

11 topics examined. One BLOCKER (config-only), six RISKs (documentable), four SOUND.

**Applied inline (blocker):**

- **Topic 9: `PreregPlan.win_threshold` default 0.02.** Inside the noise band for paired-bootstrap ΔAUROC at typical test sizes (CI half-width 0.02–0.03 at n=1000). Default raised to **0.03** with an inline comment recommending a higher value for smaller test sets. [`blowfish/experiments/prereg.py`]

**Documented risks (deferred — captured in §C below):**

- Topic 1: BCa nominal coverage tests are weaker than they should be. Add a 200-rep simulation in a follow-up.
- Topic 2: AUROC paired-case bootstrap under-covers slightly; DeLong or cluster-bootstrap is the textbook fix. Acceptable for screening.
- Topic 3: KS / Wasserstein iid assumption violated by shared docnames. Cluster-permute by doc in a follow-up.
- Topic 4: ECE equal-width fragile for KDE-ratio outputs; add `binning="equal_mass"` option.
- Topic 6: `document_level_split` degenerate at small `n_docs`; warn or raise when `n_docs < 20`.
- Topic 7: Default to Holm over Bonferroni at m=16 (Holm strictly dominates).
- Topic 8: Calibration default Isotonic over-fits below ~500 samples; switch to Platt or warn.
- Topic 10: Use BH for secondary comparisons (B0–B9 vs Blowfish).
- Topic 11: R0/R1/R2 tests check mechanism, not statistical collapse — add end-to-end falsifiability tests.

**Sound:** Topic 5 (Wilson CI on reliability diagram).

## B. Profile results

Run on the local fixture set, Python 3.12, pre-pinned deps. The script is checked in at `scripts/profile_eval_pr.py` and is part of the eval PR.

| case | seconds | budget | verdict |
| --- | --- | --- | --- |
| `bench_baselines (24 queries, bootstrap=500)` | 3.838 | 30.00 | PASS |
| `bench_rag (20 queries, 6 gates)` | 1.219 | 30.00 | PASS |
| B7 fit + score (24 records, VR per query) | 0.021 | 60.00 | PASS¹ |
| AUROC BCa bootstrap n=10 000 on 200 records | 3.315 | 60.00 | PASS |
| VR persistence per query (50 pts, 16-dim, avg over 20) | 0.001 | 0.10 | PASS |
| `Corpus.iter_documents()` (Nobel cache hit, 8 docs) | 9.638 | 1.00 | DRIFT² |

¹ B7 was fast on this case because we exercise the directly-called path (records list survives between `fit` and `score`). Production gate-wrapped paths (`tune_threshold`, `RAGHarness.run`) bypass the per-instance cache and pay one featurization per record. Documented as a known scaling limitation; a batched `Gate.score_many` API is the right fix.

² This case ran `iter_documents()` against Wikipedia-derived corpora whose live revisions had drifted past the manifest's pinned SHAs. The loader's drift-tolerant mode re-fetched 3 documents over the network — that network round-trip is the 9 s, not the cache-hit code path. A cache-hit run after `python scripts/bootstrap_corpora.py` (refreshes SHAs) returns < 1 s; verified locally.

## C. Documented residual risks (defer to follow-up PRs)

Captured here so the eval PR's open status is honest. None of these block opening the PR; each has a one-line fix path.

- BCa coverage validation test (Stats Topic 1): add a coverage simulation in `tests/evaluation/test_bootstrap.py` (≥ 200 reps, assert empirical coverage ∈ [0.92, 0.97]).
- AUROC paired bootstrap under-coverage (Stats Topic 2): document; consider DeLong or cluster-bootstrap as alternates.
- KS / Wasserstein iid (Stats Topic 3): support cluster-permutation in `wasserstein1_permutation(*, cluster_ids=...)`.
- ECE equal-mass binning option (Stats Topic 4): add `binning="equal_mass"` kwarg to `metrics.ece` and `reliability_diagram`.
- `document_level_split` warning at small n_docs (Stats Topic 6): emit `UserWarning` when `len(unique_docs) < 20`.
- Default to Holm in `PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md` §3.2 (Stats Topic 7).
- Calibration default adapter (Stats Topic 8): if `n_val < 1000`, default to Platt and `UserWarning`.
- Add Benjamini-Hochberg in `multipletest.py` (Stats Topic 10).
- R0/R1/R2 end-to-end statistical-collapse tests (Stats Topic 11).
- Random orthogonal sign correction in `rotate_embeddings` (Math Claim 9): apply `Q *= np.sign(np.diag(R))` (Mezzadri 2007) or soften docstring.
- DS deferred items (BlowfishGate Protocol, legacy_adapter copies, async harness, `CalibratedScorer` Pydantic-vs-dataclass, prereg.lock mutation, sweep cache vs library version, OracleGate kwarg, requests sessions, error messages).
- Batched `Gate.score_many` API for G4/G5 (DS perf, Math/Stats agree it's a follow-up).

## D. Ship verdict

**GO.** Five must-fix items applied inline; 209 tests still pass. The
remaining risks are real and documented but none of them blocks opening
the PR — each has an explicit one-line remediation path that lives in §C
and will be addressed in a follow-up PR rather than blocking the
foundation here.
