# Technical review (implementation)

One section per module under [`blowfish/`](../../blowfish/). Template: **Responsibility → Actual behavior → Problems → Recommended changes → Agent tasks.**

---

## Module: [`blowfish/__init__.py`](../../blowfish/__init__.py)

### Responsibility
Package marker.

### Actual behavior
Empty module.

### Problems found
- No public re-exports; README imports `blowfish.ingestion`, `blowfish.training`, etc. manually.

### Recommended changes
- Optionally export stable API in `__all__` or document canonical import paths.

### Agent tasks
- TASK-205 (P3)

---

## Module: [`blowfish/calculations/__init__.py`](../../blowfish/calculations/__init__.py)

### Responsibility
Subpackage marker.

### Actual behavior
Empty.

### Problems found
- No re-export of `calculate_relevant_features` at package boundary.

### Recommended changes
- `from blowfish.calculations.calculations import calculate_relevant_features` in `__init__.py`.

### Agent tasks
- TASK-205 (P3)

---

## Module: [`blowfish/calculations/calculations.py`](../../blowfish/calculations/calculations.py)

### Responsibility
Engineer per-query features: retrieval score spread, Vietoris–Rips summaries, silhouette spread, doc/topic spread.

### Actual behavior
- `calculate_scaled_distance_distribution`: sorts distance-like `score`, drops first, divides by a nonzero closest-distance denominator; exact nearest-neighbor distance `0.0` no longer emits infinities.
- `calculate_vr_persistence_features`: VR persistence on unit-normalized `(chunk - query)`; emits paper-aligned `w1_h0` and `lt_max_h1`, plus legacy alias metrics for backward compatibility.
- `calculate_silhouette_score_distribution`: mean/std of per-row silhouette from topics dataframe.
- `calculate_doc_spread` / `calculate_topic_spread`: set cardinalities divided by k.

### Problems found
- **P0 remediated in code:** Paper metric helpers now exist for `W₁(H₀)` and `LT_max(H₁)`; requires end-to-end replication before paper claims are considered validated (TASK-001).
- **P0:** No ε sweep (TASK-004).
- **P1 partially remediated:** cosine evaluator path converts similarity to distance, but direct inference callers must provide distance-like `topn_scores` (TASK-006).

### Recommended changes
- Add ε parameterization or drop paper-aligned claims.
- Add a small end-to-end fixture that verifies `run_scoring` returns a one-row feature frame with finite features.

### Agent tasks
- TASK-001, TASK-004, TASK-006, TASK-108, TASK-103 (baselines for comparison)

---

## Module: [`blowfish/ingestion/__init__.py`](../../blowfish/ingestion/__init__.py)

### Responsibility
Exports ingestion symbols.

### Actual behavior
Minimal.

### Problems found
- Verify consistency with public API named in README.

### Recommended changes
- Align `__all__` with documented classes.

### Agent tasks
- TASK-205

---

## Module: [`blowfish/ingestion/chunk_embeddings.py`](../../blowfish/ingestion/chunk_embeddings.py)

### Responsibility
Embed text chunks; persist `*_chunk_embeddings.pkl`.

### Actual behavior
- Instantiates embedder from `EmbeddingModelHooks`; encodes `Text` column; saves subset of columns.

### Problems found
- Optional SSL disable passes `verify=False` — security footgun (document).
- No validation of embedding dimension vs `vdb_vector_size` at this layer.

### Recommended changes
- Validate shape of first batch against indexer config in CI or runtime assert.

### Agent tasks
- TASK-201 (P2 validation)

---

## Module: [`blowfish/ingestion/topic_clustering.py`](../../blowfish/ingestion/topic_clustering.py)

### Responsibility
UMAP 2D projection + HDBSCAN labeling + per-point silhouette; save topics pickle.

### Actual behavior
- `detect_umap_nneighbours_optimum`: loops `u in np.arange(2,40,2)`, each run fits UMAP, builds **full pairwise distance matrix** on 2D projections for histogram (O(n²) per u).
- `run_hdbscan`: loops `i in 2..99`, fits HDBSCAN, tracks silhouette − outlier fraction heuristic; refits at argmax.
- Broad `except Exception` prints and may leave partial state.

### Problems found
- **P1:** O(n²) inner loop per UMAP trial (TASK-105).
- **P2:** Parameter `range` shadows builtin; `tolerance: int = 0.1` is misleading type.
- **P2:** `print` instead of logging; tqdm on nested loops.

### Recommended changes
- Replace histogram heuristic with subsampled or incremental criterion; cache projections.
- Use `logging`; rename `range` → `neighbor_grid`.

### Agent tasks
- TASK-105, TASK-202 (P2)

---

## Module: [`blowfish/ingestion/vdb_indexing.py`](../../blowfish/ingestion/vdb_indexing.py)

### Responsibility
FAISS flat index persistence; append vectors and hash mapping.

### Actual behavior
- Supports legacy `IndexFlatL2` and cosine mode via normalized vectors + `IndexFlatIP`; JSON sidecar records `hash_key` order and metric.

### Problems found
- **P0 partially remediated:** cosine index path exists; experiments must still record metric and compare L2 vs cosine (TASK-006).
- **P0 remediated:** duplicate `hash_key` values are rejected both within an input batch and against an existing index mapping (TASK-007).

### Recommended changes
- Optional future upsert policy for intentional re-indexing.

### Agent tasks
- TASK-006, TASK-007

---

## Module: [`blowfish/training/__init__.py`](../../blowfish/training/__init__.py)

### Responsibility
Package marker.

### Actual behavior
Minimal.

### Agent tasks
- TASK-205

---

## Module: [`blowfish/training/queries_embeddings.py`](../../blowfish/training/queries_embeddings.py)

### Responsibility
Batch-encode queries; assign `query_id`; pickle dump.

### Actual behavior
- Column assertions; `encode` full list; saves `{save_file}.pkl`.

### Problems found
- Large in-memory list of embeddings; no chunking for huge CSVs (P2 engineering).

### Recommended changes
- Optional batched encoding with configurable batch size.

### Agent tasks
- TASK-106 (related)

---

## Module: [`blowfish/training/queries_evaluation.py`](../../blowfish/training/queries_evaluation.py)

### Responsibility
Per-query FAISS search; join metadata from `topics_df`; compute doc/chunk match flags.

### Actual behavior
- `iterrows` loop; `search` single query at a time.
- Fails fast on duplicate `topics_df.hash_key`; uses scalar lookup after unique index setup.
- `evaluate_isin`: Levenshtein with `weights=(0,1,0)` — insertion/deletion free, substitutions counted.
- Cosine evaluator path normalizes query vectors and converts FAISS inner-product similarity to cosine distance.

### Problems found
- **P0 remediated:** Non-unique `hash_key` now raises `ValueError` (TASK-007).
- **P1:** No batched search (TASK-106).
- **P2:** `print` on assertion failure only.

### Recommended changes
- Batched `index.search` for all queries.

### Agent tasks
- TASK-007, TASK-106

---

## Module: [`blowfish/training/disambiguator_training.py`](../../blowfish/training/disambiguator_training.py)

### Responsibility
Flatten eval JSON-like columns; per-query feature aggregation; class-balanced sample; fit KDE; pickle model.

### Actual behavior
- `format_qa_eval_df`: explode lists; drop `label == -1`; build `topic_label`.
- `generate_queries_features`: groupby query; skip if <2 chunks; `correct_prediction` from first row's `chunk_match`.
- **Joint KDE** on `correct_prediction` + features with **bandwidth=0.2**; `shuffle` balance.

### Problems found
- **P1:** Fixed bandwidth, no scaling (TASK-101).
- **P2:** `print` on save; path concatenation without `os.path.join`.
- Drop columns by name but KDE sees first column as label — order-dependent (fragile if column order changes).

### Recommended changes
- Explicit feature matrix builder with column order contract + tests.
- Bandwidth selection / `StandardScaler` pipeline.

### Agent tasks
- TASK-101, TASK-103, TASK-104

---

## Module: [`blowfish/inference/__init__.py`](../../blowfish/inference/__init__.py)

### Responsibility
Package marker.

### Actual behavior
Single line / empty.

### Agent tasks
- TASK-205

---

## Module: [`blowfish/inference/scorer.py`](../../blowfish/inference/scorer.py)

### Responsibility
Join retrieval rows with topics; compute feature vector; KDE-based `p_correct`.

### Actual behavior
- `format_topics` fails fast on duplicate `hash_key`; `join_topics_to_query_and_chunks` merges query rows to topic metadata.
- `get_correctness_probability`: builds `[1]+sample` and `[0]+sample`, truncates to `n_features_in_`, uses a log-domain ratio with two `score_samples` calls.
- `run_scoring`: single-row path.

### Problems found
- **P1 remediated:** Numerical stability and redundant KDE calls improved; still needs extreme-tail fixture (TASK-102).
- **P1:** No batch API (TASK-106).
- **P2:** Assertion failures return `None` rather than raising — error-prone for callers.

### Recommended changes
- `score_batch(X)` accepting `(n_queries, n_features)`.

### Agent tasks
- TASK-102, TASK-106, TASK-203

---

## Module: [`blowfish/inference/decider.py`](../../blowfish/inference/decider.py)

### Responsibility
SHAP KernelExplainer over KDE surrogate; map negative SHAP mass to "topicspread" / "docspread" / "dataspread".

### Actual behavior
- `__init__` samples KDE until 50 "high confidence" feature rows (expensive).
- `KDE_prediction`: column-wise concat of ones + samples, log-domain ratio from KDE scores.
- `explain_query`: compares feature names and maps `top_k_doc_spread` to `"docspread"`.

### Problems found
- **P0 remediated:** Dead `docspread` branch fixed (TASK-002).
- **P1:** Construction cost remains high (TASK-107).

### Recommended changes
- Lazy-init explainer; cache background samples; add direct unit around `top_k_doc_spread` SHAP path if SHAP is installed.

### Agent tasks
- TASK-002, TASK-107, TASK-102

---

## Module: [`blowfish/utils/__init__.py`](../../blowfish/utils/__init__.py)

### Responsibility
Package marker.

### Agent tasks
- TASK-205

---

## Module: [`blowfish/utils/constants.py`](../../blowfish/utils/constants.py)

### Responsibility
Default KDE feature name ordering.

### Actual behavior
- `DEFAULT_KDE_FEATURES` list — must match training column order after `drop query_id`.

### Problems found
- Name `docspread` vs `top_k_doc_spread` inconsistency with decider (TASK-002).

### Recommended changes
- Single source of truth for feature keys + display names.

### Agent tasks
- TASK-002

---

## Module: [`blowfish/utils/embedding_models_factory.py`](../../blowfish/utils/embedding_models_factory.py)

### Responsibility
Map string keys to embedding classes (SentenceTransformer, OpenAI wrappers).

### Actual behavior
- Thin `encode()` wrappers returning numpy arrays.

### Problems found
- No dimension metadata surface per model for validation.

### Recommended changes
- `expected_dim` property or registry.

### Agent tasks
- TASK-201

---

## Module: [`blowfish/utils/vdb_factory.py`](../../blowfish/utils/vdb_factory.py)

### Responsibility
Dispatch `faiss_index` string to `FaissVDBIndexing`.

### Actual behavior
One hook class.

### Problems found
- No cosine index variant.

### Recommended changes
- Register `faiss_index_cosine` or metric flag.

### Agent tasks
- TASK-006

---

## Module: [`blowfish/requirements.txt`](../../blowfish/requirements.txt)

### Responsibility
Pinned dependencies for development/runtime.

### Actual behavior
- Lists runtime dependencies such as `giotto-tda` and `faiss-cpu`; SHAP has been split into `requirements-explain.txt`.

### Problems found
- **P0 remediated:** `setup.py` now reads package requirements into `install_requires` (TASK-005).

### Recommended changes
- Longer term: migrate to `pyproject.toml` and keep optional explain dependencies isolated.

### Agent tasks
- TASK-005

---

## Module: [`setup.py`](../../setup.py) (repo root)

### Responsibility
Package metadata.

### Actual behavior
- `install_requires` is populated from `blowfish/requirements.txt`.
- `extras_require["explain"]` is populated from `blowfish/requirements-explain.txt`.

### Problems found
- **P0 remediated:** TASK-005.

### Agent tasks
- TASK-005

---

## Summary task index (P0–P3)

| Priority | Task IDs |
|----------|----------|
| P0 | TASK-001 – TASK-008 |
| P1 | TASK-101 – TASK-109 |
| P2 | TASK-201 – TASK-204 (defined in AGENTIC_REMEDIATION_SPEC) |
| P3 | TASK-205 – TASK-206 |

See [`AGENTIC_REMEDIATION_SPEC.md`](./AGENTIC_REMEDIATION_SPEC.md) for executable task blocks.
