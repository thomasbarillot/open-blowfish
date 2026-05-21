# Blowfish

Library for quantifying ambiguity in semantic search via topological signatures of
the query neighborhood, with a Gaussian KDE ratio used at inference time. Official
implementation of [arXiv:2406.07990](https://arxiv.org/abs/2406.07990) (Barillot &
De Castro).

> **Read this first:**
> [`PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md`](./PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md)
> is the canonical scientific entry point — it covers the paper-level critique,
> the upstream RAG experimental design, and the open scientific gaps. This README
> is install + quickstart only.

## Installation

```bash
pip install -r blowfish/requirements.txt
pip install -e .
```

Optional SHAP explanations (`FeedbackDecider`):

```bash
pip install -e ".[explain]"
```

**Breaking change in 0.2.0** — KDE feature names use the paper-aligned persistence
metrics (`w1_h0`, `lt_max_h1`). Retrain KDE models produced before this release.

## Quickstart

```py
import pandas as pd

from blowfish.ingestion import NaiveChunksEmbedding, TopicClusterGenerator, FaissVDBIndexing
from blowfish.training import BulkQueriesEmbedder, BulkQueriesEvaluator, DisambiguationModelGenerator
from blowfish.inference import AmbiguityScorer, FeedbackDecider

configuration = {
    "llm_encoder_config": {"model_name_or_path": "sentence-transformers/all-mpnet-base-v2"},
    "llm_encoder_type": "sentence_transformer",
    "vdb_vector_size": 768,
    "vdb_type": "faiss_index",
    "vdb_metric": "cosine",        # paper convention; "l2" remains for legacy indexes
    "top_k_results": 50,           # paper k; warning fires below 15
}

# --- Ingestion: build the chunk embedding store -----------------------------
chunks_df = pd.read_csv(...)                          # columns: Text, docname, hash_key (unique!)
embedded_chunks = NaiveChunksEmbedding(**configuration)(chunks_df, docname="example_doc")
topics_df = TopicClusterGenerator()(embedded_chunks, docname="example_doc")
vector_index, vector_mapping = FaissVDBIndexing(**configuration)(topics_df)

# --- Training: fit the KDE on a Q&A set -------------------------------------
qa_training_data = pd.read_csv(...)                   # columns: query, answer, docname
embedded_queries = BulkQueriesEmbedder(**configuration)(qa_training_data)
query_eval = BulkQueriesEvaluator(**configuration)(embedded_queries, topics_df)
queries_features, balanced_features, kde = DisambiguationModelGenerator()(query_eval)

# --- Inference: score a new query ------------------------------------------
scorer = AmbiguityScorer(kde, topics_df)
decider = FeedbackDecider(kde)

features_df = ...                                     # topn_docname, topn_scores, topn_rank,
                                                      # query_embedding, chunk_embeddings, hash_key
clarity_score, query_features, chunks_with_topics = scorer.run_scoring(features_df)
explanation = decider.explain_query(query_features)
```

A higher `clarity_score` is better. `explanation` is one of `topicspread`,
`docspread`, or `dataspread`, indicating the largest SHAP contributor.

### Paper Eq. (1) ε neighborhood (TASK-004)

`AmbiguityScorer(..., epsilon=...)` and
`DisambiguationModelGenerator(epsilon=...)` accept an optional ε that
subsamples retrieved neighbors before VR persistence runs (paper §3.2).
`epsilon=None` (default) preserves pre-0.3 behavior. The paper's axis
convention is `ε = d(i,q) − 1`; this kwarg takes `d(i,q)` directly, so the
paper's `ε = 0.4` corresponds to `epsilon=1.4` here.

## Model support

Out of the box: `SentenceTransformer`, `OpenAIEmbeddings`, `AzureOpenAIEmbeddings`.
Custom models: subclass in
[`blowfish/utils/embedding_models_factory.py`](./blowfish/utils/embedding_models_factory.py)
implementing `encode() -> np.ndarray`.

## Configuration reference

The configuration dictionary is shared across the ingestion, training, and
inference modules. Required keys: `llm_encoder_config`, `llm_encoder_type`,
`top_k_results`, `vdb_type`, `vdb_vector_size`. Use the same `top_k_results`
and `vdb_metric` end-to-end; rebuild the index if `vdb_metric` changes (`l2`
vs. `cosine`).

| Optional key | Default | Notes |
|---|---|---|
| `embeddings_storage_dir` | `./` | Where chunk embedding pickles go |
| `topics_storage_dir` | `./` | Where topic pickles go |
| `vdb_path` | `./faiss.index` | FAISS index path |
| `json_index_path` | `./index.json` | hash_key → faiss-row sidecar |
| `vdb_reset_faiss_index` | `False` | Overwrite existing index |
| `vdb_metric` | `l2` | `cosine` aligns with the paper; `l2` is legacy |
| `kde_storage_name` | `disambiguator_kde.pkl` | KDE pickle filename |
| `disable_ssl` | `False` | For OpenAI client behind corporate proxies |

## KDE features

`scale_mean`, `scale_min`, `iq25-75_scale`, `w1_h0`, `lt_max_h1`,
`top_k_doc_spread`, `top_k_topic_spread`, `silhouette_score_mean`,
`silhouette_score_std`.

`w1_h0` and `lt_max_h1` follow arXiv:2406.07990 (`W₁(H₀)` with `(N-1)^{-1}`
normalization; `LT_max(H₁)` half-life). Older keys
(`max_homology_birth`, `mean_homology1st_lifetime`, …) are still computed if
you keep them in `kde_features_order`.

## Tests

```bash
pip install -r requirements-dev.txt
pip install -e .
pytest tests/
```

CI is described in [`.github/workflows/README.md`](./.github/workflows/README.md).

## Documentation map

- [`PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md`](./PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md) — paper-level critique, statistical-test program, RAG experimental design (**start here**).
- [`docs/review/`](./docs/review/README.md) — per-module engineering review, TASK-001…TASK-206 spec, methodology trace.

## Contribution, license, citation

Pull requests welcome — please ensure `pytest` passes locally.

Copyright ©2024 BlackRock, Inc. Distributed under the
[Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

```bibtex
@misc{barillot2024blowfishtopologicalstatisticalsignatures,
  title         = {Blowfish: Topological and statistical signatures for quantifying ambiguity in semantic search},
  author        = {Thomas Roland Barillot and Alex De Castro},
  year          = {2024},
  eprint        = {2406.07990},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2406.07990}
}
```
