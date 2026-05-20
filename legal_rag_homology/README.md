# Legal RAG Hallucinations Reproduction

Reproduction of Magesh et al., "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools" (Stanford RegLab, 2024), with a pluggable retriever so a homology-based variant can be compared against the baseline.

See the design spec in `docs/superpowers/specs/2026-04-22-legal-rag-hallucinations-reproduction-design.md`.

## Pipeline

All commands are run with `poetry run` from the repo root.

### 1. Build corpus

```bash
# Download dataset and extract questions + gold citations
poetry run python -m scripts.legal_rag_homology.run_build_corpus --step dataset

# Fetch opinions from CourtListener (gold + distractors)
poetry run python -m scripts.legal_rag_homology.run_build_corpus --step corpus
```

Options: `--distractor-count N` (default 5000), `--seed N` (default 42).

Gold opinion lookups are checkpointed to `data/corpus/gold_opinions_checkpoint.parquet` so re-runs skip the lookup step. Delete the file to force a refresh.

### 2. Build indices

```bash
# Chunk opinions, build BM25 + dense (FAISS) indices
# TORCHDYNAMO_DISABLE=1 is required to prevent segfaults from torch.compile on macOS
TORCHDYNAMO_DISABLE=1 TOKENIZERS_PARALLELISM=false \
  poetry run python -m scripts.legal_rag_homology.run_index
```

Options: `--step {chunks,bm25,dense,all}` (default all), `--batch-size N` (default 64).

Steps can be run individually if only one index needs rebuilding.

### 3. Run RAG + evaluate

```bash
poetry run python -m scripts.legal_rag_homology.run_rag --retriever hybrid --run-name baseline-hybrid
poetry run python -m scripts.legal_rag_homology.run_evaluate --run baseline-hybrid
```

Options: `--sample N` to evaluate a subset of questions.

Evaluation combines:
- **Groundedness** (deterministic): compares predicted citations against gold. A response is *misgrounded* if it contains fabricated citations.
- **Correctness** (LLM judge via Bedrock): rates legal reasoning as correct/partially_correct/incorrect.
- **Hallucination**: a response is hallucinated if it is incorrect OR misgrounded.

## Environment

Requires `COURTLISTENER_API_TOKEN` env var (falls back to a default token if unset).

All artifacts live under `scripts/legal_rag_homology/data/` (gitignored).
