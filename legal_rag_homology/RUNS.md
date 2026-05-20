# Run registry

One row per evaluation run. Update when you launch a new run.

Columns:
- **Profile**: `chunk512` (default) or `chunk128`. Determined by `CHUNK_PROFILE` env var.
- **Retriever**: `hybrid` / `homology` / `bm25` / `dense` / `random_homology` / `tail_homology`.
- **Topk**: `--topk` flag passed to `run_rag` (homology retriever ignores topk; field reads `n/a`).
- **Shared tol** / **Noise tol**: `--shared-tolerance` / `--noisiness-tolerance` (homology retriever only).
- **mean n_retrieved**: pulled from `metrics.json -> retrieval_chunk_metrics.mean_n_retrieved`. Useful sanity check.
- **Notes**: free-form. Use this for anything that distinguishes the run from a sibling.

If a row says `?`, the parameter wasn't recorded at launch — fill in from shell history (`history | grep run_rag`) when you can.

## chunk512 (CHUNK_TOKENS=512, CHUNK_OVERLAP_TOKENS=100)

| Run name | Retriever | Topk | Shared tol | Noise tol | mean n_retrieved | Notes |
|---|---|---|---|---|---|---|
| baseline-hybrid-sonnet | hybrid | 50 | n/a | n/a | 50.0 | Full candidate set, no rerank trim |
| baseline-hybrid-sonnet-k5 | hybrid | 5 | n/a | n/a | 5.0 | Production-style k=5 baseline |
| homology-sonnet-1 | homology | n/a | 0.5 | 0.5 | 5.6 | Confirmed by user 2026-05-19 |
| homology-sonnet-2 | homology | n/a | ? | ? | 11.4 | Backfill tolerances |
| homology-sonnet-3 | homology | n/a | ? | ? | 9.5 | Backfill tolerances |
| homology-sonnet-4 | homology | n/a | ? | ? | 11.3 | Backfill tolerances |
| homology-sonnet-5 | homology | n/a | ? | ? | 3.0 | Backfill tolerances |
| homology-sonnet-6 | homology | n/a | ? | ? | 17.0 | Backfill tolerances |

## chunk128 (CHUNK_TOKENS=128, CHUNK_OVERLAP_TOKENS=32)

| Run name | Retriever | Topk | Shared tol | Noise tol | mean n_retrieved | Notes |
|---|---|---|---|---|---|---|
| baseline-hybrid-sonnet-k5 | hybrid | 5 | n/a | n/a | 5.0 | Mirror of chunk512 baseline |
| homology-sonnet-1 | homology | n/a | 0.5 | 0.5 | 13.1 | Same tolerances as chunk512 sibling, but the finer chunks pushed mean_n_retrieved up sharply |
| homology-sonnet-2 | homology | n/a | 0.3 | 0.3 | 8.1 | Tighter tolerances pulled retrieved set from 13.1 -> 8.1, still above chunk512 sibling (5.6) |

## Conventions

- Run names are reused across profiles; the `runs_chunk128/` vs `runs/` directory tells you which profile produced them.
- Retrieval-only re-evaluation: `python -m scripts.legal_rag_homology.run_evaluate --patch-retrieval --run <name>` (or `--all`). No LLM judge calls; only retrieval metrics refresh.
- Full re-evaluation: `python -m scripts.legal_rag_homology.run_evaluate --run <name>`. Re-runs the LLM judge — expensive.
- To launch under chunk128, set the three env vars: `CHUNK_PROFILE=chunk128 CHUNK_TOKENS=128 CHUNK_OVERLAP_TOKENS=32`.
