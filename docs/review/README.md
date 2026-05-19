# Blowfish review bundle

Agent-executable documentation for repository and companion paper (arXiv:2406.07990) alignment, remediation, and validation.

> Outward-facing scientific critique + RAG experimental design lives one level up
> at [`PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md`](../../PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md).
> This bundle is the per-module engineering review with TASK IDs.

## Priority buckets (rubric)

| Bucket | Maps to | Meaning |
|--------|---------|---------|
| **Critical** | **P0** | Correctness bugs, code↔paper breakage, packaging that hides deps, unsupported scientific claims |
| **High** | **P1** | Methodology gaps (bandwidth, baselines, metrics), performance hot paths, missing batch APIs |
| **Medium** | **P2** | Logging, validation, tests, warning hygiene, strict keys |
| **Low** | **P3** | Naming, exports, polish |

## Canonical reading order

1. [`MATH_HONESTY_CHECK.md`](./MATH_HONESTY_CHECK.md) — terminology → operations (no laundering)
2. [`METHODOLOGY_REVIEW.md`](./METHODOLOGY_REVIEW.md) — KDE + topology + experiments vs paper
3. [`TECHNICAL_REVIEW.md`](./TECHNICAL_REVIEW.md) — per-module implementation review + task index
4. [`PERFORMANCE_REVIEW.md`](./PERFORMANCE_REVIEW.md) — hotspots + profiling gates
5. [`TESTING_AND_PROFILING_PLAN.md`](./TESTING_AND_PROFILING_PLAN.md) — coverage matrix
6. [`BASELINE_RESULTS.md`](./BASELINE_RESULTS.md) — *fill after experiments*
7. [`ABLATION_PLAN.md`](./ABLATION_PLAN.md) — feature and placebo ablations
8. [`AGENTIC_REMEDIATION_SPEC.md`](./AGENTIC_REMEDIATION_SPEC.md) — **TASK-001…TASK-206** full executable blocks

## P0 task index (headlines)

| ID | Title | Current status |
|----|-------|----------------|
| TASK-001 | Paper-aligned VR metrics or honest rename | **Implemented in code; needs end-to-end experiment validation** |
| TASK-002 | Fix `docspread` vs `top_k_doc_spread` | **Implemented** |
| TASK-003 | Default `top_k` vs paper k=50 | **Implemented in README/config warning** |
| TASK-004 | ε neighborhood scaling | **Implemented (approach A: ε-threshold subsample); end-to-end ε sweep on real data still open** |
| TASK-005 | `setup.py` / packaging dependencies | **Implemented** |
| TASK-006 | Cosine vs L2 FAISS + `scale_*` semantics | **Partially implemented; cosine index path exists, ε/metric experiments still open** |
| TASK-007 | Unique `hash_key` enforcement | **Implemented for indexer, evaluator, and scorer** |
| TASK-008 | Paper length confound control | **Open; paper/data task** |

## Non-negotiable review answer ([`METHODOLOGY_REVIEW.md`](./METHODOLOGY_REVIEW.md) §5)

**Are topology + KDE genuinely improving ambiguity estimation vs simpler heuristics?**

**Current evidence in repo:** *Not demonstrated.*

**Next experiment:** see [`ABLATION_PLAN.md`](./ABLATION_PLAN.md) + [`BASELINE_RESULTS.md`](./BASELINE_RESULTS.md).

## Current verification snapshot

As of this review pass, focused remediation tests pass locally:

```text
35 passed, 11 warnings
```

The warnings are dependency/runtime hygiene items, not failed assertions: Pydantic v2 class-based `Config` deprecations and giotto-tda dimensionality warnings on tiny test fixtures.

## Deliverables checklist (charter §2)

| File | Status |
|------|--------|
| `TECHNICAL_REVIEW.md` | Done |
| `METHODOLOGY_REVIEW.md` | Done |
| `PERFORMANCE_REVIEW.md` | Done |
| `AGENTIC_REMEDIATION_SPEC.md` | Done |
| `TESTING_AND_PROFILING_PLAN.md` | Done |
| `BASELINE_RESULTS.md` | Template |
| `ABLATION_PLAN.md` | Template |
| `MATH_HONESTY_CHECK.md` | Done |
