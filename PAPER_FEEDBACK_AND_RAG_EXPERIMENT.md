# Paper feedback and RAG experiment design

> Canonical scientific entry point for this repository. README.md retains install
> and quickstart only. Per-module engineering review lives under
> [`docs/review/`](./docs/review/README.md); this document is the **paper-level
> critique and forward experimental program** for arXiv:2406.07990 (Barillot &
> De Castro, *Topological quantification of ambiguity in semantic search*).
>
> Audience: paper authors, reviewers, and anyone planning to use Blowfish in a
> RAG system. The recommendations below are constructive; they assume the
> direction of the paper is sound and aim to harden the evidence base.

## Contents

1. [Why this document](#1-why-this-document)
2. [Statistical-plot critique with concrete replacements](#2-statistical-plot-critique-with-concrete-replacements)
3. [Statistical-test methodology — moving from visual to falsifiable](#3-statistical-test-methodology--moving-from-visual-to-falsifiable)
4. [Baselines and ablations the paper omits](#4-baselines-and-ablations-the-paper-omits)
5. [Upstream RAG experimental design](#5-upstream-rag-experimental-design)
6. [Pre-registration template](#6-pre-registration-template)
7. [Status and dependencies](#7-status-and-dependencies)

---

## 1. Why this document

The arXiv preprint demonstrates the *direction* — topological signatures of the
local neighborhood around a query embedding carry information correlated with
retrieval ambiguity — but the current evidence is **visual and qualitative**.
Reviewers and downstream users need:

- **Numerical evidence with uncertainty.** KDE plots without sample sizes, bandwidth
  disclosure, or confidence bands cannot be reproduced or stress-tested.
- **Falsifiable hypotheses.** "These distributions look different" is not a test.
  AUROC / AUPRC / Wasserstein / KS with bootstrap CIs and pre-registered
  thresholds are.
- **Head-to-head against simple baselines.** The non-negotiable question
  (`docs/review/METHODOLOGY_REVIEW.md` §5) — are topology + KDE genuinely better
  than score-gap, retrieval entropy, kNN density, or calibrated logistic on the
  same features? — is currently unanswered.
- **Confounder control.** Chunk length confounds the topic-count proxy
  (`TASK-008`); this needs to be eliminated or acknowledged as a limitation in
  the abstract.
- **Downstream usefulness.** Detecting ambiguity in isolation is interesting;
  improving an *end-to-end* RAG system's answer quality is what justifies a
  production deployment.

The four sections below propose concrete remediations.

---

## 2. Statistical-plot critique with concrete replacements

### 2.1 What the paper figures currently show

The paper relies primarily on:

- **Stacked KDE / histogram plots** of W₁(H₀) and LT_max(H₁) split by
  "ambiguous" (e.g. C₂₅₀ queries → C₇₅₀ corpus) and "clear" (C₇₅₀ queries →
  C₂₅₀ corpus) conditions.
- **Curves vs. ε** showing how the feature distributions evolve with the
  neighborhood scaling factor.
- **2D embedding visualizations** (UMAP / HDBSCAN) to motivate the topic-count
  intuition.

These plots show that something is happening, but they do not show *how much*,
*how reliably*, or *how it compares to a simpler baseline*.

### 2.2 What each plot is missing

| Issue | Why it matters | Recommended fix |
|---|---|---|
| Sample size `n` not stated per condition | Cannot judge power; KDE shape on small `n` is dominated by bandwidth | Annotate each panel with `n_ambiguous` and `n_clear`; report total query count |
| KDE bandwidth not disclosed | Different `h` produces different "separation"; reader cannot reproduce | State bandwidth rule (Scott, Silverman, CV-selected) and the resulting numeric `h` |
| No bootstrap bands around the KDE curves | Apparent separation may be sampling noise | Overlay 95% pointwise bootstrap CIs (B ≥ 1000 resamples) |
| No effect size | Visual separation is not a magnitude | Report Cohen's *d*, Cliff's δ, or AUROC with 95% CI in each panel title |
| No per-embedding-model panels with CIs | The paper averages across models; behavior may be model-specific | Faceted figure: one row per model, CI band per condition |
| ε sweeps shown as single curves | Cannot tell whether monotonicity is real | Bootstrap-CI ribbons around each ε curve; mark statistical significance |
| Histograms without overlaid summaries | Eye is biased by binning | Add median ± IQR markers; show mean with 95% CI as inset |
| No calibration / reliability diagram | If `p_correct` is to be used as a probability, it must be calibrated | Add a reliability diagram (binned mean vs. observed accuracy) and report ECE |
| Categorical "ambiguous vs clear" without intermediate | Binary labels hide gradient | If possible, replace with continuous difficulty score (e.g. annotated human grade) |
| Color choice and ink density | Some panels overplot heavily | Use semi-transparent fills + thicker median lines; colorblind-safe palette |

### 2.3 Per-figure replacement specs

For each existing figure, the replacement should preserve the same intuition
but add the missing rigor. Concretely:

**Figure F1 (KDE of W₁(H₀) and LT_max(H₁) by condition):**
- 2×4 grid: rows = {W₁(H₀), LT_max(H₁)}, cols = {model₁, …, model₄}.
- Each panel: filled KDE per condition with 95% bootstrap-CI ribbon.
- Inset: AUROC ± 95% CI for the single-feature classifier (this condition vs.
  the other).
- Footer: `n_ambiguous = …`, `n_clear = …`, `h = …` (with selection rule).

**Figure F2 (curves vs. ε):**
- One panel per (feature, model). x-axis = ε (with axis convention disclosed —
  paper's `ε = d(i,q) − 1` or the implementation's `d(i,q)`).
- y-axis = feature statistic (mean ± 95% bootstrap band).
- Vertical dashed line at the paper's headline operating point (`ε = 0.4` in
  the v2 preprint).
- Report AUROC vs. ε as a *second axis* or twin plot so the reader sees not
  just feature evolution but discriminative-power evolution.

**Figure F3 (UMAP + HDBSCAN clusters):**
- Keep as motivation; **add** an explicit caption acknowledging that 2D
  UMAP-with-HDBSCAN is an engineering heuristic, not the substrate of the
  topological claims. Cross-link `docs/review/METHODOLOGY_REVIEW.md` §3.5.

### 2.4 What to drop

- Any plot that depends on visual judgment alone without a paired numerical
  metric in the caption.
- Heatmaps with categorical axes where a table would be more honest.
- "Marker color = condition" plots without an accompanying conditional
  distribution test.

---

## 3. Statistical-test methodology — moving from visual to falsifiable

### 3.1 The current implicit hypothesis

The paper's narrative argument is roughly:

> H1: The distributions of W₁(H₀) and LT_max(H₁) differ between ambiguous and
> clear queries.
> H2: This difference is large enough to be useful for ranking / explanation.
> H3: The mechanism is the underlying manifold complexity, not chunk length.

H1 is testable with simple distributional tests; H2 needs a discriminative
metric with CIs; H3 needs confounder control.

### 3.2 Concrete test program

| Hypothesis | Test | Output | Multiple-testing |
|---|---|---|---|
| H1 (distributional difference) | Two-sample Kolmogorov-Smirnov; Wasserstein-1 with permutation p-value | KS statistic + p, W₁ + 95% CI | Bonferroni across the 4 features × 4 models = 16 tests |
| H1 (mean / median shift) | Permutation test of mean and median difference | Δ̄ ± 95% bootstrap CI, p | Same correction |
| H2 (discriminative power) | AUROC, AUPRC, Brier score, ECE; per-feature and stacked KDE | Each ± 95% bootstrap CI (B = 10 000) | n/a (primary endpoints) |
| H2 vs. baselines | Paired bootstrap of AUROC difference (Blowfish − baseline) | Δ AUROC ± CI, p | Bonferroni across baselines |
| H3 (length confound) | Stratify by chunk-length bucket; test H1/H2 within stratum; report interaction effect | Stratified AUROC and Δ̄ tables | Bonferroni across strata |

### 3.3 Sensitivity grids

Each headline result must be reported across a sensitivity grid, not at a
single hyperparameter choice:

- **k (top-k)** — `k ∈ {15, 25, 50, 100}` (paper uses 50).
- **ε** — `ε ∈ {0.2, 0.4, 0.8, 1.6}` in the paper's convention.
- **KDE bandwidth `h`** — `{Scott, Silverman, 0.1, 0.2, 0.5}`.
- **Embedding model** — all four from Table 2 + at least one open model.
- **Random seed** — at least 5 seeds for any sampling-based step (train/val
  split, bootstrap, balancing).

A result that flips sign within the sensitivity grid is not a result.

### 3.4 Held-out evaluation and pre-registration

- Split queries into **train / val / test** at the *document* level (not query
  level — same document in both splits is leakage).
- Pre-register on OSF (or equivalent) the primary metric (AUROC of Blowfish vs.
  best baseline on the test split), the pre-defined "win" threshold, and the
  exclusion criteria, *before* touching the test set.
- All hyperparameter selection on `val`; test set touched once for the final
  numbers reported in the paper.

### 3.5 Calibration

If Blowfish output is to be called `p_correct`, it must be calibrated. Report:

- **Reliability diagram** with 10 equal-mass bins, observed accuracy ± Wilson CI
  per bin.
- **Expected Calibration Error (ECE)** with 95% bootstrap CI.
- **Brier score** alongside AUROC.

If raw KDE-ratio outputs are not calibrated (likely), report a calibrated
variant (isotonic or Platt scaling on `val`) and a one-line caveat in the
abstract.

---

## 4. Baselines and ablations the paper omits

### 4.1 Baseline matrix

| ID | Method | Why include |
|---|---|---|
| B0 | Random scoring | Sanity floor |
| B1 | Top-1 retrieval score | Cheapest baseline available in any RAG stack |
| B2 | Score gap: `score_1 - score_2` | Common heuristic, ~free |
| B3 | Top-k score entropy | Distribution-aware, ~free |
| B4 | Mean kNN distance (Euclidean offset norm mean) | Closest geometric analogue without topology |
| B5 | kNN density estimate (1 / mean kNN distance) | Density baseline for KDE comparison |
| B6 | Mahalanobis distance from query to retrieved cluster centroid | Covariance-aware |
| B7 | Calibrated logistic regression on the **same** Blowfish features | Isolates KDE vs. discriminative on same inputs |
| B8 | Gradient-boosted trees on the same features | Stronger discriminative baseline |
| B9 | Retrieval rank of the gold chunk (oracle) | Upper bound for the gating use case |

Reporting AUROC ± 95% CI for B0–B9 on the same test set is the minimum bar.
Blowfish's claim of being "topological" is only meaningful if it beats B7/B8 on
its own features — otherwise the lift comes from features alone, not topology.

### 4.2 Feature ablations

Strip features one block at a time and re-run the full pipeline:

| Variant | Features kept |
|---|---|
| Full | All 9 KDE features |
| − topology | Drop `w1_h0`, `lt_max_h1` (and legacy aliases) |
| − spread | Drop `top_k_doc_spread`, `top_k_topic_spread` |
| − silhouette | Drop silhouette mean/std |
| − scale | Drop `scale_mean`, `scale_min`, `iq25-75_scale` |
| Topology-only | Only `w1_h0`, `lt_max_h1` |
| Score-only | Only `scale_*` (no topology, no clustering) |

For each variant report AUROC ± CI. If `Full − topology` ≈ `Full` within CIs,
the topology features are decorative for *this* downstream task; that's a
legitimate finding and should be reported honestly rather than masked.

### 4.3 Randomized topology controls

- **Permuted neighborhoods.** Replace the retrieved top-k chunks with k random
  chunks from the corpus; recompute features; re-evaluate AUROC. Any
  discriminative power in this control is a confounder (e.g. corpus-wide
  geometry leaking through).
- **Rotated embeddings.** Apply a random orthogonal rotation to all embeddings;
  topological features are rotation-invariant by construction → AUROC must be
  unchanged. Failing this test signals an implementation bug.

### 4.4 Falsification criteria

Pre-declare what would refute the topology contribution:

- If `Δ AUROC (Full − Topology-only) ≤ 0.01` with 95% CI overlapping zero on
  the test set, the paper's topology contribution is **not demonstrated**.
- If B7 (calibrated logistic on same features) is within 0.01 AUROC of the
  KDE, the "KDE is the modelling choice" claim is **not justified** — a simpler
  classifier is equivalent.

These thresholds should be agreed *before* running the experiment.

---

## 5. Upstream RAG experimental design

### 5.1 Research question

> Does gating retrieval-augmented generation with Blowfish's `p_correct`
> improve end-to-end answer quality, measured against (i) ungated RAG and (ii)
> simpler gates, on public open-domain QA benchmarks?

Detecting ambiguity is interesting; *acting on it* is what makes Blowfish a
production-worthy tool.

### 5.2 Datasets

Run on at least two of:

| Dataset | Why |
|---|---|
| HotpotQA (distractor + full-wiki) | Multi-hop; explicit hard / easy splits |
| Natural Questions (NQ-open) | Realistic open-domain queries; calibrated short / long answers |
| MS MARCO v2.1 (passage / document) | Industrial-scale; binary-relevance judgements |
| TriviaQA (unfiltered web) | Long-tail entities; tests robustness |

Pick one as the **primary** benchmark and pre-register; the others provide
external validity.

### 5.3 System under test

Fixed within a single experimental cell; swept across cells per §5.3.1:

- **Corpus chunker** (see §5.3.1).
- **Embedding model** (one of: `text-embedding-3-small`, `bge-large`, `e5-mistral`).
- **Retriever**: FAISS `IndexFlatIP` on unit-normalized embeddings (matches paper
  cosine convention).
- **Generator**: a frozen LLM (e.g. `gpt-4o-mini`, `claude-haiku-4-5`,
  `llama-3.1-8b-instruct`).
- **Prompt template**: published verbatim in the paper appendix.
- `k = 50` retrieved chunks per query.

The **gating policy** is the primary IV (§5.4); the chunker and embedding
model are secondary IVs whose effects are measured in the sensitivity grid.

### 5.3.1 Text segmentation as a swept axis

Chunking matters: the paper's `C₂₅₀` vs. `C₇₅₀` comparison conflates *token
budget* with *number of curated topics*. To disentangle them — and to make
the Blowfish topology features comparable across granularities — chunking
becomes an explicit experimental dimension. Four concrete chunkers cover
the axes; all four implement the package's `Chunker` Protocol so the
experiment harness sweeps them uniformly.

| Chunker | Implementation | Why this one |
|---|---|---|
| `SentenceChunker` | `pysbd` (default) or `spaCy` opt-in via `[datasets-spacy]` extra | Smallest principled semantic unit; deterministic, multilingual, no data download. Matches paper's "fine" chunking axis. |
| `ParagraphChunker` | In-repo regex on `\n\n+` separators | Mid-grain, structure-respecting, zero deps. Matches "coarse" axis. |
| `RecursiveCharacterChunker` | `langchain-text-splitters.RecursiveCharacterTextSplitter` | The community baseline; required for cross-paper comparability. Tries paragraph → sentence → word → character in order. |
| `TokenSlidingWindowChunker` | `tiktoken` (cl100k or o200k) | Token-budget aware; `size` and `stride` directly sweep the paper's `C_N` axis. Token-boundary-correct against modern LLM context windows. |

**Sweep grid (locked):**

- `chunker ∈ {Sentence, Paragraph, Recursive(size=512, overlap=64), Token(size=250), Token(size=500), Token(size=750)}` — six cells covering the paper's chunk-length range plus the two structural extremes.
- `tokenizer ∈ {cl100k_base, o200k_base}` only for the token chunker. `o200k` is the modern OpenAI / Anthropic-compatible BPE; `cl100k` is the legacy default.
- Embedding × chunker is a full Cartesian product in the sensitivity grid (§3.3) for the top-3 embedding models; for the long tail we report a single chunker for cost.

**Documented but deferred to a follow-up PR** (rationale in parentheses):

- **Semantic chunkers** — `LlamaIndex.SemanticSplitterNodeParser`, `langchain.text_splitter.SemanticChunker`, `chonkie.SemanticChunker` (introduces embedder-dependence into a primitive that should be embedder-independent; complicates the sensitivity grid).
- **Late chunking** — Jina-AI 2024 (embed the full document with a long-context model, pool over chunk boundaries). Ties the experiment to a specific embedding-model family; defer until Jina or a comparable open model is locked.
- **Hierarchical chunkers** — multi-resolution (small for retrieval, large for context). Worth ablating but doubles the matrix; deferred.
- **Markdown / HTML structure-aware splitters** — Nobel and Fields lecture text isn't reliably structured; not load-bearing for this corpus.

**Reproducibility**: every chunker writes its `chunker_params` (size, overlap, separators, library, library version) into the `DatasetManifest`, which is sha256-hashed into the cache key. Re-running with the same params hits the cache; changing any param invalidates it. This makes every reported number in §5.5 reconstructable from its manifest hash alone.

### 5.4 Gating policies (the IV)

| Policy | Description |
|---|---|
| **G0 — No gate** | Always pass top-k to the generator (baseline) |
| **G1 — Score gap** | If `score_1 − score_2 < τ`, route to *abstain / re-query*; else generate |
| **G2 — Retrieval entropy** | If entropy(softmax(top-k scores)) > τ, abstain |
| **G3 — Mean kNN distance** | If mean Euclidean offset norm > τ, abstain |
| **G4 — Blowfish `p_correct`** | If `p_correct < τ`, abstain |
| **G5 — Calibrated logistic on Blowfish features** | Discriminative baseline at the same gate point |
| **G6 — Oracle gate** | Use gold chunk presence in top-k as the gate (upper bound) |

Tune τ on `val` to a fixed *abstain rate* (e.g. 20%) across all gates so they
are comparable at constant cost. Then report quality at multiple abstain
rates (10 / 20 / 30%) — a Pareto frontier, not a single point.

### 5.5 Metrics (the DVs)

Primary:

- **Answered subset accuracy** (Exact Match and F1) — quality of the answers
  the system *did not* abstain on.
- **Abstain precision / recall vs. ground truth answerability** — when the
  gate fires, how often was it right to fire?
- **Net utility under a cost model** — for a per-query cost of {correct answer
  = +1, wrong answer = −c, abstain = 0}, compute expected utility per query.
  Vary `c ∈ {1, 3, 10}`.

Secondary:

- **Calibration of `p_correct`** (Brier, ECE) on the answerability label.
- **Latency overhead** of running Blowfish features per query (the gate must
  not dominate retrieval latency).
- **Robustness** — repeat under permuted retrieval order; results should be
  rank-invariant.

### 5.6 Statistical analysis

- Pair queries across gating policies (same query, different gate) and use
  **paired bootstrap** for all pairwise comparisons.
- Report Δ-EM, Δ-F1, Δ-utility with 95% CI per pairwise comparison.
- Bonferroni-correct over the 6 pairwise comparisons against G4.
- **Stratified analysis** by query difficulty (easy / medium / hard from the
  dataset's annotations or a fixed-baseline proxy). Blowfish's value is
  expected to concentrate on the harder strata — pre-register this prediction
  and test it explicitly.

### 5.7 Reproducibility checklist

- All seeds fixed and published; at least 5 seeds for the answer-set
  bootstrap.
- Embedding model and generator versions pinned (date-stamped).
- Prompts and retrieval params published verbatim.
- Code: a single CLI in `blowfish/experiments/` that takes `--dataset
  --gate --tau` and produces a CSV row per query.
- Results: a single notebook in `docs/review/BASELINE_RESULTS.md` that reads
  the CSVs and produces every published table / figure.

### 5.8 What would falsify the value of Blowfish gating

Pre-declare:

- **G4 vs. G0** — if Δ-utility ≤ 0 with 95% CI overlapping zero across all
  three cost models, gating with Blowfish is **not useful for this task at
  this abstain rate**. Report as a null result.
- **G4 vs. G1/G2/G3** — if Blowfish gating is within 0.01 utility of any of
  the cheap heuristic gates, the cost of the topology pipeline is **not
  justified for this task**. Report and recommend the cheap gate.
- **G4 vs. G5** — if calibrated-logistic on the same features is within 0.005
  utility of Blowfish, the **KDE** is not the load-bearing component;
  reformulate as a discriminative classifier.

---

## 6. Pre-registration template

Copy the block below into the project's OSF / aspredicted entry before
running the test-set evaluation. Once the test set is touched, this is
locked.

```
Title: Blowfish v2 — Topological ambiguity scoring on open-domain RAG

Primary hypothesis (H_primary):
  On the held-out test split of {dataset}, gating policy G4 (Blowfish
  p_correct) achieves higher expected utility under cost model c=3 than
  policy G0 (no gate), as measured by paired-bootstrap Δ-utility with 95%
  CI excluding zero (Bonferroni-corrected across 6 pairwise comparisons).

Secondary hypotheses:
  H_s1: G4 > G1, G2, G3 (cheap gates) at the same abstain rate.
  H_s2: G4 ≈ G5 (discriminative on same features) — null preferred; rejection
        triggers a follow-up to replace KDE with the discriminative model.
  H_s3: G4's advantage concentrates on the hard difficulty stratum.

Pre-defined "win" threshold:
  Δ-utility CI excludes zero AND point estimate ≥ 0.02 utility.

Sensitivity grid (each result reported across):
  k ∈ {25, 50, 100}; ε ∈ {0.2, 0.4, 0.8}; KDE bandwidth ∈ {Scott, 0.2};
  seed ∈ {0..4}.

Exclusion criteria:
  - Queries with no gold passage in the corpus (unanswerable by design)
  - Queries where the retriever returns < k chunks
  - Queries flagged by the dataset as ambiguous-by-construction

Analysis plan:
  Frozen Jupyter notebook in docs/review/BASELINE_RESULTS.md commit
  hash {HASH}. Test set evaluated exactly once.

Falsification criteria:
  Failure of H_primary → report null result; the abstract is rewritten to
  describe the topology as a diagnostic, not a production gate.
```

---

## 7. Status and dependencies

| Workstream | Status | Blocks |
|---|---|---|
| Plot / test rigor (§2, §3) | **Not started** — needs paper rewrite + figure regeneration | Paper v3 |
| Baselines B0–B9 (§4.1) | **Spec only** (`docs/review/AGENTIC_REMEDIATION_SPEC.md` TASK-103) | RAG experiment, all comparisons |
| Feature ablations (§4.2) | **Spec only** (`docs/review/ABLATION_PLAN.md`, template) | Topology contribution claim |
| Randomized topology controls (§4.3) | **Not started** | Honesty check |
| RAG experiment harness (§5) | **Not started** — package has no `experiments/` module yet | All downstream claims |
| Pre-registration (§6) | **Not started** | Reviewer credibility |
| Length-confound stratification (`TASK-008`) | **Open — paper task** | H3 in §3.1 |
| ε sweep on real corpora | **Code ready** (`TASK-004` landed); needs data harness | F2 replacement figure |

Single most valuable next step: implement the **B0 / B4 / B5 / B9** baselines
(cheap, cover the discriminative-vs-density question) and run them on a small
public split (e.g. NQ-open dev) end-to-end. The result either validates the
paper direction with numbers or surfaces the gap honestly — both are
publishable.
