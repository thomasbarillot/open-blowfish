from __future__ import annotations

import argparse
import json
import logging
import re

import boto3
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from . import config

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """You are a legal expert grading an AI-generated legal research response
against a reference answer that human experts have already accepted as accurate.

You will be given:
- The legal question
- The REFERENCE answer (human-accepted, treat as ground truth)
- The CANDIDATE AI response (to be graded)

Grade the candidate against the reference:
- Does the candidate reach the same legal conclusion as the reference?
- Does it rest on the same controlling authorities and reasoning?
- Are there material legal errors or omissions relative to the reference?

Be conservative: prefer 'partially_correct' over 'correct' when the candidate
agrees in conclusion but diverges in authorities or reasoning. Prefer
'incorrect' when the candidate reaches a different legal conclusion.

Rate as one of: correct, partially_correct, incorrect

Respond in JSON format exactly like this:
{
  "correctness": "correct|partially_correct|incorrect",
  "reasoning": "Brief explanation (2-3 sentences max)"
}"""


def _judge_correctness(client, question: str, candidate: str, reference: str) -> dict:
    user_content = (
        f"Question:\n{question}\n\n"
        f"REFERENCE answer (human-accepted):\n{reference}\n\n"
        f"CANDIDATE AI response:\n{candidate}"
    )
    messages = [{"role": "user", "content": user_content}]
    resp = client.invoke_model(
        modelId=config.JUDGE_MODEL_ID,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "system": _JUDGE_PROMPT,
                "messages": messages,
            }
        ),
        contentType="application/json",
    )
    body = json.loads(resp["body"].read())
    text = body["content"][0]["text"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"correctness": "error", "reasoning": text}


def _normalize_citation_spacing(cite: str) -> str:
    """Collapse internal spacing differences (e.g. 'Md. App.' vs 'Md.App.')."""
    return re.sub(r"\s+", " ", re.sub(r"\.(\w)", r". \1", cite)).strip()


_QUOTE_NORMALIZE_RE = re.compile(r"\s+")
_PREFIX_MATCH_LEN = 60
_MIN_FULL_MATCH_LEN = 12
_FUZZY_MIN_LEN = 16
_FUZZY_THRESHOLD = 90


def _normalize_for_quote_match(text: str) -> str:
    """Collapse whitespace and unify quote glyphs so passage-matching tolerates
    chunking line breaks and curly-vs-straight quotes."""
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return _QUOTE_NORMALIZE_RE.sub(" ", text).strip()


def _quote_in_chunk(needle: str, haystack: str) -> bool:
    """Substring match with a prefix fallback for chunk-boundary truncation.

    Many opinion quotes in released answers run longer than 100 tokens and can
    straddle our 512-token chunks. If the full quote doesn't appear as-is, try
    the leading 60 chars — that's usually unique enough to avoid false positives
    while catching quotes whose tail got cut off by a chunk break.

    Final fallback for needles >= _FUZZY_MIN_LEN: rapidfuzz partial_ratio at
    >= _FUZZY_THRESHOLD. Catches quotes whose punctuation/glyphs differ from
    the chunk text or whose copy in the response is slightly off. Threshold
    set conservatively to limit false positives on short quotes.
    """
    if needle in haystack:
        return True
    if len(needle) > _PREFIX_MATCH_LEN and needle[:_PREFIX_MATCH_LEN] in haystack:
        return True
    if len(needle) >= _FUZZY_MIN_LEN:
        from rapidfuzz import fuzz
        if fuzz.partial_ratio(needle, haystack) >= _FUZZY_THRESHOLD:
            return True
    return False


def _build_gold_chunk_sets() -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, str],
    dict[str, bool],
]:
    """Map each question_id to chunks of the *cited opinion* whose text contains
    the released answer's adjacent_quote.

    Gold-relevance here means "this chunk grounds the legal claim": it sits
    inside the cited document and contains the supporting passage. If the quote
    cannot be located in any chunk of the cited opinion, that gold citation
    contributes no chunks (per design choice — strict passage matching).

    Returns:
        gold_chunks_by_q: qid -> set of chunk_ids that ground that question.
        gold_citations_by_chunk: cid -> set of normalized cites whose quote
            this chunk is grounding (kept for compatibility; eval loop now uses
            chunk_to_cluster_cites for the groundedness free-pass set).
        gold_source_by_q: qid -> "primary" if any chunk was matched via the
            adjacent_quote pass, else "paraphrase_only" if only the paraphrase
            fallback contributed grounding. Qids absent from gold_chunks_by_q
            are not present here.
        has_adjacent_quote_by_q: qid -> True if any gold row for that qid has
            a non-empty adjacent_quote (regardless of whether it matched). False
            means the qid depended on the paraphrase fallback because the source
            answer had no quote at all.
    """
    gold = pd.read_parquet(config.DATASET_DIR / "gold_citations.parquet")
    chunks = pd.read_parquet(config.CORPUS_DIR / "chunks.parquet")

    if "opinion_id" not in gold.columns or "adjacent_quote" not in gold.columns:
        raise RuntimeError(
            "gold_citations.parquet is missing 'opinion_id' / 'adjacent_quote' columns. "
            "Re-run: python -m scripts.legal_rag_homology.run_build_corpus --step dataset"
        )

    chunks_by_opinion: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cid, op_id, text in zip(
        chunks["chunk_id"].values,
        chunks["opinion_id"].values,
        chunks["text"].fillna("").values,
    ):
        chunks_by_opinion[op_id].append((cid, _normalize_for_quote_match(text)))

    gold_chunks_by_q: dict[str, set[str]] = {}
    gold_citations_by_chunk: dict[str, set[str]] = defaultdict(set)
    primary_qids: set[str] = set()
    has_quote_qids: set[str] = set()
    quote_hits = 0
    quote_misses = 0
    no_quote = 0
    no_opinion = 0

    for _, row in gold.iterrows():
        qid = row["source_question_id"]
        op_id = row["opinion_id"]
        quote = row["adjacent_quote"] or ""
        if quote.strip():
            has_quote_qids.add(qid)
        if not op_id:
            no_opinion += 1
            continue
        if not quote.strip():
            no_quote += 1
            continue
        needle = _normalize_for_quote_match(quote)
        if len(needle) < _MIN_FULL_MATCH_LEN:
            no_quote += 1
            continue
        op_chunks = chunks_by_opinion.get(op_id, [])
        if not op_chunks:
            no_opinion += 1
            continue

        matched_any = False
        for cid, normalized_text in op_chunks:
            if _quote_in_chunk(needle, normalized_text):
                gold_chunks_by_q.setdefault(qid, set()).add(cid)
                gold_citations_by_chunk[cid].add(row["normalized_citation"])
                matched_any = True
        if matched_any:
            quote_hits += 1
            primary_qids.add(qid)
        else:
            quote_misses += 1

    logger.info(
        "Gold chunk build (adjacent_quote): %d cites with quote hit, %d miss, %d no-quote, %d no-opinion",
        quote_hits, quote_misses, no_quote, no_opinion,
    )

    paraphrase_path = config.DATASET_DIR / "paraphrase_groundings.parquet"
    if paraphrase_path.exists():
        para = pd.read_parquet(paraphrase_path)
        para_hits = 0
        para_misses = 0
        for _, row in para.iterrows():
            qid = row["source_question_id"]
            op_id = row["opinion_id"]
            passage = row["passage"] or ""
            if not op_id or not passage.strip():
                continue
            needle = _normalize_for_quote_match(passage)
            if len(needle) < _MIN_FULL_MATCH_LEN:
                continue
            already_grounded = (
                qid in gold_chunks_by_q
                and any(
                    cid in gold_chunks_by_q[qid]
                    for cid, _ in chunks_by_opinion.get(op_id, [])
                )
            )
            matched_any = False
            for cid, normalized_text in chunks_by_opinion.get(op_id, []):
                if _quote_in_chunk(needle, normalized_text):
                    gold_chunks_by_q.setdefault(qid, set()).add(cid)
                    gold_citations_by_chunk[cid].add(row["normalized_citation"])
                    matched_any = True
            if matched_any and not already_grounded:
                para_hits += 1
            elif not matched_any:
                para_misses += 1
        logger.info(
            "Gold chunk build (paraphrase fallback): %d cites added grounding, %d returned passage but no chunk match",
            para_hits, para_misses,
        )

    gold_source_by_q = {
        qid: "primary" if qid in primary_qids else "paraphrase_only"
        for qid in gold_chunks_by_q
    }
    has_adjacent_quote_by_q = {
        qid: qid in has_quote_qids for qid in gold_chunks_by_q
    }
    return (
        gold_chunks_by_q,
        gold_citations_by_chunk,
        gold_source_by_q,
        has_adjacent_quote_by_q,
    )


def _build_chunk_to_cluster_cites() -> dict[str, set[str]]:
    """Map each chunk_id to the normalized citations of its parent opinion's cluster.

    Used as the 'retrieved_extra' free-pass set in groundedness scoring: a model that
    cites a parallel reporter for a case we retrieved (e.g. S. Ct. instead of U.S.)
    should not be marked as hallucinating.
    """
    opinions = pd.read_parquet(config.CORPUS_DIR / "opinions.parquet")
    chunks = pd.read_parquet(config.CORPUS_DIR / "chunks.parquet")

    op_cites: dict[str, set[str]] = {}
    for _, row in opinions.iterrows():
        cites = row["all_citations"]
        if cites is None or len(cites) == 0:
            continue
        normalized: set[str] = set()
        for c in cites:
            if not c:
                continue
            normalized.add(c)
            normalized.add(_normalize_citation_spacing(c))
        if normalized:
            op_cites[row["opinion_id"]] = normalized

    chunk_to_cites: dict[str, set[str]] = {}
    for cid, op_id in zip(chunks["chunk_id"].values, chunks["opinion_id"].values):
        cites = op_cites.get(op_id)
        if cites:
            chunk_to_cites[cid] = cites
    return chunk_to_cites


_NEIGHBOR_RADIUS = 1


def _parse_chunk_idx(chunk_id: str) -> tuple[str, int] | None:
    """Split chunk_id of the form '{opinion_id}::{idx}' into (opinion_id, idx).

    Returns None if the suffix isn't an integer — keeps the metric robust to
    any non-conforming ids that might land in retrieved/gold sets.
    """
    op, sep, idx = chunk_id.rpartition("::")
    if not sep or not idx.lstrip("-").isdigit():
        return None
    return op, int(idx)


def _retrieval_metrics(retrieved_chunk_ids: list[str], gold_chunk_ids: set[str]) -> dict:
    """Precision/recall of retrieved chunks against gold chunks.

    Reports both a strict variant (exact chunk_id match) and a neighbor-tolerant
    variant that credits retrieved chunks within +/- _NEIGHBOR_RADIUS chunk
    indices of a gold chunk in the same opinion. Chunks are overlapping windows
    of the same opinion (see corpus/chunking.py), so a +/-1 neighbor often
    contains most of the same passage.
    """
    retrieved = set(retrieved_chunk_ids) if retrieved_chunk_ids is not None else set()
    n_retrieved = len(retrieved)
    n_gold = len(gold_chunk_ids)
    if n_gold == 0:
        return {
            "retrieval_n_retrieved": n_retrieved,
            "retrieval_n_gold": 0,
            "retrieval_tp": 0,
            "retrieval_tp_neighbor": 0,
            "retrieval_precision": None,
            "retrieval_recall": None,
            "retrieval_f1": None,
            "retrieval_precision_at_neighbor_1": None,
            "retrieval_recall_at_neighbor_1": None,
            "retrieval_f1_at_neighbor_1": None,
        }
    tp = len(retrieved & gold_chunk_ids)
    precision = tp / n_retrieved if n_retrieved > 0 else 0.0
    recall = tp / n_gold
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    retr_idx = {p for p in (_parse_chunk_idx(c) for c in retrieved) if p is not None}
    gold_idx = {p for p in (_parse_chunk_idx(c) for c in gold_chunk_ids) if p is not None}
    strict_idx = retr_idx & gold_idx
    tp_neighbor = sum(
        1
        for (op, i) in retr_idx
        if (op, i) not in gold_idx
        and any(
            (op, i + d) in gold_idx
            for d in range(-_NEIGHBOR_RADIUS, _NEIGHBOR_RADIUS + 1)
            if d != 0
        )
    )
    gold_hit_neighbor = sum(
        1
        for g in gold_idx
        if g in retr_idx
        or any(
            (g[0], g[1] + d) in retr_idx
            for d in range(-_NEIGHBOR_RADIUS, _NEIGHBOR_RADIUS + 1)
            if d != 0
        )
    )
    tp_neighbor_total = len(strict_idx) + tp_neighbor
    precision_n = tp_neighbor_total / n_retrieved if n_retrieved > 0 else 0.0
    recall_n = gold_hit_neighbor / n_gold
    f1_n = (
        2 * precision_n * recall_n / (precision_n + recall_n)
        if (precision_n + recall_n) > 0
        else 0.0
    )
    return {
        "retrieval_n_retrieved": n_retrieved,
        "retrieval_n_gold": n_gold,
        "retrieval_tp": tp,
        "retrieval_tp_neighbor": tp_neighbor,
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "retrieval_f1": f1,
        "retrieval_precision_at_neighbor_1": precision_n,
        "retrieval_recall_at_neighbor_1": recall_n,
        "retrieval_f1_at_neighbor_1": f1_n,
    }


def _groundedness_metrics(
    predicted: set,
    expected: set,
    retrieved_extra: set,
    retrieved_secondary: set,
) -> dict:
    """Score predicted citations against expected gold cites.

    A predicted cite is a TRUE POSITIVE if it matches an expected gold cite.
    A predicted cite is FREE-PASSED (not FP, not TP) if it appears in either:
      - retrieved_extra: parallel reporters of the cited opinion's cluster
      - retrieved_secondary: secondary cites mentioned in retrieved chunk text
    Anything else predicted is a FALSE POSITIVE (hallucination).
    """
    tp = len(predicted & expected)
    free_pass = retrieved_extra | retrieved_secondary
    secondary_hits = len((predicted - expected) & retrieved_secondary)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    if fp == 0:
        grounded = "grounded"
    elif tp > 0:
        grounded = "partially_grounded"
    else:
        grounded = "ungrounded"
    return {
        "n_predicted": len(predicted),
        "n_expected": len(expected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "secondary_hits": secondary_hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "groundedness": grounded,
        "predicted_citations": sorted(predicted),
        "expected_citations": sorted(expected),
        "retrieved_citations": sorted(retrieved_extra),
        "secondary_citations": sorted(retrieved_secondary),
        "hallucinated_citations": sorted(predicted - expected - free_pass),
    }


def evaluate_run(run_name: str, sample_size: int | None = None) -> pd.DataFrame:
    run_dir = config.RUNS_DIR / run_name
    results = pd.read_parquet(run_dir / "results.parquet")
    gold = pd.read_parquet(config.DATASET_DIR / "gold_citations.parquet")
    gold_by_q = gold.groupby("source_question_id")["normalized_citation"].apply(set).to_dict()
    if "cite_class" in gold.columns:
        cite_classes_by_q = (
            gold.groupby("source_question_id")["cite_class"]
            .apply(lambda s: sorted({v for v in s if v}))
            .to_dict()
        )
    else:
        logger.warning(
            "gold_citations.parquet has no 'cite_class' column. "
            "Re-run: python -m scripts.legal_rag_homology.run_build_corpus --step dataset"
        )
        cite_classes_by_q = {}
    gold_chunks_by_q, _, gold_source_by_q, has_quote_by_q = _build_gold_chunk_sets()

    chunks_df = pd.read_parquet(config.CORPUS_DIR / "chunks.parquet")
    if "normalized_citation" in chunks_df.columns:
        chunk_to_norm_cite = {
            cid: cite for cid, cite in zip(chunks_df["chunk_id"], chunks_df["normalized_citation"])
            if cite
        }
    else:
        logger.warning(
            "chunks.parquet has no 'normalized_citation' column; retrieved set will be empty. "
            "Re-run: python -m scripts.legal_rag_homology.run_index --step chunks"
        )
        chunk_to_norm_cite = {}

    if "secondary_citations" in chunks_df.columns:
        chunk_to_secondary = {
            cid: set(cites) if cites is not None and len(cites) > 0 else set()
            for cid, cites in zip(chunks_df["chunk_id"], chunks_df["secondary_citations"])
        }
    else:
        logger.warning(
            "chunks.parquet has no 'secondary_citations' column; secondary set will be empty. "
            "Re-run: python -m scripts.legal_rag_homology.run_index --step chunks"
        )
        chunk_to_secondary = {}

    questions = pd.read_parquet(config.DATASET_DIR / "questions.parquet")
    if "response_text" not in questions.columns:
        raise RuntimeError(
            "questions.parquet is missing 'response_text' (the human-accepted reference). "
            "Re-run: python -m scripts.legal_rag_homology.run_build_corpus --step questions"
        )
    reference_by_q = dict(zip(questions["question_id"], questions["response_text"]))

    if sample_size and sample_size < len(results):
        results = results.sample(n=sample_size, random_state=config.DEFAULT_SEED)
        logger.info("Sampled %d questions for evaluation", sample_size)

    bedrock = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)

    checkpoint = run_dir / "evaluation_checkpoint.parquet"
    done_ids = set()
    rows = []
    if checkpoint.exists():
        cached = pd.read_parquet(checkpoint)
        rows = cached.to_dict("records")
        done_ids = {r["question_id"] for r in rows}
        logger.info("Resuming: %d questions already evaluated", len(done_ids))

    for _, r in tqdm(results.iterrows(), total=len(results), desc="evaluating"):
        qid = r["question_id"]
        if qid in done_ids:
            continue

        retrieved_citations = set()
        retrieved_secondary = set()
        for cid in r["retrieved_chunk_ids"]:
            primary = chunk_to_norm_cite.get(cid, "")
            if primary:
                retrieved_citations.add(primary)
            retrieved_secondary.update(chunk_to_secondary.get(cid, set()))

        cited = r["cited_normalized"]
        predicted = set(cited) if cited is not None and len(cited) > 0 else set()
        expected = gold_by_q.get(qid, set())
        ground = _groundedness_metrics(predicted, expected, retrieved_citations, retrieved_secondary)

        retrieved_chunks = r["retrieved_chunk_ids"]
        if isinstance(retrieved_chunks, np.ndarray):
            retrieved_chunks = retrieved_chunks.tolist()
        retrieval = _retrieval_metrics(retrieved_chunks, gold_chunks_by_q.get(qid, set()))

        reference = reference_by_q.get(qid, "")
        if not reference:
            logger.warning("No reference answer for %s; judge will see empty REFERENCE", qid)
        judge = _judge_correctness(bedrock, r["question_text"], r["response_text"], reference)
        correctness = judge.get("correctness", "error")
        groundedness = ground["groundedness"]

        misgrounded = groundedness == "ungrounded"
        incorrect = correctness in ("incorrect", "error")
        partially_correct = correctness == "partially_correct"
        partially_grounded = groundedness == "partially_grounded"

        if incorrect or misgrounded:
            verdict = "hallucinated"
        elif partially_correct or partially_grounded:
            verdict = "incomplete"
        else:
            verdict = "accurate"

        rows.append(
            {
                "question_id": qid,
                "category": r["category"],
                "correctness": correctness,
                "judge_reasoning": judge.get("reasoning", ""),
                **ground,
                **retrieval,
                "gold_source": gold_source_by_q.get(qid, "none"),
                "has_adjacent_quote": bool(has_quote_by_q.get(qid, False)),
                "cite_classes": cite_classes_by_q.get(qid, []),
                "verdict": verdict,
            }
        )

        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_parquet(checkpoint, index=False)

    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "evaluation.parquet", index=False)
    if checkpoint.exists():
        checkpoint.unlink()

    _print_summary(run_name, df)
    _save_metrics(run_dir, run_name, df)
    logger.info("Saved evaluation to %s", run_dir / "evaluation.parquet")
    return df


def _save_metrics(run_dir, run_name: str, df: pd.DataFrame) -> None:
    n = len(df)
    verdict_counts = {v: int((df["verdict"] == v).sum()) for v in ["accurate", "incomplete", "hallucinated"]}
    correctness_counts = df["correctness"].value_counts().to_dict()
    groundedness_counts = df["groundedness"].value_counts().to_dict()

    agg_dict = {
        "n": ("verdict", "count"),
        "accurate": ("verdict", lambda s: (s == "accurate").mean()),
        "incomplete": ("verdict", lambda s: (s == "incomplete").mean()),
        "hallucinated": ("verdict", lambda s: (s == "hallucinated").mean()),
        "precision": ("precision", "mean"),
        "recall": ("recall", "mean"),
        "f1": ("f1", "mean"),
    }
    if "retrieval_precision" in df.columns:
        agg_dict["retrieval_precision"] = ("retrieval_precision", "mean")
        agg_dict["retrieval_recall"] = ("retrieval_recall", "mean")
        agg_dict["retrieval_f1"] = ("retrieval_f1", "mean")

    by_cat = df.groupby("category").agg(**agg_dict).reset_index()

    ret_df = df.dropna(subset=["retrieval_precision"])
    retrieval_chunk_metrics = {}
    if len(ret_df) > 0:
        retrieval_chunk_metrics = {
            "macro_precision": float(ret_df["retrieval_precision"].mean()),
            "macro_recall": float(ret_df["retrieval_recall"].mean()),
            "macro_f1": float(ret_df["retrieval_f1"].mean()),
            "macro_precision_at_neighbor_1": float(
                ret_df["retrieval_precision_at_neighbor_1"].mean()
            ),
            "macro_recall_at_neighbor_1": float(
                ret_df["retrieval_recall_at_neighbor_1"].mean()
            ),
            "macro_f1_at_neighbor_1": float(ret_df["retrieval_f1_at_neighbor_1"].mean()),
            "mean_n_retrieved": float(ret_df["retrieval_n_retrieved"].mean()),
            "mean_n_gold": float(ret_df["retrieval_n_gold"].mean()),
            "mean_tp": float(ret_df["retrieval_tp"].mean()),
            "mean_tp_neighbor": float(ret_df["retrieval_tp_neighbor"].mean()),
            "n_questions_with_gold": len(ret_df),
        }

    metrics = {
        "run_name": run_name,
        "n_questions": n,
        "verdict": verdict_counts,
        "verdict_rates": {k: v / n for k, v in verdict_counts.items()},
        "correctness": {k: int(v) for k, v in correctness_counts.items()},
        "groundedness": {k: int(v) for k, v in groundedness_counts.items()},
        "citation_metrics": {
            "macro_precision": float(df["precision"].mean()),
            "macro_recall": float(df["recall"].mean()),
            "macro_f1": float(df["f1"].mean()),
            "mean_secondary_hits": (
                float(df["secondary_hits"].mean()) if "secondary_hits" in df.columns else 0.0
            ),
            "total_secondary_hits": (
                int(df["secondary_hits"].sum()) if "secondary_hits" in df.columns else 0
            ),
        },
        "retrieval_chunk_metrics": retrieval_chunk_metrics,
        "by_category": by_cat.to_dict(orient="records"),
    }

    path = run_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved aggregate metrics to %s", path)


def _print_summary(run_name: str, df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n--- Evaluation: {run_name} ({n} questions) ---")

    print("\nVerdict:")
    for val in ["accurate", "incomplete", "hallucinated"]:
        count = (df["verdict"] == val).sum()
        print(f"  {val}: {count} ({count / n:.1%})")

    print("\nCorrectness (LLM judge):")
    for val, count in df["correctness"].value_counts().items():
        print(f"  {val}: {count} ({count / n:.1%})")

    print("\nGroundedness (citation matching):")
    for val, count in df["groundedness"].value_counts().items():
        print(f"  {val}: {count} ({count / n:.1%})")
    print(f"  Macro precision: {df['precision'].mean():.3f}")
    print(f"  Macro recall:    {df['recall'].mean():.3f}")
    print(f"  Macro F1:        {df['f1'].mean():.3f}")
    if "secondary_hits" in df.columns:
        print(f"  Secondary hits:  {int(df['secondary_hits'].sum())} total "
              f"({df['secondary_hits'].mean():.2f} per question)")

    ret_df = df.dropna(subset=["retrieval_precision"])
    if len(ret_df) > 0:
        print(f"\nRetrieval (chunk-level, {len(ret_df)} questions with gold chunks):")
        print(f"  Mean chunks retrieved: {ret_df['retrieval_n_retrieved'].mean():.1f}")
        print(f"  Mean gold chunks:      {ret_df['retrieval_n_gold'].mean():.1f}")
        print(f"  Mean strict TP:        {ret_df['retrieval_tp'].mean():.2f}")
        print(f"  Mean +/-1 neighbor TP: {ret_df['retrieval_tp_neighbor'].mean():.2f}")
        print(f"  Macro precision:       {ret_df['retrieval_precision'].mean():.3f}")
        print(f"  Macro recall:          {ret_df['retrieval_recall'].mean():.3f}")
        print(f"  Macro F1:              {ret_df['retrieval_f1'].mean():.3f}")
        print(
            f"  Macro precision (+/-1):{ret_df['retrieval_precision_at_neighbor_1'].mean():.3f}"
        )
        print(
            f"  Macro recall (+/-1):   {ret_df['retrieval_recall_at_neighbor_1'].mean():.3f}"
        )
        print(
            f"  Macro F1 (+/-1):       {ret_df['retrieval_f1_at_neighbor_1'].mean():.3f}"
        )

    by_cat = (
        df.groupby("category")
        .agg(
            accurate=("verdict", lambda s: (s == "accurate").mean()),
            incomplete=("verdict", lambda s: (s == "incomplete").mean()),
            hallucinated=("verdict", lambda s: (s == "hallucinated").mean()),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
        )
        .reset_index()
    )
    print(f"\nBy category:")
    print(by_cat.to_string(index=False))


def patch_retrieval_metrics(run_name: str) -> None:
    """Add chunk-level retrieval metrics to an existing run without re-running the LLM judge."""
    run_dir = config.RUNS_DIR / run_name
    results = pd.read_parquet(run_dir / "results.parquet")
    gold_chunks_by_q, _, gold_source_by_q, has_quote_by_q = _build_gold_chunk_sets()

    gold = pd.read_parquet(config.DATASET_DIR / "gold_citations.parquet")
    if "cite_class" in gold.columns:
        cite_classes_by_q = (
            gold.groupby("source_question_id")["cite_class"]
            .apply(lambda s: sorted({v for v in s if v}))
            .to_dict()
        )
    else:
        cite_classes_by_q = {}

    eval_path = run_dir / "evaluation.parquet"
    if eval_path.exists():
        eval_df = pd.read_parquet(eval_path)
    else:
        eval_df = None

    retrieval_rows = []
    for _, r in results.iterrows():
        qid = r["question_id"]
        retrieved_chunks = r["retrieved_chunk_ids"]
        if isinstance(retrieved_chunks, np.ndarray):
            retrieved_chunks = retrieved_chunks.tolist()
        retrieval = _retrieval_metrics(retrieved_chunks, gold_chunks_by_q.get(qid, set()))
        retrieval["question_id"] = qid
        retrieval["gold_source"] = gold_source_by_q.get(qid, "none")
        retrieval["has_adjacent_quote"] = bool(has_quote_by_q.get(qid, False))
        retrieval["cite_classes"] = cite_classes_by_q.get(qid, [])
        retrieval_rows.append(retrieval)

    ret_df = pd.DataFrame(retrieval_rows)

    if eval_df is not None:
        for col in ret_df.columns:
            if col != "question_id" and col in eval_df.columns:
                eval_df = eval_df.drop(columns=[col])
        eval_df = eval_df.merge(ret_df, on="question_id", how="left")
        eval_df.to_parquet(eval_path, index=False)
        logger.info("Patched evaluation.parquet with retrieval metrics")
        _print_summary(run_name, eval_df)
    else:
        logger.info("No evaluation.parquet found, saving retrieval-only metrics")
        ret_df.to_parquet(run_dir / "retrieval_metrics.parquet", index=False)

    # Update metrics.json
    scored = ret_df.dropna(subset=["retrieval_precision"])
    retrieval_chunk_metrics = {}
    if len(scored) > 0:
        retrieval_chunk_metrics = {
            "macro_precision": float(scored["retrieval_precision"].mean()),
            "macro_recall": float(scored["retrieval_recall"].mean()),
            "macro_f1": float(scored["retrieval_f1"].mean()),
            "macro_precision_at_neighbor_1": float(
                scored["retrieval_precision_at_neighbor_1"].mean()
            ),
            "macro_recall_at_neighbor_1": float(
                scored["retrieval_recall_at_neighbor_1"].mean()
            ),
            "macro_f1_at_neighbor_1": float(scored["retrieval_f1_at_neighbor_1"].mean()),
            "mean_n_retrieved": float(scored["retrieval_n_retrieved"].mean()),
            "mean_n_gold": float(scored["retrieval_n_gold"].mean()),
            "mean_tp": float(scored["retrieval_tp"].mean()),
            "mean_tp_neighbor": float(scored["retrieval_tp_neighbor"].mean()),
            "n_questions_with_gold": len(scored),
        }

    # By-category retrieval metrics
    if eval_df is not None and "category" in eval_df.columns:
        cat_scored = eval_df.dropna(subset=["retrieval_precision"])
        if len(cat_scored) > 0:
            by_cat_ret = (
                cat_scored.groupby("category")
                .agg(
                    retrieval_precision=("retrieval_precision", "mean"),
                    retrieval_recall=("retrieval_recall", "mean"),
                    retrieval_f1=("retrieval_f1", "mean"),
                )
                .reset_index()
            )

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["retrieval_chunk_metrics"] = retrieval_chunk_metrics
        if eval_df is not None and "category" in eval_df.columns and len(cat_scored) > 0:
            for entry in metrics.get("by_category", []):
                cat_row = by_cat_ret[by_cat_ret["category"] == entry["category"]]
                if len(cat_row) > 0:
                    entry["retrieval_precision"] = float(cat_row["retrieval_precision"].iloc[0])
                    entry["retrieval_recall"] = float(cat_row["retrieval_recall"].iloc[0])
                    entry["retrieval_f1"] = float(cat_row["retrieval_f1"].iloc[0])
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Updated metrics.json with retrieval_chunk_metrics")
    else:
        with open(metrics_path, "w") as f:
            json.dump({"run_name": run_name, "retrieval_chunk_metrics": retrieval_chunk_metrics}, f, indent=2)

    print(f"\nRetrieval chunk metrics for '{run_name}':")
    for k, v in retrieval_chunk_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


def _discover_runs() -> list[str]:
    """Return sorted run names that have a results.parquet under RUNS_DIR."""
    if not config.RUNS_DIR.exists():
        return []
    return sorted(
        d.name for d in config.RUNS_DIR.iterdir()
        if d.is_dir() and (d / "results.parquet").exists()
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="Run name. Required unless --all is set.")
    p.add_argument(
        "--all",
        action="store_true",
        help="Apply to every run under RUNS_DIR. Only valid with --patch-retrieval.",
    )
    p.add_argument("--sample", type=int, default=None, help="Number of questions to sample (default: all)")
    p.add_argument(
        "--patch-retrieval",
        action="store_true",
        help="Only add retrieval chunk metrics to existing evaluation (no LLM judge)",
    )
    args = p.parse_args()

    if args.all and not args.patch_retrieval:
        p.error("--all is only supported with --patch-retrieval")
    if not args.all and not args.run:
        p.error("--run is required unless --all is set")

    if args.patch_retrieval:
        run_names = _discover_runs() if args.all else [args.run]
        if not run_names:
            print(f"No runs found under {config.RUNS_DIR}")
            return
        for i, name in enumerate(run_names, 1):
            if len(run_names) > 1:
                logger.info("[%d/%d] Patching retrieval metrics for %s", i, len(run_names), name)
            try:
                patch_retrieval_metrics(name)
            except Exception:
                logger.exception("Failed to patch %s; continuing", name)
    else:
        evaluate_run(args.run, sample_size=args.sample)


if __name__ == "__main__":
    main()
