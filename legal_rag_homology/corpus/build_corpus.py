from __future__ import annotations

import logging

import pandas as pd
from eyecite import get_citations
from eyecite.models import CaseCitation
from tqdm import tqdm

from .. import config, dataset
from ..evaluation.citation_parser import extract_citations
from . import courtlistener

logger = logging.getLogger(__name__)

_CITE_CLASSES = ("federal", "state", "lexis", "specialty", "unknown")


def _classify_cite(normalized_citation: str) -> str:
    """Classify a normalized citation by what kind of source it points to.

    Buckets:
      - "lexis"   : vendor LEXIS / Westlaw IDs (e.g. "2023 U.S. Dist. LEXIS 159954").
                    These are not lookupable in CourtListener or CAP because they
                    use proprietary indexes.
      - "federal" : federal court reporters (F. Supp., F.3d, S. Ct., F.R.D., B.R.,
                    F. App'x, etc.). Should resolve in CourtListener; failures
                    here typically mean slip opinions or mis-OCRed cite strings.
      - "state"   : state-court reporters (Nev., Cal. App. 5th, So. 3d, etc.).
                    The CourtListener gap that CAP would best fill.
      - "specialty": tax courts, military, vendor-specific specialty reporters.
      - "unknown" : eyecite couldn't recognize it as a CaseCitation at all
                    (e.g. "1999 T.C. Memo 387").
    """
    if "LEXIS" in normalized_citation or "WL" in normalized_citation.split():
        return "lexis"
    cits = list(get_citations(normalized_citation))
    case_cits = [c for c in cits if isinstance(c, CaseCitation)]
    if not case_cits:
        return "unknown"
    c = case_cits[0]
    edition = c.all_editions[0] if getattr(c, "all_editions", None) else None
    cite_type = ""
    if edition is not None:
        reporter = getattr(edition, "reporter", None) or edition
        cite_type = str(getattr(reporter, "cite_type", "") or "").lower()
    if cite_type == "federal":
        return "federal"
    if cite_type.startswith("state"):
        return "state"
    if "specialty" in cite_type:
        return "specialty"
    return "unknown"

_GOLD_CHECKPOINT = config.CORPUS_DIR / "gold_opinions_checkpoint.parquet"
_CITE_LOOKUP_CHECKPOINT = config.DATASET_DIR / "gold_citation_lookups.parquet"
_DISTRACTOR_CHECKPOINT = config.CORPUS_DIR / "distractor_checkpoint.parquet"
_DISTRACTOR_DONE_FILE = config.CORPUS_DIR / "distractor_courts_done.txt"


def build_questions_artifact() -> pd.DataFrame:
    questions = dataset.extract_questions()
    questions.to_parquet(config.DATASET_DIR / "questions.parquet", index=False)
    return questions


def build_gold_citations_artifact() -> pd.DataFrame:
    """Build gold_citations with the released-answer quote adjacent to each cite.

    For every (question, citation) we keep the nearest quoted passage from the
    released response. At evaluation time, that passage is what defines a chunk
    as gold-relevant: a chunk of the cited opinion grounds the claim only if it
    contains the quoted text. The opinion_id is resolved here too (single
    CourtListener round-trip per cite) so the eval-time join is a pure parquet
    operation.
    """
    responses = dataset.extract_released_responses()
    rows = []
    for _, row in responses.iterrows():
        if row["human_label"] != "Accurate":
            continue
        text = row["response_text"] or ""
        for cite in extract_citations(text):
            rows.append({
                "citation_string": cite.raw,
                "normalized_citation": cite.normalized,
                "source_question_id": row["question_id"],
                "adjacent_quote": cite.adjacent_quote or "",
                "response_text": row["response_text"]
            })

    df = pd.DataFrame(rows)
    df["_quote_len"] = df["adjacent_quote"].str.len()
    df = (
        df.sort_values("_quote_len", ascending=False)
        .drop_duplicates(subset="normalized_citation")
        .drop(columns="_quote_len")
        .sort_values("source_question_id")
        .reset_index(drop=True)
    )

    cached: dict[str, str] = {}
    if _CITE_LOOKUP_CHECKPOINT.exists():
        prior = pd.read_parquet(_CITE_LOOKUP_CHECKPOINT)
        cached = dict(zip(prior["normalized_citation"], prior["opinion_id"]))
        logger.info("Loaded %d cached citation lookups from %s",
                    len(cached), _CITE_LOOKUP_CHECKPOINT)

    opinion_ids: list[str] = []
    new_lookups = 0
    for cite in tqdm(df["normalized_citation"], desc="resolve gold opinion_ids"):
        if cite in cached:
            opinion_ids.append(cached[cite])
            continue
        op_id = courtlistener.lookup_citation(cite)
        resolved = str(op_id) if op_id is not None else ""
        cached[cite] = resolved
        opinion_ids.append(resolved)
        new_lookups += 1
    df["opinion_id"] = opinion_ids
    df["cite_class"] = [_classify_cite(c) for c in df["normalized_citation"]]

    if new_lookups > 0:
        pd.DataFrame(
            {"normalized_citation": list(cached.keys()),
             "opinion_id": list(cached.values())}
        ).to_parquet(_CITE_LOOKUP_CHECKPOINT, index=False)
        logger.info("Saved %d citation lookups to %s (%d new)",
                    len(cached), _CITE_LOOKUP_CHECKPOINT, new_lookups)

    df.to_parquet(config.DATASET_DIR / "gold_citations.parquet", index=False)
    return df


def apply_llm_quote_validation() -> pd.DataFrame:
    """Apply cached LLM validation results to gold_citations.parquet.

    For each (qid, cite) row in llm_validated_quotes.parquet whose
    candidate_quote matches the current heuristic adjacent_quote:
      - action="keep"   : leave as-is.
      - action="drop"   : clear adjacent_quote.
      - action="replace": substitute the corrected verbatim passage.

    No-op if the validation artifact doesn't exist. Idempotent — running
    twice yields the same result because the action key is recomputed
    from the candidate_quote each time.
    """
    from . import quote_validation

    validations = quote_validation.load_validation()
    if not validations:
        logger.info("No llm_validated_quotes.parquet found; skipping validation apply")
        return pd.read_parquet(config.DATASET_DIR / "gold_citations.parquet")

    df = pd.read_parquet(config.DATASET_DIR / "gold_citations.parquet")
    if "_llm_validated_action" in df.columns:
        df = df.drop(columns=["_llm_validated_action"])
    counts = {"keep": 0, "drop": 0, "replace": 0, "stale": 0, "missing": 0}
    actions: list[str] = []
    new_quotes: list[str] = []
    for _, row in df.iterrows():
        key = (row["source_question_id"], row["normalized_citation"])
        v = validations.get(key)
        if v is None:
            counts["missing"] += 1
            actions.append("")
            new_quotes.append(row["adjacent_quote"])
            continue
        if v["candidate_quote"] != row["adjacent_quote"]:
            counts["stale"] += 1
            actions.append("")
            new_quotes.append(row["adjacent_quote"])
            continue
        action = v["action"]
        counts[action] = counts.get(action, 0) + 1
        if action == "keep":
            actions.append("keep")
            new_quotes.append(row["adjacent_quote"])
        elif action == "drop":
            actions.append("drop")
            new_quotes.append("")
        elif action == "replace":
            actions.append("replace")
            new_quotes.append(v["passage"])
        else:
            actions.append("")
            new_quotes.append(row["adjacent_quote"])
    df["adjacent_quote"] = new_quotes
    df["_llm_validated_action"] = actions
    df.to_parquet(config.DATASET_DIR / "gold_citations.parquet", index=False)
    logger.info(
        "Applied LLM validation: keep=%d drop=%d replace=%d stale=%d missing=%d",
        counts.get("keep", 0), counts.get("drop", 0), counts.get("replace", 0),
        counts["stale"], counts["missing"],
    )
    return df


def _fetch_gold_opinions(gold_df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    if _GOLD_CHECKPOINT.exists():
        logger.info("Loading gold opinions from checkpoint: %s", _GOLD_CHECKPOINT)
        cached = pd.read_parquet(_GOLD_CHECKPOINT)
        return cached.to_dict("records"), []

    gold_opinions = []
    missing = []
    lookup_failures = 0
    fetch_failures = 0
    for _, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="gold"):
        cite = row["normalized_citation"]
        op_id = courtlistener.lookup_citation(cite)
        if op_id is None:
            lookup_failures += 1
            missing.append(cite)
            continue
        op = courtlistener.fetch_opinion(op_id)
        if op is None:
            fetch_failures += 1
            missing.append(cite)
            continue
        op["source"] = "gold"
        gold_opinions.append(op)
    logger.info("Gold results: %d found, %d lookup failures, %d fetch failures",
                len(gold_opinions), lookup_failures, fetch_failures)

    if gold_opinions:
        pd.DataFrame(gold_opinions).to_parquet(_GOLD_CHECKPOINT, index=False)
        logger.info("Saved gold checkpoint (%d opinions) to %s", len(gold_opinions), _GOLD_CHECKPOINT)

    return gold_opinions, missing


def build_corpus(distractor_count: int, seed: int = config.DEFAULT_SEED) -> pd.DataFrame:
    """Build the retrieval corpus: gold-cited opinions + federal distractors.

    seed is currently unused (CourtListener returns opinions in its own order);
    it is reserved for future sampling-based corpus variants.
    """
    gold_df = pd.read_parquet(config.DATASET_DIR / "gold_citations.parquet")
    gold_opinions, missing = _fetch_gold_opinions(gold_df)

    if missing:
        pd.DataFrame({"citation": missing}).to_csv(
            config.CORPUS_DIR / "missing_citations.csv", index=False)

    gold_ids = {o["opinion_id"] for o in gold_opinions}

    per_court_counts = _stratified_counts(config.FEDERAL_COURTS, distractor_count)
    distractors, courts_done = _load_distractor_checkpoint()
    if distractors:
        logger.info("Resuming distractors: %d cached opinions across %d courts",
                    len(distractors), len(courts_done))

    for court, n in tqdm(per_court_counts.items(), desc="distractor courts"):
        if court in courts_done:
            continue
        court_count = 0
        for op in courtlistener.fetch_bulk_court_opinions(court, max_records=n * 2):
            if op["opinion_id"] in gold_ids:
                continue
            op["source"] = "distractor"
            distractors.append(op)
            court_count += 1
            if court_count >= n:
                break
        courts_done.add(court)
        _save_distractor_checkpoint(distractors, courts_done)

    all_ops = gold_opinions + distractors
    df = pd.DataFrame(all_ops)
    df.to_parquet(config.CORPUS_DIR / "opinions.parquet", index=False)
    _clear_distractor_checkpoint()
    return df


def _load_distractor_checkpoint() -> tuple[list[dict], set[str]]:
    if not _DISTRACTOR_CHECKPOINT.exists() or not _DISTRACTOR_DONE_FILE.exists():
        return [], set()
    try:
        cached = pd.read_parquet(_DISTRACTOR_CHECKPOINT).to_dict("records")
        done = {c.strip() for c in _DISTRACTOR_DONE_FILE.read_text().splitlines() if c.strip()}
        return cached, done
    except (OSError, ValueError) as e:
        logger.warning("Distractor checkpoint unreadable, starting fresh: %s", e)
        return [], set()


def _save_distractor_checkpoint(distractors: list[dict], courts_done: set[str]) -> None:
    pd.DataFrame(distractors).to_parquet(_DISTRACTOR_CHECKPOINT, index=False)
    _DISTRACTOR_DONE_FILE.write_text("\n".join(sorted(courts_done)))


def _clear_distractor_checkpoint() -> None:
    if _DISTRACTOR_CHECKPOINT.exists():
        _DISTRACTOR_CHECKPOINT.unlink()
    if _DISTRACTOR_DONE_FILE.exists():
        _DISTRACTOR_DONE_FILE.unlink()


def _stratified_counts(courts: list[str], total: int) -> dict[str, int]:
    weights = {"scotus": 3.0}
    for c in courts:
        weights.setdefault(c, 1.0)
    s = sum(weights[c] for c in courts)
    out = {c: max(1, int(round(total * weights[c] / s))) for c in courts}
    diff = total - sum(out.values())
    if diff != 0:
        out[courts[0]] = max(1, out[courts[0]] + diff)
    return out
