"""Validate heuristic adjacent_quote against the cited opinion using Sonnet.

For every (qid, citation) where build_corpus extracted a non-empty
adjacent_quote via the positional-nearest-quote regex, ask Sonnet whether
the quote is verbatim from the cited opinion *and* supports the surrounding
response paragraph. Sonnet returns one of:

    {"action": "keep",    "passage": ""}
    {"action": "drop",    "passage": ""}
    {"action": "replace", "passage": "<verbatim substring from opinion>"}

`keep` leaves gold_citations.parquet unchanged. `drop` clears the quote so
the cite falls through to the broader paraphrase pass. `replace` substitutes
a corrected verbatim passage that the eval-time chunk substring match can
use directly.

Conservative on `keep`: when in doubt, prefer `drop` — the paraphrase pass
in paraphrase_grounding.py is the safety net.

Cached via parquet: re-runs only call Sonnet for new (qid, cite) pairs. The
heuristic quote is part of the cache key so a rebuild of gold_citations
(which can change the heuristic quote) invalidates the cached row.
"""
from __future__ import annotations

import json
import logging

import boto3
import pandas as pd
from tqdm import tqdm

from .. import config, dataset
from .paraphrase_grounding import _build_paragraph_index

logger = logging.getLogger(__name__)

_VALIDATION_PATH = config.DATASET_DIR / "llm_validated_quotes.parquet"

_MAX_OPINION_CHARS = 80_000
_MAX_PARAGRAPH_CHARS = 4_000

_SYSTEM_PROMPT = """You verify whether a heuristic-extracted candidate quote
is actually a verbatim passage from a cited legal opinion that supports a
specific claim in a research answer.

You will receive:
- A paragraph from a research answer that cites a particular case
- The exact citation that was used
- A CANDIDATE quote our heuristic extracted from the answer paragraph
- The full text of the cited opinion

Decide one of three actions:

- "keep": the candidate quote appears verbatim in the cited opinion AND
  supports the claim made in the paragraph. Return {"action":"keep","passage":""}.

- "replace": the candidate is wrong (e.g. it is a case name, a doctrinal
  phrase the answer is paraphrasing, or a quote from a different case),
  BUT the cited opinion DOES contain a verbatim passage that supports the
  paragraph's claim. Return that verbatim passage in `passage`. Copy it
  exactly — same whitespace and punctuation as the opinion text.
  10-400 characters, substantive (not boilerplate, not the case header).

- "drop": the candidate is wrong AND no verbatim passage in the opinion
  clearly supports the claim. Return {"action":"drop","passage":""}.
  An empty answer is correct when the opinion does not address the
  specific claim head-on.

Critical rules:
1. Be conservative. Prefer "drop" over "keep" when uncertain. Prefer
   "drop" over "replace" when the supporting passage isn't clearly there.
2. Any "passage" value MUST be a verbatim substring of the OPINION TEXT —
   no paraphrase, no edits.
3. Do NOT keep candidates that are case names ("X v. Y", "In re X") or
   that are short doctrinal phrases the answer is using rhetorically
   rather than quoting from the opinion.
4. Output JSON only. Format: {"action":"keep|drop|replace","passage":"..."}"""


def _build_prompt(
    paragraph: str, citation: str, candidate: str, opinion_text: str
) -> list[dict]:
    """Bedrock messages payload with prompt-caching on opinion_text.

    Opinion text is the largest reusable block; many cites share an opinion,
    so caching it lets back-to-back calls reuse the prefix.
    """
    opinion_block = opinion_text[:_MAX_OPINION_CHARS]
    paragraph_block = paragraph[-_MAX_PARAGRAPH_CHARS:]
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"OPINION CITED ({citation}):\n\n{opinion_block}",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": (
                        "PARAGRAPH FROM RESEARCH ANSWER (which cites the "
                        f"opinion above):\n\n{paragraph_block}\n\n"
                        f"CANDIDATE QUOTE (heuristic extraction):\n{candidate!r}\n\n"
                        'Return JSON: {"action":"keep|drop|replace","passage":"..."}'
                    ),
                },
            ],
        }
    ]


def _validate_one(
    client, paragraph: str, citation: str, candidate: str, opinion_text: str
) -> tuple[str, str]:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 600,
        "system": _SYSTEM_PROMPT,
        "messages": _build_prompt(paragraph, citation, candidate, opinion_text),
    }
    resp = client.invoke_model(
        modelId=config.GENERATOR_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
    )
    parsed = json.loads(resp["body"].read())
    text = parsed["content"][0]["text"]

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return ("drop", "")
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            return ("drop", "")

    action = (obj.get("action") or "").strip().lower()
    passage = (obj.get("passage") or "").strip()
    if action not in ("keep", "drop", "replace"):
        return ("drop", "")
    if action == "replace" and not passage:
        return ("drop", "")
    if action != "replace":
        passage = ""
    return (action, passage)


def _flush(rows: dict[tuple[str, str], dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        df = pd.DataFrame(columns=[
            "source_question_id", "normalized_citation",
            "candidate_quote", "action", "passage",
        ])
    df.to_parquet(_VALIDATION_PATH, index=False)
    return df


def build_llm_validation_artifact() -> pd.DataFrame:
    """Run Sonnet validation over every gold cite that has a non-empty
    heuristic adjacent_quote and a resolved opinion_id. Cached resumable.
    """
    gold = pd.read_parquet(config.DATASET_DIR / "gold_citations.parquet")
    opinions = pd.read_parquet(config.CORPUS_DIR / "opinions.parquet")
    opinion_text_by_id = dict(zip(opinions["opinion_id"], opinions["text"]))
    paragraph_index = _build_paragraph_index(dataset.extract_released_responses())

    cached: dict[tuple[str, str], dict] = {}
    if _VALIDATION_PATH.exists():
        prior = pd.read_parquet(_VALIDATION_PATH)
        for _, r in prior.iterrows():
            cached[(r["source_question_id"], r["normalized_citation"])] = r.to_dict()
        logger.info("Loaded %d cached validation rows", len(cached))

    todo: list[tuple[str, str, str, str, str]] = []
    for _, row in gold.iterrows():
        qid = row["source_question_id"]
        cite = row["normalized_citation"]
        op_id = row["opinion_id"]
        candidate = row["adjacent_quote"] or ""
        if not candidate or not op_id:
            continue
        opinion_text = opinion_text_by_id.get(op_id, "")
        if not opinion_text or not opinion_text.strip():
            continue
        paragraph = paragraph_index.get((qid, cite), "")
        if not paragraph:
            continue
        prior = cached.get((qid, cite))
        if prior and prior.get("candidate_quote") == candidate:
            continue
        todo.append((qid, cite, op_id, candidate, paragraph))

    logger.info(
        "LLM quote validation: %d cached, %d to query",
        len(cached), len(todo),
    )
    if not todo:
        return _flush(cached)

    bedrock = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)
    todo.sort(key=lambda t: t[2])  # group by opinion_id for cache hits

    save_every = 25
    new_rows = 0
    for qid, cite, op_id, candidate, paragraph in tqdm(todo, desc="validate quotes"):
        opinion_text = opinion_text_by_id.get(op_id, "")
        try:
            action, passage = _validate_one(
                bedrock, paragraph, cite, candidate, opinion_text
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Validation parse error for %s/%s: %s", qid, cite, e)
            action, passage = ("drop", "")
        cached[(qid, cite)] = {
            "source_question_id": qid,
            "normalized_citation": cite,
            "candidate_quote": candidate,
            "action": action,
            "passage": passage,
        }
        new_rows += 1
        if new_rows % save_every == 0:
            _flush(cached)

    return _flush(cached)


def load_validation() -> dict[tuple[str, str], dict]:
    """Return cached validation results keyed by (qid, normalized_citation).

    Empty dict if the artifact doesn't exist — callers should treat that as
    "no validation yet, fall back to heuristic quote".
    """
    if not _VALIDATION_PATH.exists():
        return {}
    df = pd.read_parquet(_VALIDATION_PATH)
    return {
        (r["source_question_id"], r["normalized_citation"]): r.to_dict()
        for _, r in df.iterrows()
    }
