"""Use Claude (Sonnet) to extract verbatim grounding passages for paraphrased citations.

For each gold (qid, citation) pair where:
  - opinion_id is resolved
  - the released-answer adjacent_quote is missing OR didn't substring-match the
    cited opinion's chunks

we ask the model to return a verbatim passage from the cited opinion's text
that supports the released-answer's paragraph, or empty if nothing in the
opinion clearly grounds the claim. The extracted passage is then substring-
matched against chunks at evaluation time.

Conservative by design: empty answer is the right answer when in doubt.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import boto3
import pandas as pd
from tqdm import tqdm

from .. import config, dataset
from ..evaluation.citation_parser import extract_citations

logger = logging.getLogger(__name__)

_GROUNDINGS_PATH = config.DATASET_DIR / "paraphrase_groundings.parquet"

_MAX_OPINION_CHARS = 80_000
_MAX_PARAGRAPH_CHARS = 4_000

_QUOTE_NORMALIZE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return _QUOTE_NORMALIZE_RE.sub(" ", text).strip()


_SYSTEM_PROMPT = """You identify the verbatim passage from a legal opinion that
grounds a specific claim made in a legal research answer.

You will receive:
- A paragraph from a legal research answer that cites a particular case
- The full text of the cited opinion
- The exact citation that was used

Return JSON with one field, "passage":
- A verbatim substring (10-400 characters) copied EXACTLY from the cited
  opinion's text that supports the paragraph's claim about that case.
- Empty string "" if the opinion does not clearly support the specific claim.

Critical rules:
1. The passage MUST be copy-pasted verbatim from the opinion text — no
   paraphrase, no edits, no normalization. Exact whitespace and punctuation.
2. Be conservative: if you cannot find a passage that clearly supports the
   *specific* claim made in the paragraph, return "". An empty answer is
   correct and expected for claims that the opinion does not address head-on.
3. Do not return generic boilerplate (case-name reference, citation, court
   header). The passage must contain substantive holding/reasoning relevant
   to the claim.
4. Output JSON only, no commentary. Format: {"passage": "..."}"""


def _build_prompt(paragraph: str, citation: str, opinion_text: str) -> list[dict]:
    """Returns a Bedrock messages payload with prompt-caching on opinion_text.

    The opinion text is the largest reusable chunk: many gold rows share a
    cited opinion (different questions citing Brown v. Board, etc.). We cache
    it so back-to-back calls for the same opinion read instead of write.
    """
    opinion_block = opinion_text[:_MAX_OPINION_CHARS]
    paragraph_block = paragraph[-_MAX_PARAGRAPH_CHARS:]
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"OPINION CITED ({citation}):\n\n{opinion_block}"
                    ),
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": (
                        "PARAGRAPH FROM RESEARCH ANSWER (which cites the "
                        f"opinion above):\n\n{paragraph_block}\n\n"
                        "Return JSON: {\"passage\": \"verbatim substring or empty\"}"
                    ),
                },
            ],
        }
    ]


def _extract_passage(client, paragraph: str, citation: str, opinion_text: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 600,
        "system": _SYSTEM_PROMPT,
        "messages": _build_prompt(paragraph, citation, opinion_text),
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
            return ""
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            return ""
    return (obj.get("passage") or "").strip()


def _build_paragraph_index(responses: pd.DataFrame) -> dict[tuple[str, str], str]:
    """For each (question_id, normalized_citation), pick the response paragraph
    that mentions the citation. If multiple responses cite it, keep the longest
    surrounding context."""
    out: dict[tuple[str, str], str] = {}
    for _, row in responses.iterrows():
        qid = row["question_id"]
        text = row["response_text"] or ""
        if not text.strip():
            continue
        for cite in extract_citations(text):
            key = (qid, cite.normalized)
            s, e = cite.span
            paragraph_start = text.rfind("\n\n", 0, s)
            if paragraph_start < 0:
                paragraph_start = max(0, s - 1500)
            paragraph_end = text.find("\n\n", e)
            if paragraph_end < 0:
                paragraph_end = min(len(text), e + 500)
            paragraph = text[paragraph_start:paragraph_end].strip()
            if not paragraph:
                continue
            if key not in out or len(paragraph) > len(out[key]):
                out[key] = paragraph
    return out


def _build_quote_match_set() -> set[tuple[str, str]]:
    """Re-run the cheap adjacent_quote substring match to identify (qid, cite)
    pairs that already have grounding chunks — we skip those to save LLM calls."""
    gold = pd.read_parquet(config.DATASET_DIR / "gold_citations.parquet")
    chunks = pd.read_parquet(config.CORPUS_DIR / "chunks.parquet")

    chunks_by_op: dict[str, list[str]] = defaultdict(list)
    for op_id, text in zip(chunks["opinion_id"].values, chunks["text"].fillna("").values):
        chunks_by_op[op_id].append(_normalize(text))

    matched: set[tuple[str, str]] = set()
    for _, row in gold.iterrows():
        op_id = row["opinion_id"]
        quote = row["adjacent_quote"] or ""
        if not op_id or not quote.strip():
            continue
        needle = _normalize(quote)
        if len(needle) < 12:
            continue
        for haystack in chunks_by_op.get(op_id, []):
            full_or_prefix = (
                needle in haystack
                or (len(needle) > 60 and needle[:60] in haystack)
            )
            if full_or_prefix:
                matched.add((row["source_question_id"], row["normalized_citation"]))
                break
    return matched


def build_paraphrase_groundings_artifact() -> pd.DataFrame:
    """Build paraphrase_groundings.parquet, one row per (qid, citation) we
    asked Sonnet about. Caching: if the file exists, skip rows already in it."""
    gold = pd.read_parquet(config.DATASET_DIR / "gold_citations.parquet")
    opinions = pd.read_parquet(config.CORPUS_DIR / "opinions.parquet")
    opinion_text_by_id = dict(zip(opinions["opinion_id"], opinions["text"]))

    already_matched = _build_quote_match_set()
    paragraph_index = _build_paragraph_index(dataset.extract_released_responses())

    cached_rows: dict[tuple[str, str], dict] = {}
    if _GROUNDINGS_PATH.exists():
        prior = pd.read_parquet(_GROUNDINGS_PATH)
        for _, r in prior.iterrows():
            cached_rows[(r["source_question_id"], r["normalized_citation"])] = r.to_dict()
        logger.info("Loaded %d cached paraphrase groundings", len(cached_rows))

    todo: list[tuple[str, str, str, str]] = []
    for _, row in gold.iterrows():
        qid = row["source_question_id"]
        cite = row["normalized_citation"]
        op_id = row["opinion_id"]
        if not op_id:
            continue
        if (qid, cite) in already_matched:
            continue
        if (qid, cite) in cached_rows:
            continue
        opinion_text = opinion_text_by_id.get(op_id, "")
        if not opinion_text or not opinion_text.strip():
            continue
        paragraph = paragraph_index.get((qid, cite), "")
        if not paragraph:
            continue
        todo.append((qid, cite, op_id, paragraph))

    logger.info(
        "Paraphrase grounding: %d cached, %d already-matched skipped, %d to query",
        len(cached_rows), len(already_matched), len(todo),
    )

    if not todo:
        logger.info("Nothing to do — paraphrase_groundings is up to date")
        return _flush(cached_rows)

    bedrock = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)

    todo.sort(key=lambda t: t[2])

    save_every = 25
    new_rows = 0
    for qid, cite, op_id, paragraph in tqdm(todo, desc="paraphrase grounding"):
        opinion_text = opinion_text_by_id.get(op_id, "")
        try:
            passage = _extract_passage(bedrock, paragraph, cite, opinion_text)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Paraphrase grounding parse error for %s/%s: %s", qid, cite, e)
            passage = ""
        cached_rows[(qid, cite)] = {
            "source_question_id": qid,
            "normalized_citation": cite,
            "opinion_id": op_id,
            "passage": passage,
        }
        new_rows += 1
        if new_rows % save_every == 0:
            _flush(cached_rows)

    return _flush(cached_rows)


def _flush(rows: dict[tuple[str, str], dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        df = pd.DataFrame(columns=[
            "source_question_id", "normalized_citation", "opinion_id", "passage"
        ])
    df.to_parquet(_GROUNDINGS_PATH, index=False)
    return df
