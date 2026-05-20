from __future__ import annotations

import argparse
import json
import logging
import re

import boto3
import pandas as pd
from tqdm import tqdm

from . import config, dataset
from .evaluation.citation_parser import extract_citations
from .retrieval.bm25_retriever import BM25Retriever
from .retrieval.chunk_store import ChunkStore
from .retrieval.dense_retriever import DenseRetriever
from .retrieval.homology_retriever import (
    HomologyRetriever,
    RandomHomologyRetriever,
    TailHomologyRetriever,
)
from .retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a legal research assistant. Answer the user's legal question "
    "using ONLY the provided court opinion excerpts.\n\n"
    "Each excerpt is introduced by a header of the form '[Excerpt N] (CITE)' "
    "where CITE is the canonical citation of the opinion.\n\n"
    "When you rely on an excerpt, copy its CITE VERBATIM from the header — "
    "do not reformat, do not abbreviate, do not invent citations not present "
    "in the excerpt headers above.\n\n"
    "If the provided excerpts do not contain enough information to answer "
    "the question, say so explicitly rather than speculating.\n\n"
    "After your answer, on a final line, output exactly:\n"
    "Citations used: [<cite1>; <cite2>; ...]\n"
    "where each <citeN> is copied verbatim from an excerpt header you used. "
    "If you used no excerpts, output 'Citations used: []'."
)


def _parse_citations_used_line(text: str) -> list[str]:
    """Pull the bracketed citation list from the trailing 'Citations used:' marker.

    Falls back to empty list if the marker is missing or malformed. Citations
    are separated by ';' inside the brackets.
    """
    match = re.search(r"Citations used:\s*\[(.*?)\]", text, re.DOTALL)
    if not match:
        return []
    inner = match.group(1).strip()
    if not inner:
        return []
    return [c.strip() for c in inner.split(";") if c.strip()]


def _build_retriever(name: str, **kwargs) -> object:
    if name == "bm25":
        return BM25Retriever.load(
            config.INDEX_DIR / "bm25",
            config.CORPUS_DIR / "chunks.parquet",
        )
    if name == "dense":
        return DenseRetriever.load(
            config.INDEX_DIR / "dense.faiss",
            config.CORPUS_DIR / "chunks.parquet",
        )
    if name in ("hybrid", "homology", "random_homology", "tail_homology"):
        bm25 = BM25Retriever.load(
            config.INDEX_DIR / "bm25",
            config.CORPUS_DIR / "chunks.parquet",
        )
        dense = DenseRetriever.load(
            config.INDEX_DIR / "dense.faiss",
            config.CORPUS_DIR / "chunks.parquet",
        )
        hybrid = HybridRetriever(
            bm25,
            dense,
            k_candidates=config.BM25_TOPK,
            rrf_k=config.RRF_K,
        )
        if name == "hybrid":
            return hybrid
        if name == "homology":
            return HomologyRetriever(hybrid, ChunkStore(), homology_weight=1.0)
        if name == "random_homology":
            return RandomHomologyRetriever(
                hybrid,
                ChunkStore(),
                homology_weight=1.0,
                noise_std=kwargs.get("noise_std", 0.0),
                seed=kwargs.get("seed", 42),
            )
        if name == "tail_homology":
            return TailHomologyRetriever(hybrid, ChunkStore(), homology_weight=1.0)
    raise ValueError(f"Unknown retriever: {name}")


def _build_context(chunks: list, opinions: pd.DataFrame | None = None) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        cite = getattr(c, "normalized_citation", "") or c.metadata.get("normalized_citation", "")
        header = f"[Excerpt {i}]" + (f" ({cite})" if cite else "")
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


def _call_bedrock(client, question: str, context: str) -> str:
    messages = [
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    resp = client.invoke_model(
        modelId=config.GENERATOR_MODEL_ID,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "system": _SYSTEM_PROMPT,
                "messages": messages,
            }
        ),
        contentType="application/json",
    )
    body = json.loads(resp["body"].read())
    return body["content"][0]["text"]


def run_rag(
    retriever_name: str,
    run_name: str,
    topk: int,
    shared_tolerance: float = 0.0,
    noisiness_tolerance: float = 0.2,
    noise_std: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    retriever = _build_retriever(retriever_name, noise_std=noise_std, seed=seed)
    questions = pd.read_parquet(config.DATASET_DIR / "questions.parquet")
    bedrock = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)

    run_dir = config.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = run_dir / "results_checkpoint.parquet"
    done_ids = set()
    rows = []
    if checkpoint.exists():
        cached = pd.read_parquet(checkpoint)
        rows = cached.to_dict("records")
        done_ids = {r["question_id"] for r in rows}
        logger.info("Resuming: %d questions already done", len(done_ids))

    retrieve_kwargs = {}
    if retriever_name in ("homology", "random_homology", "tail_homology"):
        retrieve_kwargs = {
            "shared_tolerance": shared_tolerance,
            "noisiness_tolerance": noisiness_tolerance,
        }

    for _, q in tqdm(questions.iterrows(), total=len(questions), desc="rag"):
        if q["question_id"] in done_ids:
            continue

        chunks = retriever.retrieve(q["question_text"], topk, **retrieve_kwargs)
        context = _build_context(chunks)
        response = _call_bedrock(bedrock, q["question_text"], context)

        eyecite_cites = [c.normalized for c in extract_citations(response)]
        marker_cites = _parse_citations_used_line(response)
        cited_union = sorted(set(eyecite_cites) | set(marker_cites))

        rows.append(
            {
                "question_id": q["question_id"],
                "category": q["category"],
                "question_text": q["question_text"],
                "response_text": response,
                "retrieved_chunk_ids": [c.chunk_id for c in chunks],
                "retrieved_opinion_ids": [c.opinion_id for c in chunks],
                "cited_normalized": cited_union,
                "cited_marker": marker_cites,
                "cited_eyecite": eyecite_cites,
                "retriever": retriever_name,
            }
        )

        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_parquet(checkpoint, index=False)

    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "results.parquet", index=False)
    if checkpoint.exists():
        checkpoint.unlink()
    logger.info("Saved %d results to %s", len(df), run_dir / "results.parquet")
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid", "homology", "random_homology", "tail_homology"],
        default="hybrid",
    )
    p.add_argument("--run-name", required=True)
    p.add_argument("--topk", type=int, default=config.FINAL_TOPK)
    p.add_argument(
        "--shared-tolerance",
        type=float,
        default=0.0,
        help="Fraction of shared-count range to accept below max (0.0 = only max, 1.0 = all)",
    )
    p.add_argument(
        "--noisiness-tolerance",
        type=float,
        default=0.2,
        help="Fraction of extra-count range to accept above min (0.0 = only min, 1.0 = all)",
    )
    p.add_argument(
        "--noise-std", type=float, default=0.0, help="Gaussian noise std added to entropy map (random_homology only)"
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for random_homology retriever")
    args = p.parse_args()

    df = run_rag(
        args.retriever,
        args.run_name,
        args.topk,
        shared_tolerance=args.shared_tolerance,
        noisiness_tolerance=args.noisiness_tolerance,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    print(f"Run complete: {len(df)} results -> {config.RUNS_DIR / args.run_name}")


if __name__ == "__main__":
    main()
