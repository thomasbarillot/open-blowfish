from __future__ import annotations
import argparse
import logging
from .corpus import build_corpus as build_corpus_module
from .corpus import paraphrase_grounding, quote_validation
from . import config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--step",
        choices=[
            "questions", "dataset", "corpus",
            "validate-quotes", "paraphrase", "all",
        ],
        default="all",
    )
    p.add_argument("--distractor-count", type=int, default=config.DEFAULT_DISTRACTOR_COUNT)
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.step in ("questions", "all"):
        q = build_corpus_module.build_questions_artifact()
        print(f"questions: {len(q)} rows -> {config.DATASET_DIR}/questions.parquet")

    if args.step in ("dataset", "all"):
        q = build_corpus_module.build_questions_artifact()
        print(f"questions: {len(q)} rows -> {config.DATASET_DIR}/questions.parquet")
        g = build_corpus_module.build_gold_citations_artifact()
        print(f"gold citations: {len(g)} rows -> {config.DATASET_DIR}/gold_citations.parquet")

    if args.step in ("corpus", "all"):
        df = build_corpus_module.build_corpus(distractor_count=args.distractor_count, seed=args.seed)
        print(f"corpus: {len(df)} opinions -> {config.CORPUS_DIR}/opinions.parquet")

    if args.step in ("validate-quotes", "all"):
        v = quote_validation.build_llm_validation_artifact()
        print(
            f"llm validation: {len(v)} rows -> "
            f"{config.DATASET_DIR}/llm_validated_quotes.parquet"
        )
        g = build_corpus_module.apply_llm_quote_validation()
        print(
            f"applied validation to gold_citations: {len(g)} rows -> "
            f"{config.DATASET_DIR}/gold_citations.parquet"
        )

    if args.step in ("paraphrase", "all"):
        pg = paraphrase_grounding.build_paraphrase_groundings_artifact()
        non_empty = (pg["passage"].str.len() > 0).sum() if len(pg) else 0
        print(
            f"paraphrase groundings: {len(pg)} rows ({non_empty} non-empty) -> "
            f"{config.DATASET_DIR}/paraphrase_groundings.parquet"
        )


if __name__ == "__main__":
    main()
