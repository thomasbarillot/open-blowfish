from __future__ import annotations
from functools import lru_cache
import pandas as pd
from datasets import load_dataset
from . import config


@lru_cache(maxsize=1)
def _load_hf_dataset() -> pd.DataFrame:
    ds = load_dataset(config.HF_DATASET_NAME, split="train")
    return ds.to_pandas()


def extract_questions() -> pd.DataFrame:
    df = _load_hf_dataset()
    # Filter on AccurateQA only and store the accepted response
    df = df[df.Label == "Accurate"]
    out = (
        df[["Question ID", "Question Category", "Question", "Response"]]
        .drop_duplicates(subset="Question ID")
        .rename(columns={
            "Question ID": "question_id",
            "Question Category": "category",
            "Question": "question_text",
            "Response": "response_text"
        })
        .reset_index(drop=True)
    )
    return out


def extract_released_responses() -> pd.DataFrame:
    df = _load_hf_dataset()
    out = df.rename(columns={
        "Question ID": "question_id",
        "Question Category": "category",
        "Question": "question_text",
        "Model": "tool",
        "Response": "response_text",
        "Correctness": "human_correctness",
        "Groundedness": "human_groundedness",
        "Label": "human_label",
    })
    return out[[
        "question_id", "category", "question_text", "tool",
        "response_text", "human_correctness", "human_groundedness", "human_label",
    ]].reset_index(drop=True)
