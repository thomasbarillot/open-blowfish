"""Document-level train/val/test split — no docname leakage across splits."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from blowfish.evaluation.types import RetrievalRecord


def _modal_docname(record: RetrievalRecord) -> str:
    if not record.top_k:
        return "__no_chunks__"
    counts: dict[str, int] = {}
    for c in record.top_k:
        counts[c.docname] = counts.get(c.docname, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def document_level_split(
    records: Sequence[RetrievalRecord],
    *,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> tuple[list[RetrievalRecord], list[RetrievalRecord], list[RetrievalRecord]]:
    """Split records into (train, val, test) so no docname appears in two splits.

    Each record is bucketed by the modal docname across its top-k chunks. The
    unique docnames are shuffled with ``seed`` and partitioned by ``ratios``;
    records inherit their docname's split.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0; got {ratios}")
    if len(records) == 0:
        return [], [], []
    docnames = [_modal_docname(r) for r in records]
    unique_docs = sorted(set(docnames))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_docs)
    n_docs = len(unique_docs)
    n_train = int(round(n_docs * ratios[0]))
    n_val = int(round(n_docs * ratios[1]))
    train_docs = set(unique_docs[:n_train])
    val_docs = set(unique_docs[n_train : n_train + n_val])
    train: list[RetrievalRecord] = []
    val: list[RetrievalRecord] = []
    test: list[RetrievalRecord] = []
    for r, doc in zip(records, docnames):
        if doc in train_docs:
            train.append(r)
        elif doc in val_docs:
            val.append(r)
        else:
            test.append(r)
    return train, val, test
