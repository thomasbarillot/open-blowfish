"""B9 — Oracle: gold chunk present in top-k (upper bound for the gating task)."""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from blowfish.baselines.base import Baseline
from blowfish.evaluation.types import RetrievalRecord, ScoreVector


class OracleBaseline(Baseline):
    name: ClassVar[str] = "B9"

    def score(self, records: Sequence[RetrievalRecord]) -> ScoreVector:
        out = np.empty(len(records), dtype=np.float64)
        for i, r in enumerate(records):
            if r.gold_chunk_hash is None:
                out[i] = 0.0
                continue
            found = any(c.hash_key == r.gold_chunk_hash for c in r.top_k)
            out[i] = 1.0 if found else 0.0
        return out
