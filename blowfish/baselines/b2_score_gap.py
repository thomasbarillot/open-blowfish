"""B2 — Top-1 / Top-2 score gap. Distance-like score: gap = score[1] - score[0]."""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from blowfish.baselines.base import Baseline
from blowfish.evaluation.types import RetrievalRecord, ScoreVector


class ScoreGapBaseline(Baseline):
    name: ClassVar[str] = "B2"

    def score(self, records: Sequence[RetrievalRecord]) -> ScoreVector:
        out = np.empty(len(records), dtype=np.float64)
        for i, r in enumerate(records):
            if len(r.top_k) < 2:
                out[i] = 0.0
                continue
            sorted_chunks = sorted(r.top_k, key=lambda c: c.rank)
            out[i] = float(sorted_chunks[1].score - sorted_chunks[0].score)
        return out
