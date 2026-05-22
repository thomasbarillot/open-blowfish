"""B4 — Mean kNN distance (negated so higher = closer cluster = more confident)."""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from blowfish.baselines.base import Baseline
from blowfish.evaluation.types import RetrievalRecord, ScoreVector


class MeanKnnDistanceBaseline(Baseline):
    name: ClassVar[str] = "B4"

    def score(self, records: Sequence[RetrievalRecord]) -> ScoreVector:
        out = np.empty(len(records), dtype=np.float64)
        for i, r in enumerate(records):
            if not r.top_k:
                out[i] = 0.0
                continue
            out[i] = -float(np.mean([c.score for c in r.top_k]))
        return out
