"""B5 — kNN density = 1 / mean kNN distance. Higher density = more confident."""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from blowfish.baselines.base import Baseline
from blowfish.evaluation.types import RetrievalRecord, ScoreVector


class KnnDensityBaseline(Baseline):
    name: ClassVar[str] = "B5"

    def score(self, records: Sequence[RetrievalRecord]) -> ScoreVector:
        out = np.empty(len(records), dtype=np.float64)
        for i, r in enumerate(records):
            if not r.top_k:
                out[i] = 0.0
                continue
            mean_d = float(np.mean([c.score for c in r.top_k]))
            out[i] = 1.0 / (mean_d + 1e-12)
        return out
