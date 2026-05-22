"""B1 — Top-1 retrieval score. Smaller distance = more confident → negate."""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from blowfish.baselines.base import Baseline
from blowfish.evaluation.types import RetrievalRecord, ScoreVector


class Top1ScoreBaseline(Baseline):
    name: ClassVar[str] = "B1"

    def score(self, records: Sequence[RetrievalRecord]) -> ScoreVector:
        out = np.empty(len(records), dtype=np.float64)
        for i, r in enumerate(records):
            if not r.top_k:
                out[i] = 0.0
                continue
            top1 = min(r.top_k, key=lambda c: c.rank)
            out[i] = -float(top1.score)
        return out
