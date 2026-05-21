"""B0 — Random scoring (sanity floor; AUROC should be ~0.5)."""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from blowfish.baselines.base import Baseline
from blowfish.evaluation.types import RetrievalRecord, ScoreVector


class RandomBaseline(Baseline):
    name: ClassVar[str] = "B0"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    def score(self, records: Sequence[RetrievalRecord]) -> ScoreVector:
        rng = np.random.default_rng(self.seed)
        return rng.random(size=len(records))
