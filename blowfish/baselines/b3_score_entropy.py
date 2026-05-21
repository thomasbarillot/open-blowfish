"""B3 — Negated softmax entropy of top-k scores. Lower entropy → more confident."""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from blowfish.baselines.base import Baseline
from blowfish.evaluation.types import RetrievalRecord, ScoreVector


class ScoreEntropyBaseline(Baseline):
    name: ClassVar[str] = "B3"

    def __init__(self, *, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def score(self, records: Sequence[RetrievalRecord]) -> ScoreVector:
        out = np.empty(len(records), dtype=np.float64)
        for i, r in enumerate(records):
            if not r.top_k:
                out[i] = 0.0
                continue
            # Negate distance → similarity logits, divide by T, stabilize, softmax.
            logits = -np.asarray([c.score for c in r.top_k], dtype=np.float64) / self.temperature
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            entropy = -float(np.sum(p * np.log(p + 1e-12)))
            out[i] = -entropy
        return out
