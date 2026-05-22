"""ABC for ambiguity-scoring baselines (B0..B9).

All baselines return a per-query ``ScoreVector`` where **higher = more confident
the retrieval is correct**, paired with the AUROC convention. Distance-like
signals (e.g. mean kNN distance) are negated inside the concrete baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Sequence

from blowfish.evaluation.types import LabelVector, RetrievalRecord, ScoreVector


class Baseline(ABC):
    """Higher score = more confident the system is correct."""

    name: ClassVar[str]
    requires_fit: ClassVar[bool] = False

    def fit(self, records: Sequence[RetrievalRecord], labels: LabelVector) -> None:
        """No-op default; trainable baselines (B7, B8) override."""
        return None

    @abstractmethod
    def score(self, records: Sequence[RetrievalRecord]) -> ScoreVector:
        """Return per-query scores; shape ``(len(records),)``."""
        ...
