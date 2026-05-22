"""B8 — GradientBoostingClassifier on the existing Blowfish features."""

from __future__ import annotations

from typing import ClassVar, Optional, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from blowfish.baselines.base import Baseline
from blowfish.calculations import calculate_relevant_features
from blowfish.evaluation.legacy_adapter import to_legacy_query_df
from blowfish.evaluation.types import LabelVector, RetrievalRecord, ScoreVector
from blowfish.utils.constants import DEFAULT_KDE_FEATURES


class GBMBaseline(Baseline):
    name: ClassVar[str] = "B8"
    requires_fit: ClassVar[bool] = True

    def __init__(
        self,
        *,
        feature_order: Optional[list[str]] = None,
        random_state: int = 0,
    ) -> None:
        self.feature_order = list(feature_order or DEFAULT_KDE_FEATURES)
        self.random_state = int(random_state)
        self.classifier: Optional[GradientBoostingClassifier] = None

    def _featurize(self, records: Sequence[RetrievalRecord]) -> np.ndarray:
        # No id()-keyed cache — see b7_calibrated_logistic._featurize for
        # the rationale (cache was both unsafe and ineffective on the slow
        # path).
        rows = []
        for r in records:
            df = to_legacy_query_df(r)
            feats = calculate_relevant_features(df, self.feature_order)
            rows.append([feats[k] for k in self.feature_order])
        return np.asarray(rows, dtype=np.float64)

    def fit(self, records: Sequence[RetrievalRecord], labels: LabelVector) -> None:
        X = self._featurize(records)
        y = np.asarray(labels, dtype=int)
        self.classifier = GradientBoostingClassifier(random_state=self.random_state)
        self.classifier.fit(X, y)

    def score(self, records: Sequence[RetrievalRecord]) -> ScoreVector:
        if self.classifier is None:
            raise RuntimeError("B8 must be fit() before score().")
        X = self._featurize(records)
        return self.classifier.predict_proba(X)[:, 1]
