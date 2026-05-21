"""B7 — Calibrated logistic regression on the existing Blowfish features."""

from __future__ import annotations

from typing import ClassVar, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from blowfish.baselines.base import Baseline
from blowfish.calculations import calculate_relevant_features
from blowfish.evaluation.calibration import CalibratedScorer
from blowfish.evaluation.legacy_adapter import to_legacy_query_df
from blowfish.evaluation.types import LabelVector, RetrievalRecord, ScoreVector
from blowfish.utils.constants import DEFAULT_KDE_FEATURES


class CalibratedLogisticBaseline(Baseline):
    name: ClassVar[str] = "B7"
    requires_fit: ClassVar[bool] = True

    def __init__(
        self,
        *,
        feature_order: Optional[list[str]] = None,
        random_state: int = 0,
    ) -> None:
        self.feature_order = list(feature_order or DEFAULT_KDE_FEATURES)
        self.random_state = int(random_state)
        self.scaler: Optional[StandardScaler] = None
        self.classifier: Optional[LogisticRegression] = None
        self.calibrator: Optional[CalibratedScorer] = None
        self._feature_cache: dict[int, np.ndarray] = {}

    def _featurize(self, records: Sequence[RetrievalRecord]) -> np.ndarray:
        key = id(records)
        if key in self._feature_cache:
            return self._feature_cache[key]
        rows = []
        for r in records:
            df = to_legacy_query_df(r)
            feats = calculate_relevant_features(df, self.feature_order)
            rows.append([feats[k] for k in self.feature_order])
        out = np.asarray(rows, dtype=np.float64)
        self._feature_cache[key] = out
        return out

    def fit(self, records: Sequence[RetrievalRecord], labels: LabelVector) -> None:
        X = self._featurize(records)
        y = np.asarray(labels, dtype=int)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.classifier = LogisticRegression(
            random_state=self.random_state, max_iter=1000, solver="lbfgs"
        )
        self.classifier.fit(Xs, y)
        raw = self.classifier.predict_proba(Xs)[:, 1]
        self.calibrator = CalibratedScorer(method="isotonic").fit(raw, y)

    def score(self, records: Sequence[RetrievalRecord]) -> ScoreVector:
        if self.classifier is None or self.scaler is None or self.calibrator is None:
            raise RuntimeError("B7 must be fit() before score().")
        X = self.scaler.transform(self._featurize(records))
        raw = self.classifier.predict_proba(X)[:, 1]
        return self.calibrator.transform(raw)
