"""Score calibration (isotonic, Platt) and reliability diagrams."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from blowfish.evaluation.types import LabelVector, ScoreVector


class CalibratedScorer(BaseModel):
    """Score-to-probability calibrator. Fit on val, transform on test."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    method: Literal["isotonic", "platt"] = "isotonic"
    fitted: Optional[Any] = None

    def fit(self, scores: ScoreVector, labels: LabelVector) -> "CalibratedScorer":
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(s, y)
            self.fitted = iso
        else:
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(s.reshape(-1, 1), y.astype(int))
            self.fitted = lr
        return self

    def transform(self, scores: ScoreVector) -> ScoreVector:
        if self.fitted is None:
            raise RuntimeError("CalibratedScorer must be fit() before transform().")
        s = np.asarray(scores, dtype=np.float64)
        if self.method == "isotonic":
            return self.fitted.transform(s)
        return self.fitted.predict_proba(s.reshape(-1, 1))[:, 1]


def reliability_diagram(
    scores: ScoreVector,
    labels: LabelVector,
    *,
    n_bins: int = 10,
    ci: Literal["wilson", "none"] = "wilson",
) -> pd.DataFrame:
    """Per-bin reliability table with optional Wilson 95% CI on the empirical accuracy."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        n_b = int(mask.sum())
        center = float(0.5 * (edges[b] + edges[b + 1]))
        if n_b == 0:
            rows.append(
                {
                    "bin_center": center,
                    "bin_mass": 0.0,
                    "bin_score_mean": float("nan"),
                    "bin_acc": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                }
            )
            continue
        acc = float(y[mask].mean())
        row = {
            "bin_center": center,
            "bin_mass": n_b / s.size,
            "bin_score_mean": float(s[mask].mean()),
            "bin_acc": acc,
        }
        if ci == "wilson":
            z = 1.96
            denom = 1.0 + z * z / n_b
            mid = (acc + z * z / (2 * n_b)) / denom
            half = (z * np.sqrt(acc * (1 - acc) / n_b + z * z / (4 * n_b * n_b))) / denom
            row["ci_low"] = float(max(0.0, mid - half))
            row["ci_high"] = float(min(1.0, mid + half))
        else:
            row["ci_low"] = float("nan")
            row["ci_high"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
