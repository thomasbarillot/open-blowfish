"""Phase 1 — evaluation/calibration.py."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from blowfish.evaluation.calibration import CalibratedScorer, reliability_diagram
from blowfish.evaluation.metrics import ece


def test_isotonic_reduces_ece_on_miscalibrated_logistic():
    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(0, 1, size=n).reshape(-1, 1)
    y = (x[:, 0] > 0).astype(int)
    lr = LogisticRegression().fit(x, y)
    raw = lr.predict_proba(x)[:, 1]
    # Deliberately miscalibrate by squaring (pulls probabilities toward 0).
    miscal = raw ** 2
    ece_before = ece(miscal, y, n_bins=15)
    cal = CalibratedScorer(method="isotonic").fit(miscal, y)
    fixed = cal.transform(miscal)
    ece_after = ece(fixed, y, n_bins=15)
    assert ece_after < ece_before


def test_platt_runs_end_to_end_in_unit_interval():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (scores > 0.5).astype(int)
    cal = CalibratedScorer(method="platt").fit(scores, labels)
    out = cal.transform(scores)
    assert out.shape == scores.shape
    assert (out >= 0).all() and (out <= 1).all()


def test_calibrated_scorer_raises_before_fit():
    cal = CalibratedScorer(method="isotonic")
    with pytest.raises(RuntimeError):
        cal.transform(np.array([0.1, 0.2]))


def test_reliability_diagram_mass_sums_to_one():
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    labels = rng.integers(0, 2, size=500)
    df = reliability_diagram(scores, labels, n_bins=10)
    assert df["bin_mass"].sum() == pytest.approx(1.0)


def test_reliability_diagram_wilson_ci_brackets_accuracy():
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    labels = rng.integers(0, 2, size=500)
    df = reliability_diagram(scores, labels, n_bins=10, ci="wilson")
    valid = df.dropna(subset=["ci_low", "ci_high"])
    assert (valid["ci_low"] <= valid["bin_acc"]).all()
    assert (valid["bin_acc"] <= valid["ci_high"]).all()
