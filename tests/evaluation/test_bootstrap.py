"""Phase 1 — evaluation/bootstrap.py."""

import numpy as np
import pytest

from blowfish.evaluation.bootstrap import bootstrap_metric, paired_bootstrap_diff
from blowfish.evaluation.metrics import auroc


def test_bootstrap_metric_seeded_deterministic():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = rng.integers(0, 2, size=200)
    r1 = bootstrap_metric(auroc, scores, labels, n=500, method="percentile", seed=42)
    r2 = bootstrap_metric(auroc, scores, labels, n=500, method="percentile", seed=42)
    assert r1.point == r2.point
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high


def test_bootstrap_metric_bca_returns_finite_ci():
    rng = np.random.default_rng(0)
    scores = rng.random(100)
    labels = rng.integers(0, 2, size=100)
    r = bootstrap_metric(auroc, scores, labels, n=200, method="bca", seed=0)
    assert np.isfinite(r.point)
    assert np.isfinite(r.ci_low) and np.isfinite(r.ci_high)
    assert r.ci_low <= r.point <= r.ci_high or abs(r.point - (r.ci_low + r.ci_high) / 2) < 0.1
    assert r.method == "bca"


def test_paired_bootstrap_diff_zero_for_identical_inputs():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = rng.integers(0, 2, size=200)
    r = paired_bootstrap_diff(auroc, scores, scores, labels, n=500, seed=0)
    assert r.point == pytest.approx(0.0)
    assert r.ci_low <= 0.0 <= r.ci_high


def test_paired_bootstrap_diff_detects_known_lift():
    rng = np.random.default_rng(0)
    n = 500
    labels = rng.integers(0, 2, size=n).astype(float)
    # ``a`` is strongly correlated with labels; ``b`` is random noise.
    a = labels + rng.normal(0, 0.05, size=n)
    b = rng.random(size=n)
    r = paired_bootstrap_diff(auroc, a, b, labels, n=500, seed=0)
    assert r.point > 0.3
    assert r.ci_low > 0.0
