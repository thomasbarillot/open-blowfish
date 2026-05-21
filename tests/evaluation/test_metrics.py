"""Phase 1 — evaluation/metrics.py."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from blowfish.evaluation.metrics import auprc, auroc, brier, ece, fpr_at_tpr, nll


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    scores = rng.random(size=200)
    assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))


def test_auroc_constant_labels_returns_nan():
    labels = np.zeros(50, dtype=int)
    scores = np.random.default_rng(0).random(50)
    assert np.isnan(auroc(scores, labels))


def test_auprc_constant_labels_returns_nan():
    assert np.isnan(auprc(np.array([0.1, 0.2]), np.array([0, 0])))


def test_brier_closed_form():
    # Predictions: [0, 1, 0.5, 0.5]; labels [0,1,0,1] → ((0)+(0)+(0.25)+(0.25))/4 = 0.125
    scores = np.array([0.0, 1.0, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert brier(scores, labels) == pytest.approx(0.125)


def test_nll_low_on_confident_correct_predictions():
    scores = np.array([0.001, 0.999, 0.001, 0.999])
    labels = np.array([0, 1, 0, 1])
    assert nll(scores, labels) < 0.01


def test_ece_perfect_calibration_close_to_zero():
    # Bin 0.2: 20% positive; bin 0.8: 80% positive — perfectly calibrated.
    scores = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
    labels = np.concatenate(
        [np.array([1] * 20 + [0] * 80), np.array([1] * 80 + [0] * 20)]
    )
    assert ece(scores, labels, n_bins=10) < 0.01


def test_ece_known_miscalibration():
    # All predictions 0.9 but ground-truth accuracy is 0.5 → ECE = 0.4.
    scores = np.full(100, 0.9)
    labels = np.array([1] * 50 + [0] * 50)
    assert ece(scores, labels, n_bins=10) == pytest.approx(0.4)


def test_fpr_at_tpr_perfect_separation_is_zero():
    scores = np.arange(100, dtype=float)
    labels = np.array([0] * 50 + [1] * 50)
    assert fpr_at_tpr(scores, labels, tpr=0.95) == pytest.approx(0.0)
