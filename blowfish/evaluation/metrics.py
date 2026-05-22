"""Single-number evaluation metrics computed from (scores, labels) vectors."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from blowfish.evaluation.types import LabelVector, ScoreVector


def auroc(scores: ScoreVector, labels: LabelVector) -> float:
    """Area under the ROC curve. Higher is better. NaN if labels are constant."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def auprc(scores: ScoreVector, labels: LabelVector) -> float:
    """Area under the precision-recall curve. Higher is better."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def brier(scores: ScoreVector, labels: LabelVector) -> float:
    """Brier score on probabilistic scores in [0, 1]. Lower is better."""
    return float(brier_score_loss(labels, scores))


def nll(scores: ScoreVector, labels: LabelVector, *, eps: float = 1e-12) -> float:
    """Negative log-likelihood (log-loss). Lower is better."""
    s = np.clip(np.asarray(scores, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(s) + (1.0 - y) * np.log(1.0 - s)))


def ece(scores: ScoreVector, labels: LabelVector, *, n_bins: int = 10) -> float:
    """Expected Calibration Error, equal-width binned. Lower is better."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if s.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        bin_mass = float(mask.mean())
        bin_acc = float(y[mask].mean())
        bin_conf = float(s[mask].mean())
        total += bin_mass * abs(bin_acc - bin_conf)
    return total


def fpr_at_tpr(scores: ScoreVector, labels: LabelVector, *, tpr: float = 0.95) -> float:
    """False positive rate at a fixed true positive rate. Lower is better."""
    fprs, tprs, _ = roc_curve(labels, scores)
    idx = int(np.searchsorted(tprs, tpr))
    if idx >= len(fprs):
        return float(fprs[-1])
    return float(fprs[idx])
