"""Two-sample distributional comparison tests."""

from __future__ import annotations

import numpy as np
from scipy import stats


def ks_two_sample(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test → (statistic, p-value)."""
    res = stats.ks_2samp(np.asarray(x), np.asarray(y))
    return float(res.statistic), float(res.pvalue)


def wasserstein1_permutation(
    x: np.ndarray, y: np.ndarray, *, n_perm: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Wasserstein-1 distance + one-sided permutation p-value.

    W₁ is non-negative and the permutation null centers near 0, so the
    natural test is one-sided (``H_a: W₁ > 0``). The returned p-value is
    the add-one (Phipson–Smyth) estimator ``(n_perm_ge + 1) / (n_perm + 1)``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    observed = float(stats.wasserstein_distance(x_arr, y_arr))
    pooled = np.concatenate([x_arr, y_arr])
    n_x = x_arr.size
    rng = np.random.default_rng(seed)
    n_ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if stats.wasserstein_distance(pooled[:n_x], pooled[n_x:]) >= observed:
            n_ge += 1
    pvalue = (n_ge + 1) / (n_perm + 1)
    return observed, float(pvalue)


def kl_divergence(p_samples: np.ndarray, q_samples: np.ndarray, *, n_bins: int = 50) -> float:
    """Histogram KL(p || q) with Laplace smoothing — diagnostic, not a formal test."""
    p_arr = np.asarray(p_samples, dtype=np.float64)
    q_arr = np.asarray(q_samples, dtype=np.float64)
    lo = float(min(p_arr.min(), q_arr.min()))
    hi = float(max(p_arr.max(), q_arr.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p_arr, bins=edges)
    q_hist, _ = np.histogram(q_arr, bins=edges)
    p = (p_hist + 1.0) / (p_hist.sum() + n_bins)
    q = (q_hist + 1.0) / (q_hist.sum() + n_bins)
    return float(np.sum(p * np.log(p / q)))
