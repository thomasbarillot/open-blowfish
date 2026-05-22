"""Bootstrap CIs for single-sample metrics and paired metric differences."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import norm

from blowfish.evaluation.types import LabelVector, ScoreVector


MetricFn = Callable[[ScoreVector, LabelVector], float]


class BootstrapResult(BaseModel):
    """Output of a bootstrap procedure: point estimate plus CI and the raw samples."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    point: float
    ci_low: float
    ci_high: float
    samples: np.ndarray
    method: Literal["percentile", "bca"]
    alpha: float


def _bca_quantiles(
    samples: np.ndarray, point: float, jackknife: np.ndarray, alpha: float
) -> tuple[float, float]:
    """Compute BCa-adjusted percentile quantiles."""
    prop_lt = float((samples < point).mean())
    if 0.0 < prop_lt < 1.0:
        z0 = float(norm.ppf(prop_lt))
    else:
        z0 = 0.0
    jack_mean = jackknife.mean()
    num = float(np.sum((jack_mean - jackknife) ** 3))
    den = float(6.0 * (np.sum((jack_mean - jackknife) ** 2) ** 1.5))
    a = num / den if den > 0 else 0.0
    z_lo = float(norm.ppf(alpha / 2.0))
    z_hi = float(norm.ppf(1.0 - alpha / 2.0))

    def _adjust(z: float) -> float:
        denom = 1.0 - a * (z0 + z)
        if denom == 0.0:
            return float(norm.cdf(z0))
        return float(norm.cdf(z0 + (z0 + z) / denom))

    return (
        float(np.quantile(samples, _adjust(z_lo))),
        float(np.quantile(samples, _adjust(z_hi))),
    )


def bootstrap_metric(
    metric_fn: MetricFn,
    scores: ScoreVector,
    labels: LabelVector,
    *,
    n: int = 10_000,
    method: Literal["percentile", "bca"] = "bca",
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap CI for a single-sample metric (e.g. AUROC, AUPRC, ECE)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n_obs = s.size
    rng = np.random.default_rng(seed)
    point = float(metric_fn(s, y))
    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        samples[i] = metric_fn(s[idx], y[idx])
    if method == "percentile":
        ci_low = float(np.quantile(samples, alpha / 2.0))
        ci_high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    else:
        jackknife = np.empty(n_obs, dtype=np.float64)
        all_idx = np.arange(n_obs)
        for i in range(n_obs):
            mask = all_idx != i
            jackknife[i] = metric_fn(s[mask], y[mask])
        ci_low, ci_high = _bca_quantiles(samples, point, jackknife, alpha)
    return BootstrapResult(
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        samples=samples,
        method=method,
        alpha=alpha,
    )


def paired_bootstrap_diff(
    metric_fn: MetricFn,
    a_scores: ScoreVector,
    b_scores: ScoreVector,
    labels: LabelVector,
    *,
    n: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Paired bootstrap CI for ``metric_fn(a) - metric_fn(b)`` on the same labels."""
    a = np.asarray(a_scores, dtype=np.float64)
    b = np.asarray(b_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n_obs = y.size
    rng = np.random.default_rng(seed)
    point = float(metric_fn(a, y) - metric_fn(b, y))
    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        samples[i] = metric_fn(a[idx], y[idx]) - metric_fn(b[idx], y[idx])
    ci_low = float(np.quantile(samples, alpha / 2.0))
    ci_high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return BootstrapResult(
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        samples=samples,
        method="percentile",
        alpha=alpha,
    )
