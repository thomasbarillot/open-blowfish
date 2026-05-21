"""Multiple-testing correction helpers (Bonferroni, Holm step-down)."""

from __future__ import annotations

import numpy as np


def bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Bonferroni-corrected p-values: ``p_i * m`` clipped to ``[0, 1]``."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    return np.clip(p * m, 0.0, 1.0)


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down corrected p-values (monotone in raw p)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted
