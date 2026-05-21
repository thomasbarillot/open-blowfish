import numpy as np
import pandas as pd
import pytest

from blowfish.calculations.calculations import (
    calculate_relevant_features,
    calculate_vr_persistence_features,
    paper_lt_max_h1,
    paper_w1_h0_from_birth_death,
)
from blowfish.utils.constants import DEFAULT_KDE_FEATURES


def test_paper_w1_h0_closed_form_two_intervals():
    """Eq. (2) with N-1 = number of finite summands => mean of |d-b|/2."""
    b = np.array([0.0, 0.0])
    d = np.array([1.0, 2.0])
    # contrib per bar: 0.5, 1.0; mean over n=2 finite features = 0.75.
    assert paper_w1_h0_from_birth_death(b, d) == pytest.approx(0.75)


def test_paper_w1_h0_handles_nonzero_birth():
    """For a generic (b, d) point the half-distance |d - (b+d)/2| = (d-b)/2."""
    b = np.array([0.2, 0.5])
    d = np.array([0.6, 1.5])
    # bars: 0.2, 0.5; mean = 0.35
    assert paper_w1_h0_from_birth_death(b, d) == pytest.approx(0.35)


def test_paper_w1_h0_empty():
    assert paper_w1_h0_from_birth_death(np.array([]), np.array([])) == 0.0


def test_paper_lt_max_h1():
    """Eq. (3) is sup |y - gamma_perp(y)| = max((d-b)/2)."""
    b = np.array([0.0, 0.1])
    d = np.array([0.5, 0.7])
    # bars: 0.5, 0.6 -> half: 0.25, 0.30 -> max 0.30
    assert paper_lt_max_h1(b, d) == pytest.approx(0.30)


def test_paper_lt_max_h1_empty():
    assert paper_lt_max_h1(np.array([]), np.array([])) == 0.0


def test_calculate_vr_smoke_includes_paper_keys():
    sub = pd.DataFrame({
        "chunk_embeddings": [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        "query_embedding": [[0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
    })
    out = calculate_vr_persistence_features(sub)
    assert "w1_h0" in out and "lt_max_h1" in out
    assert np.isfinite(out["w1_h0"])
    assert np.isfinite(out["lt_max_h1"])


def test_calculate_relevant_features_default_order_filters():
    sub = pd.DataFrame({
        "score": [0.1, 0.2, 0.3],
        "chunk_embeddings": [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        "query_embedding": [[0.0, 0.0]] * 3,
        "docname": ["a", "a", "b"],
        "topic_label": ["a_1", "a_1", "b_2"],
        "silhouette_score": [0.5, 0.6, 0.7],
    })
    feats = calculate_relevant_features(sub, DEFAULT_KDE_FEATURES)
    assert set(feats.keys()) == set(DEFAULT_KDE_FEATURES)
    assert "w1_h0" in feats
    assert "max_homology_birth" not in feats
