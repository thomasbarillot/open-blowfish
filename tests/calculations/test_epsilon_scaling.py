"""TASK-004: paper Eq. (1) ε neighborhood scaling for VR persistence features."""

import numpy as np
import pandas as pd
import pytest

from blowfish.calculations.calculations import (
    DEFAULT_EPSILON_N_MIN,
    calculate_vr_persistence_features,
    paper_eq1_scaled_neighbor_distances,
)


# ---------------------------------------------------------------------------
# paper_eq1_scaled_neighbor_distances
# ---------------------------------------------------------------------------

def test_eq1_nearest_neighbor_has_unit_distance():
    """d(nearest, q) = 1 by construction; farther neighbors scale linearly."""
    query = np.array([0.0, 0.0])
    chunks = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    d = paper_eq1_scaled_neighbor_distances(chunks, query)
    assert d[0] == pytest.approx(1.0)
    assert d[1] == pytest.approx(3.0)
    assert d[2] == pytest.approx(5.0)


def test_eq1_accepts_stacked_query_input():
    """Helper accepts the (k, d) query layout used by the DataFrame pipeline."""
    query = np.tile([0.0, 0.0], (3, 1))
    chunks = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    d = paper_eq1_scaled_neighbor_distances(chunks, query)
    assert d == pytest.approx([1.0, 2.0, 3.0])


def test_eq1_safe_zero_falls_back_to_next_nonzero():
    """Query coincides with the nearest chunk → use the next strictly-positive offset norm."""
    query = np.array([0.0, 0.0])
    chunks = np.array([
        [0.0, 0.0],   # zero offset; excluded from denominator
        [2.0, 0.0],   # next nonzero → denominator
        [4.0, 0.0],
    ])
    d = paper_eq1_scaled_neighbor_distances(chunks, query)
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(1.0)
    assert d[2] == pytest.approx(2.0)


def test_eq1_all_zero_offsets_returns_zeros():
    """Degenerate: every chunk coincides with the query → zeros (no scaling possible)."""
    query = np.array([0.0, 0.0])
    chunks = np.zeros((3, 2))
    d = paper_eq1_scaled_neighbor_distances(chunks, query)
    assert np.allclose(d, 0.0)


def test_eq1_empty_chunks_returns_empty():
    d = paper_eq1_scaled_neighbor_distances(np.zeros((0, 3)), np.zeros(3))
    assert d.shape == (0,)


# ---------------------------------------------------------------------------
# ε filtering inside calculate_vr_persistence_features
# ---------------------------------------------------------------------------

# Ten neighbors at increasing radii on evenly-spaced angles — gives a
# non-degenerate VR complex after unit-normalization (10 distinct directions)
# while still allowing ε to select strict subsets by radius.
_RADII = np.array([1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
_ANGLES = np.linspace(0.0, 2.0 * np.pi, _RADII.size, endpoint=False)


def _radial_sub_df() -> pd.DataFrame:
    chunks = [[r * np.cos(a), r * np.sin(a), 0.0, 0.0] for r, a in zip(_RADII, _ANGLES)]
    return pd.DataFrame({
        "chunk_embeddings": chunks,
        "query_embedding": [[0.0, 0.0, 0.0, 0.0]] * _RADII.size,
    })


def test_epsilon_none_matches_default_behavior():
    """ε=None must reproduce the pre-TASK-004 output (feature-flag rollback path)."""
    sub = _radial_sub_df()
    base = calculate_vr_persistence_features(sub)
    flagged = calculate_vr_persistence_features(sub, epsilon=None)
    for key, value in base.items():
        assert flagged[key] == pytest.approx(value)


def test_epsilon_inclusive_matches_no_filter():
    """ε large enough to include all neighbors matches ε=None bitwise."""
    sub = _radial_sub_df()
    base = calculate_vr_persistence_features(sub, epsilon=None)
    inclusive = calculate_vr_persistence_features(sub, epsilon=1e6)
    for key, value in base.items():
        assert inclusive[key] == pytest.approx(value)


def test_epsilon_below_n_min_falls_back_to_full_neighborhood():
    """ε so small the filter would leave < n_min points → use all neighbors."""
    sub = _radial_sub_df()
    base = calculate_vr_persistence_features(sub, epsilon=None)
    # ε < 1 keeps zero scaled neighbors (nearest has d=1); fallback engages.
    fallback = calculate_vr_persistence_features(sub, epsilon=0.5)
    for key, value in base.items():
        assert fallback[key] == pytest.approx(value)


def test_epsilon_n_min_threshold_respected():
    """ε that keeps exactly n_min points must run (not fall back); ε just below must fall back."""
    sub = _radial_sub_df()
    # _RADII = [1.0, 1.2, 1.5, 2.0, ...]. With scaling by the nearest (1.0),
    # scaled distances == _RADII. ε=2.0 keeps the 4 nearest (== n_min default).
    eps_keep = 2.0
    eps_below = 1.4   # keeps only 2 → fallback
    base = calculate_vr_persistence_features(sub, epsilon=None)
    keep4 = calculate_vr_persistence_features(sub, epsilon=eps_keep, n_min=DEFAULT_EPSILON_N_MIN)
    fallback = calculate_vr_persistence_features(sub, epsilon=eps_below, n_min=DEFAULT_EPSILON_N_MIN)
    # At n_min, features should be allowed to differ from the full-neighborhood baseline.
    differ = (not np.isclose(keep4["w1_h0"], base["w1_h0"])) or \
             (not np.isclose(keep4["lt_max_h1"], base["lt_max_h1"]))
    assert differ, "keeping exactly n_min neighbors should change paper-aligned features"
    # Below n_min: identical to base
    for key, value in base.items():
        assert fallback[key] == pytest.approx(value)


def test_epsilon_sweep_produces_finite_smooth_curve():
    """Acceptance criterion: ε sweep gives finite values, varies, and converges to the baseline."""
    sub = _radial_sub_df()
    eps_grid = [2.0, 3.0, 5.0, 10.0, 1e6]
    feats = [calculate_vr_persistence_features(sub, epsilon=e) for e in eps_grid]
    w1 = np.array([f["w1_h0"] for f in feats])
    lt = np.array([f["lt_max_h1"] for f in feats])

    # All values finite (no inf/nan from the filter path).
    assert np.all(np.isfinite(w1)) and np.all(np.isfinite(lt))

    # The largest ε converges to the no-filter baseline.
    baseline = calculate_vr_persistence_features(sub, epsilon=None)
    assert w1[-1] == pytest.approx(baseline["w1_h0"])
    assert lt[-1] == pytest.approx(baseline["lt_max_h1"])

    # The sweep should actually move the features (otherwise ε is inert here).
    assert np.unique(np.round(w1, 8)).size > 1 or np.unique(np.round(lt, 8)).size > 1
