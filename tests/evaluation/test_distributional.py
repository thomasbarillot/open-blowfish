"""Phase 1 — evaluation/distributional.py."""

import numpy as np

from blowfish.evaluation.distributional import (
    kl_divergence,
    ks_two_sample,
    wasserstein1_permutation,
)


def test_ks_rejects_shifted_normal():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=300)
    y = rng.normal(0.5, 1.0, size=300)
    stat, p = ks_two_sample(x, y)
    assert stat > 0
    assert p < 0.05


def test_ks_does_not_reject_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=300)
    y = rng.normal(0.0, 1.0, size=300)
    _, p = ks_two_sample(x, y)
    assert p > 0.01


def test_wasserstein1_permutation_rejects_shifted():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=200)
    y = rng.normal(0.5, 1.0, size=200)
    d, p = wasserstein1_permutation(x, y, n_perm=200, seed=0)
    assert d > 0.2
    assert p < 0.05


def test_kl_divergence_small_on_identical_samples():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=500)
    assert kl_divergence(x, x, n_bins=20) < 0.05


def test_kl_divergence_positive_on_shifted_samples():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=500)
    y = rng.normal(2.0, 1.0, size=500)
    assert kl_divergence(x, y, n_bins=20) > 0.1
