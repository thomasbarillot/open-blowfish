"""Phase 2 — B0 random baseline."""

import numpy as np

from blowfish.baselines.b0_random import RandomBaseline


def test_random_baseline_seeded_deterministic(synthetic_records):
    a = RandomBaseline(seed=42).score(synthetic_records)
    b = RandomBaseline(seed=42).score(synthetic_records)
    assert np.array_equal(a, b)


def test_random_baseline_shape_and_range(synthetic_records):
    s = RandomBaseline(seed=0).score(synthetic_records)
    assert s.shape == (len(synthetic_records),)
    assert (s >= 0).all() and (s <= 1).all()
