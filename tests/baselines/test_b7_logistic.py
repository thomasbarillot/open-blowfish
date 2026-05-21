"""Phase 2 — B7 calibrated logistic regression on Blowfish features."""

import numpy as np
import pytest

from blowfish.baselines import BaselineHooks
from blowfish.evaluation.metrics import auroc


def test_b7_requires_fit_before_score(synthetic_records):
    b = BaselineHooks.B7()
    with pytest.raises(RuntimeError):
        b.score(synthetic_records)


def test_b7_overfits_on_train_set(synthetic_records, synthetic_labels):
    """Sanity: B7 should achieve high AUROC when scored on its own training set."""
    b = BaselineHooks.B7(random_state=0)
    b.fit(synthetic_records, synthetic_labels)
    s = b.score(synthetic_records)
    assert auroc(s, synthetic_labels) > 0.7


def test_b7_deterministic_under_seed(synthetic_records, synthetic_labels):
    a = BaselineHooks.B7(random_state=0)
    a.fit(synthetic_records, synthetic_labels)
    s_a = a.score(synthetic_records)
    b = BaselineHooks.B7(random_state=0)
    b.fit(synthetic_records, synthetic_labels)
    s_b = b.score(synthetic_records)
    assert np.allclose(s_a, s_b)
