"""Phase 2 — B1..B6 stateless geometric baselines."""

import numpy as np
import pytest

from blowfish.baselines import BaselineHooks
from blowfish.evaluation.metrics import auroc


@pytest.mark.parametrize("baseline_id", ["B1", "B2", "B3", "B4", "B5", "B6"])
def test_baseline_returns_finite_scores(baseline_id, synthetic_records):
    baseline = getattr(BaselineHooks, baseline_id)()
    s = baseline.score(synthetic_records)
    assert s.shape == (len(synthetic_records),)
    assert np.isfinite(s).all()


@pytest.mark.parametrize("baseline_id", ["B1", "B2", "B4", "B5"])
def test_distance_based_baselines_discriminate_correct_from_incorrect(
    baseline_id, synthetic_records, synthetic_labels
):
    """B1/B2/B4/B5 should achieve clearly better than chance AUROC on the strongly-signaled fixture."""
    baseline = getattr(BaselineHooks, baseline_id)()
    s = baseline.score(synthetic_records)
    a = auroc(s, synthetic_labels)
    assert a > 0.6, f"{baseline_id} AUROC={a:.3f} on synthetic (expected > 0.6)"


def test_b3_entropy_runs_and_is_finite(synthetic_records):
    """B3 entropy on small-k synthetic data is a noisy signal — only assert it runs cleanly.

    The discriminative claim for B3 is exercised in the full RAG benchmark in Phase 5; on
    a 20-chunk synthetic fixture both correct and incorrect queries can produce peaky
    softmax distributions, so AUROC is not a reliable assertion target here.
    """
    s = BaselineHooks.B3(temperature=0.1).score(synthetic_records)
    assert s.shape == (len(synthetic_records),)
    assert np.isfinite(s).all()
