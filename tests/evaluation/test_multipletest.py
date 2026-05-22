"""Phase 1 — evaluation/multipletest.py."""

import numpy as np
import pytest

from blowfish.evaluation.multipletest import bonferroni, holm


def test_bonferroni_multiplies_by_m():
    p = np.array([0.01, 0.05, 0.1])
    corrected = bonferroni(p)
    assert corrected[0] == pytest.approx(0.03)
    assert corrected[1] == pytest.approx(0.15)
    assert corrected[2] == pytest.approx(0.3)


def test_bonferroni_clips_to_one():
    assert (bonferroni(np.array([0.5, 0.6])) == 1.0).all()


def test_bonferroni_empty():
    assert bonferroni(np.array([])).size == 0


def test_holm_is_monotone_in_ranked_p():
    p = np.array([0.001, 0.01, 0.04, 0.1])
    corrected = holm(p)
    assert (corrected[:-1] <= corrected[1:]).all()


def test_holm_single_test_passes_through():
    assert holm(np.array([0.03]))[0] == pytest.approx(0.03)


def test_holm_no_value_exceeds_one():
    p = np.array([0.4, 0.6, 0.8])
    assert (holm(p) <= 1.0).all()
