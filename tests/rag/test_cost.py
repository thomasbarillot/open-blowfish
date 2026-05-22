"""Phase 4 — CostModel."""

from __future__ import annotations

import pytest

from blowfish.rag.cost import CostModel


def test_default_cost_model_payoffs():
    c = CostModel()
    assert c.payoff("correct") == 1.0
    assert c.payoff("wrong") == -3.0
    assert c.payoff("abstain") == 0.0


def test_expected_utility_all_correct():
    c = CostModel()
    assert c.expected_utility(["correct"] * 10) == pytest.approx(1.0)


def test_expected_utility_all_abstain_is_zero():
    c = CostModel()
    assert c.expected_utility(["abstain"] * 10) == pytest.approx(0.0)


def test_expected_utility_mixed():
    c = CostModel(correct=1.0, wrong=-3.0, abstain=0.0)
    # 4 correct + 3 wrong + 3 abstain over 10 queries
    outcomes = ["correct"] * 4 + ["wrong"] * 3 + ["abstain"] * 3
    expected = (4 * 1.0 + 3 * -3.0 + 3 * 0.0) / 10
    assert c.expected_utility(outcomes) == pytest.approx(expected)


def test_unknown_outcome_raises():
    c = CostModel()
    with pytest.raises(ValueError, match="Unknown outcome"):
        c.payoff("maybe")  # type: ignore[arg-type]


def test_expected_utility_empty_returns_zero():
    assert CostModel().expected_utility([]) == 0.0


def test_cost_model_sweep_param_overrides():
    cheap = CostModel(wrong=-1.0)
    expensive = CostModel(wrong=-10.0)
    # Same outcomes, different cost models → different utility
    out = ["correct"] * 5 + ["wrong"] * 5
    assert cheap.expected_utility(out) > expensive.expected_utility(out)
