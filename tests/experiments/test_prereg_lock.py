"""Phase 5 — PreregPlan lock + verify."""

from __future__ import annotations

import pytest

from blowfish.experiments.prereg import (
    PreregPlan,
    PreregViolation,
    lock,
    verify_lock,
)
from blowfish.rag.cost import CostModel


def _mk_plan(**overrides) -> PreregPlan:
    defaults = dict(
        title="Test plan",
        primary_hypothesis="G4 beats G0 on Δ-utility CI excluding zero.",
        gate_under_test="G4",
        comparison_gates=["G0", "G1", "G6"],
        cost_model=CostModel(),
        sensitivity_grid={"k": [25, 50]},
        exclusion_criteria=["no gold in corpus"],
        win_threshold=0.02,
    )
    defaults.update(overrides)
    return PreregPlan(**defaults)


def test_lock_then_verify_succeeds(tmp_cache_dir):
    plan = _mk_plan()
    path = lock(plan)
    assert path.exists()
    verify_lock(plan)  # no exception


def test_verify_without_lock_raises(tmp_cache_dir):
    plan = _mk_plan()
    with pytest.raises(PreregViolation, match="No pre-registration lock"):
        verify_lock(plan)


def test_mutating_plan_after_lock_raises(tmp_cache_dir):
    plan = _mk_plan()
    lock(plan)
    mutated = _mk_plan(win_threshold=0.001)  # changed
    with pytest.raises(PreregViolation, match="mutated since lock"):
        verify_lock(mutated)


def test_lock_is_idempotent(tmp_cache_dir):
    plan = _mk_plan()
    path1 = lock(plan)
    contents1 = path1.read_text()
    path2 = lock(plan)
    assert path1 == path2
    assert path2.read_text() == contents1  # not rewritten


def test_allow_test_set_env_var_skips_verify(tmp_cache_dir, monkeypatch):
    plan = _mk_plan()
    # No lock written. With the escape hatch, verify_lock should warn but not raise.
    monkeypatch.setenv("BLOWFISH_ALLOW_TEST_SET", "1")
    with pytest.warns(UserWarning, match="BLOWFISH_ALLOW_TEST_SET"):
        verify_lock(plan)


def test_plan_hash_excludes_lock_timestamps(tmp_cache_dir):
    """A fresh plan and a locked plan with the same content share the same hash."""
    plan_a = _mk_plan()
    h_a = plan_a.hash
    lock(plan_a)
    # Locking populates locked_at / locked_git_sha, but the hash must not change.
    assert plan_a.hash == h_a
