"""Phase 2 — B9 oracle (gold chunk in top-k)."""

from blowfish.baselines import BaselineHooks
from blowfish.evaluation.metrics import auroc


def test_b9_perfect_when_gold_always_in_top_k(synthetic_records, synthetic_labels):
    """Gold chunk is always rank 0 in correct records, absent in incorrect → AUROC = 1.0."""
    s = BaselineHooks.B9().score(synthetic_records)
    assert auroc(s, synthetic_labels) == 1.0


def test_b9_score_matches_gold_membership(synthetic_records):
    s = BaselineHooks.B9().score(synthetic_records)
    for record, score in zip(synthetic_records, s):
        if record.gold_chunk_hash is None:
            assert score == 0.0
        else:
            assert score == 1.0
