"""Phase 5 — R0 / R1 / R2 randomized topology controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blowfish.experiments.controls import (
    permute_neighborhoods,
    rotate_embeddings,
    shuffle_feature_block,
)


def test_r0_permute_neighborhoods_destroys_gold_membership(experiment_records):
    """After R0, the gold chunk is no longer associated with its query — the
    paper's claim should collapse on the permuted records."""
    out = permute_neighborhoods(experiment_records, seed=0)
    for r in out:
        # gold_chunk_hash is cleared (no longer meaningful after pooling)
        assert r.gold_chunk_hash is None


def test_r0_preserves_query_embedding(experiment_records):
    out = permute_neighborhoods(experiment_records, seed=0)
    assert len(out) == len(experiment_records)
    for original, shuffled in zip(experiment_records, out):
        assert np.allclose(original.query_embedding, shuffled.query_embedding)


def test_r0_preserves_k(experiment_records):
    out = permute_neighborhoods(experiment_records, seed=0)
    k = len(experiment_records[0].top_k)
    for r in out:
        assert len(r.top_k) == k


def test_r1_rotation_preserves_pairwise_distances(experiment_records):
    """All Euclidean distances must be unchanged — this is the topology-
    invariance property R1 falsifies."""
    out = rotate_embeddings(experiment_records, seed=0)
    for original, rotated in zip(experiment_records, out):
        d_orig = np.linalg.norm(
            np.asarray([c.chunk_embedding for c in original.top_k])
            - np.asarray(original.query_embedding),
            axis=1,
        )
        d_rot = np.linalg.norm(
            np.asarray([c.chunk_embedding for c in rotated.top_k])
            - np.asarray(rotated.query_embedding),
            axis=1,
        )
        assert np.allclose(d_orig, d_rot, atol=1e-10)


def test_r1_rotation_preserves_norms(experiment_records):
    out = rotate_embeddings(experiment_records, seed=0)
    for original, rotated in zip(experiment_records, out):
        assert np.allclose(
            np.linalg.norm(original.query_embedding),
            np.linalg.norm(rotated.query_embedding),
            atol=1e-10,
        )


def test_r1_actually_rotates(experiment_records):
    """Sanity: R1 should NOT be a no-op — the embeddings differ from the original."""
    out = rotate_embeddings(experiment_records, seed=0)
    diffs = []
    for original, rotated in zip(experiment_records, out):
        d = np.linalg.norm(
            np.asarray(original.query_embedding) - np.asarray(rotated.query_embedding)
        )
        diffs.append(d)
    assert max(diffs) > 1e-6


def test_r2_shuffle_feature_block_swaps_only_specified_columns():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame(
        {
            "topology_a": rng.normal(size=n),
            "topology_b": rng.normal(size=n),
            "spread_a": rng.normal(size=n),
            "label": (rng.random(n) > 0.5).astype(int),
        }
    )
    out = shuffle_feature_block(df, ["topology_a", "topology_b"], seed=0)
    # spread_a and label are unchanged
    assert (out["spread_a"] == df["spread_a"]).all()
    assert (out["label"] == df["label"]).all()
    # topology columns are permuted (very unlikely to be identical)
    assert not (out["topology_a"] == df["topology_a"]).all()


def test_r2_unknown_column_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError, match="missing"):
        shuffle_feature_block(df, ["does_not_exist"], seed=0)
