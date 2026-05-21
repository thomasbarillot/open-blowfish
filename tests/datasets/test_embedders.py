"""Phase 3 — embedders."""

from __future__ import annotations

import numpy as np

from blowfish.datasets.embedders import StubEmbedder


def test_stub_embedder_deterministic_across_calls():
    e = StubEmbedder(dim=8)
    a = e.encode(["hello", "world"])
    b = e.encode(["hello", "world"])
    assert np.allclose(a, b)


def test_stub_embedder_distinct_for_different_text():
    e = StubEmbedder(dim=8)
    a = e.encode(["hello"])
    b = e.encode(["world"])
    assert not np.allclose(a, b)


def test_stub_embedder_shape_matches_dim():
    e = StubEmbedder(dim=16)
    out = e.encode(["a", "b", "c"])
    assert out.shape == (3, 16)
    assert e.dim == 16


def test_stub_embedder_params_serializable():
    import json
    p = StubEmbedder(dim=8).params()
    assert p == {"backend": "stub", "dim": 8}
    json.dumps(p)
