import json
import os

import numpy as np
import pandas as pd
import pytest

from blowfish.ingestion.vdb_indexing import FaissVDBIndexing


def _df(vectors, hash_keys):
    return pd.DataFrame({"chunk_embedding": list(vectors), "hash_key": hash_keys})


def test_cosine_index_roundtrip(tmp_path):
    d = 4
    vdb_path = str(tmp_path / "faiss.index")
    json_path = str(tmp_path / "index.json")

    indexer = FaissVDBIndexing(
        vdb_vector_size=d,
        vdb_path=vdb_path,
        json_index_path=json_path,
        vdb_metric="cosine",
        vdb_reset_faiss_index=True,
    )
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((6, d)).astype(np.float32)
    indexer(_df(raw, [f"h{i}" for i in range(6)]))

    with open(json_path, "r") as f:
        payload = json.load(f)
    assert payload["metric"] == "cosine"

    matching = FaissVDBIndexing(
        vdb_vector_size=d,
        vdb_path=vdb_path,
        json_index_path=json_path,
        vdb_metric="cosine",
    )
    idx, mapping = matching.load_index()
    assert mapping == [f"h{i}" for i in range(6)]
    # Search should return self with IP ~ 1 for query equal to a stored unit vector.
    q = raw[0:1] / max(np.linalg.norm(raw[0]), 1e-12)
    D, I = idx.search(q.astype(np.float32), 1)
    assert I[0][0] == 0
    assert D[0][0] == pytest.approx(1.0, abs=1e-5)


def test_metric_mismatch_on_load_raises(tmp_path):
    d = 3
    vdb_path = str(tmp_path / "faiss.index")
    json_path = str(tmp_path / "index.json")
    built = FaissVDBIndexing(
        vdb_vector_size=d,
        vdb_path=vdb_path,
        json_index_path=json_path,
        vdb_metric="l2",
        vdb_reset_faiss_index=True,
    )
    built(_df(np.eye(d, dtype=np.float32), ["h0", "h1", "h2"]))

    wrong = FaissVDBIndexing(
        vdb_vector_size=d,
        vdb_path=vdb_path,
        json_index_path=json_path,
        vdb_metric="cosine",
    )
    with pytest.raises(ValueError, match="vdb_metric"):
        wrong.load_index()


def test_invalid_metric_rejected(tmp_path):
    d = 2
    bad = FaissVDBIndexing(
        vdb_vector_size=d,
        vdb_path=str(tmp_path / "x.index"),
        json_index_path=str(tmp_path / "x.json"),
        vdb_metric="cosineish",
        vdb_reset_faiss_index=True,
    )
    with pytest.raises(ValueError, match="cosine"):
        bad.reset_index()


def test_bad_vector_shape_rejected(tmp_path):
    d = 4
    indexer = FaissVDBIndexing(
        vdb_vector_size=d,
        vdb_path=str(tmp_path / "y.index"),
        json_index_path=str(tmp_path / "y.json"),
        vdb_metric="l2",
        vdb_reset_faiss_index=True,
    )
    bad = pd.DataFrame({
        "chunk_embedding": [np.array([1.0, 2.0, 3.0])],
        "hash_key": ["h"],
    })
    with pytest.raises(ValueError, match=r"shape"):
        indexer(bad)


def test_duplicate_hash_key_in_input_rejected(tmp_path):
    indexer = FaissVDBIndexing(
        vdb_vector_size=2,
        vdb_path=str(tmp_path / "dup.index"),
        json_index_path=str(tmp_path / "dup.json"),
        vdb_reset_faiss_index=True,
    )

    with pytest.raises(ValueError, match="unique"):
        indexer(_df(np.eye(2, dtype=np.float32), ["same", "same"]))


def test_duplicate_hash_key_against_existing_mapping_rejected(tmp_path):
    indexer = FaissVDBIndexing(
        vdb_vector_size=2,
        vdb_path=str(tmp_path / "existing.index"),
        json_index_path=str(tmp_path / "existing.json"),
        vdb_reset_faiss_index=True,
    )
    indexer(_df(np.eye(2, dtype=np.float32), ["h0", "h1"]))

    append = FaissVDBIndexing(
        vdb_vector_size=2,
        vdb_path=str(tmp_path / "existing.index"),
        json_index_path=str(tmp_path / "existing.json"),
    )
    with pytest.raises(ValueError, match="already contains"):
        append(_df(np.array([[1.0, 1.0]], dtype=np.float32), ["h1"]))
