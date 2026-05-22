"""Phase 3 — sweep() Cartesian product cache."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from blowfish.datasets.chunkers import ParagraphChunker, RecursiveCharacterChunker
from blowfish.datasets.embedders import StubEmbedder
from blowfish.datasets.sweep import sweep
from blowfish.datasets.types import Document


class _FixtureCorpus:
    """Tiny in-memory corpus used by sweep tests (no network, no cache writes)."""

    name = "fixture_corpus"
    version = "v1"

    def iter_documents(self):
        yield Document(
            doc_id="d1",
            title="t1",
            text="Para one body.\n\nPara two body.\n\nPara three body.",
        )


def test_sweep_writes_parquet_and_manifest(tmp_cache_dir):
    manifests = sweep([_FixtureCorpus()], [ParagraphChunker()], [StubEmbedder(dim=8)])
    assert len(manifests) == 1
    m = manifests[0]
    cache_dir = Path(tmp_cache_dir) / m.hash[:16]
    parquet_path = cache_dir / "embeddings.parquet"
    manifest_path = cache_dir / "manifest.json"
    assert parquet_path.exists()
    assert manifest_path.exists()
    table = pq.read_table(parquet_path)
    assert table.num_rows == 3
    assert "embedding" in table.column_names


def test_sweep_is_idempotent_with_cache_hit(tmp_cache_dir):
    corpus = _FixtureCorpus()
    chunker = ParagraphChunker()
    embedder = StubEmbedder(dim=8)
    m1 = sweep([corpus], [chunker], [embedder])[0]
    m2 = sweep([corpus], [chunker], [embedder])[0]
    assert m1.hash == m2.hash


def test_sweep_different_chunker_yields_different_hash(tmp_cache_dir):
    corpus = _FixtureCorpus()
    embedder = StubEmbedder(dim=8)
    m_para = sweep([corpus], [ParagraphChunker()], [embedder])[0]
    m_rec = sweep(
        [corpus], [RecursiveCharacterChunker(chunk_size=20, chunk_overlap=4)], [embedder]
    )[0]
    assert m_para.hash != m_rec.hash


def test_sweep_cartesian_product_size(tmp_cache_dir):
    corpus = _FixtureCorpus()
    chunkers = [ParagraphChunker(), RecursiveCharacterChunker(chunk_size=20, chunk_overlap=4)]
    embedders = [StubEmbedder(dim=4), StubEmbedder(dim=8)]
    manifests = sweep([corpus], chunkers, embedders)
    assert len(manifests) == 4  # 1 × 2 × 2
    assert len({m.hash for m in manifests}) == 4
