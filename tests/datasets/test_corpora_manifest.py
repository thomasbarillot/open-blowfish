"""Phase 3 — corpora loaders + DatasetManifest hashing."""

from __future__ import annotations

import json

from blowfish.datasets import Corpus, list_corpora, load_corpus
from blowfish.datasets.corpora import (
    ALL_CORPUS_NAMES,
    CorpusHooks,
    FieldsMedalists,
    NobelPhysics,
)
from blowfish.datasets.manifest import DatasetManifest


def test_corpus_hooks_factories_return_named_corpus():
    for name in ALL_CORPUS_NAMES:
        factory = getattr(CorpusHooks, name)
        instance = factory()
        assert instance.name == name


def test_list_corpora_includes_all_shipped_names():
    discovered = list_corpora()
    for name in ALL_CORPUS_NAMES:
        assert name in discovered


def test_load_corpus_round_trip():
    c = load_corpus("nobel_physics")
    assert isinstance(c, Corpus)
    assert c.name == "nobel_physics"


def test_corpus_metadata_exposes_top_level_fields():
    c = NobelPhysics()
    md = c.metadata()
    assert md.get("name") == "nobel_physics"
    assert "documents" not in md  # metadata() drops the document list


def test_corpus_inferred_license_from_documents():
    """When all documents share the same license string, ``Corpus.license`` reflects it."""
    c = NobelPhysics()
    if c.documents:
        assert c.license == "CC-BY-SA-4.0"


def test_nobel_physics_manifest_entries_loadable_from_cache():
    docs = list(NobelPhysics().iter_documents())
    assert len(docs) >= 1, (
        "NobelPhysics manifest is empty or cache not bootstrapped. "
        "Run: python scripts/bootstrap_corpora.py"
    )
    first = docs[0]
    assert first.title
    assert len(first.text) > 100
    assert first.metadata.get("license") == "CC-BY-SA-4.0"


def test_fields_medalists_manifest_entries_loadable_from_cache():
    docs = list(FieldsMedalists().iter_documents())
    assert len(docs) >= 1, (
        "FieldsMedalists manifest is empty or cache not bootstrapped. "
        "Run: python scripts/bootstrap_corpora.py"
    )
    assert len(docs[0].text) > 100


def test_unknown_corpus_raises_with_help_text():
    import pytest
    with pytest.raises(FileNotFoundError, match="list_corpora"):
        Corpus("does_not_exist")


def test_manifest_hash_deterministic():
    m1 = DatasetManifest(
        corpus="x",
        corpus_version="v1",
        chunker="sentence",
        chunker_params={"language": "en"},
        embedder="stub",
        embedder_dim=8,
    )
    m2 = DatasetManifest(
        corpus="x",
        corpus_version="v1",
        chunker="sentence",
        chunker_params={"language": "en"},
        embedder="stub",
        embedder_dim=8,
    )
    assert m1.hash == m2.hash
    assert m1.key() == m2.key()


def test_manifest_hash_changes_with_chunker_params():
    base = DatasetManifest(
        corpus="x",
        corpus_version="v1",
        chunker="recursive",
        chunker_params={"chunk_size": 256, "chunk_overlap": 32},
        embedder="stub",
        embedder_dim=8,
    )
    altered = DatasetManifest(
        corpus="x",
        corpus_version="v1",
        chunker="recursive",
        chunker_params={"chunk_size": 512, "chunk_overlap": 32},
        embedder="stub",
        embedder_dim=8,
    )
    assert base.hash != altered.hash


def test_manifest_round_trips_json():
    m = DatasetManifest(
        corpus="x",
        corpus_version="v1",
        chunker="paragraph",
        embedder="stub",
        embedder_dim=8,
    )
    blob = m.model_dump_json()
    loaded = DatasetManifest.model_validate_json(blob)
    assert loaded.hash == m.hash
