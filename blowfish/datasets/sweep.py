"""Cartesian sweep over ``(corpus × chunker × embedder)`` with parquet cache."""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from blowfish.datasets.cache import cache_path_for
from blowfish.datasets.chunkers import Chunker
from blowfish.datasets.corpora import Corpus
from blowfish.datasets.embedders import Embedder
from blowfish.datasets.manifest import DatasetManifest


def sweep(
    corpora: Sequence[Corpus],
    chunkers: Sequence[Chunker],
    embedders: Sequence[Embedder],
    *,
    force: bool = False,
) -> list[DatasetManifest]:
    """For each ``(corpus, chunker, embedder)`` cell, ensure a cached parquet
    of chunks + embeddings exists. Returns the list of manifests (one per cell)."""
    results: list[DatasetManifest] = []
    for corpus, chunker, embedder in product(corpora, chunkers, embedders):
        manifest = DatasetManifest(
            corpus=corpus.name,
            corpus_version=corpus.version,
            chunker=chunker.name,
            chunker_params=chunker.params(),
            embedder=embedder.name,
            embedder_dim=embedder.dim,
        )
        cache_dir = cache_path_for(manifest.hash)
        cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = cache_dir / "embeddings.parquet"
        manifest_path = cache_dir / "manifest.json"
        if parquet_path.exists() and manifest_path.exists() and not force:
            results.append(manifest)
            continue
        all_chunks = []
        for doc in corpus.iter_documents():
            for c in chunker.chunk(doc):
                all_chunks.append(c)
        if all_chunks:
            texts = [c.text for c in all_chunks]
            embeddings = embedder.encode(texts)
        else:
            embeddings = np.zeros((0, embedder.dim), dtype=np.float32)
        table = pa.Table.from_pydict(
            {
                "chunk_id": [c.chunk_id for c in all_chunks],
                "doc_id": [c.doc_id for c in all_chunks],
                "text": [c.text for c in all_chunks],
                "start": [c.start for c in all_chunks],
                "end": [c.end for c in all_chunks],
                "unit": [c.unit for c in all_chunks],
                "embedding": [list(v) for v in embeddings],
            }
        )
        pq.write_table(table, parquet_path)
        manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        results.append(manifest)
    return results
