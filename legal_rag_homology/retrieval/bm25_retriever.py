from __future__ import annotations
from pathlib import Path
import bm25s
import pandas as pd
from .base import Retriever, RetrievedChunk


class BM25Retriever(Retriever):
    def __init__(self, retriever: bm25s.BM25, chunk_ids: list[str], texts: list[str],
                 opinion_ids: list[str], normalized_citations: list[str]):
        self._r = retriever
        self._chunk_ids = chunk_ids
        self._texts = texts
        self._opinion_ids = opinion_ids
        self._normalized_citations = normalized_citations

    @classmethod
    def from_chunks(cls, chunks: list[dict]) -> "BM25Retriever":
        texts = [c["text"] for c in chunks]
        tokens = bm25s.tokenize(texts, show_progress=False)
        r = bm25s.BM25()
        r.index(tokens, show_progress=False)
        return cls(r, [c["chunk_id"] for c in chunks], texts, [c["opinion_id"] for c in chunks], [c.get("normalized_citation","") for c in chunks])

    @classmethod
    def load(cls, index_dir: Path, chunks_parquet: Path) -> "BM25Retriever":
        r = bm25s.BM25.load(str(index_dir), load_corpus=False)
        df = pd.read_parquet(chunks_parquet)
        if "normalized_citation" in df.columns:
            chunk_citations = df["normalized_citation"].fillna("").tolist()
        else:
            chunk_citations = [""] * len(df)
        return cls(r, df["chunk_id"].tolist(), df["text"].tolist(),
                   df["opinion_id"].tolist(), chunk_citations)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        self._r.save(str(index_dir))

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        q_tokens = bm25s.tokenize([query], show_progress=False)
        doc_idx, scores = self._r.retrieve(q_tokens, k=k, show_progress=False)
        out: list[RetrievedChunk] = []
        for rank, (i, s) in enumerate(zip(doc_idx[0], scores[0])):
            out.append(RetrievedChunk(
                chunk_id=self._chunk_ids[int(i)],
                opinion_id=self._opinion_ids[int(i)],
                normalized_citation=self._normalized_citations[int(i)],
                text=self._texts[int(i)],
                score=float(s),
                source="bm25",
                metadata={"rank": rank},
            ))
        return out
