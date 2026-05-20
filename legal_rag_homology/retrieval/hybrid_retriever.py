from __future__ import annotations

from collections import defaultdict

from .base import RetrievedChunk, Retriever


class HybridRetriever(Retriever):
    def __init__(self, bm25: Retriever, dense: Retriever, k_candidates: int, rrf_k: int):
        self._bm25 = bm25
        self._dense = dense
        self._k_cand = k_candidates
        self._rrf_k = rrf_k

    def _get_and_rerank(self, query: str, k: int):
        bm25_hits = self._bm25.retrieve(query, self._k_cand)
        dense_hits = self._dense.retrieve(query, self._k_cand)

        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, RetrievedChunk] = {}
        for rank, hit in enumerate(bm25_hits):
            scores[hit.chunk_id] += 1.0 / (self._rrf_k + rank + 1)
            chunks.setdefault(hit.chunk_id, hit)
        for rank, hit in enumerate(dense_hits):
            scores[hit.chunk_id] += 1.0 / (self._rrf_k + rank + 1)
            chunks.setdefault(hit.chunk_id, hit)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        ranked_chunks = [chunks[cid] for cid, _ in ranked]
        return ranked, ranked_chunks, chunks

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:

        ranked, ranked_chunks, chunks = self._get_and_rerank(query, k)

        out = []
        for cid, s in ranked:
            base = chunks[cid]
            out.append(
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    opinion_id=base.opinion_id,
                    normalized_citation=base.normalized_citation,
                    text=base.text,
                    score=s,
                    source="hybrid",
                    metadata={**base.metadata, "fused_score": s},
                )
            )
        return out
