import numpy as np
from scripts.legal_rag_homology.retrieval.bm25_retriever import BM25Retriever


def _tiny_corpus():
    return [
        {"chunk_id": "c1", "opinion_id": "op1", "text": "the rule against perpetuities applies"},
        {"chunk_id": "c2", "opinion_id": "op2", "text": "inevitable discovery exception to the exclusionary rule"},
        {"chunk_id": "c3", "opinion_id": "op3", "text": "unrelated patent law discussion"},
    ]


def test_bm25_ranks_most_relevant_first():
    r = BM25Retriever.from_chunks(_tiny_corpus())
    out = r.retrieve("inevitable discovery", k=2)
    assert out[0].chunk_id == "c2"
    assert len(out) == 2
    assert out[0].score >= out[1].score


def test_bm25_source_is_bm25():
    r = BM25Retriever.from_chunks(_tiny_corpus())
    out = r.retrieve("perpetuities", k=1)
    assert out[0].source == "bm25"


def test_dense_retriever_ranks_semantic_match_first():
    import faiss, numpy as np
    from scripts.legal_rag_homology.retrieval.dense_retriever import DenseRetriever

    embs = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype="float32")
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(3)
    index.add(embs)
    chunk_ids = ["c1", "c2", "c3"]
    opinion_ids = ["op1", "op2", "op3"]
    texts = ["t1", "t2", "t3"]
    q = np.array([[1.0, 0.05, 0.0]], dtype="float32")
    faiss.normalize_L2(q)

    class _FakeEncoder:
        def encode(self, _text): return q[0]

    r = DenseRetriever(index=index, encoder=_FakeEncoder(), chunk_ids=chunk_ids,
                       opinion_ids=opinion_ids, texts=texts,
                       normalized_citations=["", "", ""])
    out = r.retrieve("anything", k=2)
    assert out[0].chunk_id == "c1"
    assert out[1].chunk_id == "c2"
    assert out[0].source == "dense"


def test_hybrid_rrf_fuses_bm25_and_dense():
    from scripts.legal_rag_homology.retrieval.base import Retriever, RetrievedChunk
    from scripts.legal_rag_homology.retrieval.hybrid_retriever import HybridRetriever

    class _Stub(Retriever):
        def __init__(self, ranking):
            self._ranking = ranking
        def retrieve(self, query, k):
            return [RetrievedChunk(chunk_id=cid, opinion_id=cid, normalized_citation="",
                                   text="", score=1.0 - i*0.1, source="stub")
                    for i, cid in enumerate(self._ranking[:k])]

    bm25 = _Stub(["c2", "c1", "c3", "c4"])
    dense = _Stub(["c1", "c2", "c4", "c3"])
    h = HybridRetriever(bm25=bm25, dense=dense, k_candidates=4, rrf_k=60)
    out = h.retrieve("query", k=2)
    assert [c.chunk_id for c in out[:2]] == ["c1", "c2"] or [c.chunk_id for c in out[:2]] == ["c2", "c1"]
    assert out[0].source == "hybrid"
