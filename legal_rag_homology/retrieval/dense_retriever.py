from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .. import config
from .base import Retriever, RetrievedChunk


class DenseRetriever(Retriever):
    def __init__(self, index, model, tokenizer, chunk_ids: list[str],
                 opinion_ids: list[str], texts: list[str], normalized_citations: list[str]):
        self._index = index
        self._model = model
        self._tokenizer = tokenizer
        self._chunk_ids = chunk_ids
        self._opinion_ids = opinion_ids
        self._texts = texts
        self._normalized_citations = normalized_citations

    @classmethod
    def load(cls, index_path: Path, chunks_parquet: Path,
             model_name: str = config.EMBEDDING_MODEL) -> "DenseRetriever":
        df = pd.read_parquet(chunks_parquet)
        if "normalized_citation" in df.columns:
            chunk_citations = df["normalized_citation"].fillna("").tolist()
        else:
            chunk_citations = [""] * len(df)

        model_config = AutoConfig.from_pretrained(model_name)
        model_config.reference_compile = False
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, config=model_config)
        model.eval()
        # Load faiss AFTER torch to avoid OpenMP library conflict on Apple Silicon
        import faiss
        index = faiss.read_index(str(index_path))
        return cls(index, model, tokenizer, df["chunk_id"].tolist(),
                   df["opinion_id"].tolist(), df["text"].tolist(), chunk_citations)

    def _encode(self, text: str) -> np.ndarray:
        inputs = self._tokenizer(
            text, padding=True, truncation=True,
            return_tensors="pt", max_length=config.CHUNK_TOKENS,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
        emb = emb.numpy().astype("float32")
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norm, 1e-12)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        import faiss
        q = self._encode(query)
        faiss.normalize_L2(q)
        scores, idxs = self._index.search(q, k)
        out = []
        for rank, (i, s) in enumerate(zip(idxs[0], scores[0])):
            if i == -1:
                continue
            out.append(RetrievedChunk(
                chunk_id=self._chunk_ids[int(i)],
                opinion_id=self._opinion_ids[int(i)],
                normalized_citation=self._normalized_citations[int(i)],
                text=self._texts[int(i)],
                score=float(s),
                source="dense",
                metadata={"rank": rank},
            ))
        return out
