from __future__ import annotations
import numpy as np
import pandas as pd
from .base import RetrievedChunk
from .. import config


class ChunkStore:
    def __init__(self):
        self.chunks = pd.read_parquet(config.CORPUS_DIR / "chunks.parquet")
        self.opinions = pd.read_parquet(config.CORPUS_DIR / "opinions.parquet")[
            ["opinion_id", "primary_citation", "normalized_citation", "court", "date_filed"]
        ]
        self._embeddings = np.load(config.INDEX_DIR / "embeddings.npy", mmap_mode="r")
        row_map = pd.read_parquet(config.INDEX_DIR / "chunk_id_to_row.parquet")
        self._id_to_row = dict(zip(row_map["chunk_id"], row_map["row"]))
        self._chunks_by_id = self.chunks.set_index("chunk_id")
        self._ops_by_id = self.opinions.set_index("opinion_id")

    def all_embeddings(self) -> np.ndarray:
        return np.asarray(self._embeddings)

    def get_embeddings(self, chunk_ids: list[str]) -> np.ndarray:
        rows = [self._id_to_row[cid] for cid in chunk_ids]
        return np.asarray(self._embeddings[rows])

    def chunk_id_at_row(self, row_idx: int) -> str:
        return self.chunks.iloc[row_idx]["chunk_id"]

    def get_chunks(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        out = []
        for cid in chunk_ids:
            row = self._chunks_by_id.loc[cid]
            op = self._ops_by_id.loc[row["opinion_id"]]
            out.append(RetrievedChunk(
                chunk_id=cid,
                opinion_id=row["opinion_id"],
                text=row["text"],
                score=0.0,
                source="store",
                metadata={
                    "primary_citation": op["primary_citation"],
                    "normalized_citation": op["normalized_citation"],
                    "court": op["court"],
                    "date_filed": op["date_filed"],
                },
            ))
        return out
