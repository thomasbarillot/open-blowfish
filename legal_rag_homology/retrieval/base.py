from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: str
    opinion_id: str
    normalized_citation: str
    text: str
    score: float
    source: str
    metadata: dict = field(default_factory=dict)


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...
