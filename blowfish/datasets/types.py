"""Source-side dataset types: a ``Document`` is a raw input; a ``Chunk`` is a
piece produced by a chunker. Distinct from ``evaluation.types.RetrievedChunk``,
which is a chunk *retrieved* for a query at inference time.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """Source document fed to a chunker."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """One semantic unit produced by a chunker."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    unit: str
    params: dict[str, Any] = Field(default_factory=dict)
