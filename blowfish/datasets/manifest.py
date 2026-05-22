"""``DatasetManifest`` — content-addressed key for one ``corpus × chunker × embedder`` cell."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from blowfish.utils.constants import MANIFEST_SCHEMA_VERSION


class DatasetManifest(BaseModel):
    """Spec of a ``(corpus × chunker × embedder)`` artifact. The sha256 of its
    canonical JSON serialization (excluding ``created_at``) is the cache key."""

    model_config = ConfigDict(extra="ignore")

    corpus: str
    corpus_version: str
    chunker: str
    chunker_params: dict[str, Any] = Field(default_factory=dict)
    embedder: str
    embedder_dim: int
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = MANIFEST_SCHEMA_VERSION

    @property
    def hash(self) -> str:
        payload = self.model_dump(exclude={"created_at"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def key(self) -> str:
        return self.hash
