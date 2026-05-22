"""Embedder Protocol with a stub backend for tests and a real adapter that
delegates to ``EmbeddingModelHooks`` for production embedding models.
"""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def params(self) -> dict[str, Any]: ...


class StubEmbedder:
    """Deterministic test embedder: sha256-of-text seeds a per-text RNG."""

    name: ClassVar[str] = "stub"

    def __init__(self, *, dim: int = 8) -> None:
        self.dim = int(dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed % (2**31))
            out[i] = rng.normal(size=self.dim).astype(np.float32)
        return out

    def params(self) -> dict[str, Any]:
        return {"backend": "stub", "dim": self.dim}


class _EmbeddingAdapter:
    """Production adapter delegating to a backend resolved via ``EmbeddingModelHooks``."""

    def __init__(
        self,
        *,
        encoder_type: str,
        encoder_config: dict[str, Any],
        display_name: Optional[str] = None,
    ) -> None:
        from blowfish.utils.embedding_models_factory import EmbeddingModelHooks

        self.encoder_type = encoder_type
        self.encoder_config = dict(encoder_config)
        cls = getattr(EmbeddingModelHooks, encoder_type)
        self._encoder = cls(**encoder_config)
        sample = self._encoder.encode(["dimension probe"])
        arr = np.asarray(sample, dtype=np.float32)
        self.dim = int(arr.shape[-1])
        self._name = display_name or encoder_type

    @property
    def name(self) -> str:
        return self._name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._encoder.encode(list(texts)), dtype=np.float32)

    def params(self) -> dict[str, Any]:
        return {"encoder_type": self.encoder_type, "encoder_config": self.encoder_config}


def make_embedder(
    encoder_type: str,
    encoder_config: dict[str, Any],
    *,
    name: Optional[str] = None,
) -> _EmbeddingAdapter:
    """Build a production embedder from a config dict.

    Example::

        emb = make_embedder(
            "sentence_transformer",
            {"model_name_or_path": "sentence-transformers/all-mpnet-base-v2"},
        )
    """
    return _EmbeddingAdapter(
        encoder_type=encoder_type, encoder_config=encoder_config, display_name=name
    )
