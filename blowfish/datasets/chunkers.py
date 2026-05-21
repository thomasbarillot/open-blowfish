"""Text chunkers under a common ``Chunker`` Protocol.

Implements the four chunkers locked in ``PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md``
§5.3.1:

- :class:`SentenceChunker` — ``pysbd``, deterministic sentence boundary detection.
- :class:`ParagraphChunker` — regex on ``\\n\\n+``, zero deps.
- :class:`RecursiveCharacterChunker` — wraps ``langchain-text-splitters``.
- :class:`TokenSlidingWindowChunker` — token-aware via ``tiktoken``.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Optional, Protocol, runtime_checkable

from blowfish.datasets.types import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    """Common chunker interface."""

    name: ClassVar[str]

    def chunk(self, doc: Document) -> list[Chunk]: ...

    def params(self) -> dict[str, Any]: ...


class _ChunkerBase:
    name: ClassVar[str] = "base"

    def params(self) -> dict[str, Any]:
        return {}

    def _mk_chunk(
        self, doc: Document, idx: int, text: str, start: int, end: int, unit: str
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{doc.doc_id}::{self.name}::{idx:05d}",
            doc_id=doc.doc_id,
            text=text,
            start=start,
            end=end,
            unit=unit,
            params=self.params(),
        )


class SentenceChunker(_ChunkerBase):
    """``pysbd``-backed sentence chunker (default ``language='en'``)."""

    name: ClassVar[str] = "sentence"

    def __init__(self, *, language: str = "en") -> None:
        self.language = language
        try:
            import pysbd
        except ImportError as e:
            raise ImportError(
                "SentenceChunker requires the [datasets] extra: "
                "pip install -e '.[datasets]'"
            ) from e
        self._segmenter = pysbd.Segmenter(language=language, clean=False, char_span=True)

    def params(self) -> dict[str, Any]:
        return {"language": self.language, "library": "pysbd"}

    def chunk(self, doc: Document) -> list[Chunk]:
        spans = self._segmenter.segment(doc.text)
        out: list[Chunk] = []
        for i, span in enumerate(spans):
            out.append(
                self._mk_chunk(
                    doc, i, span.sent, int(span.start), int(span.end), "sentence"
                )
            )
        return out


_PARAGRAPH_RE = re.compile(r"\n\s*\n+")


class ParagraphChunker(_ChunkerBase):
    """Paragraph chunker splitting on ``\\n\\n+`` separators. Zero deps."""

    name: ClassVar[str] = "paragraph"

    def params(self) -> dict[str, Any]:
        return {"separator": r"\n\s*\n+"}

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        out: list[Chunk] = []
        pos = 0
        idx = 0
        for match in _PARAGRAPH_RE.finditer(text):
            piece = text[pos : match.start()].strip()
            if piece:
                out.append(self._mk_chunk(doc, idx, piece, pos, match.start(), "paragraph"))
                idx += 1
            pos = match.end()
        tail = text[pos:].strip()
        if tail:
            out.append(self._mk_chunk(doc, idx, tail, pos, len(text), "paragraph"))
        return out


class RecursiveCharacterChunker(_ChunkerBase):
    """Wraps ``langchain_text_splitters.RecursiveCharacterTextSplitter``."""

    name: ClassVar[str] = "recursive"

    def __init__(self, *, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as e:
            raise ImportError(
                "RecursiveCharacterChunker requires the [datasets] extra: "
                "pip install -e '.[datasets]'"
            ) from e
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def params(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "library": "langchain-text-splitters",
        }

    def chunk(self, doc: Document) -> list[Chunk]:
        texts = self._splitter.split_text(doc.text)
        out: list[Chunk] = []
        cursor = 0
        for i, t in enumerate(texts):
            loc = doc.text.find(t, cursor)
            if loc < 0:
                loc = doc.text.find(t)
            if loc < 0:
                loc = cursor
            start = loc
            end = loc + len(t)
            out.append(self._mk_chunk(doc, i, t, start, end, "recursive"))
            cursor = max(0, end - self.chunk_overlap)
        return out


class TokenSlidingWindowChunker(_ChunkerBase):
    """Token-aware sliding window using ``tiktoken``.

    ``size`` is the window length in tokens; ``stride`` is the step. Char
    offsets in the produced chunks are *approximate* — tiktoken's BPE
    decoding does not preserve precise char spans for every input — but the
    chunk text itself round-trips exactly.
    """

    name: ClassVar[str] = "token"

    def __init__(
        self,
        *,
        size: int = 250,
        stride: Optional[int] = None,
        encoding: str = "o200k_base",
    ) -> None:
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "TokenSlidingWindowChunker requires the [datasets] extra: "
                "pip install -e '.[datasets]'"
            ) from e
        self.size = size
        self.stride = stride if stride is not None else size
        self.encoding = encoding
        self._encoder = tiktoken.get_encoding(encoding)

    def params(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "stride": self.stride,
            "encoding": self.encoding,
            "library": "tiktoken",
        }

    def chunk(self, doc: Document) -> list[Chunk]:
        tokens = self._encoder.encode(doc.text)
        out: list[Chunk] = []
        idx = 0
        pos = 0
        while pos < len(tokens):
            end = min(pos + self.size, len(tokens))
            slice_tokens = tokens[pos:end]
            slice_text = self._encoder.decode(slice_tokens)
            probe = slice_text[:32] if len(slice_text) >= 32 else slice_text
            char_start = doc.text.find(probe) if probe else 0
            if char_start < 0:
                char_start = 0
            char_end = char_start + len(slice_text)
            out.append(
                self._mk_chunk(
                    doc, idx, slice_text, max(0, char_start), char_end, "token"
                )
            )
            idx += 1
            if end == len(tokens):
                break
            pos += self.stride
        return out


class ChunkerHooks:
    sentence = SentenceChunker
    paragraph = ParagraphChunker
    recursive = RecursiveCharacterChunker
    token = TokenSlidingWindowChunker


ALL_CHUNKER_NAMES = ("sentence", "paragraph", "recursive", "token")
