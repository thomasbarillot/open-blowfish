"""LLM generator Protocol + ``EchoGenerator`` test double.

The RAG harness depends on a narrow generator interface so end users can plug
any LLM (Anthropic, OpenAI, a local model, a mock) into the same evaluation
pipeline. Optional SDK adapters live under ``blowfish/rag/adapters/`` and are
lazy-imported behind the ``[anthropic]`` / ``[openai]`` extras.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class GenerationResult(BaseModel):
    """Single LLM response."""

    model_config = ConfigDict(extra="ignore")

    text: str
    latency_ms: float = 0.0
    model: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Generator(Protocol):
    """Common interface for any LLM backend."""

    name: ClassVar[str]

    def generate(
        self, prompt: str, context: Sequence[str], **kwargs: Any
    ) -> GenerationResult: ...


class EchoGenerator:
    """Deterministic test double: returns the first context chunk as the answer.

    Useful for harness end-to-end tests without an LLM dependency. If the first
    context chunk contains the gold answer (as it does in our synthetic
    fixtures for "correct" queries), EM/F1 judges will mark the row correct.
    """

    name: ClassVar[str] = "echo"

    def __init__(self, *, prefix: str = "") -> None:
        self.prefix = prefix

    def generate(
        self, prompt: str, context: Sequence[str], **kwargs: Any
    ) -> GenerationResult:
        start = time.perf_counter()
        text = context[0] if len(context) > 0 else "[no context]"
        if self.prefix:
            text = f"{self.prefix} {text}"
        latency_ms = (time.perf_counter() - start) * 1000.0
        return GenerationResult(text=text, latency_ms=latency_ms, model=self.name, usage={})


class _GeneratorHooksMeta(type):
    """Metaclass providing class-level ``__getattr__`` for lazy adapter imports.

    Python only consults ``__getattr__`` on instances by default; for
    class-attribute lookup (``GeneratorHooks.anthropic``) the resolver
    has to live on the metaclass.
    """

    def __getattr__(cls, name: str) -> Any:  # noqa: D401
        if name == "anthropic":
            from blowfish.rag.adapters.anthropic_adapter import AnthropicGenerator
            return AnthropicGenerator
        if name == "openai":
            from blowfish.rag.adapters.openai_adapter import OpenAIGenerator
            return OpenAIGenerator
        raise AttributeError(
            f"No generator registered under {name!r}. "
            f"Known: {ALL_GENERATOR_NAMES}"
        )


class GeneratorHooks(metaclass=_GeneratorHooksMeta):
    """Class-attribute registry mirroring ``VDBHooks``.

    Optional adapters are lazy-imported via the metaclass: accessing
    ``GeneratorHooks.anthropic`` triggers the import only on first use, and
    raises ``ImportError`` with a remediation hint if the ``[anthropic]``
    extra is not installed.
    """

    echo: ClassVar = EchoGenerator


ALL_GENERATOR_NAMES = ("echo", "anthropic", "openai")
