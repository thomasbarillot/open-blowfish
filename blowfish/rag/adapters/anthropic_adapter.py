"""Anthropic SDK adapter for the :class:`Generator` Protocol.

Lazy-imported by ``GeneratorHooks.anthropic``; raises ``ImportError`` with a
remediation hint if the ``anthropic`` package is not installed (``[anthropic]``
extra).
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Sequence

from blowfish.rag.generator import GenerationResult


class AnthropicGenerator:
    """Thin wrapper around ``anthropic.Anthropic().messages.create``.

    Construction:

    >>> gen = AnthropicGenerator(model="claude-haiku-4-5", max_tokens=512)

    Uses the standard ``anthropic`` SDK; the client honors ``ANTHROPIC_API_KEY``
    from the environment.
    """

    name: ClassVar[str] = "anthropic"

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "AnthropicGenerator requires the [anthropic] extra: "
                "pip install -e '.[anthropic]'"
            ) from exc
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._client = client if client is not None else anthropic.Anthropic()

    def generate(
        self, prompt: str, context: Sequence[str], **kwargs: Any
    ) -> GenerationResult:
        start = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        # Anthropic content is a list of blocks; we concatenate text blocks.
        text_parts: list[str] = []
        for block in getattr(message, "content", []) or []:
            block_text = getattr(block, "text", None)
            if block_text:
                text_parts.append(block_text)
        text = "".join(text_parts)
        usage = getattr(message, "usage", None)
        usage_dict: dict[str, Any] = {}
        if usage is not None:
            for attr in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                value = getattr(usage, attr, None)
                if value is not None:
                    usage_dict[attr] = value
        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            model=self.model,
            usage=usage_dict,
        )
