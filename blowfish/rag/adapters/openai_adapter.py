"""OpenAI SDK adapter for the :class:`Generator` Protocol.

Lazy-imported by ``GeneratorHooks.openai``; raises ``ImportError`` with a
remediation hint if the ``openai`` package is not installed (``[openai]``
extra).
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Sequence

from blowfish.rag.generator import GenerationResult


class OpenAIGenerator:
    """Thin wrapper around ``openai.OpenAI().chat.completions.create``.

    Construction:

    >>> gen = OpenAIGenerator(model="gpt-4o-mini", max_tokens=512)

    Uses the standard ``openai`` SDK; the client honors ``OPENAI_API_KEY``
    from the environment.
    """

    name: ClassVar[str] = "openai"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenAIGenerator requires the [openai] extra: "
                "pip install -e '.[openai]'"
            ) from exc
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._client = client if client is not None else openai.OpenAI()

    def generate(
        self, prompt: str, context: Sequence[str], **kwargs: Any
    ) -> GenerationResult:
        start = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        choices = getattr(completion, "choices", []) or []
        text = choices[0].message.content if choices else ""
        usage = getattr(completion, "usage", None)
        usage_dict: dict[str, Any] = {}
        if usage is not None:
            for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, attr, None)
                if value is not None:
                    usage_dict[attr] = value
        return GenerationResult(
            text=text or "",
            latency_ms=latency_ms,
            model=self.model,
            usage=usage_dict,
        )
