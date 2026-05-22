"""Phase 4 — OpenAI adapter test (mocked client; skipped if extra missing)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

openai = pytest.importorskip("openai")

from blowfish.rag.adapters.openai_adapter import OpenAIGenerator
from blowfish.rag.generator import GenerationResult


def _mock_client_returning(text: str, *, total_tokens: int = 12):
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    completion.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 7
    usage.total_tokens = total_tokens
    completion.usage = usage
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    return client


def test_openai_adapter_returns_generation_result():
    client = _mock_client_returning("the answer")
    gen = OpenAIGenerator(model="gpt-4o-mini", client=client)
    result = gen.generate("prompt", ["ctx1", "ctx2"])
    assert isinstance(result, GenerationResult)
    assert result.text == "the answer"
    assert result.model == "gpt-4o-mini"
    assert result.usage["total_tokens"] == 12


def test_openai_adapter_passes_prompt_to_client():
    client = _mock_client_returning("ok")
    gen = OpenAIGenerator(client=client, max_tokens=64, temperature=0.0)
    gen.generate("the prompt body", ["ctx"])
    call_args = client.chat.completions.create.call_args.kwargs
    assert call_args["messages"][0]["content"] == "the prompt body"
    assert call_args["max_tokens"] == 64
