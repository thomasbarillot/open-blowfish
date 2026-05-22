"""Phase 4 — Anthropic adapter test (mocked client; skipped if extra missing)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

anthropic = pytest.importorskip("anthropic")

from blowfish.rag.adapters.anthropic_adapter import AnthropicGenerator
from blowfish.rag.generator import GenerationResult


def _mock_client_returning(text: str, *, input_tokens: int = 5, output_tokens: int = 7):
    msg = MagicMock()
    block = MagicMock()
    block.text = text
    msg.content = [block]
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    msg.usage = usage
    client = MagicMock()
    client.messages.create.return_value = msg
    return client


def test_anthropic_adapter_returns_generation_result():
    client = _mock_client_returning("the answer")
    gen = AnthropicGenerator(model="claude-haiku-4-5", client=client)
    result = gen.generate("prompt", ["ctx1", "ctx2"])
    assert isinstance(result, GenerationResult)
    assert result.text == "the answer"
    assert result.model == "claude-haiku-4-5"
    assert result.usage["input_tokens"] == 5
    assert result.usage["output_tokens"] == 7


def test_anthropic_adapter_passes_prompt_to_client():
    client = _mock_client_returning("ok")
    gen = AnthropicGenerator(client=client, max_tokens=128, temperature=0.0)
    gen.generate("the prompt body", ["ctx"])
    call_args = client.messages.create.call_args.kwargs
    assert call_args["messages"][0]["content"] == "the prompt body"
    assert call_args["max_tokens"] == 128
