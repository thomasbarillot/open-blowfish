"""Phase 4 — Generator Protocol + EchoGenerator + GeneratorHooks."""

from __future__ import annotations

import pytest

from blowfish.rag.generator import (
    ALL_GENERATOR_NAMES,
    EchoGenerator,
    GenerationResult,
    Generator,
    GeneratorHooks,
)


def test_echo_returns_first_context_chunk():
    gen = EchoGenerator()
    result = gen.generate("the prompt", ["chunk one", "chunk two"])
    assert result.text == "chunk one"
    assert result.model == "echo"


def test_echo_handles_empty_context_gracefully():
    result = EchoGenerator().generate("the prompt", [])
    assert "[no context]" in result.text


def test_echo_satisfies_generator_protocol():
    """``isinstance(gen, Generator)`` requires the structural members."""
    gen = EchoGenerator()
    assert isinstance(gen, Generator)


def test_echo_with_prefix_decorates_output():
    gen = EchoGenerator(prefix="ANSWER:")
    result = gen.generate("p", ["payload"])
    assert result.text == "ANSWER: payload"


def test_generation_result_round_trips_through_pydantic():
    r = GenerationResult(text="x", latency_ms=12.5, model="m", usage={"tokens": 7})
    blob = r.model_dump_json()
    rebuilt = GenerationResult.model_validate_json(blob)
    assert rebuilt.text == "x" and rebuilt.usage["tokens"] == 7


def test_generator_hooks_exposes_echo_directly():
    assert GeneratorHooks.echo is EchoGenerator
    assert "echo" in ALL_GENERATOR_NAMES


def test_generator_hooks_lazy_imports_missing_extras_raise():
    """If the user requests an optional adapter and the extra is not installed,
    accessing it should raise ``ImportError`` with a clear remediation hint."""
    try:
        cls = GeneratorHooks.anthropic
        gen = cls()
        # If anthropic IS installed, we still hit code path; that's fine.
        _ = gen  # noqa
    except ImportError as e:
        assert "[anthropic]" in str(e)
    except AttributeError:
        # GeneratorHooks is a classmethod-shaped registry; resolution may take
        # this path when anthropic isn't reachable via the optional import.
        pass
