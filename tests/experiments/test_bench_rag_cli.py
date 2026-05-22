"""Phase 5 — bench_rag CLI smoke test."""

from __future__ import annotations

import subprocess
import sys


def test_bench_rag_dummy_runs_and_prints_table():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "blowfish.experiments.bench_rag",
            "--dummy",
            "--gates",
            "G0,G1,G6",
            "--seed",
            "0",
            "--abstain-rate",
            "0.2",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    for gid in ("G0", "G1", "G6"):
        assert gid in result.stdout


def test_bench_rag_g4_skipped_with_explanation():
    """G4 needs an AmbiguityScorer; the CLI skips it with a note on stderr."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "blowfish.experiments.bench_rag",
            "--dummy",
            "--gates",
            "G0,G4",
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "G4" in result.stderr
    assert "skip" in result.stderr.lower()
