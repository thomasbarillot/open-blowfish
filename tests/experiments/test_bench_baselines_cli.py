"""Phase 5 — bench_baselines CLI smoke test."""

from __future__ import annotations

import subprocess
import sys


def test_bench_baselines_dummy_runs_and_prints_table():
    """Invoke the CLI as a subprocess; expect exit 0 and a parsable table."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "blowfish.experiments.bench_baselines",
            "--dummy",
            "--bootstrap",
            "100",
            "--baselines",
            "B0,B1,B2,B4,B5,B9",
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    # Every requested baseline ID should appear in the table.
    for bid in ("B0", "B1", "B2", "B4", "B5", "B9"):
        assert bid in result.stdout


def test_bench_baselines_no_dummy_errors_out():
    result = subprocess.run(
        [sys.executable, "-m", "blowfish.experiments.bench_baselines"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "dummy" in (result.stdout + result.stderr).lower()
