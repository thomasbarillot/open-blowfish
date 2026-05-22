"""Phase 5 — sensitivity sweep_grid."""

from __future__ import annotations

import pandas as pd
import pytest

from blowfish.experiments.sensitivity import grid_size, sweep_grid


def test_sweep_grid_cartesian_product_size():
    out = sweep_grid(
        lambda k, eps: {"metric": k * eps},
        grid={"k": [10, 20], "eps": [0.1, 0.5, 1.0]},
    )
    assert len(out) == 6
    assert set(out.columns) == {"k", "eps", "metric"}


def test_sweep_grid_carries_grid_columns_to_output():
    out = sweep_grid(
        lambda x: {"y": x + 1},
        grid={"x": [1, 2, 3]},
    )
    assert (out["y"] == out["x"] + 1).all()


def test_sweep_grid_handles_dataframe_runner():
    """Runner returns multiple rows per combo — sweep_grid annotates each row
    with the grid keys."""

    def runner(k):
        return pd.DataFrame({"row": [0, 1, 2], "value": [k, k * 2, k * 3]})

    out = sweep_grid(runner, grid={"k": [10, 100]})
    assert len(out) == 6  # 2 combos × 3 rows each
    assert set(out["k"].unique()) == {10, 100}


def test_grid_size_no_run():
    assert grid_size({"k": [10, 20], "eps": [0.1, 0.5, 1.0]}) == 6


def test_sweep_grid_empty_grid_returns_empty_df():
    assert sweep_grid(lambda: {"x": 1}, grid={}).empty
