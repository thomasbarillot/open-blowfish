"""Cartesian-product sensitivity sweep over k / ε / bandwidth / seed (and any
other knobs a runner declares).

Pattern: a sensitivity grid is a ``dict[str, list[Any]]``. The runner is any
callable that takes those keys as kwargs and returns either a scalar dict or
a per-row DataFrame. ``sweep_grid`` does the Cartesian product and returns
one long-format DataFrame with the grid columns plus the runner's output
columns — ready for plotting or for filtering at a single operating point.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Mapping

import pandas as pd


def sweep_grid(
    runner: Callable[..., Mapping[str, Any] | pd.DataFrame],
    grid: Mapping[str, list[Any]],
) -> pd.DataFrame:
    """Run ``runner(**combo)`` for every combination in ``grid``.

    Each combo's output is annotated with the grid keys + values that
    produced it. Returns a single DataFrame; rows are runner outputs, columns
    are the union of grid keys and runner-output keys.
    """
    if not grid:
        return pd.DataFrame()
    keys = list(grid.keys())
    rows: list[dict[str, Any]] = []
    for values in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))
        result = runner(**params)
        if isinstance(result, pd.DataFrame):
            block = result.copy()
            for k, v in params.items():
                block[k] = v
            rows.extend(block.to_dict("records"))
        else:
            rows.append({**params, **dict(result)})
    return pd.DataFrame(rows)


def grid_size(grid: Mapping[str, list[Any]]) -> int:
    """Cell count without running anything — useful for cost estimates."""
    n = 1
    for v in grid.values():
        n *= max(1, len(v))
    return n
