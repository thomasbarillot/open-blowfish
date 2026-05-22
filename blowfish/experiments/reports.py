"""Tabular reports from baselines / RAG benchmark results.

Self-contained markdown formatter to avoid pulling in ``tabulate`` (a
transitive dep of ``pd.DataFrame.to_markdown``); the floatfmt logic mirrors
its default behavior at the precision we care about.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import pandas as pd


def _format_cell(value, floatfmt: str) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return format(value, floatfmt)
    return str(value)


def _to_markdown(df: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_no rows_"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(_format_cell(v, floatfmt) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def baseline_table(
    rows: Iterable[dict],
    *,
    columns: Sequence[str] = ("name", "auroc", "ci_low", "ci_high", "auprc", "n"),
    sort_by: Optional[str] = "auroc",
    descending: bool = True,
    floatfmt: str = ".4f",
) -> str:
    """Format a list of baseline-result dicts as a markdown table."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return "_no rows_"
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending)
    kept = [c for c in columns if c in df.columns]
    return _to_markdown(df[kept], floatfmt=floatfmt)


def rag_table(
    rows: Iterable[dict],
    *,
    columns: Sequence[str] = (
        "gate",
        "abstain_rate",
        "em",
        "f1",
        "expected_utility",
        "n_answered",
    ),
    sort_by: Optional[str] = "expected_utility",
    descending: bool = True,
    floatfmt: str = ".4f",
) -> str:
    """Format a list of per-gate RAG-result dicts as a markdown table."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return "_no rows_"
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending)
    kept = [c for c in columns if c in df.columns]
    return _to_markdown(df[kept], floatfmt=floatfmt)


def write_csv(rows: Iterable[dict], path: str) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False)
