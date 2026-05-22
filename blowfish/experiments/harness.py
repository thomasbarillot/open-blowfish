"""Top-level experiment runner: takes records + baselines, returns a
bootstrap-CI baseline table; takes records + gates + generator, returns a
per-gate RAG benchmark table.

The two CLIs (``bench_baselines.py`` and ``bench_rag.py``) are thin wrappers
around ``ExperimentRunner``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from blowfish.baselines.base import Baseline
from blowfish.evaluation.bootstrap import bootstrap_metric
from blowfish.evaluation.metrics import auprc, auroc
from blowfish.evaluation.types import RetrievalRecord
from blowfish.rag.cost import CostModel
from blowfish.rag.gates import Gate, tune_threshold
from blowfish.rag.generator import Generator
from blowfish.rag.harness import RAGHarness
from blowfish.rag.judge import F1Judge
from blowfish.rag.metrics import summarize


class ExperimentRunner(BaseModel):
    """Glue between records, baselines, and the RAG harness.

    ``run_baselines`` reports per-baseline AUROC ± 95% bootstrap CI.
    ``run_rag`` reports per-gate {abstain_rate, EM, F1, utility} on the
    answered subset, optionally also stratified by query difficulty.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    records: list[RetrievalRecord]
    bootstrap_n: int = 1000
    seed: int = 0

    def labels(self) -> np.ndarray:
        return np.asarray(
            [int(r.correct_prediction) if r.correct_prediction is not None else 0 for r in self.records],
            dtype=int,
        )

    def run_baselines(self, baselines: Sequence[Baseline]) -> pd.DataFrame:
        y = self.labels()
        rows: list[dict[str, Any]] = []
        for b in baselines:
            if b.requires_fit:
                b.fit(self.records, y)
            scores = b.score(self.records)
            ci = bootstrap_metric(
                auroc,
                scores,
                y,
                n=self.bootstrap_n,
                method="percentile",
                seed=self.seed,
            )
            rows.append(
                {
                    "name": b.name,
                    "auroc": ci.point,
                    "ci_low": ci.ci_low,
                    "ci_high": ci.ci_high,
                    "auprc": float(auprc(scores, y)),
                    "n": int(len(self.records)),
                }
            )
        return pd.DataFrame(rows)

    def run_rag(
        self,
        gates: Sequence[Gate],
        generator: Generator,
        *,
        val_records: Optional[Sequence[RetrievalRecord]] = None,
        cost_model: Optional[CostModel] = None,
        abstain_rate: float = 0.2,
    ) -> pd.DataFrame:
        """For each gate: tune threshold on val_records (default: first 30%
        of records), run the harness on the remaining records, return summary
        row. G0 (no-gate) and G6 (oracle) skip tuning."""
        cost = cost_model or CostModel()
        if val_records is None:
            n_val = max(1, int(0.3 * len(self.records)))
            val_records = list(self.records[:n_val])
            test_records = list(self.records[n_val:])
        else:
            test_records = list(self.records)
        val_labels = np.asarray(
            [int(r.correct_prediction) if r.correct_prediction is not None else 0 for r in val_records],
            dtype=int,
        )
        rows: list[dict[str, Any]] = []
        for gate in gates:
            if getattr(gate, "requires_fit", False):
                gate.fit(val_records, val_labels)
            if gate.name not in ("G0", "G6"):
                tune_threshold(gate, val_records, abstain_rate=abstain_rate)
            harness = RAGHarness(
                generator=generator,
                gate=gate,
                judge=F1Judge(),
                cost_model=cost,
            )
            run_df = harness.run(test_records)
            summary = summarize(run_df, records=test_records)
            rows.append({"gate": gate.name, **summary})
        return pd.DataFrame(rows)
