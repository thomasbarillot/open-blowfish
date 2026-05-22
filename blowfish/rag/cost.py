"""Cost model for the RAG benchmark.

A single Pydantic class with three knobs and an ``expected_utility(outcomes)``
helper. The penalty for ``wrong`` is the headline knob — sweeping it across
``{-1, -3, -10}`` simulates "consumer accepts noise" → "regulated production"
operating regimes.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict


Outcome = Literal["correct", "wrong", "abstain"]


class CostModel(BaseModel):
    """Per-query payoff. Defaults match the headline operating point of
    ``PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md`` §5.5 (cost ``c=3`` for wrong)."""

    model_config = ConfigDict(extra="ignore")

    correct: float = 1.0
    wrong: float = -3.0
    abstain: float = 0.0

    def payoff(self, outcome: Outcome) -> float:
        if outcome == "correct":
            return self.correct
        if outcome == "wrong":
            return self.wrong
        if outcome == "abstain":
            return self.abstain
        raise ValueError(f"Unknown outcome: {outcome!r}")

    def expected_utility(self, outcomes: Iterable[Outcome]) -> float:
        """Mean payoff over the supplied outcome sequence."""
        payoffs = [self.payoff(o) for o in outcomes]
        if not payoffs:
            return 0.0
        return float(np.mean(payoffs))
