"""Gating policies G0–G6 + ``tune_threshold`` helper.

All gates implement the same :class:`Gate` Protocol: ``score(record)`` returns
a per-query confidence (higher = more confident the system is correct);
``should_abstain(record)`` returns True when the score falls below the gate's
threshold. Stateless gates (G0, G1, G2, G3, G6) work immediately; trainable
gates (G4, G5) require ``fit(records, labels)`` first.

Gate ↔ baseline mapping (per ``PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md`` §5.4):

- G0 No-gate    — never abstains; baseline reference.
- G1 Score gap  — wraps :class:`ScoreGapBaseline` (B2).
- G2 Entropy    — wraps :class:`ScoreEntropyBaseline` (B3).
- G3 Mean kNN   — wraps :class:`MeanKnnDistanceBaseline` (B4).
- G4 Blowfish   — wraps :class:`AmbiguityScorer` via legacy-DataFrame adapter.
- G5 Logistic   — wraps :class:`CalibratedLogisticBaseline` (B7).
- G6 Oracle     — gold-chunk presence in top-k (upper-bound reference).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from blowfish.baselines.b2_score_gap import ScoreGapBaseline
from blowfish.baselines.b3_score_entropy import ScoreEntropyBaseline
from blowfish.baselines.b4_knn_distance import MeanKnnDistanceBaseline
from blowfish.baselines.b7_calibrated_logistic import CalibratedLogisticBaseline
from blowfish.evaluation.legacy_adapter import to_legacy_query_df
from blowfish.evaluation.types import LabelVector, RetrievalRecord, ScoreVector


@runtime_checkable
class Gate(Protocol):
    """Common gate interface."""

    name: ClassVar[str]
    requires_fit: ClassVar[bool]
    threshold: Optional[float]

    def fit(self, records: Sequence[RetrievalRecord], labels: LabelVector) -> None: ...

    def score(self, record: RetrievalRecord) -> float: ...

    def should_abstain(self, record: RetrievalRecord) -> bool: ...


class _GateBase:
    name: ClassVar[str] = "_base"
    requires_fit: ClassVar[bool] = False

    def __init__(self, *, threshold: Optional[float] = None) -> None:
        self.threshold = threshold

    def fit(self, records: Sequence[RetrievalRecord], labels: LabelVector) -> None:
        return None

    def should_abstain(self, record: RetrievalRecord) -> bool:
        if self.threshold is None:
            raise RuntimeError(
                f"{self.name} threshold is unset. Call "
                f"tune_threshold(gate, val_records, abstain_rate=...) first."
            )
        return self.score(record) < self.threshold


class NoGate(_GateBase):
    """G0 — never abstains; baseline reference."""

    name: ClassVar[str] = "G0"

    def score(self, record: RetrievalRecord) -> float:
        return math.inf

    def should_abstain(self, record: RetrievalRecord) -> bool:
        return False


class ScoreGapGate(_GateBase):
    """G1 — abstain when top-1/top-2 retrieval distances are too close."""

    name: ClassVar[str] = "G1"

    def __init__(self, *, threshold: Optional[float] = None) -> None:
        super().__init__(threshold=threshold)
        self._baseline = ScoreGapBaseline()

    def score(self, record: RetrievalRecord) -> float:
        return float(self._baseline.score([record])[0])


class EntropyGate(_GateBase):
    """G2 — abstain when softmax-entropy of top-k retrieval scores is high (flat)."""

    name: ClassVar[str] = "G2"

    def __init__(self, *, temperature: float = 1.0, threshold: Optional[float] = None) -> None:
        super().__init__(threshold=threshold)
        self._baseline = ScoreEntropyBaseline(temperature=temperature)

    def score(self, record: RetrievalRecord) -> float:
        return float(self._baseline.score([record])[0])


class MeanKnnDistanceGate(_GateBase):
    """G3 — abstain when the average top-k distance is large (loose cluster)."""

    name: ClassVar[str] = "G3"

    def __init__(self, *, threshold: Optional[float] = None) -> None:
        super().__init__(threshold=threshold)
        self._baseline = MeanKnnDistanceBaseline()

    def score(self, record: RetrievalRecord) -> float:
        return float(self._baseline.score([record])[0])


class BlowfishGate(_GateBase):
    """G4 — abstain when ``AmbiguityScorer.p_correct`` is low.

    The scorer is supplied at construction time; tests use a stub. The gate
    invokes the legacy ``calculate_query_correctness_probability`` API via the
    DataFrame adapter so existing pickled KDE models keep working.
    """

    name: ClassVar[str] = "G4"

    def __init__(self, scorer: Any, *, threshold: Optional[float] = None) -> None:
        super().__init__(threshold=threshold)
        self.scorer = scorer

    def score(self, record: RetrievalRecord) -> float:
        df = to_legacy_query_df(record)
        result = self.scorer.calculate_query_correctness_probability(df)
        return float(result.p_correct.iloc[0])


class CalibratedLogisticGate(_GateBase):
    """G5 — calibrated logistic regression on Blowfish features (wraps B7)."""

    name: ClassVar[str] = "G5"
    requires_fit: ClassVar[bool] = True

    def __init__(
        self,
        *,
        random_state: int = 0,
        threshold: Optional[float] = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self._baseline = CalibratedLogisticBaseline(random_state=random_state)

    def fit(self, records: Sequence[RetrievalRecord], labels: LabelVector) -> None:
        self._baseline.fit(records, labels)

    def score(self, record: RetrievalRecord) -> float:
        return float(self._baseline.score([record])[0])


class OracleGate(_GateBase):
    """G6 — abstain iff gold chunk is **not** in the top-k.

    Upper-bound reference: anything below this gate's quality is loss
    attributable to the generator or the retriever, not the gating policy.
    """

    name: ClassVar[str] = "G6"

    def __init__(self) -> None:
        super().__init__(threshold=0.5)

    def score(self, record: RetrievalRecord) -> float:
        if record.gold_chunk_hash is None:
            return 0.0
        return 1.0 if any(c.hash_key == record.gold_chunk_hash for c in record.top_k) else 0.0

    def should_abstain(self, record: RetrievalRecord) -> bool:
        return self.score(record) < 0.5


class GateHooks:
    """Class-attribute registry of gates."""

    G0: ClassVar = NoGate
    G1: ClassVar = ScoreGapGate
    G2: ClassVar = EntropyGate
    G3: ClassVar = MeanKnnDistanceGate
    G4: ClassVar = BlowfishGate
    G5: ClassVar = CalibratedLogisticGate
    G6: ClassVar = OracleGate


ALL_GATE_IDS = ("G0", "G1", "G2", "G3", "G4", "G5", "G6")


def tune_threshold(
    gate: Gate,
    val_records: Sequence[RetrievalRecord],
    *,
    abstain_rate: float = 0.2,
) -> float:
    """Pick the gate threshold that hits the requested abstain rate on the val set.

    Computes ``gate.score(r)`` for every val record, sorts ascending, and sets
    ``gate.threshold`` to the score at the ``abstain_rate`` quantile — so on
    average that fraction of records falls below the threshold (= abstains).
    """
    if not 0.0 <= abstain_rate <= 1.0:
        raise ValueError(f"abstain_rate must be in [0, 1]; got {abstain_rate}")
    scores = np.array([gate.score(r) for r in val_records], dtype=np.float64)
    if scores.size == 0:
        gate.threshold = 0.0
        return 0.0
    threshold = float(np.quantile(scores, abstain_rate))
    gate.threshold = threshold
    return threshold
