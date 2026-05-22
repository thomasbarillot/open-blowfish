"""Code-level pre-registration lock for RAG experiments.

The experiment harness refuses to read the test set without a corresponding
``PreregPlan`` that has been ``lock()``'d. This is the falsifiable-evidence
hygiene called for in ``PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md`` §3.4 / §6:
once a plan is locked, mutating any field (the win threshold, the comparison
gates, the exclusion criteria, …) is detected at verify time and raises
:class:`PreregViolation`.

The lock file is keyed by the plan's **title** (slugified) so that mutating
the plan's *content* still routes verification to the same lock file — and
the content-hash comparison inside surfaces the mutation. Rotating a plan
(rather than mutating one) means giving it a fresh title and re-locking.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from blowfish.datasets.cache import cache_root
from blowfish.rag.cost import CostModel


_ALLOW_TEST_SET_ENV = "BLOWFISH_ALLOW_TEST_SET"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class PreregViolation(RuntimeError):
    """Raised by :func:`verify_lock` when the plan is unlocked or mutated."""


class PreregPlan(BaseModel):
    """OSF-style pre-registration of one RAG experiment.

    Mirrors the template in ``PAPER_FEEDBACK_AND_RAG_EXPERIMENT.md`` §6.
    """

    model_config = ConfigDict(extra="ignore")

    title: str
    primary_hypothesis: str
    gate_under_test: str
    comparison_gates: list[str] = Field(default_factory=list)
    cost_model: CostModel = Field(default_factory=CostModel)
    sensitivity_grid: dict[str, list[Any]] = Field(default_factory=dict)
    exclusion_criteria: list[str] = Field(default_factory=list)
    win_threshold: float = 0.02
    schema_version: int = 1
    locked_at: Optional[str] = None
    locked_git_sha: Optional[str] = None

    @property
    def hash(self) -> str:
        """SHA-256 of the canonical JSON serialization, excluding lock-time stamps."""
        payload = self.model_dump(exclude={"locked_at", "locked_git_sha"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def slug(self) -> str:
        """Title-derived stable id used as the lock filename. Mutating other
        fields keeps the slug; rotating to a fresh plan requires a new title."""
        return _SLUG_RE.sub("_", self.title.lower()).strip("_") or "untitled"


def _prereg_dir(cache_dir: Optional[Path] = None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else (cache_root() / "prereg")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _current_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def lock(plan: PreregPlan, *, cache_dir: Optional[Path] = None) -> Path:
    """Write a lock file for ``plan``. Idempotent: locking the same plan twice
    is a no-op (the existing file is preserved). Returns the lock path.
    """
    target = _prereg_dir(cache_dir) / f"{plan.slug}.lock"
    if target.exists():
        return target
    plan.locked_at = datetime.now(timezone.utc).isoformat()
    plan.locked_git_sha = _current_git_sha()
    target.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return target


def verify_lock(plan: PreregPlan, *, cache_dir: Optional[Path] = None) -> None:
    """Raise :class:`PreregViolation` if ``plan`` has not been locked, or if it
    has been locked but mutated since. Returns silently on success.

    Honors ``$BLOWFISH_ALLOW_TEST_SET=1`` as a dev escape hatch: emits a
    :class:`UserWarning` and returns. CI must NOT set this env var.
    """
    if os.environ.get(_ALLOW_TEST_SET_ENV) == "1":
        warnings.warn(
            "BLOWFISH_ALLOW_TEST_SET=1 — pre-registration verification skipped. "
            "Do NOT set this in CI or in published evaluation runs.",
            UserWarning,
            stacklevel=2,
        )
        return
    target = _prereg_dir(cache_dir) / f"{plan.slug}.lock"
    if not target.exists():
        raise PreregViolation(
            f"No pre-registration lock for plan {plan.title!r} "
            f"(content-hash {plan.hash[:12]}…). "
            f"Call blowfish.experiments.prereg.lock(plan) before touching the test set."
        )
    locked = PreregPlan.model_validate_json(target.read_text(encoding="utf-8"))
    if locked.hash != plan.hash:
        raise PreregViolation(
            f"Pre-registration plan {plan.title!r} mutated since lock "
            f"({locked.hash[:12]}… → {plan.hash[:12]}…). "
            f"Either revert the changes or rotate to a fresh PreregPlan "
            f"(give it a new title) and re-lock."
        )
