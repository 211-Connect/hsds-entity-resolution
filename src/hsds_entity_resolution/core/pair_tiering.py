"""Shared threshold-band policy for scored entity-pair outcomes."""

from __future__ import annotations

from typing import Literal

PairOutcome = Literal["duplicate", "maybe", "below_maybe"]


def classify_pair_outcome(
    *,
    final_score: float,
    duplicate_threshold: float,
    maybe_threshold: float,
) -> PairOutcome:
    """Classify one score into duplicate/maybe/below-maybe threshold bands."""
    if final_score >= duplicate_threshold:
        return "duplicate"
    if final_score >= maybe_threshold:
        return "maybe"
    return "below_maybe"


def is_review_eligible_outcome(pair_outcome: str) -> bool:
    """Return whether the pair outcome should appear in steward review queue."""
    return pair_outcome in {"duplicate", "maybe"}


def is_review_eligible_score(
    *,
    final_score: float,
    duplicate_threshold: float,
    maybe_threshold: float,
) -> bool:
    """Return review eligibility derived from score bands using shared policy."""
    outcome = classify_pair_outcome(
        final_score=final_score,
        duplicate_threshold=duplicate_threshold,
        maybe_threshold=maybe_threshold,
    )
    return is_review_eligible_outcome(outcome)
