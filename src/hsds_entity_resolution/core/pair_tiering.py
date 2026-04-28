"""Shared threshold-band policy for scored entity-pair outcomes."""

from __future__ import annotations

from typing import Literal

PairOutcome = Literal["duplicate", "maybe", "below_maybe"]


def classify_pair_outcome(
    *,
    final_score: float,
    duplicate_threshold: float,
    maybe_threshold: float,
    low_maybe_threshold: float,
) -> PairOutcome:
    """Classify one score into duplicate/maybe/below-maybe threshold bands.

    ``duplicate_threshold`` (strict high-confidence line) is retained for
    diagnostics and downstream metrics; pair tiering uses ``maybe_threshold``
    as the minimum score for the merged **predicted-duplicate** band (scores
    at or above it are ``pair_outcome == "duplicate"`` and
    ``predicted_duplicate``). The softer review band is
    ``low_maybe_threshold <= score < maybe_threshold`` with outcome ``maybe``.
    """
    _ = duplicate_threshold
    if final_score >= maybe_threshold:
        return "duplicate"
    if final_score >= low_maybe_threshold:
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
    low_maybe_threshold: float,
) -> bool:
    """Return review eligibility derived from score bands using shared policy."""
    _ = duplicate_threshold
    outcome = classify_pair_outcome(
        final_score=final_score,
        duplicate_threshold=duplicate_threshold,
        maybe_threshold=maybe_threshold,
        low_maybe_threshold=low_maybe_threshold,
    )
    return is_review_eligible_outcome(outcome)
