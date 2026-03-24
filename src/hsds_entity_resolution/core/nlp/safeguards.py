"""Centralized safeguard policies for NLP similarity scoring."""

from __future__ import annotations

import re
from collections.abc import Callable

from hsds_entity_resolution.core.nlp.types import NlpSafeguardContext, NlpSafeguardOutcome

NlpSafeguard = Callable[[NlpSafeguardContext], NlpSafeguardOutcome]


def apply_nlp_safeguards(
    *,
    similarity: float,
    context: NlpSafeguardContext,
    safeguards: tuple[NlpSafeguard, ...] | None = None,
) -> float:
    """Apply safeguard policies in deterministic order."""
    active_safeguards = safeguards or DEFAULT_NLP_SAFEGUARDS
    adjusted_similarity = similarity
    for safeguard in active_safeguards:
        outcome = safeguard(context)
        if outcome.veto:
            return 0.0
        adjusted_similarity = max(0.0, adjusted_similarity - outcome.penalty)
    return adjusted_similarity


def number_mismatch_safeguard(context: NlpSafeguardContext) -> NlpSafeguardOutcome:
    """Veto fuzzy contribution when both names include different numbers."""
    if not context.config.scoring.nlp.number_mismatch_veto_enabled:
        return NlpSafeguardOutcome()
    left_numbers = set(re.findall(r"\d+", context.left_name))
    right_numbers = set(re.findall(r"\d+", context.right_name))
    if left_numbers and right_numbers and left_numbers != right_numbers:
        return NlpSafeguardOutcome(veto=True)
    return NlpSafeguardOutcome()


DEFAULT_NLP_SAFEGUARDS: tuple[NlpSafeguard, ...] = (number_mismatch_safeguard,)
