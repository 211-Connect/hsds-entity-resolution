"""NLP section scoring for candidate pairs."""

from __future__ import annotations

from typing import Any

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.dataframe_utils import clean_text_scalar
from hsds_entity_resolution.core.nlp.algorithms import resolve_fuzzy_similarity
from hsds_entity_resolution.core.nlp.safeguards import apply_nlp_safeguards
from hsds_entity_resolution.core.nlp.types import NlpSafeguardContext


def compute_nlp_score(
    *,
    left: dict[str, Any],
    right: dict[str, Any],
    config: EntityResolutionRunConfig,
    deterministic_score: float,
) -> tuple[float, float]:
    """Compute NLP contribution and raw fuzzy similarity for reason payloads."""
    left_name = clean_text_scalar(left.get("name"))
    right_name = clean_text_scalar(right.get("name"))
    base_similarity = resolve_fuzzy_similarity(
        left_name=left_name,
        right_name=right_name,
        algorithm=config.scoring.nlp.fuzzy_algorithm,
        strict_validation_mode=config.execution.strict_validation_mode,
    )
    similarity = apply_nlp_safeguards(
        similarity=base_similarity,
        context=NlpSafeguardContext(
            left_name=left_name,
            right_name=right_name,
            config=config,
        ),
    )
    weighted = 0.0
    if similarity >= config.scoring.nlp.fuzzy_threshold:
        weighted = similarity
    if deterministic_score <= 0.0 and weighted < config.scoring.nlp.standalone_fuzzy_threshold:
        weighted = 0.0
    return weighted, similarity
