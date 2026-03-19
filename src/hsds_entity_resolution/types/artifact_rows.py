"""Typed row payloads for stage-internal artifact construction."""

from __future__ import annotations

from typing import Literal, TypedDict

from hsds_entity_resolution.types.domain import EntityType


class CandidatePairRow(TypedDict):
    """Canonical candidate pair row."""

    pair_key: str
    entity_a_id: str
    entity_b_id: str
    entity_type: EntityType
    embedding_similarity: float
    candidate_reason_codes: list[str]
    source_schema_a: str
    source_schema_b: str


class PairReasonRow(TypedDict):
    """Explainability reason row for one pair."""

    pair_key: str
    match_type: str
    raw_contribution: float
    weighted_contribution: float
    signal_weight: float


class ScoredPairRow(TypedDict):
    """Scored pair row emitted by `score_candidates`."""

    pair_key: str
    entity_a_id: str
    entity_b_id: str
    source_schema_a: str
    source_schema_b: str
    entity_type: EntityType
    policy_version: str
    model_version: str
    deterministic_section_score: float
    nlp_section_score: float
    ml_section_score: float | None
    final_score: float
    predicted_duplicate: bool
    pair_outcome: Literal["duplicate", "maybe", "below_maybe"]
    review_eligible: bool
    embedding_similarity: float


class MitigationEventRow(TypedDict):
    """Mitigation evidence event row."""

    pair_key: str
    mitigation_reason: str
    evidence: dict[str, float | int]
    pre_mitigation_prediction: bool
    post_mitigation_prediction: bool
