"""Polars DataFrame column schemas for every pipeline artifact.

Each constant is a ``dict[str, <Polars type>]`` that can be passed directly
to ``pl.DataFrame(schema=...)``, ``pl.DataFrame(schema_overrides=...)``, or
``frame_with_schema(rows, schema)``.  Keeping all schemas here gives a single
place to inspect or evolve the shape of every artifact the pipeline produces.

Stages import only the schemas they need; they no longer define private schema
dicts locally.
"""

from __future__ import annotations

from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Stage 1 — Clean Entities
# ---------------------------------------------------------------------------

CLEAN_ENTITY_SCHEMA: dict[str, Any] = {
    "entity_id": pl.String,
    "entity_type": pl.String,
    "source_schema": pl.String,
    "embedding_vector": pl.List(pl.Float64),
    "content_hash": pl.String,
    "name": pl.String,
    "description": pl.String,
    "emails": pl.List(pl.String),
    "phones": pl.List(pl.String),
    "websites": pl.List(pl.String),
    # Struct shape varies by source schema — stored as Object so Polars never
    # runs struct inference across rows with inconsistent field sets.
    "locations": pl.Object,
    "taxonomies": pl.Object,
    "identifiers": pl.Object,
    "services_rollup": pl.Object,
    "organization_name": pl.String,
    "organization_id": pl.String,
    # Display-quality passthrough fields: original casing, passed through unchanged.
    "display_name": pl.String,
    "display_description": pl.String,
    "alternate_name": pl.String,
    "short_description": pl.String,
    "application_process": pl.String,
    "fees_description": pl.String,
    "eligibility_description": pl.String,
    "resource_writer_name": pl.String,
    "assured_date": pl.String,
    "assurer_email": pl.String,
    "original_id": pl.String,
}

ENTITY_INDEX_SCHEMA: dict[str, Any] = {
    "entity_id": pl.String,
    "entity_type": pl.String,
    "content_hash": pl.String,
    "active_flag": pl.Boolean,
}

# ---------------------------------------------------------------------------
# Stage 2 — Generate Candidates
# ---------------------------------------------------------------------------

CANDIDATE_PAIR_SCHEMA: dict[str, Any] = {
    "pair_key": pl.String,
    "entity_a_id": pl.String,
    "entity_b_id": pl.String,
    "entity_type": pl.String,
    "embedding_similarity": pl.Float64,
    "candidate_reason_codes": pl.List(pl.String),
    "source_schema_a": pl.String,
    "source_schema_b": pl.String,
}

# ---------------------------------------------------------------------------
# Stage 3 — Score Candidates
# ---------------------------------------------------------------------------

SCORED_PAIRS_SCHEMA: dict[str, Any] = {
    "pair_key": pl.String,
    "entity_a_id": pl.String,
    "entity_b_id": pl.String,
    "source_schema_a": pl.String,
    "source_schema_b": pl.String,
    "entity_type": pl.String,
    "policy_version": pl.String,
    "model_version": pl.String,
    "deterministic_section_score": pl.Float64,
    "nlp_section_score": pl.Float64,
    "ml_section_score": pl.Float64,
    "final_score": pl.Float64,
    "predicted_duplicate": pl.Boolean,
    "pair_outcome": pl.String,
    "review_eligible": pl.Boolean,
    "embedding_similarity": pl.Float64,
}

PAIR_REASONS_SCHEMA: dict[str, Any] = {
    "pair_key": pl.String,
    "match_type": pl.String,
    "raw_contribution": pl.Float64,
    "weighted_contribution": pl.Float64,
    "signal_weight": pl.Float64,
    "matched_value": pl.String,
    "entity_a_value": pl.String,
    "entity_b_value": pl.String,
    "similarity_score": pl.Float64,
}

# Used by evidence_policy.count_contributing_reasons as the empty-fallback schema.
CONTRIBUTING_REASONS_SCHEMA: dict[str, Any] = {
    "pair_key": pl.String,
    "reason_count": pl.Int64,
}

# ---------------------------------------------------------------------------
# Stage 4 — Apply Mitigation
# ---------------------------------------------------------------------------

# FINALIZED_PAIRS omits source_schema_a/b (dropped during finalization) and
# adds mitigation_reason compared to SCORED_PAIRS.
FINALIZED_PAIRS_SCHEMA: dict[str, Any] = {
    "pair_key": pl.String,
    "entity_a_id": pl.String,
    "entity_b_id": pl.String,
    "entity_type": pl.String,
    "policy_version": pl.String,
    "model_version": pl.String,
    "deterministic_section_score": pl.Float64,
    "nlp_section_score": pl.Float64,
    "ml_section_score": pl.Float64,
    "final_score": pl.Float64,
    "predicted_duplicate": pl.Boolean,
    "pair_outcome": pl.String,
    "review_eligible": pl.Boolean,
    "embedding_similarity": pl.Float64,
    "mitigation_reason": pl.String,
}

MITIGATION_EVENTS_SCHEMA: dict[str, Any] = {
    "pair_key": pl.String,
    "mitigation_reason": pl.String,
    "evidence": pl.Struct({"embedding_similarity": pl.Float64, "reason_count": pl.Int64}),
    "pre_mitigation_prediction": pl.Boolean,
    "post_mitigation_prediction": pl.Boolean,
}

REMOVED_PAIR_IDS_SCHEMA: dict[str, Any] = {
    "pair_key": pl.String,
    "cleanup_reason": pl.String,
}

PAIR_ID_REMAP_SCHEMA: dict[str, Any] = {
    "old_pair_key": pl.String,
    "new_pair_key": pl.String,
    "reason": pl.String,
}

PAIR_STATE_INDEX_SCHEMA: dict[str, Any] = {
    "pair_key": pl.String,
    "entity_a_id": pl.String,
    "entity_b_id": pl.String,
    "entity_type": pl.String,
    "scope_id": pl.String,
    "retained_flag": pl.Boolean,
}

# ---------------------------------------------------------------------------
# Stage 5 — Cluster Pairs
# ---------------------------------------------------------------------------

CLUSTERS_SCHEMA: dict[str, Any] = {
    "cluster_id": pl.String,
    "entity_type": pl.String,
    "cluster_size": pl.Int64,
    "pair_count": pl.Int64,
    "avg_confidence_score": pl.Float64,
    "max_confidence_score": pl.Float64,
    "min_confidence_score": pl.Float64,
    "objective_score": pl.Float64,
    "positive_edge_sum": pl.Float64,
    "negative_edge_penalty": pl.Float64,
    "cluster_risk_score": pl.Float64,
    "algorithm_version": pl.String,
}

CLUSTER_PAIRS_SCHEMA: dict[str, Any] = {
    "cluster_id": pl.String,
    "pair_key": pl.String,
    "is_reviewed": pl.Boolean,
    "review_decision": pl.Boolean,
    "assignment_score": pl.Float64,
    "assignment_method": pl.String,
}

# ---------------------------------------------------------------------------
# Stage 6 — Materialize Review Queue
# ---------------------------------------------------------------------------

REVIEW_QUEUE_SCHEMA: dict[str, Any] = {
    "pair_key": pl.String,
    "entity_a_id": pl.String,
    "entity_b_id": pl.String,
    "final_score": pl.Float64,
    "cluster_risk_score": pl.Float64,
    "priority_score": pl.Float64,
    "team_id": pl.String,
    "scope_id": pl.String,
    "entity_type": pl.String,
}
