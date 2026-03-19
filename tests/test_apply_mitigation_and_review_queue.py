"""Coverage-focused tests for mitigation and review-queue stages."""

from __future__ import annotations

import polars as pl

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.apply_mitigation import apply_mitigation
from hsds_entity_resolution.core.cluster_pairs import cluster_pairs
from hsds_entity_resolution.core.materialize_review_queue import materialize_review_queue


def test_apply_mitigation_emits_reconciliation_without_clusters() -> None:
    """Mitigation should emit removals and leave clustering to a dedicated stage."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1",
        scope_id="scope-1",
        entity_type="organization",
    )
    scored_pairs = pl.DataFrame(
        {
            "pair_key": ["a::b", "a::c", "x::y"],
            "entity_a_id": ["a", "a", "x"],
            "entity_b_id": ["b", "c", "y"],
            "entity_type": ["organization", "organization", "organization"],
            "policy_version": ["hsds-er-v1", "hsds-er-v1", "hsds-er-v1"],
            "model_version": ["m", "m", "m"],
            "deterministic_section_score": [0.9, 0.1, 0.9],
            "nlp_section_score": [0.9, 0.1, 0.9],
            "ml_section_score": [0.9, 0.1, 0.9],
            "final_score": [0.95, 0.9, 0.9],
            "predicted_duplicate": [True, True, True],
            "embedding_similarity": [0.95, 0.2, 0.9],
        }
    )
    pair_reasons = pl.DataFrame(
        {
            "pair_key": ["a::b", "x::y"],
            "match_type": ["name_similarity", "name_similarity"],
            "raw_contribution": [0.9, 0.9],
            "weighted_contribution": [0.9, 0.9],
            "signal_weight": [1.0, 1.0],
        }
    )
    removed_entity_ids = pl.DataFrame(
        {
            "entity_id": ["x"],
            "entity_type": ["organization"],
            "cleanup_reason": ["entity_deleted"],
        }
    )

    result = apply_mitigation(
        scored_pairs=scored_pairs,
        pair_reasons=pair_reasons,
        removed_entity_ids=removed_entity_ids,
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )

    removed = {row["pair_key"]: row["cleanup_reason"] for row in result.removed_pair_ids.to_dicts()}
    assert removed["a::c"] == "low_evidence_override"
    assert removed["x::y"] == "entity_deleted"


def test_apply_mitigation_counts_only_contributing_reasons() -> None:
    """Zero-weight reasons must not prevent low-evidence mitigation."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1b",
        scope_id="scope-1b",
        entity_type="organization",
    )
    scored_pairs = pl.DataFrame(
        {
            "pair_key": ["a::b"],
            "entity_a_id": ["a"],
            "entity_b_id": ["b"],
            "entity_type": ["organization"],
            "policy_version": ["hsds-er-v1"],
            "model_version": ["m"],
            "deterministic_section_score": [0.7],
            "nlp_section_score": [0.0],
            "ml_section_score": [None],
            "final_score": [0.9],
            "predicted_duplicate": [True],
            "embedding_similarity": [0.2],
        }
    )
    pair_reasons = pl.DataFrame(
        {
            "pair_key": ["a::b"],
            "match_type": ["name_similarity"],
            "raw_contribution": [0.25],
            "weighted_contribution": [0.0],
            "signal_weight": [1.0],
        }
    )

    result = apply_mitigation(
        scored_pairs=scored_pairs,
        pair_reasons=pair_reasons,
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )

    finalized = result.finalized_scored_pairs.row(0, named=True)
    assert finalized["predicted_duplicate"] is False
    assert finalized["mitigation_reason"] == "low_evidence_override"
    mitigation_event = result.mitigation_events.row(0, named=True)
    assert mitigation_event["evidence"]["reason_count"] == 0


def test_apply_mitigation_keeps_low_embedding_pair_with_contributing_reason() -> None:
    """Positive weighted evidence should satisfy reason-count gate."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1c",
        scope_id="scope-1c",
        entity_type="organization",
    )
    scored_pairs = pl.DataFrame(
        {
            "pair_key": ["a::b"],
            "entity_a_id": ["a"],
            "entity_b_id": ["b"],
            "entity_type": ["organization"],
            "policy_version": ["hsds-er-v1"],
            "model_version": ["m"],
            "deterministic_section_score": [0.7],
            "nlp_section_score": [0.1],
            "ml_section_score": [None],
            "final_score": [0.9],
            "predicted_duplicate": [True],
            "embedding_similarity": [0.2],
        }
    )
    pair_reasons = pl.DataFrame(
        {
            "pair_key": ["a::b"],
            "match_type": ["shared_phone"],
            "raw_contribution": [1.0],
            "weighted_contribution": [0.2],
            "signal_weight": [0.2],
        }
    )

    result = apply_mitigation(
        scored_pairs=scored_pairs,
        pair_reasons=pair_reasons,
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )

    finalized = result.finalized_scored_pairs.row(0, named=True)
    assert finalized["predicted_duplicate"] is True
    assert finalized["mitigation_reason"] is None
    assert result.mitigation_events.is_empty()


def test_cluster_pairs_emits_stable_membership_hash_ids() -> None:
    """Correlation clustering should emit deterministic stable IDs from member sets."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1a",
        scope_id="scope-1a",
        entity_type="organization",
    )
    finalized = pl.DataFrame(
        {
            "pair_key": ["a::b", "b::c", "a::c"],
            "entity_a_id": ["a", "b", "a"],
            "entity_b_id": ["b", "c", "c"],
            "entity_type": ["organization", "organization", "organization"],
            "policy_version": ["hsds-er-v1", "hsds-er-v1", "hsds-er-v1"],
            "model_version": ["m", "m", "m"],
            "deterministic_section_score": [0.8, 0.8, 0.2],
            "nlp_section_score": [0.8, 0.8, 0.2],
            "ml_section_score": [0.8, 0.8, 0.2],
            "final_score": [0.95, 0.94, 0.10],
            "predicted_duplicate": [True, True, False],
            "embedding_similarity": [0.95, 0.94, 0.10],
            "mitigation_reason": [None, None, None],
        }
    )
    clustered_first = cluster_pairs(
        finalized_scored_pairs=finalized,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )
    clustered_second = cluster_pairs(
        finalized_scored_pairs=finalized,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )
    assert clustered_first.clusters.height == 1
    assert clustered_second.clusters.height == 1
    first_id = clustered_first.clusters.row(0, named=True)["cluster_id"]
    second_id = clustered_second.clusters.row(0, named=True)["cluster_id"]
    assert first_id == second_id
    assert str(first_id).startswith("ccv1::")


def test_materialize_review_queue_filters_and_orders_pairs() -> None:
    """Review queue should exclude removed keys and sort by priority score."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-2",
        scope_id="scope-2",
        entity_type="organization",
    )
    finalized = pl.DataFrame(
        {
            "pair_key": ["a::b", "b::c", "d::e"],
            "entity_a_id": ["a", "b", "d"],
            "entity_b_id": ["b", "c", "e"],
            "predicted_duplicate": [True, True, False],
            "final_score": [0.85, 0.95, 0.99],
            "embedding_similarity": [0.8, 0.9, 0.95],
        }
    )
    removed = pl.DataFrame({"pair_key": ["a::b"], "cleanup_reason": ["entity_deleted"]})

    clusters = pl.DataFrame(
        {
            "cluster_id": ["cluster-1"],
            "entity_type": ["organization"],
            "cluster_size": [2],
            "pair_count": [1],
            "avg_confidence_score": [0.95],
            "max_confidence_score": [0.95],
            "min_confidence_score": [0.95],
            "objective_score": [0.5],
            "positive_edge_sum": [0.5],
            "negative_edge_penalty": [0.0],
            "cluster_risk_score": [0.0],
            "algorithm_version": ["correlative_greedy_v1"],
        }
    )
    cluster_pairs_frame = pl.DataFrame(
        {
            "cluster_id": ["cluster-1"],
            "pair_key": ["b::c"],
            "is_reviewed": [None],
            "review_decision": [None],
            "assignment_score": [0.5],
            "assignment_method": ["correlative_greedy_v1"],
        }
    )
    result = materialize_review_queue(
        finalized_scored_pairs=finalized,
        removed_pair_ids=removed,
        clusters=clusters,
        cluster_pairs=cluster_pairs_frame,
        config=config,
    )

    assert result.review_queue_items.height == 1
    queue_row = result.review_queue_items.row(0, named=True)
    assert queue_row["pair_key"] == "b::c"
    assert queue_row["team_id"] == "team-2"


def test_materialize_review_queue_includes_maybe_tier_and_excludes_below_maybe() -> None:
    """Review queue eligibility should include duplicate+maybe tiers at threshold boundaries."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-2b",
        scope_id="scope-2b",
        entity_type="organization",
    )
    finalized = pl.DataFrame(
        {
            "pair_key": ["dup::edge", "maybe::edge", "below::edge"],
            "entity_a_id": ["a", "c", "e"],
            "entity_b_id": ["b", "d", "f"],
            "predicted_duplicate": [True, False, False],
            "pair_outcome": ["duplicate", "maybe", "below_maybe"],
            "review_eligible": [True, True, False],
            "final_score": [
                config.scoring.duplicate_threshold,
                config.scoring.maybe_threshold,
                config.scoring.maybe_threshold - 0.01,
            ],
            "embedding_similarity": [0.9, 0.8, 0.8],
        }
    )

    result = materialize_review_queue(
        finalized_scored_pairs=finalized,
        removed_pair_ids=pl.DataFrame(),
        clusters=pl.DataFrame(),
        cluster_pairs=pl.DataFrame(),
        config=config,
    )

    queue_keys = set(result.review_queue_items.get_column("pair_key").to_list())
    assert queue_keys == {"dup::edge", "maybe::edge"}


def test_apply_mitigation_emits_candidate_lost_from_previous_state() -> None:
    """Previously retained pairs missing from current run should be marked candidate_lost."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-3",
        scope_id="scope-3",
        entity_type="organization",
    )
    previous_pair_state_index = pl.DataFrame(
        {
            "pair_key": ["a::b"],
            "entity_a_id": ["a"],
            "entity_b_id": ["b"],
            "entity_type": ["organization"],
            "scope_id": ["scope-3"],
            "retained_flag": [True],
        }
    )
    result = apply_mitigation(
        scored_pairs=pl.DataFrame(),
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state_index,
        config=config,
    )
    assert result.removed_pair_ids.height == 1
    assert result.removed_pair_ids.row(0, named=True) == {
        "pair_key": "a::b",
        "cleanup_reason": "candidate_lost",
    }


def test_apply_mitigation_emits_score_dropped_from_previous_state() -> None:
    """Previously retained pairs scored below duplicate threshold should be marked score_dropped."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-4",
        scope_id="scope-4",
        entity_type="organization",
    )
    scored_pairs = pl.DataFrame(
        {
            "pair_key": ["a::b"],
            "entity_a_id": ["a"],
            "entity_b_id": ["b"],
            "entity_type": ["organization"],
            "policy_version": ["hsds-er-v1"],
            "model_version": ["m"],
            "deterministic_section_score": [0.1],
            "nlp_section_score": [0.1],
            "ml_section_score": [None],
            "final_score": [0.2],
            "predicted_duplicate": [False],
            "embedding_similarity": [0.1],
        }
    )
    previous_pair_state_index = pl.DataFrame(
        {
            "pair_key": ["a::b"],
            "entity_a_id": ["a"],
            "entity_b_id": ["b"],
            "entity_type": ["organization"],
            "scope_id": ["scope-4"],
            "retained_flag": [True],
        }
    )
    result = apply_mitigation(
        scored_pairs=scored_pairs,
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state_index,
        config=config,
    )
    removed = {row["pair_key"]: row["cleanup_reason"] for row in result.removed_pair_ids.to_dicts()}
    assert removed["a::b"] == "score_dropped"


def test_apply_mitigation_emits_scope_removed_when_scope_is_decommissioned() -> None:
    """Scope cleanup mode should classify all previously retained pairs as scope_removed."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-5",
        scope_id="scope-5",
        entity_type="organization",
    )
    previous_pair_state_index = pl.DataFrame(
        {
            "pair_key": ["a::b", "c::d"],
            "entity_a_id": ["a", "c"],
            "entity_b_id": ["b", "d"],
            "entity_type": ["organization", "organization"],
            "scope_id": ["scope-5", "scope-5"],
            "retained_flag": [True, True],
        }
    )
    result = apply_mitigation(
        scored_pairs=pl.DataFrame(),
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state_index,
        config=config,
        scope_removed=True,
    )
    removed = {row["pair_key"]: row["cleanup_reason"] for row in result.removed_pair_ids.to_dicts()}
    assert removed["a::b"] == "scope_removed"
    assert removed["c::d"] == "scope_removed"
