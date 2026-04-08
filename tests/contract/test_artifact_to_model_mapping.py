"""Contract tests for mapping component artifacts to destination row models."""

from __future__ import annotations

import polars as pl
from consumer.consumer_adapter.artifact_contracts import assert_valid_artifacts
from consumer.consumer_adapter.mapper import (
    ConsumerRunContext,
    _map_deduplication_run,
    _map_duplicate_reasons,
    map_artifacts_to_tables,
)
from consumer.consumer_adapter.row_model_validation import validate_table_rows

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.pipeline import run_incremental


def test_artifacts_map_cleanly_to_runtime_models() -> None:
    """Mapped rows should satisfy all runtime row-model contracts."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1",
        scope_id="scope-1",
        entity_type="organization",
    )
    org = pl.DataFrame(
        {
            "entity_id": ["org-a", "org-b"],
            "source_schema": ["TENANT_A", "TENANT_B"],
            "name": ["Alpha Clinic", "Alpha Clinic"],
            "description": ["A", "A"],
            "emails": [["one@example.org"], ["one@example.org"]],
            "phones": [["555-0000"], ["555-0000"]],
            "websites": [["example.org"], ["example.org"]],
            "locations": [[], []],
            "taxonomies": [
                [
                    {
                        "taxonomy_term_id": "tax-org-1",
                        "code": "BD-1800",
                        "name": "Health",
                        "description": "Health support",
                    }
                ],
                [
                    {
                        "taxonomy_term_id": "tax-org-1",
                        "code": "BD-1800",
                        "name": "Health",
                        "description": "Health support",
                    }
                ],
            ],
            "identifiers": [[], []],
            "services_rollup": [
                [
                    {
                        "id": "svc-1",
                        "name": "Case Management",
                        "description": "Navigation support",
                        "taxonomy_codes": [
                            {
                                "taxonomy_term_id": "tax-svc-1",
                                "code": "PH-0300",
                                "name": "Case Management",
                                "description": "Case Management",
                            }
                        ],
                    }
                ],
                [
                    {
                        "id": "svc-1",
                        "name": "Case Management",
                        "description": "Navigation support",
                        "taxonomy_codes": [
                            {
                                "taxonomy_term_id": "tax-svc-1",
                                "code": "PH-0300",
                                "name": "Case Management",
                                "description": "Case Management",
                            }
                        ],
                    }
                ],
            ],
            "embedding_vector": [[1.0, 0.0], [0.99, 0.01]],
        }
    )
    result = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    artifacts = {
        "denormalized_organization": result.denormalized_organization,
        "denormalized_service": result.denormalized_service,
        "entity_delta_summary": result.entity_delta_summary,
        "removed_entity_ids": result.removed_entity_ids,
        "candidate_pairs": result.candidate_pairs,
        "scored_pairs": result.scored_pairs,
        "pair_reasons": result.pair_reasons,
        "mitigation_events": result.mitigation_events,
        "removed_pair_ids": result.removed_pair_ids,
        "pair_id_remap": result.pair_id_remap,
        "clusters": result.clusters,
        "cluster_pairs": result.cluster_pairs,
        "pair_state_index": result.pair_state_index,
        "review_queue_items": result.review_queue_items,
        "run_summary": result.run_summary,
    }
    assert_valid_artifacts(artifacts)
    mapped = map_artifacts_to_tables(
        artifacts=artifacts,
        context=ConsumerRunContext(
            run_id="run-1",
            team_id="team-1",
            scope_id="scope-1",
            target_schemas=["NE211", "DUPAGEC211"],
            entity_type="organization",
            job_name="fixture_job",
            config=config,
        ),
    )
    validation = validate_table_rows(mapped.table_frames)
    assert validation.is_valid
    assert mapped.table_frames["DUPLICATE_PAIRS"].height == 1
    assert mapped.table_frames["DUPLICATE_PAIR_SCORES"].height == 1


def test_service_artifacts_with_taxonomy_term_ids_map_cleanly_to_runtime_models() -> None:
    """Service cache rows must retain taxonomy_term_id after normalization."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1",
        scope_id="scope-1",
        entity_type="service",
    )
    service = pl.DataFrame(
        {
            "entity_id": ["svc-a", "svc-b"],
            "source_schema": ["TENANT_A", "TENANT_B"],
            "name": ["Alpha Clinic", "Alpha Clinic"],
            "description": ["A", "A"],
            "emails": [["one@example.org"], ["one@example.org"]],
            "phones": [["555-0000"], ["555-0000"]],
            "websites": [["example.org"], ["example.org"]],
            "locations": [[], []],
            "taxonomies": [
                [
                    {
                        "taxonomy_term_id": "tax-svc-1",
                        "code": "PH-0300",
                        "name": "Case Management",
                        "description": "Case Management",
                    }
                ],
                [
                    {
                        "taxonomy_term_id": "tax-svc-1",
                        "code": "PH-0300",
                        "name": "Case Management",
                        "description": "Case Management",
                    }
                ],
            ],
            "organization_name": ["Org", "Org"],
            "organization_id": ["org-1", "org-1"],
            "embedding_vector": [[1.0, 0.0], [0.99, 0.01]],
        }
    )
    result = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=service,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    artifacts = {
        "denormalized_organization": result.denormalized_organization,
        "denormalized_service": result.denormalized_service,
        "entity_delta_summary": result.entity_delta_summary,
        "removed_entity_ids": result.removed_entity_ids,
        "candidate_pairs": result.candidate_pairs,
        "scored_pairs": result.scored_pairs,
        "pair_reasons": result.pair_reasons,
        "mitigation_events": result.mitigation_events,
        "removed_pair_ids": result.removed_pair_ids,
        "pair_id_remap": result.pair_id_remap,
        "clusters": result.clusters,
        "cluster_pairs": result.cluster_pairs,
        "pair_state_index": result.pair_state_index,
        "review_queue_items": result.review_queue_items,
        "run_summary": result.run_summary,
    }
    assert_valid_artifacts(artifacts)
    mapped = map_artifacts_to_tables(
        artifacts=artifacts,
        context=ConsumerRunContext(
            run_id="run-1",
            team_id="team-1",
            scope_id="scope-1",
            target_schemas=["NE211", "DUPAGEC211"],
            entity_type="service",
            job_name="fixture_job",
            config=config,
        ),
    )
    validation = validate_table_rows(mapped.table_frames)
    assert validation.is_valid
    service_rows = mapped.table_frames["DENORMALIZED_SERVICE_CACHE"].to_dicts()
    assert service_rows[0]["TAXONOMIES"][0]["taxonomy_term_id"] == "tax-svc-1"


def test_deduplication_run_mapping_uses_duplicate_count_for_threshold_metrics() -> None:
    """Threshold metrics should remain duplicate-only even when maybe-tier retention is higher."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1",
        scope_id="scope-1",
        entity_type="organization",
    )
    run_summary = pl.DataFrame(
        {
            "candidate_count": [9],
            "duplicate_count": [2],
            "maybe_count": [3],
            "retained_count": [5],
        }
    )
    mapped = _map_deduplication_run(
        run_summary=run_summary,
        context=ConsumerRunContext(
            run_id="run-1",
            team_id="team-1",
            scope_id="scope-1",
            target_schemas=["NE211", "DUPAGEC211"],
            entity_type="organization",
            job_name="il211_regional",
            config=config,
        ),
        now_value="2026-03-06T00:00:00+00:00",
    )
    row = mapped.row(0, named=True)
    assert row["JOB_NAME"] == "il211_regional"
    assert row["TARGET_SCHEMAS"] == ["NE211", "DUPAGEC211"]
    assert row["PAIRS_ABOVE_THRESHOLD"] == 2
    assert row["PAIRS_PREDICTED_DUPLICATE"] == 2


def test_duplicate_reason_ids_are_scoped_to_the_score_row() -> None:
    """Same pair reason across different runs should map to different score-scoped IDs."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1",
        scope_id="scope-1",
        entity_type="service",
    )
    pair_reasons = pl.DataFrame(
        {
            "pair_key": ["pair-1"],
            "match_type": ["shared_email"],
            "raw_contribution": [1.0],
            "weighted_contribution": [0.16],
            "signal_weight": [0.16],
            "matched_value": ["alpha@example.org"],
            "entity_a_value": ["alpha@example.org"],
            "entity_b_value": ["alpha@example.org"],
            "similarity_score": [None],
        }
    )
    scored_pairs = pl.DataFrame(
        {
            "pair_key": ["pair-1"],
            "entity_type": ["service"],
        }
    )
    first = _map_duplicate_reasons(
        pair_reasons=pair_reasons,
        scored_pairs=scored_pairs,
        context=ConsumerRunContext(
            run_id="run-1",
            team_id="team-1",
            scope_id="scope-1",
            target_schemas=["TENANT_A"],
            entity_type="service",
            job_name="fixture_job",
            config=config,
        ),
        created_at="2026-03-27T00:00:00+00:00",
    )
    second = _map_duplicate_reasons(
        pair_reasons=pair_reasons,
        scored_pairs=scored_pairs,
        context=ConsumerRunContext(
            run_id="run-2",
            team_id="team-1",
            scope_id="scope-1",
            target_schemas=["TENANT_A"],
            entity_type="service",
            job_name="fixture_job",
            config=config,
        ),
        created_at="2026-03-27T01:00:00+00:00",
    )

    first_row = first.row(0, named=True)
    second_row = second.row(0, named=True)
    assert first_row["ID"] != second_row["ID"]
    assert first_row["DUPLICATE_PAIR_SCORE_ID"] != second_row["DUPLICATE_PAIR_SCORE_ID"]
    assert first_row["DEDUPLICATION_RUN_ID"] == "run-1"
    assert second_row["DEDUPLICATION_RUN_ID"] == "run-2"


def test_duplicate_reason_ids_ignore_mutable_scores_within_one_score_row() -> None:
    """Score-local value changes should keep the same score-scoped reason ID."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1",
        scope_id="scope-1",
        entity_type="service",
    )
    scored_pairs = pl.DataFrame(
        {
            "pair_key": ["pair-1"],
            "entity_type": ["service"],
        }
    )
    first = _map_duplicate_reasons(
        pair_reasons=pl.DataFrame(
            {
                "pair_key": ["pair-1"],
                "match_type": ["name_similarity"],
                "raw_contribution": [1.0],
                "weighted_contribution": [1.0],
                "signal_weight": [1.0],
                "matched_value": ["teen pregnancy prevention"],
                "entity_a_value": ["teen pregnancy prevention"],
                "entity_b_value": ["teen pregnancy prevention"],
                "similarity_score": [1.0],
            }
        ),
        scored_pairs=scored_pairs,
        context=ConsumerRunContext(
            run_id="run-1",
            team_id="team-1",
            scope_id="scope-1",
            target_schemas=["TENANT_A"],
            entity_type="service",
            job_name="fixture_job",
            config=config,
        ),
        created_at="2026-03-27T00:00:00+00:00",
    )
    second = _map_duplicate_reasons(
        pair_reasons=pl.DataFrame(
            {
                "pair_key": ["pair-1"],
                "match_type": ["name_similarity"],
                "raw_contribution": [0.92],
                "weighted_contribution": [0.92],
                "signal_weight": [0.85],
                "matched_value": ["teen pregnancy prevention"],
                "entity_a_value": ["teen pregnancy prevention"],
                "entity_b_value": ["teen pregnancy prevention"],
                "similarity_score": [0.92],
            }
        ),
        scored_pairs=scored_pairs,
        context=ConsumerRunContext(
            run_id="run-1",
            team_id="team-1",
            scope_id="scope-1",
            target_schemas=["TENANT_A"],
            entity_type="service",
            job_name="fixture_job",
            config=config,
        ),
        created_at="2026-03-27T01:00:00+00:00",
    )

    first_row = first.row(0, named=True)
    second_row = second.row(0, named=True)
    assert first_row["ID"] == second_row["ID"]
    assert second_row["RAW_CONTRIBUTION"] == 0.92
    assert second_row["WEIGHTED_CONTRIBUTION"] == 0.92
    assert second_row["SIGNAL_WEIGHT"] == 0.85
    assert second_row["SIMILARITY_SCORE"] == 0.92
