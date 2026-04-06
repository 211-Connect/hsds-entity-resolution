"""Tests for incremental entity-resolution orchestration."""

from __future__ import annotations

import polars as pl
import pytest

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.pipeline import run_incremental


def test_config_rejects_invalid_threshold_ordering() -> None:
    """Run config must reject duplicate thresholds below maybe threshold."""
    base = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="t1",
        scope_id="s1",
        entity_type="organization",
    ).model_dump()
    base["scoring"]["duplicate_threshold"] = 0.5
    base["scoring"]["maybe_threshold"] = 0.7
    with pytest.raises(ValueError, match="duplicate_threshold"):
        _ = EntityResolutionRunConfig.model_validate(base)


def test_pipeline_emits_canonical_candidate_pairs() -> None:
    """Candidate pairs must be canonicalized as `entity_a_id < entity_b_id`."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1",
        scope_id="scope-1",
        entity_type="organization",
    )
    org = pl.DataFrame(
        {
            "entity_id": ["Org-B", "org-a"],
            "source_schema": ["IL211", "IL211"],
            "name": ["North Clinic", "North Clinic LLC"],
            "description": ["Primary care", "Primary care services"],
            "emails": [["hello@north.org"], ["hello@north.org"]],
            "phones": [["555-0100"], ["555-0100"]],
            "websites": [["north.org"], ["north.org"]],
            "locations": [[], []],
            "taxonomies": [[{"code": "BD"}], [{"code": "BD"}]],
            "identifiers": [[], []],
            "services_rollup": [[], []],
            "embedding_vector": [[0.9, 0.1], [0.92, 0.08]],
        }
    )
    result = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    assert result.candidate_pairs.height == 1
    pair = result.candidate_pairs.row(0, named=True)
    assert pair["entity_a_id"] < pair["entity_b_id"]
    assert pair["pair_key"] == f"{pair['entity_a_id']}__{pair['entity_b_id']}"


def test_pipeline_normalizes_mixed_taxonomy_shapes_to_canonical_contract() -> None:
    """Mixed legacy taxonomy variants should normalize into one canonical row shape."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-taxonomy-shape",
        scope_id="scope-taxonomy-shape",
        entity_type="organization",
    )
    org = pl.DataFrame(
        {
            "entity_id": ["org-a", "org-b"],
            "source_schema": ["IL211", "IL211"],
            "name": ["North Clinic", "North Clinic LLC"],
            "description": ["Primary care", "Primary care services"],
            "emails": [["hello@north.org"], ["hello@north.org"]],
            "phones": [["555-0100"], ["555-0100"]],
            "websites": [["north.org"], ["north.org"]],
            "locations": [[], []],
            "taxonomies": [[{"CODE": "261Q00000X"}], ["261q00000x"]],
            "identifiers": [[], []],
            "services_rollup": [
                [{"name": "Case Management", "taxonomy_codes": ["T1017"]}],
                [{"name": "Case Management", "taxonomy_codes": ["t1017"]}],
            ],
            "embedding_vector": [[0.9, 0.1], [0.92, 0.08]],
        }
    )
    result = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )

    denormalized = result.denormalized_organization.sort("entity_id").to_dicts()
    assert denormalized[0]["taxonomies"] == [{"code": "261q00000x"}]
    assert denormalized[1]["taxonomies"] == [{"code": "261q00000x"}]
    assert denormalized[0]["services_rollup"] == [
        {"name": "case management", "taxonomies": [{"code": "t1017"}]}
    ]
    assert denormalized[1]["services_rollup"] == [
        {"name": "case management", "taxonomies": [{"code": "t1017"}]}
    ]
    reasons = result.candidate_pairs.row(0, named=True)["candidate_reason_codes"]
    assert "shared_taxonomy" in reasons


def test_pipeline_short_circuits_on_no_change() -> None:
    """Pipeline should emit no downstream pairs when entity deltas are empty."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-2",
        scope_id="scope-2",
        entity_type="organization",
    )
    org = pl.DataFrame(
        {
            "entity_id": ["org-a"],
            "source_schema": ["IL211"],
            "name": ["North Clinic"],
            "description": ["Primary care"],
            "emails": [["hello@north.org"]],
            "phones": [["555-0100"]],
            "websites": [["north.org"]],
            "locations": [[]],
            "taxonomies": [[]],
            "identifiers": [[]],
            "services_rollup": [[]],
            "embedding_vector": [[1.0, 0.0]],
        }
    )
    first_run = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    previous_entity_index = first_run.denormalized_organization.select(
        ["entity_id", "entity_type", "content_hash"]
    ).with_columns(pl.lit(True).alias("active_flag"))
    second_run = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=previous_entity_index,
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    summary = second_run.run_summary.row(0, named=True)
    assert summary["no_change"] is True
    assert second_run.candidate_pairs.is_empty()


def test_pipeline_no_change_preserves_previous_pair_state_without_cleanup() -> None:
    """No-change runs must not emit cleanup removals for previously retained pairs."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-2b",
        scope_id="scope-2b",
        entity_type="organization",
    )
    org = pl.DataFrame(
        {
            "entity_id": ["org-a", "org-b"],
            "source_schema": ["IL211", "IL211"],
            "name": ["North Clinic", "North Clinic LLC"],
            "description": ["Primary care", "Primary care services"],
            "emails": [["hello@north.org"], ["hello@north.org"]],
            "phones": [["555-0100"], ["555-0100"]],
            "websites": [["north.org"], ["north.org"]],
            "locations": [[], []],
            "taxonomies": [[{"code": "BD"}], [{"code": "BD"}]],
            "identifiers": [[], []],
            "services_rollup": [[], []],
            "embedding_vector": [[0.9, 0.1], [0.92, 0.08]],
        }
    )
    first_run = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    previous_entity_index = first_run.denormalized_organization.select(
        ["entity_id", "entity_type", "content_hash"]
    ).with_columns(pl.lit(True).alias("active_flag"))
    second_run = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=previous_entity_index,
        previous_pair_state_index=first_run.pair_state_index,
        config=config,
    )
    assert second_run.removed_pair_ids.is_empty()
    assert second_run.pair_state_index.height == first_run.pair_state_index.height
    assert second_run.pair_state_index.select("pair_key").to_series().to_list() == (
        first_run.pair_state_index.select("pair_key").to_series().to_list()
    )


def test_pipeline_backfill_generates_candidates_without_deltas() -> None:
    """Explicit backfill should regenerate candidates even when no entity deltas exist."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-3",
        scope_id="scope-3",
        entity_type="organization",
    )
    org = pl.DataFrame(
        {
            "entity_id": ["org-a", "org-b"],
            "source_schema": ["IL211", "IL211"],
            "name": ["North Clinic", "North Clinic LLC"],
            "description": ["Primary care", "Primary care services"],
            "emails": [["hello@north.org"], ["hello@north.org"]],
            "phones": [["555-0100"], ["555-0100"]],
            "websites": [["north.org"], ["north.org"]],
            "locations": [[], []],
            "taxonomies": [[{"code": "BD"}], [{"code": "BD"}]],
            "identifiers": [[], []],
            "services_rollup": [[], []],
            "embedding_vector": [[0.9, 0.1], [0.92, 0.08]],
        }
    )
    first_run = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    previous_entity_index = first_run.denormalized_organization.select(
        ["entity_id", "entity_type", "content_hash"]
    ).with_columns(pl.lit(True).alias("active_flag"))
    second_run = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=previous_entity_index,
        previous_pair_state_index=first_run.pair_state_index,
        config=config,
        explicit_backfill=True,
    )
    assert second_run.candidate_pairs.height == 1


def test_pipeline_force_rescore_generates_candidates_without_deltas() -> None:
    """Version-triggered rescore should regenerate candidates without entity deltas."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-3b",
        scope_id="scope-3b",
        entity_type="organization",
    )
    org = pl.DataFrame(
        {
            "entity_id": ["org-a", "org-b"],
            "source_schema": ["IL211", "IL211"],
            "name": ["North Clinic", "North Clinic LLC"],
            "description": ["Primary care", "Primary care services"],
            "emails": [["hello@north.org"], ["hello@north.org"]],
            "phones": [["555-0100"], ["555-0100"]],
            "websites": [["north.org"], ["north.org"]],
            "locations": [[], []],
            "taxonomies": [[{"code": "BD"}], [{"code": "BD"}]],
            "identifiers": [[], []],
            "services_rollup": [[], []],
            "embedding_vector": [[0.9, 0.1], [0.92, 0.08]],
        }
    )
    first_run = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    previous_entity_index = first_run.denormalized_organization.select(
        ["entity_id", "entity_type", "content_hash"]
    ).with_columns(pl.lit(True).alias("active_flag"))
    second_run = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=previous_entity_index,
        previous_pair_state_index=first_run.pair_state_index,
        config=config,
        force_rescore=True,
    )
    assert second_run.candidate_pairs.height == 1
    assert second_run.scored_pairs.height == 1
    assert second_run.removed_pair_ids.is_empty()


def test_pipeline_marks_changed_when_non_contact_payload_changes() -> None:
    """Hash should change when non-contact dedupe-relevant payload fields change."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-4",
        scope_id="scope-4",
        entity_type="organization",
    )
    org_day_one = pl.DataFrame(
        {
            "entity_id": ["org-a"],
            "source_schema": ["IL211"],
            "name": ["North Clinic"],
            "description": ["Primary care"],
            "emails": [["hello@north.org"]],
            "phones": [["555-0100"]],
            "websites": [["north.org"]],
            "locations": [[{"city": "Chicago"}]],
            "taxonomies": [[{"code": "261Q00000X"}]],
            "identifiers": [[{"system": "npi", "value": "1234"}]],
            "services_rollup": [[{"name": "Case Management"}]],
            "embedding_vector": [[1.0, 0.0]],
        }
    )
    first_run = run_incremental(
        organization_entities=org_day_one,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    previous_entity_index = first_run.denormalized_organization.select(
        ["entity_id", "entity_type", "content_hash"]
    ).with_columns(pl.lit(True).alias("active_flag"))
    org_day_two = org_day_one.with_columns(
        pl.lit([[{"system": "npi", "value": "9999"}]]).alias("identifiers")
    )
    second_run = run_incremental(
        organization_entities=org_day_two,
        service_entities=pl.DataFrame(),
        previous_entity_index=previous_entity_index,
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    changed = second_run.entity_delta_summary.row(0, named=True)
    assert changed["changed_count"] == 1


def test_pipeline_scope_removed_emits_scope_removed_cleanup_signals() -> None:
    """Scope-removed mode should emit scope_removed for all previously retained pairs."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-5",
        scope_id="scope-5",
        entity_type="organization",
    )
    previous_pair_state_index = pl.DataFrame(
        {
            "pair_key": ["a__b"],
            "entity_a_id": ["a"],
            "entity_b_id": ["b"],
            "entity_type": ["organization"],
            "scope_id": ["scope-5"],
            "retained_flag": [True],
        }
    )
    result = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state_index,
        config=config,
        scope_removed=True,
    )
    assert result.scored_pairs.is_empty()
    assert result.pair_state_index.is_empty()
    assert result.removed_pair_ids.row(0, named=True) == {
        "pair_key": "a__b",
        "cleanup_reason": "scope_removed",
    }
