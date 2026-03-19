"""Tests for candidate overlap prefilter behavior."""

from __future__ import annotations

import polars as pl

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.generate_candidates import generate_candidates


def test_generate_candidates_keeps_taxonomy_only_overlap() -> None:
    """Taxonomy-only overlap should pass prefilter for candidate retention."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-taxonomy",
        scope_id="scope-taxonomy",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[[{"code": "101-201-301"}], [{"code": "101-201"}]],
        locations=[[], []],
    )
    result = generate_candidates(
        denormalized_organization=organization_entities,
        denormalized_service=_empty_entity_frame(),
        changed_entities=_changed_entities("organization"),
        config=config,
        explicit_backfill=False,
    )

    assert result.candidate_pairs.height == 1
    reason_codes = result.candidate_pairs.row(0, named=True)["candidate_reason_codes"]
    assert "shared_taxonomy" in reason_codes


def test_generate_candidates_keeps_location_only_overlap() -> None:
    """Location-only overlap should pass prefilter for candidate retention."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-location",
        scope_id="scope-location",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[[], []],
        locations=[
            [{"city": "Chicago", "state": "IL"}],
            [{"city": "chicago", "state": "il"}],
        ],
    )
    result = generate_candidates(
        denormalized_organization=organization_entities,
        denormalized_service=_empty_entity_frame(),
        changed_entities=_changed_entities("organization"),
        config=config,
        explicit_backfill=False,
    )

    assert result.candidate_pairs.height == 1
    reason_codes = result.candidate_pairs.row(0, named=True)["candidate_reason_codes"]
    assert "shared_location" in reason_codes


def test_generate_candidates_keeps_taxonomy_overlap_for_mixed_shapes() -> None:
    """Legacy taxonomy variants should normalize into equivalent overlap semantics."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-taxonomy-mixed",
        scope_id="scope-taxonomy-mixed",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[
            [{"CODE": "261Q00000X"}],
            ["261q00000x"],
        ],
        locations=[[], []],
    )
    result = generate_candidates(
        denormalized_organization=organization_entities,
        denormalized_service=_empty_entity_frame(),
        changed_entities=_changed_entities("organization"),
        config=config,
        explicit_backfill=False,
    )

    assert result.candidate_pairs.height == 1
    reason_codes = result.candidate_pairs.row(0, named=True)["candidate_reason_codes"]
    assert "shared_taxonomy" in reason_codes


def test_generate_candidates_respects_configured_overlap_channels() -> None:
    """Configured prefilter channels should control candidate retention behavior."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-config",
        scope_id="scope-config",
        entity_type="organization",
    ).model_dump()
    payload["blocking"]["overlap_prefilter_channels"] = ["email"]
    config = EntityResolutionRunConfig.model_validate(payload)
    organization_entities = _organization_frame(
        taxonomies=[[{"code": "T101"}], [{"code": "T101"}]],
        locations=[[], []],
    )
    result = generate_candidates(
        denormalized_organization=organization_entities,
        denormalized_service=_empty_entity_frame(),
        changed_entities=_changed_entities("organization"),
        config=config,
        explicit_backfill=False,
    )

    assert result.candidate_pairs.is_empty()


def test_generate_candidates_keeps_domain_overlap_for_url_variants() -> None:
    """Website overlap should compare canonical domains across URL variants."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-domain",
        scope_id="scope-domain",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[[], []],
        locations=[[], []],
        websites=[
            ["https://www.alpha.org/about"],
            ["http://alpha.org/contact"],
        ],
    )
    result = generate_candidates(
        denormalized_organization=organization_entities,
        denormalized_service=_empty_entity_frame(),
        changed_entities=_changed_entities("organization"),
        config=config,
        explicit_backfill=False,
    )

    assert result.candidate_pairs.height == 1
    reason_codes = result.candidate_pairs.row(0, named=True)["candidate_reason_codes"]
    assert "shared_domain" in reason_codes


def test_generate_candidates_keeps_domain_overlap_for_email_and_website() -> None:
    """Email and website values should overlap when they share a domain."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-domain-mixed",
        scope_id="scope-domain-mixed",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[[], []],
        locations=[[], []],
        emails=[["hello@alpha.org"], []],
        websites=[[], ["https://www.alpha.org/path"]],
    )
    result = generate_candidates(
        denormalized_organization=organization_entities,
        denormalized_service=_empty_entity_frame(),
        changed_entities=_changed_entities("organization"),
        config=config,
        explicit_backfill=False,
    )

    assert result.candidate_pairs.height == 1
    reason_codes = result.candidate_pairs.row(0, named=True)["candidate_reason_codes"]
    assert "shared_domain" in reason_codes


def _organization_frame(
    *,
    taxonomies: list[object],
    locations: list[object],
    emails: list[object] | None = None,
    websites: list[object] | None = None,
    services_rollup: list[object] | None = None,
) -> pl.DataFrame:
    """Build normalized-organization frame with configurable overlap payloads."""
    resolved_emails = emails if emails is not None else [[], []]
    resolved_websites = websites if websites is not None else [[], []]
    resolved_services_rollup = services_rollup if services_rollup is not None else [[], []]
    return pl.DataFrame(
        {
            "entity_id": ["org-a", "org-b"],
            "entity_type": ["organization", "organization"],
            "source_schema": ["IL211", "IL211"],
            "name": ["North Clinic", "North Clinic LLC"],
            "description": ["Primary care", "Primary care services"],
            "emails": resolved_emails,
            "phones": [[], []],
            "websites": resolved_websites,
            "locations": locations,
            "taxonomies": taxonomies,
            "identifiers": [[], []],
            "services_rollup": resolved_services_rollup,
            "organization_name": ["", ""],
            "organization_id": ["", ""],
            "embedding_vector": [[1.0, 0.0], [0.99, 0.01]],
            "content_hash": ["hash-a", "hash-b"],
        }
    )


def _empty_entity_frame() -> pl.DataFrame:
    """Return empty normalized-entity frame with required schema."""
    return _organization_frame(taxonomies=[[], []], locations=[[], []]).head(0)


def _changed_entities(entity_type: str) -> pl.DataFrame:
    """Return one changed anchor entity for candidate expansion."""
    return pl.DataFrame(
        {
            "entity_id": ["org-a"],
            "entity_type": [entity_type],
            "delta_class": ["added"],
        }
    )
