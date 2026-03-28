"""Tests for candidate overlap prefilter behavior."""

from __future__ import annotations

import polars as pl
import pytest

import hsds_entity_resolution.core.generate_candidates as generate_candidates_module
from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.generate_candidates import generate_candidates


def test_generate_candidates_rejects_taxonomy_only_overlap() -> None:
    """Taxonomy-only overlap should not pass without non-taxonomy evidence."""
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

    assert result.candidate_pairs.is_empty()


def test_generate_candidates_keeps_parent_taxonomy_overlap() -> None:
    """Parent-child HSIS taxonomy matches still need non-taxonomy evidence."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-taxonomy-parent",
        scope_id="scope-taxonomy-parent",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[[{"code": "BD"}], [{"code": "B"}]],
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


def test_generate_candidates_rejects_location_only_overlap() -> None:
    """Location-only overlap should not pass without taxonomy evidence."""
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

    assert result.candidate_pairs.is_empty()


def test_generate_candidates_rejects_taxonomy_overlap_for_mixed_shapes() -> None:
    """Legacy taxonomy variants still need non-taxonomy evidence to pass blocking."""
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

    assert result.candidate_pairs.is_empty()


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


def test_generate_candidates_rejects_domain_overlap_for_url_variants_without_taxonomy() -> None:
    """Website overlap alone should not pass when taxonomy is mandatory."""
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

    assert result.candidate_pairs.is_empty()


def test_generate_candidates_rejects_domain_overlap_for_email_and_website() -> None:
    """Domain overlap alone should not pass when taxonomy is mandatory."""
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

    assert result.candidate_pairs.is_empty()


def test_generate_candidates_keeps_exact_taxonomy_plus_email_overlap() -> None:
    """Exact taxonomy plus email overlap should pass the combined gate."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-taxonomy-email",
        scope_id="scope-taxonomy-email",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[[{"code": "BD"}], [{"code": "BD"}]],
        locations=[[], []],
        emails=[["hello@north.org"], ["hello@north.org"]],
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
    assert "shared_email" in reason_codes


def test_generate_candidates_keeps_parent_taxonomy_plus_location_overlap() -> None:
    """Parent-child taxonomy plus location overlap should pass the combined gate."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-taxonomy-location",
        scope_id="scope-taxonomy-location",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[[{"code": "BD"}], [{"code": "B"}]],
        locations=[
            [{"city": "Chicago", "state": "IL"}],
            [{"city": "Chicago", "state": "IL"}],
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
    assert "shared_taxonomy" in reason_codes
    assert "shared_location" in reason_codes


def test_generate_candidates_keeps_taxonomy_plus_phone_overlap() -> None:
    """Taxonomy plus phone overlap should pass the combined gate."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-taxonomy-phone",
        scope_id="scope-taxonomy-phone",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[[{"code": "BD"}], [{"code": "BD"}]],
        locations=[[], []],
        phones=[["555-0100"], ["555-0100"]],
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
    assert "shared_phone" in reason_codes


def test_generate_candidates_rejects_sibling_taxonomy_even_with_email_overlap() -> None:
    """Sibling taxonomy matches should not pass even when another overlap exists."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-taxonomy-sibling",
        scope_id="scope-taxonomy-sibling",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[[{"code": "BD"}], [{"code": "BF"}]],
        locations=[[], []],
        emails=[["hello@north.org"], ["hello@north.org"]],
    )
    result = generate_candidates(
        denormalized_organization=organization_entities,
        denormalized_service=_empty_entity_frame(),
        changed_entities=_changed_entities("organization"),
        config=config,
        explicit_backfill=False,
    )

    assert result.candidate_pairs.is_empty()


def test_generate_candidates_rejects_grandparent_taxonomy_even_with_email_overlap() -> None:
    """Grandparent-child taxonomy matches should not pass under the stricter rule."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-taxonomy-grandparent",
        scope_id="scope-taxonomy-grandparent",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[[{"code": "B"}], [{"code": "BD-1800"}]],
        locations=[[], []],
        emails=[["hello@north.org"], ["hello@north.org"]],
    )
    result = generate_candidates(
        denormalized_organization=organization_entities,
        denormalized_service=_empty_entity_frame(),
        changed_entities=_changed_entities("organization"),
        config=config,
        explicit_backfill=False,
    )

    assert result.candidate_pairs.is_empty()


def test_generate_candidates_keeps_taxonomy_plus_domain_overlap_for_email_and_website() -> None:
    """Taxonomy plus domain overlap should pass even across email-vs-website evidence."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-taxonomy-domain-mixed",
        scope_id="scope-taxonomy-domain-mixed",
        entity_type="organization",
    )
    organization_entities = _organization_frame(
        taxonomies=[[{"code": "BD"}], [{"code": "BD"}]],
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
    assert "shared_taxonomy" in reason_codes
    assert "shared_domain" in reason_codes


def test_generate_candidates_logs_blocking_overview_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate-candidates should emit one info summary for coarse blocking tuning."""

    class _FakeLogger:
        def __init__(self) -> None:
            self.info_messages: list[str] = []
            self.debug_messages: list[str] = []

        def info(self, message: str, *args: object) -> None:
            self.info_messages.append(message % args if args else message)

        def debug(self, message: str, *args: object) -> None:
            self.debug_messages.append(message % args if args else message)

    logger = _FakeLogger()
    monkeypatch.setattr(generate_candidates_module, "get_dagster_logger", lambda: logger)

    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-overview",
        scope_id="scope-overview",
        entity_type="organization",
    ).model_dump()
    payload["blocking"]["max_candidates_per_entity"] = 1
    config = EntityResolutionRunConfig.model_validate(payload)
    organization_entities = pl.DataFrame(
        {
            "entity_id": ["org-a", "org-b", "org-c", "org-d"],
            "entity_type": ["organization"] * 4,
            "source_schema": ["IL211"] * 4,
            "name": ["A", "B", "C", "D"],
            "description": ["", "", "", ""],
            "emails": [
                ["shared@north.org"],
                [],
                ["shared@north.org"],
                [],
            ],
            "phones": [[], [], [], []],
            "websites": [[], [], [], []],
            "locations": [[], [], [], []],
            "taxonomies": [
                [{"code": "BD"}],
                [{"code": "BD"}],
                [{"code": "BD"}],
                [{"code": "ZZ"}],
            ],
            "identifiers": [[], [], [], []],
            "services_rollup": [[], [], [], []],
            "organization_name": ["", "", "", ""],
            "organization_id": ["", "", "", ""],
            "embedding_vector": [
                [1.0, 0.0],
                [0.999, 0.001],
                [0.998, 0.002],
                [0.0, 1.0],
            ],
            "content_hash": ["hash-a", "hash-b", "hash-c", "hash-d"],
        }
    )

    _ = generate_candidates(
        denormalized_organization=organization_entities,
        denormalized_service=_empty_entity_frame(),
        changed_entities=_changed_entities("organization"),
        config=config,
        explicit_backfill=False,
    )

    assert logger.info_messages
    overview_message = logger.info_messages[-1]
    assert overview_message.startswith("ℹ️ generate_candidates_overview")
    assert "threshold=0.750" in overview_message
    assert "max_candidates_per_entity=1" in overview_message
    assert "anchors_at_cap=1 (100.0%)" in overview_message
    assert "heuristic=max_candidates_may_be_low" in overview_message


def _organization_frame(
    *,
    taxonomies: list[object],
    locations: list[object],
    emails: list[object] | None = None,
    phones: list[object] | None = None,
    websites: list[object] | None = None,
    services_rollup: list[object] | None = None,
) -> pl.DataFrame:
    """Build normalized-organization frame with configurable overlap payloads."""
    resolved_emails = emails if emails is not None else [[], []]
    resolved_phones = phones if phones is not None else [[], []]
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
            "phones": resolved_phones,
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
