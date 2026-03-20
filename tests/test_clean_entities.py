"""Tests for the clean_entities stage.

Tests specify correct HSDS entity resolution behavior — they test what the
pipeline *should* do, not merely what it currently does.  Where a test is a
design-contract lock (Tier C) the docstring says so explicitly.
"""

from __future__ import annotations

import polars as pl
import pytest

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.clean_entities import clean_entities
from hsds_entity_resolution.types.contracts import CleanEntitiesResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(*, strict: bool = False) -> EntityResolutionRunConfig:
    """Return a minimal organization config for clean_entities tests.

    strict_validation_mode defaults to True in the base config, so we
    always override it explicitly here to match the requested mode.
    """
    base = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="test-team",
        scope_id="test-scope",
        entity_type="organization",
    )
    data = base.model_dump()
    data["execution"]["strict_validation_mode"] = strict
    return EntityResolutionRunConfig.model_validate(data)


def _org(
    *,
    entity_id: str = "test-org",
    source_schema: str = "TEST",
    name: str = "Test Organization",
    description: str = "A test org",
    emails: list[str] | None = None,
    phones: list[str] | None = None,
    websites: list[str] | None = None,
    locations: list[object] | None = None,
    taxonomies: list[object] | None = None,
    identifiers: list[object] | None = None,
    services_rollup: list[object] | None = None,
    embedding_vector: list[float] | None = None,
) -> pl.DataFrame:
    """Build a one-row organization DataFrame for use in clean_entities tests."""
    return pl.DataFrame(
        {
            "entity_id": [entity_id],
            "source_schema": [source_schema],
            "name": [name],
            "description": [description],
            "emails": [emails if emails is not None else []],
            "phones": [phones if phones is not None else []],
            "websites": [websites if websites is not None else []],
            "locations": [locations if locations is not None else []],
            "taxonomies": [taxonomies if taxonomies is not None else []],
            "identifiers": [identifiers if identifiers is not None else []],
            "services_rollup": [services_rollup if services_rollup is not None else []],
            "embedding_vector": [embedding_vector if embedding_vector is not None else [0.5, 0.5]],
        }
    )


def _run(frame: pl.DataFrame, previous: pl.DataFrame | None = None) -> CleanEntitiesResult:
    """Run clean_entities with an optional previous index."""
    return clean_entities(
        organization_entities=frame,
        service_entities=pl.DataFrame(),
        previous_entity_index=previous if previous is not None else pl.DataFrame(),
        config=_config(),
    )


def _hash_of(result: CleanEntitiesResult, entity_id: str) -> str:
    """Extract content_hash for a given entity_id from a clean result."""
    return result.entity_index.filter(pl.col("entity_id") == entity_id)["content_hash"][0]


# ---------------------------------------------------------------------------
# Content hash — list ordering invariance
# ---------------------------------------------------------------------------


def test_email_list_order_does_not_affect_content_hash() -> None:
    """Emails in different orders must produce the same content hash.

    Hash stability across list orderings is required so that a source-system
    export that changes the order of multi-value fields does not trigger
    unnecessary re-scoring of every pair that involved the entity.
    """
    result_z_first = _run(_org(entity_id="order-test", emails=["zebra@org.org", "alpha@org.org"]))
    result_a_first = _run(_org(entity_id="order-test", emails=["alpha@org.org", "zebra@org.org"]))
    assert _hash_of(result_z_first, "order-test") == _hash_of(result_a_first, "order-test")


def test_phone_list_order_does_not_affect_content_hash() -> None:
    """Phones in different orders must produce the same content hash."""
    result_1 = _run(_org(entity_id="phone-order", phones=["312-555-0101", "773-555-0202"]))
    result_2 = _run(_org(entity_id="phone-order", phones=["773-555-0202", "312-555-0101"]))
    assert _hash_of(result_1, "phone-order") == _hash_of(result_2, "phone-order")


def test_nested_location_list_order_does_not_affect_content_hash() -> None:
    """Location objects in different list positions must produce the same hash."""
    loc_a = {"city": "Chicago", "state": "IL"}
    loc_b = {"city": "Evanston", "state": "IL"}
    result_ab = _run(_org(entity_id="loc-order", locations=[loc_a, loc_b]))
    result_ba = _run(_org(entity_id="loc-order", locations=[loc_b, loc_a]))
    assert _hash_of(result_ab, "loc-order") == _hash_of(result_ba, "loc-order")


# ---------------------------------------------------------------------------
# Content hash — field inclusion / exclusion
# ---------------------------------------------------------------------------


def test_embedding_vector_change_does_not_affect_content_hash() -> None:
    """Changing only embedding_vector must not change the content hash.

    Embeddings are computed from the entity's semantic content; they are not
    part of the entity's identity for incremental change detection.  Including
    the embedding in the hash would cause every model upgrade or re-embed to
    trigger full pair re-scoring.
    """
    result_v1 = _run(_org(entity_id="embed-stable", embedding_vector=[0.9, 0.1]))
    result_v2 = _run(_org(entity_id="embed-stable", embedding_vector=[0.1, 0.9]))
    assert _hash_of(result_v1, "embed-stable") == _hash_of(result_v2, "embed-stable")


def test_embedding_vector_change_produces_unchanged_delta_class() -> None:
    """An entity whose only change is embedding_vector must be classified unchanged.

    Delta class drives which pairs are re-generated.  A change to
    embedding_vector alone must never trigger candidate regeneration.
    """
    run1 = _run(_org(entity_id="embed-delta", embedding_vector=[0.9, 0.1]))
    run2 = _run(
        _org(entity_id="embed-delta", embedding_vector=[0.1, 0.9]),
        previous=run1.entity_index,
    )
    delta_class = run2.changed_entities.filter(pl.col("entity_id") == "embed-delta")["delta_class"][
        0
    ]
    assert delta_class == "unchanged"


def test_name_change_does_affect_content_hash() -> None:
    """A name change must produce a different content hash (sanity check)."""
    result_v1 = _run(_org(entity_id="name-change", name="North Shelter"))
    result_v2 = _run(_org(entity_id="name-change", name="South Shelter"))
    assert _hash_of(result_v1, "name-change") != _hash_of(result_v2, "name-change")


def test_source_schema_is_part_of_content_hash() -> None:
    """Tier C: source_schema IS included in the content hash — this is intentional.

    Each source schema is a distinct data authority.  Two records with the
    same payload but different source schemas represent genuinely different
    provenance and must hash differently.  This ensures pair re-scoring is
    triggered whenever an entity migrates between schemas.

    NOTE: This is a design-contract test.  If this test starts failing, it
    means source_schema was removed from the hash computation in
    clean_entities.py, which is an intentional regression that requires
    deliberate review.
    """
    result_a = _run(_org(entity_id="schema-test", source_schema="SCHEMA_A"))
    result_b = _run(_org(entity_id="schema-test", source_schema="SCHEMA_B"))
    assert _hash_of(result_a, "schema-test") != _hash_of(result_b, "schema-test")


# ---------------------------------------------------------------------------
# Delta classification
# ---------------------------------------------------------------------------


def test_first_run_all_entities_classified_as_added() -> None:
    """With no prior index all entities are classified as added."""
    frame = pl.concat([_org(entity_id="org-a"), _org(entity_id="org-b")], how="diagonal_relaxed")
    result = _run(frame)
    delta_classes = set(result.changed_entities["delta_class"].to_list())
    assert delta_classes == {"added"}


def test_unchanged_entity_is_classified_unchanged() -> None:
    """An entity with the same hash as the prior run must be classified unchanged."""
    run1 = _run(_org(entity_id="stable-org"))
    run2 = _run(_org(entity_id="stable-org"), previous=run1.entity_index)
    delta_class = run2.changed_entities.filter(pl.col("entity_id") == "stable-org")["delta_class"][
        0
    ]
    assert delta_class == "unchanged"


def test_entity_with_changed_payload_is_classified_changed() -> None:
    """An entity whose name changed must be classified as changed."""
    run1 = _run(_org(entity_id="changing-org", name="Original Name"))
    run2 = _run(
        _org(entity_id="changing-org", name="Updated Name"),
        previous=run1.entity_index,
    )
    delta_class = run2.changed_entities.filter(pl.col("entity_id") == "changing-org")[
        "delta_class"
    ][0]
    assert delta_class == "changed"


def test_entity_absent_from_current_run_is_classified_removed() -> None:
    """An entity in the prior index that is missing from the current run must be removed."""
    run1 = _run(
        pl.concat(
            [_org(entity_id="keeper"), _org(entity_id="departing")],
            how="diagonal_relaxed",
        )
    )
    run2 = _run(_org(entity_id="keeper"), previous=run1.entity_index)
    removed_ids = run2.removed_entity_ids["entity_id"].to_list()
    assert "departing" in removed_ids


def test_new_entity_in_second_run_is_classified_added() -> None:
    """An entity that appears in run 2 but not run 1 must be classified as added."""
    run1 = _run(_org(entity_id="existing-org"))
    frame2 = pl.concat(
        [_org(entity_id="existing-org"), _org(entity_id="brand-new")],
        how="diagonal_relaxed",
    )
    run2 = _run(frame2, previous=run1.entity_index)
    delta_class = run2.changed_entities.filter(pl.col("entity_id") == "brand-new")["delta_class"][0]
    assert delta_class == "added"


def test_delta_summary_counts_match_changed_entities() -> None:
    """entity_delta_summary counts must be consistent with changed_entities content."""
    run1 = _run(
        pl.concat(
            [
                _org(entity_id="stable"),
                _org(entity_id="will-change", name="Before"),
                _org(entity_id="will-be-removed"),
            ],
            how="diagonal_relaxed",
        )
    )
    run2_frame = pl.concat(
        [
            _org(entity_id="stable"),
            _org(entity_id="will-change", name="After"),
            _org(entity_id="brand-new"),
        ],
        how="diagonal_relaxed",
    )
    run2 = _run(run2_frame, previous=run1.entity_index)
    summary = run2.entity_delta_summary.row(0, named=True)
    assert summary["added_count"] == 1
    assert summary["changed_count"] == 1
    assert summary["unchanged_count"] == 1
    assert summary["removed_count"] == 1


# ---------------------------------------------------------------------------
# no_change flag
# ---------------------------------------------------------------------------


def test_no_change_flag_true_when_everything_is_unchanged() -> None:
    """no_change must be True when all entities are unchanged and none removed."""
    run1 = _run(_org(entity_id="stable-org"))
    run2 = _run(_org(entity_id="stable-org"), previous=run1.entity_index)
    assert run2.no_change is True


def test_no_change_flag_false_when_entity_is_added() -> None:
    """no_change must be False when any entity is added."""
    run1 = _run(_org(entity_id="existing"))
    run2_frame = pl.concat(
        [_org(entity_id="existing"), _org(entity_id="new-arrival")],
        how="diagonal_relaxed",
    )
    run2 = _run(run2_frame, previous=run1.entity_index)
    assert run2.no_change is False


def test_no_change_flag_false_when_entity_is_changed() -> None:
    """no_change must be False when any entity payload has changed."""
    run1 = _run(_org(entity_id="mutable", name="Old Name"))
    run2 = _run(_org(entity_id="mutable", name="New Name"), previous=run1.entity_index)
    assert run2.no_change is False


def test_no_change_flag_false_when_entity_is_removed() -> None:
    """no_change must be False when any entity was in the prior index but is now absent."""
    run1 = _run(pl.concat([_org(entity_id="a"), _org(entity_id="b")], how="diagonal_relaxed"))
    run2 = _run(_org(entity_id="a"), previous=run1.entity_index)
    assert run2.no_change is False


def test_no_change_flag_true_on_first_run_with_no_prior_index() -> None:
    """no_change is True on a first run with an empty prior index only if no entities."""
    result = _run(pl.DataFrame())
    assert result.no_change is True


# ---------------------------------------------------------------------------
# Embedding validation
# ---------------------------------------------------------------------------


def test_empty_embedding_vector_accepted_in_non_strict_mode() -> None:
    """embedding_vector=[] is accepted in non-strict mode without raising.

    Many HSDS entities are processed before embeddings are computed.  The
    pipeline must not hard-fail on missing embeddings in non-strict mode.
    """
    result = _run(_org(entity_id="no-embed", embedding_vector=[]))
    assert result.entity_index.filter(pl.col("entity_id") == "no-embed").height == 1


def test_empty_embedding_raises_in_strict_mode() -> None:
    """empty embedding_vector must raise in strict validation mode."""
    with pytest.raises(ValueError, match="Embedding structural validation failed"):
        clean_entities(
            organization_entities=_org(entity_id="strict-no-embed", embedding_vector=[]),
            service_entities=pl.DataFrame(),
            previous_entity_index=pl.DataFrame(),
            config=_config(strict=True),
        )


def test_mismatched_embedding_lengths_padded_to_max_in_non_strict_mode() -> None:
    """Entities with shorter embedding vectors are padded to the max length.

    In non-strict mode, inconsistent vector lengths (e.g., a mix of 2D and
    1D embeddings) are resolved by right-padding shorter vectors with 0.0.
    This preserves pipeline stability when a provider returns partial results.
    """
    frame = pl.DataFrame(
        {
            "entity_id": ["long-vec", "short-vec"],
            "source_schema": ["TEST", "TEST"],
            "name": ["Long", "Short"],
            "description": ["", ""],
            "emails": [[], []],
            "phones": [[], []],
            "websites": [[], []],
            "locations": [[], []],
            "taxonomies": [[], []],
            "identifiers": [[], []],
            "services_rollup": [[], []],
            "embedding_vector": [[0.9, 0.1], [0.5]],
        }
    )
    result = clean_entities(
        organization_entities=frame,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        config=_config(strict=False),
    )
    vectors = result.denormalized_organization.select("embedding_vector").to_series().to_list()
    lengths = {len(v) for v in vectors}
    assert len(lengths) == 1, f"Expected uniform vector lengths after padding, got {lengths}"
    assert lengths == {2}


def test_mismatched_embedding_lengths_raise_in_strict_mode() -> None:
    """Inconsistent vector lengths must raise in strict validation mode."""
    frame = pl.DataFrame(
        {
            "entity_id": ["long-vec", "short-vec"],
            "source_schema": ["TEST", "TEST"],
            "name": ["Long", "Short"],
            "description": ["", ""],
            "emails": [[], []],
            "phones": [[], []],
            "websites": [[], []],
            "locations": [[], []],
            "taxonomies": [[], []],
            "identifiers": [[], []],
            "services_rollup": [[], []],
            "embedding_vector": [[0.9, 0.1], [0.5]],
        }
    )
    with pytest.raises(ValueError, match="Embedding structural validation failed"):
        clean_entities(
            organization_entities=frame,
            service_entities=pl.DataFrame(),
            previous_entity_index=pl.DataFrame(),
            config=_config(strict=True),
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_organization_frame_returns_empty_result() -> None:
    """An empty input frame must produce an empty entity index and no deltas."""
    result = _run(pl.DataFrame())
    assert result.entity_index.is_empty()
    summary = result.entity_delta_summary.row(0, named=True)
    assert summary["added_count"] == 0
    assert summary["changed_count"] == 0


def test_case_insensitive_entity_id_normalization() -> None:
    """entity_id must be lowercased for stable, case-insensitive matching."""
    result = _run(_org(entity_id="UpperCase-Org-ID"))
    entity_ids = result.entity_index["entity_id"].to_list()
    assert all(eid == eid.lower() for eid in entity_ids)


def test_whitespace_normalized_in_name() -> None:
    """Extra whitespace in name must be collapsed and trimmed."""
    result = _run(_org(entity_id="ws-org", name="  North   Shelter  Services  "))
    name = result.denormalized_organization["name"][0]
    assert name == "north shelter services"
