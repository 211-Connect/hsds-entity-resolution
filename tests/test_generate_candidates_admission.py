"""Tests for candidate admission: embedding floor, Informative Keys, Structural Exclusions."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import polars as pl
import pytest

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.admission import (
    build_informative_key_table,
    is_informative,
    key_values,
)
from hsds_entity_resolution.core.generate_candidates import generate_candidates
from hsds_entity_resolution.core.generate_candidates_sharded import (
    merge_generate_candidates_results,
)
from hsds_entity_resolution.core.pipeline import run_incremental_until_candidates
from hsds_entity_resolution.core.score_candidates import score_candidates

# Cosine of [1, 0] and [0.6, 0.8] is 0.6: below the 0.75 default floor.
_LOW_SIMILARITY = [[1.0, 0.0], [0.6, 0.8]]
# Cosine of [1, 0] and [0.99, 0.01] is ~0.9999: above the floor.
_HIGH_SIMILARITY = [[1.0, 0.0], [0.99, 0.01]]


def _config() -> EntityResolutionRunConfig:
    return EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-admission", scope_id="scope-admission", entity_type="service"
    )


def _services(
    *,
    embeddings: list[list[float]],
    source_schemas: tuple[str, str] = ("SOURCE_A", "SOURCE_A"),
    names: tuple[str, str] = ("Case Management", "Rent Assistance"),
    phones: tuple[list[str], list[str]] = ([], []),
    emails: tuple[list[str], list[str]] = ([], []),
    websites: tuple[list[str], list[str]] = ([], []),
    locations: tuple[list[dict[str, str]], list[dict[str, str]]] = ([], []),
    taxonomies: tuple[list[dict[str, str]], list[dict[str, str]]] = ([], []),
) -> pl.DataFrame:
    """Two service rows; every overlap field is empty unless given."""
    return pl.DataFrame(
        {
            "entity_id": ["svc-a", "svc-b"],
            "entity_type": ["service", "service"],
            "source_schema": list(source_schemas),
            "name": list(names),
            "description": ["Care coordination", "Help with rent"],
            "emails": list(emails),
            "phones": list(phones),
            "websites": list(websites),
            "locations": list(locations),
            "taxonomies": list(taxonomies),
            "identifiers": [[], []],
            "services_rollup": [[], []],
            "organization_name": ["North Org", "South Org"],
            "organization_id": ["org-a", "org-b"],
            "embedding_vector": embeddings,
            "content_hash": ["hash-a", "hash-b"],
        }
    )


def _empty_frame() -> pl.DataFrame:
    return _services(embeddings=_HIGH_SIMILARITY).head(0)


def _changed(*entity_ids: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": list(entity_ids),
            "entity_type": ["service"] * len(entity_ids),
            "delta_class": ["added"] * len(entity_ids),
        }
    )


def _generate(
    services: pl.DataFrame,
    *,
    informative_keys: Mapping[str, Mapping[Any, bool]] | None = None,
    structural_exclusion: Any = None,
    key_corroboration: Any = None,
    config: EntityResolutionRunConfig | None = None,
    anchor: str = "svc-a",
) -> Any:
    return generate_candidates(
        denormalized_organization=_empty_frame(),
        denormalized_service=services,
        changed_entities=_changed(anchor),
        config=config or _config(),
        explicit_backfill=False,
        informative_keys=informative_keys,
        structural_exclusion=structural_exclusion,
        key_corroboration=key_corroboration,
    )


def _only_pair(result: Any) -> dict[str, Any]:
    assert result.candidate_pairs.height == 1
    return result.candidate_pairs.row(0, named=True)


# ---------------------------------------------------------------------------
# Embedding floor
# ---------------------------------------------------------------------------


def test_similarity_at_or_above_the_floor_admits_without_any_shared_key() -> None:
    pair = _only_pair(_generate(_services(embeddings=_HIGH_SIMILARITY)))

    assert pair["blocking_rule_id"] == "embedding_floor"
    assert pair["candidate_reason_codes"] == ["embedding_floor"]


def test_similarity_below_the_floor_with_no_shared_key_is_not_a_candidate() -> None:
    result = _generate(_services(embeddings=_LOW_SIMILARITY))

    assert result.candidate_pairs.is_empty()


def test_shared_taxonomy_or_city_alone_does_not_admit() -> None:
    """Taxonomy and (city, state) are not Informative Keys."""
    result = _generate(
        _services(
            embeddings=_LOW_SIMILARITY,
            taxonomies=([{"code": "BH-1800"}], [{"code": "BH-1800"}]),
            locations=([{"city": "Chicago", "state": "IL"}], [{"city": "Chicago", "state": "IL"}]),
        )
    )

    assert result.candidate_pairs.is_empty()


def test_the_floor_is_capped_per_anchor() -> None:
    config = _config()
    raw = config.model_dump()
    raw["blocking"]["max_candidates_per_entity"] = 1
    capped = EntityResolutionRunConfig.model_validate(raw)
    three = pl.concat(
        [
            _services(embeddings=_HIGH_SIMILARITY),
            _services(embeddings=_HIGH_SIMILARITY)
            .tail(1)
            .with_columns(pl.lit("svc-c").alias("entity_id"), pl.lit("Food Pantry").alias("name")),
        ]
    )

    result = _generate(three, config=capped)

    assert result.candidate_pairs.height == 1


# ---------------------------------------------------------------------------
# Informative Keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("phone", {"phones": (["555-010-0199"], ["555-010-0199"])}),
        ("email", {"emails": (["Intake@Example.org"], ["intake@example.org"])}),
        ("website", {"websites": (["https://example.org/rent"], ["https://example.org/rent"])}),
        (
            "address",
            {
                "locations": (
                    [{"address_1": "123 N Main St", "city": "Chicago", "state": "IL"}],
                    [{"address_1": "123 North Main Street", "city": "Chicago", "state": "IL"}],
                )
            },
        ),
        ("name", {"names": ("Senior Meals", "senior  meals")}),
    ],
)
def test_a_shared_informative_key_admits_below_the_floor(
    field: str, kwargs: dict[str, Any]
) -> None:
    pair = _only_pair(_generate(_services(embeddings=_LOW_SIMILARITY, **kwargs)))

    assert pair["blocking_rule_id"] == f"informative_key:{field}"
    assert pair["candidate_reason_codes"] == [f"informative_key_{field}"]
    assert pair["embedding_similarity"] == pytest.approx(0.6)


def test_a_key_admits_only_when_informative_in_both_schemas() -> None:
    """A service name informative in one schema (program names) and not in the other
    (category labels shared by hundreds of rows) does not admit the pair."""
    services = _services(
        embeddings=_LOW_SIMILARITY,
        source_schemas=("SOURCE_CATEGORY_NAMES", "SOURCE_PROGRAM_NAMES"),
        names=("Food Pantries", "Food Pantries"),
    )
    table = {
        "SOURCE_CATEGORY_NAMES": {"name": False},
        "SOURCE_PROGRAM_NAMES": {"name": True},
    }

    assert _generate(services, informative_keys=table).candidate_pairs.is_empty()
    assert _generate(services).candidate_pairs.height == 1


def test_a_key_not_informative_in_a_schema_does_not_admit_within_it() -> None:
    services = _services(embeddings=_LOW_SIMILARITY, phones=(["5550100199"], ["5550100199"]))

    result = _generate(services, informative_keys={"source_a": {"phone": False}})

    assert result.candidate_pairs.is_empty()


def _three_sharing_a_phone() -> pl.DataFrame:
    third = (
        _services(embeddings=_LOW_SIMILARITY, phones=(["5550100199"], ["5550100199"]))
        .tail(1)
        .with_columns(
            pl.lit("svc-c").alias("entity_id"),
            pl.lit("Food Pantry").alias("name"),
            pl.Series("embedding_vector", [[0.0, 1.0]]),
        )
    )
    return pl.concat(
        [_services(embeddings=_LOW_SIMILARITY, phones=(["5550100199"], ["5550100199"])), third]
    )


def _with_chunking(**chunking: int) -> EntityResolutionRunConfig:
    raw = _config().model_dump()
    raw["chunking"].update(chunking)
    return EntityResolutionRunConfig.model_validate(raw)


def test_key_admission_is_uncapped_by_default() -> None:
    pairs = _generate(_three_sharing_a_phone()).candidate_pairs

    assert pairs.get_column("pair_key").to_list() == ["svc-a__svc-b", "svc-a__svc-c"]


def test_key_fanout_cap_keeps_a_deterministic_sorted_subset() -> None:
    """A value shared by more records than the cap keeps the first ids in sorted order."""
    pairs = _generate(
        _three_sharing_a_phone(), config=_with_chunking(max_contact_index_fanout=2)
    ).candidate_pairs

    assert pairs.get_column("pair_key").to_list() == ["svc-a__svc-b"]


def test_key_pairs_per_anchor_cap_limits_each_anchor() -> None:
    pairs = _generate(
        _three_sharing_a_phone(), config=_with_chunking(max_contact_overlap_pairs_per_anchor=1)
    ).candidate_pairs

    assert pairs.get_column("pair_key").to_list() == ["svc-a__svc-b"]


def test_embedding_and_key_admission_merge_and_the_key_names_the_rule() -> None:
    pair = _only_pair(
        _generate(
            _services(
                embeddings=_HIGH_SIMILARITY,
                phones=(["5550100199"], ["5550100199"]),
                emails=(["a@example.org"], ["a@example.org"]),
            )
        )
    )

    assert pair["blocking_rule_id"] == "informative_key:email+phone"
    assert pair["candidate_reason_codes"] == [
        "embedding_floor",
        "informative_key_email",
        "informative_key_phone",
    ]


# ---------------------------------------------------------------------------
# Key corroboration
# ---------------------------------------------------------------------------


def _reject_names(_a: Mapping[str, Any], _b: Mapping[str, Any], field: str) -> bool:
    return field != "name"


def test_a_shared_key_the_corroboration_rejects_does_not_admit() -> None:
    services = _services(embeddings=_LOW_SIMILARITY, names=("Food Pantry", "Food Pantry"))

    result = _generate(services, key_corroboration=_reject_names)

    assert result.candidate_pairs.is_empty()
    assert _generate(services).candidate_pairs.height == 1


def test_another_shared_key_still_admits_and_names_only_itself() -> None:
    services = _services(
        embeddings=_LOW_SIMILARITY,
        names=("Food Pantry", "Food Pantry"),
        phones=(["5550100199"], ["5550100199"]),
    )

    pair = _only_pair(_generate(services, key_corroboration=_reject_names))

    assert pair["blocking_rule_id"] == "informative_key:phone"
    assert pair["candidate_reason_codes"] == ["informative_key_phone"]


def test_corroboration_never_affects_the_embedding_floor() -> None:
    services = _services(embeddings=_HIGH_SIMILARITY, names=("Food Pantry", "Food Pantry"))

    pair = _only_pair(_generate(services, key_corroboration=_reject_names))

    assert pair["blocking_rule_id"] == "embedding_floor"
    assert pair["candidate_reason_codes"] == ["embedding_floor"]


@pytest.mark.parametrize("anchor", ["svc-a", "svc-b"])
def test_corroboration_sees_the_canonical_pair_whichever_record_is_the_anchor(
    anchor: str,
) -> None:
    seen: list[tuple[str, str, str]] = []

    def record(a: Mapping[str, Any], b: Mapping[str, Any], field: str) -> bool:
        seen.append((str(a["entity_id"]), str(b["entity_id"]), field))
        return True

    services = _services(embeddings=_LOW_SIMILARITY, names=("Food Pantry", "Food Pantry"))
    _generate(services, key_corroboration=record, anchor=anchor)

    assert seen == [("svc-a", "svc-b", "name")]


def test_the_pipeline_passes_the_corroboration_to_admission() -> None:
    services = _services(embeddings=_LOW_SIMILARITY, names=("Food Pantry", "Food Pantry"))

    def candidates(key_corroboration: Any) -> pl.DataFrame:
        return run_incremental_until_candidates(
            organization_entities=pl.DataFrame(),
            service_entities=services,
            previous_entity_index=pl.DataFrame(),
            previous_pair_state_index=pl.DataFrame(),
            config=_config(),
            explicit_backfill=True,
            key_corroboration=key_corroboration,
        ).candidates.candidate_pairs

    assert candidates(None).height == 1
    assert candidates(_reject_names).is_empty()


def test_build_informative_key_table_uses_cutoff_and_overrides() -> None:
    table = build_informative_key_table(
        ratios={
            "category_source": {"name": 0.19, "phone": 0.6, "email": None},
            "program_source": {"name": 0.56, "phone": 0.3},
            "override_source": {"name": 0.12},
        },
        cutoff=0.3,
        overrides={"override_source": {"name": True}, "category_source": {"phone": False}},
    )

    assert table["CATEGORY_SOURCE"] == {
        "name": False,
        "phone": False,
        "email": False,
        "website": False,
        "address": False,
    }
    assert table["PROGRAM_SOURCE"]["name"] is True
    assert table["PROGRAM_SOURCE"]["phone"] is False  # at the cutoff, not above it
    assert table["OVERRIDE_SOURCE"]["name"] is True  # override beats a low ratio


def test_is_informative_defaults_to_true_for_unknown_schemas_and_fields() -> None:
    assert is_informative(None, schema="ANY", field="name")
    assert is_informative({"A": {}}, schema="a", field="phone")
    assert is_informative({}, schema="B", field="email")
    assert not is_informative({"a": {"email": False}}, schema="A", field="email")


def test_key_values_normalise_for_exact_comparison() -> None:
    entity = {
        "name": "  Senior   MEALS ",
        "phones": ["5550100199"],
        "emails": ["A@Example.org"],
        "websites": ["https://example.org"],
        "locations": [
            {
                "address_1": "123 N Main St.",
                "city": "Chicago",
                "state": "IL",
                "postal_code": "60601",
            },
            {"city": "No Street", "state": "IL"},
        ],
    }

    assert key_values(entity, "name") == {"senior meals"}
    assert key_values(entity, "email") == {"a@example.org"}
    assert key_values(entity, "address") == {"123 north main street|chicago|il|60601"}


# ---------------------------------------------------------------------------
# Structural Exclusions
# ---------------------------------------------------------------------------


def _same_site_different_term(a: Mapping[str, Any], b: Mapping[str, Any]) -> str | None:
    """Exclude two records at one address that carry different taxonomy terms."""
    same_site = key_values(a, "address") & key_values(b, "address")
    terms_a = {term["code"] for term in a["taxonomies"]}
    terms_b = {term["code"] for term in b["taxonomies"]}
    if same_site and terms_a and terms_b and not terms_a & terms_b:
        return "same_site_different_term"
    return None


def _same_site_services(*, codes: tuple[str, str]) -> pl.DataFrame:
    site = [{"address_1": "10 Elm St", "city": "Springfield", "state": "IL"}]
    return _services(
        embeddings=_HIGH_SIMILARITY,
        locations=(site, site),
        taxonomies=([{"code": codes[0]}], [{"code": codes[1]}]),
    )


def test_an_exclusion_keeps_a_same_site_different_term_pair_out_and_records_why() -> None:
    result = _generate(
        _same_site_services(codes=("BD-1800", "BH-3800")),
        structural_exclusion=_same_site_different_term,
    )

    assert result.candidate_pairs.is_empty()
    excluded = result.excluded_pairs.row(0, named=True)
    assert excluded["pair_key"] == "svc-a__svc-b"
    assert excluded["exclusion_reason"] == "same_site_different_term"
    assert excluded["source_schema_a"] == "SOURCE_A"


def test_an_exclusion_that_returns_none_admits_as_usual() -> None:
    result = _generate(
        _same_site_services(codes=("BD-1800", "BD-1800")),
        structural_exclusion=_same_site_different_term,
    )

    assert result.candidate_pairs.height == 1
    assert result.excluded_pairs.is_empty()


def test_the_exclusion_sees_the_canonical_entity_order() -> None:
    seen: list[tuple[str, str]] = []

    def record(a: Mapping[str, Any], b: Mapping[str, Any]) -> str | None:
        seen.append((str(a["entity_id"]), str(b["entity_id"])))
        return None

    generate_candidates(
        denormalized_organization=_empty_frame(),
        denormalized_service=_services(embeddings=_HIGH_SIMILARITY),
        changed_entities=_changed("svc-b"),
        config=_config(),
        explicit_backfill=False,
        structural_exclusion=record,
    )

    assert seen == [("svc-a", "svc-b")]


def test_sharded_merge_keeps_excluded_pairs() -> None:
    result = _generate(
        _same_site_services(codes=("BD-1800", "BH-3800")),
        structural_exclusion=_same_site_different_term,
    )

    merged = merge_generate_candidates_results([result, result])

    assert merged.candidate_pairs.is_empty()
    assert merged.excluded_pairs.get_column("pair_key").to_list() == ["svc-a__svc-b"]


# ---------------------------------------------------------------------------
# Scoring is independent of how a pair was admitted
# ---------------------------------------------------------------------------


def test_a_pair_scores_identically_whatever_admitted_it() -> None:
    services = _services(
        embeddings=_HIGH_SIMILARITY,
        phones=(["5550100199"], ["5550100199"]),
        names=("Case Management", "Case Management"),
    )
    candidates = _generate(services).candidate_pairs
    as_before = candidates.with_columns(
        pl.lit("il_lane_rule").alias("blocking_rule_id"),
        pl.lit(["embedding_threshold", "shared_phone"]).alias("candidate_reason_codes"),
    )

    def score(frame: pl.DataFrame) -> dict[str, Any]:
        scored = score_candidates(
            candidate_pairs=frame,
            denormalized_organization=_empty_frame(),
            denormalized_service=services,
            config=_config(),
        ).scored_pairs
        return scored.select(
            "final_score", "pair_outcome", "review_eligible", "policy_rule_id"
        ).row(0, named=True)

    assert score(candidates) == score(as_before)


# ---------------------------------------------------------------------------
# Config migration and diagnostics
# ---------------------------------------------------------------------------


def test_admission_rules_fail_validation_naming_the_change() -> None:
    raw = _config().model_dump()
    raw["source_policy"]["admission_rules"] = []

    with pytest.raises(ValueError, match="ISS-2177.*admission_rules"):
        EntityResolutionRunConfig.model_validate(raw)


def test_overlap_prefilter_channels_fail_validation_naming_the_change() -> None:
    raw = _config().model_dump()
    raw["blocking"]["overlap_prefilter_channels"] = ["email"]

    with pytest.raises(ValueError, match="ISS-2177.*overlap_prefilter_channels"):
        EntityResolutionRunConfig.model_validate(raw)


def test_pair_rules_keep_working_without_admission_rules() -> None:
    raw = _config().model_dump()
    raw["source_policy"] = {
        "source_profiles": {"left": {"source_schemas": ["source_a"]}},
        "pair_rules": [{"rule_id": "left_only", "source_profiles": ["left"]}],
    }

    config = EntityResolutionRunConfig.model_validate(raw)

    assert config.source_policy.pair_rules[0].rule_id == "left_only"


def test_overview_log_reports_floor_keys_and_exclusions(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="hsds_entity_resolution.core.generate_candidates"):
        _generate(
            _same_site_services(codes=("BD-1800", "BH-3800")),
            structural_exclusion=_same_site_different_term,
        )

    overview = next(
        r.message for r in caplog.records if "generate_candidates_overview" in r.message
    )
    assert "floor=0.750" in overview
    assert "excluded=1" in overview
    assert "key_admitted=0" in overview


# ---------------------------------------------------------------------------
# Caller key-value filter
# ---------------------------------------------------------------------------


def _drop_category_names(entity: Mapping[str, Any], field: str, values: set[str]) -> set[str]:
    """A name that is the record's own category label never identifies it."""
    if field != "name":
        return values
    labels = {str(term.get("name", "")).strip().lower() for term in entity["taxonomies"]}
    return values - labels


_PANTRY_TERM = [{"code": "BD-1800", "name": "Food Pantries"}]


def _run_filtered(services: pl.DataFrame, key_filter: Any) -> pl.DataFrame:
    return generate_candidates(
        denormalized_organization=_empty_frame(),
        denormalized_service=services,
        changed_entities=_changed("svc-a"),
        config=_config(),
        explicit_backfill=False,
        key_value_filter=key_filter,
    ).candidate_pairs


def test_a_filtered_key_value_no_longer_admits_but_other_keys_still_do() -> None:
    labelled = _services(
        embeddings=_LOW_SIMILARITY,
        names=("Food Pantries", "Food Pantries"),
        taxonomies=(_PANTRY_TERM, _PANTRY_TERM),
    )
    with_phone = _services(
        embeddings=_LOW_SIMILARITY,
        names=("Food Pantries", "Food Pantries"),
        phones=(["5550100199"], ["5550100199"]),
        taxonomies=(_PANTRY_TERM, _PANTRY_TERM),
    )

    assert _run_filtered(labelled, None).height == 1
    assert _run_filtered(labelled, _drop_category_names).is_empty()
    pair = _run_filtered(with_phone, _drop_category_names).row(0, named=True)
    assert pair["blocking_rule_id"] == "informative_key:phone"


def test_a_key_value_filter_cannot_add_values() -> None:
    def invent(entity: Mapping[str, Any], field: str, values: set[str]) -> set[str]:
        return values | {"invented"}

    assert _run_filtered(_services(embeddings=_LOW_SIMILARITY), invent).is_empty()
