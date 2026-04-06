"""Tests for ML inference routing in candidate scoring."""

from __future__ import annotations

import polars as pl
import pytest

import hsds_entity_resolution.core.score_candidates as score_candidates_module
from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.ml_inference import to_legacy_entity


def test_score_candidates_uses_model_score_when_available(monkeypatch) -> None:
    """ML section score should use model output when scorer returns a pair score."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-ml",
        scope_id="scope-ml",
        entity_type="organization",
    )
    payload = config.model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = True
    payload["scoring"]["ml"]["ml_gate_threshold"] = 0.0
    config = EntityResolutionRunConfig.model_validate(payload)

    monkeypatch.setattr(
        score_candidates_module,
        "score_pairs_with_model",
        lambda **_: {"org-a__org-b": 0.11},
    )

    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=_normalized_org_rows(
            "Alpha Health",
            "Alpha Health Services",
            include_overlap=False,
            left_locations=[],
            right_locations=[],
            left_identifiers=[],
            right_identifiers=[],
        ),
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    row = result.scored_pairs.row(0, named=True)
    assert row["ml_section_score"] == 0.11
    assert row["embedding_similarity"] == 0.95


def test_score_candidates_falls_back_to_embedding_similarity(monkeypatch) -> None:
    """ML section score should preserve embedding fallback when scorer yields no result."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-ml",
        scope_id="scope-ml",
        entity_type="organization",
    )
    payload = config.model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = True
    payload["scoring"]["ml"]["ml_gate_threshold"] = 0.0
    config = EntityResolutionRunConfig.model_validate(payload)

    monkeypatch.setattr(
        score_candidates_module,
        "score_pairs_with_model",
        lambda **_: {},
    )

    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=_normalized_org_rows(
            "Alpha Health",
            "Alpha Health Services",
            include_overlap=False,
            left_locations=[],
            right_locations=[],
            left_identifiers=[],
            right_identifiers=[],
        ),
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    row = result.scored_pairs.row(0, named=True)
    assert row["ml_section_score"] == 0.95


def test_to_legacy_entity_normalizes_mixed_taxonomy_shapes() -> None:
    """Legacy feature payload should canonicalize mixed taxonomy and rollup variants."""
    legacy = to_legacy_entity(
        row={
            "entity_id": "org-a",
            "name": "Alpha Health",
            "description": "Primary care",
            "emails": ["hello@alpha.org"],
            "phones": ["555-0100"],
            "websites": ["alpha.org"],
            "locations": [{"CITY": "Chicago", "STATE": "IL", "ZIP": "60601"}],
            "taxonomies": [{"CODE": "261Q00000X"}, "261q00000x"],
            "identifiers": [],
            "services_rollup": [
                {"name": "Case Management", "taxonomy_codes": ["T1017"]},
                {"NAME": "Case Management", "taxonomies": [{"code": "t1017-1"}]},
            ],
            "organization_name": "",
            "organization_id": "",
            "embedding_vector": [0.1, 0.2],
        }
    )

    assert legacy["taxonomies"] == [{"code": "261q00000x"}]
    assert legacy["services_rollup"] == [
        {"name": "case management", "taxonomies": ["t1017"]},
        {"name": "case management", "taxonomies": ["t1017-1"]},
    ]


def test_score_candidates_honors_fuzzy_algorithm_setting() -> None:
    """Token-sort algorithm should outperform sequence matcher on word-order swaps."""
    sequence_config = _config_with_nlp_overrides(
        fuzzy_algorithm="sequence_matcher",
        fuzzy_threshold=0.6,
        standalone_fuzzy_threshold=0.7,
    )
    token_sort_config = _config_with_nlp_overrides(
        fuzzy_algorithm="token_sort_ratio",
        fuzzy_threshold=0.6,
        standalone_fuzzy_threshold=0.7,
    )

    sequence_result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=_normalized_org_rows(
            "Alpha Care Center",
            "Center Alpha Care",
            include_overlap=False,
        ),
        denormalized_service=pl.DataFrame(),
        config=sequence_config,
    )
    token_sort_result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=_normalized_org_rows(
            "Alpha Care Center",
            "Center Alpha Care",
            include_overlap=False,
        ),
        denormalized_service=pl.DataFrame(),
        config=token_sort_config,
    )

    assert (
        sequence_result.scored_pairs.row(0, named=True)["nlp_section_score"]
        < (token_sort_result.scored_pairs.row(0, named=True)["nlp_section_score"])
    )


def test_score_candidates_honors_fuzzy_threshold() -> None:
    """NLP contribution should be zero when similarity is below fuzzy threshold."""
    config = _config_with_nlp_overrides(
        fuzzy_algorithm="sequence_matcher",
        fuzzy_threshold=0.98,
        standalone_fuzzy_threshold=0.7,
    )

    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=_normalized_org_rows(
            "Alpha Health",
            "Beta Services",
            include_overlap=False,
        ),
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    assert result.scored_pairs.row(0, named=True)["nlp_section_score"] == 0.0
    assert "name_similarity" not in set(result.pair_reasons.get_column("match_type").to_list())


def test_score_candidates_honors_standalone_threshold() -> None:
    """Standalone fuzzy score must clear standalone threshold when deterministic score is zero."""
    config = _config_with_nlp_overrides(
        fuzzy_algorithm="token_sort_ratio",
        fuzzy_threshold=0.6,
        standalone_fuzzy_threshold=0.95,
    )

    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=_normalized_org_rows(
            "Alpha Care Health",
            "Health Alpha Care Services",
            include_overlap=False,
            left_locations=[],
            right_locations=[],
            left_identifiers=[],
            right_identifiers=[],
        ),
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    assert result.scored_pairs.row(0, named=True)["nlp_section_score"] == 0.0


def test_score_candidates_applies_number_mismatch_safeguard() -> None:
    """Number mismatch veto should zero out NLP when both names contain different numbers."""
    config = _config_with_nlp_overrides(
        fuzzy_algorithm="token_sort_ratio",
        fuzzy_threshold=0.6,
        standalone_fuzzy_threshold=0.7,
    )

    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=_normalized_org_rows(
            "Station 4 Clinic",
            "Station 6 Clinic",
            include_overlap=False,
        ),
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    assert result.scored_pairs.row(0, named=True)["nlp_section_score"] == 0.0


def test_score_candidates_rejects_unknown_fuzzy_algorithm_in_strict_mode() -> None:
    """Unknown algorithm should fail fast under strict validation."""
    config = _config_with_nlp_overrides(fuzzy_algorithm="unknown_algo")

    with pytest.raises(ValueError, match="Unsupported fuzzy_algorithm"):
        _ = score_candidates_module.score_candidates(
            candidate_pairs=_candidate_pairs(),
            denormalized_organization=_normalized_org_rows(
                "Alpha Care",
                "Alpha Care",
                include_overlap=False,
            ),
            denormalized_service=pl.DataFrame(),
            config=config,
        )


def test_score_candidates_emits_shared_address_reason_on_canonical_match() -> None:
    """Address variants that canonicalize equally should emit shared_address."""
    config = _config_with_ml_disabled()
    normalized = _normalized_org_rows(
        include_overlap=False,
        left_locations=[
            {
                "address_1": "123 N. Main St.",
                "city": "Chicago",
                "state": "IL",
                "postal_code": "60601",
            }
        ],
        right_locations=[
            {
                "address_1": "123 north main street",
                "city": " chicago ",
                "state": "il",
                "postal_code": "60601",
            }
        ],
        left_identifiers=[],
        right_identifiers=[],
    )
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=normalized,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    reasons = result.pair_reasons.to_dicts()
    shared_address = next(reason for reason in reasons if reason["match_type"] == "shared_address")
    assert shared_address["matched_value"] == "123 north main street, chicago, il, 60601"
    assert shared_address["entity_a_value"] == "123 north main street, chicago, il, 60601"
    assert shared_address["entity_b_value"] == "123 north main street, chicago, il, 60601"


def test_score_candidates_omits_shared_address_reason_on_mismatch() -> None:
    """Different canonical addresses should not emit shared_address reason rows."""
    config = _config_with_ml_disabled()
    normalized = _normalized_org_rows(
        include_overlap=False,
        left_locations=[
            {
                "address_1": "123 Main Street",
                "city": "Chicago",
                "state": "IL",
                "postal_code": "60601",
            }
        ],
        right_locations=[
            {
                "address_1": "789 Main Street",
                "city": "Chicago",
                "state": "IL",
                "postal_code": "60601",
            }
        ],
        left_identifiers=[],
        right_identifiers=[],
    )
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=normalized,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    reasons = result.pair_reasons.to_dicts()
    assert all(reason["match_type"] != "shared_address" for reason in reasons)


def test_score_candidates_emits_shared_identifier_reason_for_system_value_match() -> None:
    """Identifier match requires exact normalized system+value alignment."""
    config = _config_with_ml_disabled()
    normalized = _normalized_org_rows(
        include_overlap=False,
        left_locations=[],
        right_locations=[],
        left_identifiers=[{"system": "NPI", "value": "123456"}],
        right_identifiers=[{"system": "npi", "value": "123456"}],
    )
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=normalized,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    reasons = result.pair_reasons.to_dicts()
    shared_identifier = next(
        reason for reason in reasons if reason["match_type"] == "shared_identifier"
    )
    assert shared_identifier["matched_value"] == "npi|123456"
    assert shared_identifier["entity_a_value"] == "npi|123456"
    assert shared_identifier["entity_b_value"] == "npi|123456"


def test_score_candidates_emits_name_and_ml_explainability_fields(monkeypatch) -> None:
    """Name and ML reasons should retain the compared values and similarity score."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-ml",
        scope_id="scope-ml",
        entity_type="organization",
    )
    payload = config.model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = True
    payload["scoring"]["ml"]["ml_gate_threshold"] = 0.0
    config = EntityResolutionRunConfig.model_validate(payload)

    monkeypatch.setattr(
        score_candidates_module,
        "compute_nlp_score",
        lambda **_: (0.9, 0.9),
    )

    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=_normalized_org_rows(
            "Alpha Health",
            "Alpha Health Services",
            include_overlap=False,
            left_locations=[],
            right_locations=[],
            left_identifiers=[],
            right_identifiers=[],
        ),
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    reasons = result.pair_reasons.to_dicts()
    name_reason = next(reason for reason in reasons if reason["match_type"] == "name_similarity")
    ml_reason = next(reason for reason in reasons if reason["match_type"] == "ml_similarity")

    assert name_reason["entity_a_value"] == "alpha health"
    assert name_reason["entity_b_value"] == "alpha health services"
    assert name_reason["similarity_score"] == pytest.approx(name_reason["raw_contribution"])
    assert ml_reason["matched_value"] is None
    assert ml_reason["entity_a_value"] is None
    assert ml_reason["entity_b_value"] is None
    assert ml_reason["similarity_score"] == pytest.approx(0.95)


def test_score_candidates_emits_exact_taxonomy_reason_with_full_score() -> None:
    """Exact HSIS taxonomy matches should contribute the full taxonomy score."""
    config = _config_with_taxonomy_only()
    normalized = _normalized_org_rows(
        include_overlap=False,
        left_locations=[],
        right_locations=[],
        left_identifiers=[],
        right_identifiers=[],
        left_taxonomies=[{"code": "BD"}],
        right_taxonomies=[{"code": "BD"}],
    )
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=normalized,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    taxonomy_reason = next(
        reason
        for reason in result.pair_reasons.to_dicts()
        if reason["match_type"] == "shared_taxonomy"
    )
    assert taxonomy_reason["matched_value"] == "bd"
    assert taxonomy_reason["similarity_score"] == pytest.approx(1.0)
    assert result.scored_pairs.row(0, named=True)["deterministic_section_score"] == pytest.approx(
        1.0
    )


def test_score_candidates_emits_parent_taxonomy_reason_with_decay() -> None:
    """Parent-child HSIS taxonomy matches should decay by one level."""
    config = _config_with_taxonomy_only()
    normalized = _normalized_org_rows(
        include_overlap=False,
        left_locations=[],
        right_locations=[],
        left_identifiers=[],
        right_identifiers=[],
        left_taxonomies=[{"code": "BD"}],
        right_taxonomies=[{"code": "B"}],
    )
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=normalized,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    taxonomy_reason = next(
        reason
        for reason in result.pair_reasons.to_dicts()
        if reason["match_type"] == "shared_taxonomy"
    )
    assert taxonomy_reason["matched_value"] == "b"
    assert taxonomy_reason["similarity_score"] == pytest.approx(0.7)
    assert result.scored_pairs.row(0, named=True)["deterministic_section_score"] == pytest.approx(
        0.7
    )


def test_score_candidates_emits_grandparent_taxonomy_reason_with_decay() -> None:
    """Grandparent HSIS taxonomy matches should decay by two levels."""
    config = _config_with_taxonomy_only()
    normalized = _normalized_org_rows(
        include_overlap=False,
        left_locations=[],
        right_locations=[],
        left_identifiers=[],
        right_identifiers=[],
        left_taxonomies=[{"code": "BD-1800"}],
        right_taxonomies=[{"code": "B"}],
    )
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=normalized,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    taxonomy_reason = next(
        reason
        for reason in result.pair_reasons.to_dicts()
        if reason["match_type"] == "shared_taxonomy"
    )
    assert taxonomy_reason["matched_value"] == "b"
    assert taxonomy_reason["similarity_score"] == pytest.approx(0.49)
    assert result.scored_pairs.row(0, named=True)["deterministic_section_score"] == pytest.approx(
        0.49
    )


def test_score_candidates_omits_shared_identifier_when_system_differs() -> None:
    """Matching identifier values with different systems must not count as shared."""
    config = _config_with_ml_disabled()
    normalized = _normalized_org_rows(
        include_overlap=False,
        left_locations=[],
        right_locations=[],
        left_identifiers=[{"system": "npi", "value": "123456"}],
        right_identifiers=[{"system": "ein", "value": "123456"}],
    )
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=normalized,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    reasons = result.pair_reasons.to_dicts()
    assert all(reason["match_type"] != "shared_identifier" for reason in reasons)


def test_score_candidates_shared_address_can_flip_duplicate_prediction() -> None:
    """Address evidence should raise deterministic score enough to flip prediction outcome."""
    config = _config_for_address_flip()
    without_address = _normalized_org_rows(
        include_overlap=False,
        left_locations=[{"address_1": "100 Elm St", "city": "Chicago", "state": "IL"}],
        right_locations=[{"address_1": "900 Pine St", "city": "Chicago", "state": "IL"}],
        left_identifiers=[],
        right_identifiers=[],
    )
    with_address = _normalized_org_rows(
        include_overlap=False,
        left_locations=[{"address_1": "100 Elm St", "city": "Chicago", "state": "IL"}],
        right_locations=[{"address_1": "100 elm street", "city": "chicago", "state": "il"}],
        left_identifiers=[],
        right_identifiers=[],
    )

    without_result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=without_address,
        denormalized_service=pl.DataFrame(),
        config=config,
    )
    with_result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=with_address,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    without_row = without_result.scored_pairs.row(0, named=True)
    with_row = with_result.scored_pairs.row(0, named=True)
    assert without_row["deterministic_section_score"] < with_row["deterministic_section_score"]
    assert without_row["predicted_duplicate"] is False
    assert with_row["predicted_duplicate"] is True


def test_score_candidates_omits_zero_contribution_deterministic_reasons() -> None:
    """Deterministic reason rows should be emitted only when they contribute."""
    config = _config_with_ml_disabled()
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=_normalized_org_rows(
            include_overlap=False,
            left_locations=[],
            right_locations=[],
            left_identifiers=[],
            right_identifiers=[],
        ),
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    match_types = set(result.pair_reasons.get_column("match_type").to_list())
    assert "shared_email" not in match_types
    assert "shared_phone" not in match_types
    assert "shared_domain" not in match_types
    assert "shared_address" not in match_types
    assert "shared_identifier" not in match_types


def test_score_candidates_normalizes_deterministic_score_when_signal_disabled() -> None:
    """Deterministic section must normalize by enabled signal weight sum."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-cal-org",
        scope_id="scope-cal-org",
        entity_type="organization",
    ).model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = False
    payload["scoring"]["deterministic"]["shared_identifier"]["enabled"] = False
    config = EntityResolutionRunConfig.model_validate(payload)

    normalized = _normalized_org_rows(
        include_overlap=False,
        left_phones=["555-0100"],
        right_phones=["555-0100"],
        left_emails=[],
        right_emails=[],
        left_websites=[],
        right_websites=[],
        left_locations=[],
        right_locations=[],
        left_identifiers=[],
        right_identifiers=[],
    )
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=normalized,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    expected = 0.20 / 0.81
    assert result.scored_pairs.row(0, named=True)["deterministic_section_score"] == pytest.approx(
        expected
    )


def test_score_candidates_service_defaults_normalize_full_deterministic_match_to_one() -> None:
    """Service deterministic score should saturate at 1.0 for full overlap under defaults."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-cal-svc",
        scope_id="scope-cal-svc",
        entity_type="service",
    ).model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = False
    config = EntityResolutionRunConfig.model_validate(payload)

    result = score_candidates_module.score_candidates(
        candidate_pairs=_service_candidate_pairs(),
        denormalized_organization=pl.DataFrame(),
        denormalized_service=_normalized_service_rows(include_overlap=True),
        config=config,
    )

    assert result.scored_pairs.row(0, named=True)["deterministic_section_score"] == pytest.approx(
        1.0
    )


def test_score_candidates_service_ignores_identifier_overlap_even_when_present() -> None:
    """Service scoring should not emit shared_identifier from synthetic identifier fixtures."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-svc-identifiers",
        scope_id="scope-svc-identifiers",
        entity_type="service",
    ).model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = False
    config = EntityResolutionRunConfig.model_validate(payload)

    service_rows = _normalized_service_rows(include_overlap=True).with_columns(
        pl.Series(
            "identifiers",
            [
                [{"system": "provider_id", "value": "svc-123"}],
                [{"system": "provider_id", "value": "svc-123"}],
            ],
        )
    )

    result = score_candidates_module.score_candidates(
        candidate_pairs=_service_candidate_pairs(),
        denormalized_organization=pl.DataFrame(),
        denormalized_service=service_rows,
        config=config,
    )

    reasons = result.pair_reasons.to_dicts()
    assert all(reason["match_type"] != "shared_identifier" for reason in reasons)
    assert result.scored_pairs.row(0, named=True)["deterministic_section_score"] == pytest.approx(
        1.0
    )


def test_score_candidates_emits_shared_domain_reason_for_url_variants() -> None:
    """Website URLs should match on normalized domain, not raw string equality."""
    config = _config_with_ml_disabled()
    normalized = _normalized_org_rows(
        include_overlap=False,
        left_emails=[],
        right_emails=[],
        left_websites=["https://www.alpha.org/about"],
        right_websites=["http://alpha.org/contact"],
        left_locations=[],
        right_locations=[],
        left_identifiers=[],
        right_identifiers=[],
    )
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=normalized,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    reasons = result.pair_reasons.to_dicts()
    assert any(reason["match_type"] == "shared_domain" for reason in reasons)


def test_score_candidates_emits_shared_domain_reason_for_email_to_website_match() -> None:
    """Domain overlap should include email-vs-website combinations."""
    config = _config_with_ml_disabled()
    normalized = _normalized_org_rows(
        include_overlap=False,
        left_emails=["hello@alpha.org"],
        right_emails=[],
        left_websites=[],
        right_websites=["https://www.alpha.org/contact"],
        left_locations=[],
        right_locations=[],
        left_identifiers=[],
        right_identifiers=[],
    )
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=normalized,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    reasons = result.pair_reasons.to_dicts()
    assert any(reason["match_type"] == "shared_domain" for reason in reasons)


def test_score_candidates_treats_shared_domain_as_binary_with_partial_overlap() -> None:
    """Shared-domain raw contribution should be 1.0 if any domain overlaps."""
    config = _config_with_ml_disabled()
    normalized = _normalized_org_rows(
        include_overlap=False,
        left_emails=[],
        right_emails=[],
        left_websites=["alpha.org", "alpha-two.org", "alpha-three.org"],
        right_websites=["alpha-three.org", "beta.org", "gamma.org"],
        left_locations=[],
        right_locations=[],
        left_identifiers=[],
        right_identifiers=[],
    )
    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=normalized,
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    domain_reason = next(
        reason
        for reason in result.pair_reasons.to_dicts()
        if reason["match_type"] == "shared_domain"
    )
    assert domain_reason["raw_contribution"] == 1.0


def test_score_candidates_classifies_threshold_bands_and_review_eligibility(
    monkeypatch,
) -> None:
    """Scored rows should classify duplicate/maybe/below-maybe with boundary correctness."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-tier",
        scope_id="scope-tier",
        entity_type="organization",
    ).model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = False
    payload["scoring"]["deterministic_section_weight"] = 1.0
    payload["scoring"]["nlp_section_weight"] = 0.0
    payload["scoring"]["ml_section_weight"] = 0.0
    payload["scoring"]["duplicate_threshold"] = 0.8
    payload["scoring"]["maybe_threshold"] = 0.6
    config = EntityResolutionRunConfig.model_validate(payload)

    score_by_pair = {"a__b": 0.8, "c__d": 0.6, "e__f": 0.59}

    def _fake_pre_score_pair(
        *,
        candidate: dict[str, object],
        entity_lookup: dict[tuple[str, str], dict[str, object]],
        config: EntityResolutionRunConfig,
    ) -> score_candidates_module.PreMlPairRecord:
        _ = entity_lookup
        _ = config
        pair_key = str(candidate["pair_key"])
        score = score_by_pair[pair_key]
        return score_candidates_module.PreMlPairRecord(
            candidate=candidate,
            det_score=score,
            nlp_score=0.0,
            pre_ml_score=score,
            det_reasons=[],
            nlp_reasons=[],
        )

    monkeypatch.setattr(score_candidates_module, "_pre_score_pair", _fake_pre_score_pair)

    result = score_candidates_module.score_candidates(
        candidate_pairs=pl.DataFrame(
            {
                "pair_key": ["a__b", "c__d", "e__f"],
                "entity_a_id": ["a", "c", "e"],
                "entity_b_id": ["b", "d", "f"],
                "entity_type": ["organization", "organization", "organization"],
                "embedding_similarity": [0.8, 0.8, 0.8],
                "candidate_reason_codes": [["embedding_threshold"]] * 3,
                "source_schema_a": ["IL211", "IL211", "IL211"],
                "source_schema_b": ["IL211", "IL211", "IL211"],
            }
        ),
        denormalized_organization=pl.DataFrame(
            {
                "entity_id": ["a", "b", "c", "d", "e", "f"],
                "entity_type": ["organization"] * 6,
                "name": ["A", "B", "C", "D", "E", "F"],
                "description": [""] * 6,
                "emails": [[]] * 6,
                "phones": [[]] * 6,
                "websites": [[]] * 6,
                "locations": [[]] * 6,
                "taxonomies": [[]] * 6,
                "identifiers": [[]] * 6,
                "services_rollup": [[]] * 6,
                "organization_name": [""] * 6,
                "organization_id": [""] * 6,
                "embedding_vector": [[1.0, 0.0]] * 6,
                "source_schema": ["IL211"] * 6,
            }
        ),
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    by_key = {row["pair_key"]: row for row in result.scored_pairs.to_dicts()}
    assert by_key["a__b"]["pair_outcome"] == "duplicate"
    assert by_key["a__b"]["predicted_duplicate"] is True
    assert by_key["a__b"]["review_eligible"] is True
    assert by_key["c__d"]["pair_outcome"] == "maybe"
    assert by_key["c__d"]["predicted_duplicate"] is False
    assert by_key["c__d"]["review_eligible"] is True
    assert by_key["e__f"]["pair_outcome"] == "below_maybe"
    assert by_key["e__f"]["predicted_duplicate"] is False
    assert by_key["e__f"]["review_eligible"] is False
    summary = result.score_delta_summary.row(0, named=True)
    assert summary["duplicate_count"] == 1
    assert summary["maybe_count"] == 1
    assert summary["retained_count"] == 2


def test_shadow_confidence_preserves_legacy_score_and_avoids_trivial_saturation() -> None:
    """Shadow confidence should compress strong evidence while leaving legacy score unchanged."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-shadow",
        scope_id="scope-shadow",
        entity_type="service",
    )
    payload = config.model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = False
    config = EntityResolutionRunConfig.model_validate(payload)

    result = score_candidates_module.score_candidates(
        candidate_pairs=_service_candidate_pairs(),
        denormalized_organization=pl.DataFrame(),
        denormalized_service=_normalized_service_rows(include_overlap=True),
        config=config,
    )

    row = result.scored_pairs.row(0, named=True)
    assert row["final_score"] == pytest.approx(1.0)
    assert row["legacy_confidence_score"] == pytest.approx(row["final_score"])
    assert row["shadow_confidence_score"] < 1.0
    assert row["shadow_confidence_score"] == pytest.approx(0.7369158958)
    assert row["shadow_log_odds"] == pytest.approx(1.03)
    assert row["calibration_version"] == "shadow-log-odds-v1"


def test_shadow_confidence_increases_with_stronger_evidence() -> None:
    """Shadow confidence should increase monotonically as corroborating evidence is added."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-shadow",
        scope_id="scope-shadow",
        entity_type="service",
    )
    payload = config.model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = False
    config = EntityResolutionRunConfig.model_validate(payload)

    weaker = score_candidates_module.score_candidates(
        candidate_pairs=_service_candidate_pairs(),
        denormalized_organization=pl.DataFrame(),
        denormalized_service=_normalized_service_rows(include_overlap=False),
        config=config,
    ).scored_pairs.row(0, named=True)
    stronger = score_candidates_module.score_candidates(
        candidate_pairs=_service_candidate_pairs(),
        denormalized_organization=pl.DataFrame(),
        denormalized_service=_normalized_service_rows(include_overlap=True),
        config=config,
    ).scored_pairs.row(0, named=True)

    assert weaker["legacy_confidence_score"] < stronger["legacy_confidence_score"]
    assert weaker["shadow_confidence_score"] < stronger["shadow_confidence_score"]


def test_shadow_confidence_can_be_disabled_to_match_legacy_score() -> None:
    """Disabled calibration should fall back to the legacy confidence score."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-shadow",
        scope_id="scope-shadow",
        entity_type="organization",
    ).model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = False
    payload["scoring"]["calibration"]["enabled"] = False
    config = EntityResolutionRunConfig.model_validate(payload)

    result = score_candidates_module.score_candidates(
        candidate_pairs=_candidate_pairs(),
        denormalized_organization=_normalized_org_rows(include_overlap=True),
        denormalized_service=pl.DataFrame(),
        config=config,
    )

    row = result.scored_pairs.row(0, named=True)
    assert row["shadow_confidence_score"] == pytest.approx(row["legacy_confidence_score"])


def _config_with_nlp_overrides(**nlp_overrides: float | str) -> EntityResolutionRunConfig:
    """Return default run config with NLP-specific override values."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-nlp",
        scope_id="scope-nlp",
        entity_type="organization",
    ).model_dump()
    for key, value in nlp_overrides.items():
        payload["scoring"]["nlp"][key] = value
    payload["scoring"]["ml"]["ml_enabled"] = False
    return EntityResolutionRunConfig.model_validate(payload)


def _config_with_ml_disabled() -> EntityResolutionRunConfig:
    """Return default run config with ML disabled for deterministic-only tests."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-det",
        scope_id="scope-det",
        entity_type="organization",
    ).model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = False
    return EntityResolutionRunConfig.model_validate(payload)


def _config_for_address_flip() -> EntityResolutionRunConfig:
    """Return config where address signal can independently cross duplicate threshold."""
    payload = _config_with_ml_disabled().model_dump()
    payload["scoring"]["deterministic_section_weight"] = 1.0
    payload["scoring"]["nlp_section_weight"] = 0.0
    payload["scoring"]["ml_section_weight"] = 0.0
    payload["scoring"]["duplicate_threshold"] = 0.5
    payload["scoring"]["maybe_threshold"] = 0.3
    payload["scoring"]["deterministic"]["shared_email"]["weight"] = 0.0
    payload["scoring"]["deterministic"]["shared_phone"]["weight"] = 0.0
    payload["scoring"]["deterministic"]["shared_domain"]["weight"] = 0.0
    payload["scoring"]["deterministic"]["shared_taxonomy"]["weight"] = 0.0
    payload["scoring"]["deterministic"]["shared_identifier"]["weight"] = 0.0
    payload["scoring"]["deterministic"]["shared_address"]["weight"] = 0.6
    return EntityResolutionRunConfig.model_validate(payload)


def _config_with_taxonomy_only() -> EntityResolutionRunConfig:
    """Return config where taxonomy is the only active deterministic signal."""
    payload = _config_with_ml_disabled().model_dump()
    payload["scoring"]["deterministic_section_weight"] = 1.0
    payload["scoring"]["nlp_section_weight"] = 0.0
    payload["scoring"]["ml_section_weight"] = 0.0
    payload["scoring"]["deterministic"]["shared_email"]["weight"] = 0.0
    payload["scoring"]["deterministic"]["shared_phone"]["weight"] = 0.0
    payload["scoring"]["deterministic"]["shared_domain"]["weight"] = 0.0
    payload["scoring"]["deterministic"]["shared_taxonomy"]["weight"] = 0.6
    payload["scoring"]["deterministic"]["shared_address"]["weight"] = 0.0
    payload["scoring"]["deterministic"]["shared_identifier"]["weight"] = 0.0
    return EntityResolutionRunConfig.model_validate(payload)


def _candidate_pairs() -> pl.DataFrame:
    """Return one canonical candidate-pair row."""
    return pl.DataFrame(
        {
            "pair_key": ["org-a__org-b"],
            "entity_a_id": ["org-a"],
            "entity_b_id": ["org-b"],
            "entity_type": ["organization"],
            "embedding_similarity": [0.95],
            "candidate_reason_codes": [["embedding_threshold", "shared_email"]],
            "source_schema_a": ["IL211"],
            "source_schema_b": ["IL211"],
        }
    )


def _normalized_org_rows(
    left_name: str = "Alpha Health",
    right_name: str = "Alpha Health LLC",
    *,
    include_overlap: bool = True,
    left_locations: list[dict[str, str]] | None = None,
    right_locations: list[dict[str, str]] | None = None,
    left_identifiers: list[dict[str, str]] | None = None,
    right_identifiers: list[dict[str, str]] | None = None,
    left_emails: list[str] | None = None,
    right_emails: list[str] | None = None,
    left_phones: list[str] | None = None,
    right_phones: list[str] | None = None,
    left_websites: list[str] | None = None,
    right_websites: list[str] | None = None,
    left_taxonomies: list[dict[str, str]] | None = None,
    right_taxonomies: list[dict[str, str]] | None = None,
) -> pl.DataFrame:
    """Return normalized organization rows with all lookup columns required for scoring."""
    emails = (
        [
            left_emails if left_emails is not None else [],
            right_emails if right_emails is not None else [],
        ]
        if left_emails is not None or right_emails is not None
        else ([["hello@alpha.org"], ["hello@alpha.org"]] if include_overlap else [[], []])
    )
    phones = (
        [
            left_phones if left_phones is not None else [],
            right_phones if right_phones is not None else [],
        ]
        if left_phones is not None or right_phones is not None
        else ([["555-0100"], ["555-0100"]] if include_overlap else [[], []])
    )
    websites = (
        [
            left_websites if left_websites is not None else [],
            right_websites if right_websites is not None else [],
        ]
        if left_websites is not None or right_websites is not None
        else ([["alpha.org"], ["alpha.org"]] if include_overlap else [[], []])
    )
    resolved_locations = [
        left_locations
        if left_locations is not None
        else [{"city": "Chicago", "state": "IL", "postal_code": "60601"}],
        right_locations
        if right_locations is not None
        else [{"city": "Chicago", "state": "IL", "postal_code": "60601"}],
    ]
    resolved_identifiers = [
        left_identifiers if left_identifiers is not None else [{"system": "npi", "value": "123"}],
        right_identifiers if right_identifiers is not None else [{"system": "npi", "value": "123"}],
    ]
    resolved_taxonomies = [
        left_taxonomies
        if left_taxonomies is not None
        else ([{"code": "261Q00000X"}] if include_overlap else []),
        right_taxonomies
        if right_taxonomies is not None
        else ([{"code": "261Q00000X"}] if include_overlap else []),
    ]
    resolved_services_rollup = (
        [
            [{"name": "Case Management", "taxonomies": ["T1017"]}],
            [{"name": "Case Management", "taxonomies": ["T1017"]}],
        ]
        if include_overlap
        else [[], []]
    )
    return pl.DataFrame(
        {
            "entity_id": ["org-a", "org-b"],
            "entity_type": ["organization", "organization"],
            "name": [left_name, right_name],
            "description": ["Primary care", "Primary care services"],
            "emails": emails,
            "phones": phones,
            "websites": websites,
            "locations": resolved_locations,
            "taxonomies": resolved_taxonomies,
            "identifiers": resolved_identifiers,
            "services_rollup": resolved_services_rollup,
            "organization_name": ["", ""],
            "organization_id": ["", ""],
            "embedding_vector": [[0.9, 0.1], [0.92, 0.08]],
            "source_schema": ["IL211", "IL211"],
        }
    )


def _service_candidate_pairs() -> pl.DataFrame:
    """Return one canonical service candidate-pair row."""
    return pl.DataFrame(
        {
            "pair_key": ["svc-a__svc-b"],
            "entity_a_id": ["svc-a"],
            "entity_b_id": ["svc-b"],
            "entity_type": ["service"],
            "embedding_similarity": [0.95],
            "candidate_reason_codes": [["embedding_threshold", "shared_email"]],
            "source_schema_a": ["211HSIS"],
            "source_schema_b": ["211HSIS"],
        }
    )


def _normalized_service_rows(*, include_overlap: bool) -> pl.DataFrame:
    """Return normalized service rows with all lookup columns required for scoring."""
    emails = [["hello@alpha.org"], ["hello@alpha.org"]] if include_overlap else [[], []]
    phones = [["555-0100"], ["555-0100"]] if include_overlap else [[], []]
    websites = [["alpha.org"], ["alpha.org"]] if include_overlap else [[], []]
    locations = (
        [
            [
                {
                    "address_1": "123 Main St",
                    "city": "Chicago",
                    "state": "IL",
                    "postal_code": "60601",
                }
            ],
            [
                {
                    "address_1": "123 Main St",
                    "city": "Chicago",
                    "state": "IL",
                    "postal_code": "60601",
                }
            ],
        ]
        if include_overlap
        else [[], []]
    )
    return pl.DataFrame(
        {
            "entity_id": ["svc-a", "svc-b"],
            "entity_type": ["service", "service"],
            "name": ["Case Management", "Case Management"],
            "description": ["Care coordination service", "Care coordination service"],
            "emails": emails,
            "phones": phones,
            "websites": websites,
            "locations": locations,
            "taxonomies": [[{"code": "T1017"}], [{"code": "T1017"}]],
            # Service rows produced by the DBT-fed runtime path do not currently
            # carry identifiers, so tests should model that default shape unless
            # they are intentionally exercising a synthetic edge case.
            "identifiers": [[], []],
            "services_rollup": [
                [{"name": "Case Management", "taxonomies": ["T1017"]}],
                [{"name": "Case Management", "taxonomies": ["T1017"]}],
            ],
            "organization_name": ["Alpha Org", "Alpha Org"],
            "organization_id": ["org-1", "org-1"],
            "embedding_vector": [[0.9, 0.1], [0.91, 0.09]],
            "source_schema": ["211HSIS", "211HSIS"],
        }
    )
