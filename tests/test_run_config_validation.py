"""Validation tests for centralized run-config rules."""

from __future__ import annotations

import pytest

from hsds_entity_resolution.config.entity_resolution_run_config import (
    EntityResolutionRunConfig,
)


def test_weight_sum_validation_rejects_invalid_configuration() -> None:
    """Section weights must sum to approximately one."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="organization",
    ).model_dump()
    payload["scoring"]["deterministic_section_weight"] = 0.9
    payload["scoring"]["nlp_section_weight"] = 0.9
    payload["scoring"]["ml_section_weight"] = 0.9

    with pytest.raises(ValueError, match="Section weights"):
        _ = EntityResolutionRunConfig.model_validate(payload)


def test_overlap_prefilter_channels_reject_unknown_channel() -> None:
    """Blocking config should reject unsupported overlap prefilter channels."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="organization",
    ).model_dump()
    payload["blocking"]["overlap_prefilter_channels"] = ["email", "zipcode"]

    with pytest.raises(ValueError, match="Unsupported overlap prefilter channels"):
        _ = EntityResolutionRunConfig.model_validate(payload)


def test_source_policy_rejects_unknown_profile_reference() -> None:
    """Source policy rules must reference configured abstract profiles."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="service",
    ).model_dump()
    payload["source_policy"]["admission_rules"] = [
        {
            "rule_id": "missing-profile-rule",
            "entity_types": ["service"],
            "source_relation": "same_profile",
            "source_profiles": ["missing"],
            "all_of": ["address_exact"],
        }
    ]

    with pytest.raises(ValueError, match="Unknown source profile"):
        _ = EntityResolutionRunConfig.model_validate(payload)


def test_source_policy_rejects_unknown_signal_suppression() -> None:
    """Feature suppression rules must use known generic signal identifiers."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="service",
    ).model_dump()
    payload["source_policy"]["pair_rules"] = [
        {
            "rule_id": "bad-signal",
            "entity_types": ["service"],
            "feature_overrides": {
                "suppressions": [
                    {"signal": "private_source_fact", "when_all_present": ["shared_taxonomy"]}
                ]
            },
        }
    ]

    with pytest.raises(ValueError, match="Unsupported signal suppression"):
        _ = EntityResolutionRunConfig.model_validate(payload)


def test_source_policy_accepts_embedding_floor_overrides() -> None:
    """Pair rules can define embedding floors and explicit signal exemptions."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="service",
    ).model_dump()
    payload["source_policy"]["source_profiles"] = {
        "PROFILE_SHARED": {"source_schemas": ["SOURCE_A"]}
    }
    payload["source_policy"]["pair_rules"] = [
        {
            "rule_id": "embedding-floor",
            "entity_types": ["service"],
            "source_relation": "same_profile",
            "source_profiles": ["PROFILE_SHARED"],
            "feature_overrides": {
                "min_review_embedding_similarity": 0.76,
                "min_duplicate_embedding_similarity": 0.90,
                "embedding_floor_exempt_signals": ["shared_taxonomy"],
            },
        }
    ]

    config = EntityResolutionRunConfig.model_validate(payload)

    overrides = config.source_policy.pair_rules[0].feature_overrides
    assert overrides.min_review_embedding_similarity == 0.76
    assert overrides.min_duplicate_embedding_similarity == 0.90
    assert overrides.embedding_floor_exempt_signals == ["shared_taxonomy"]


def test_source_policy_rejects_unknown_embedding_floor_exemption() -> None:
    """Embedding floor exemptions must use known evidence signal names."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="service",
    ).model_dump()
    payload["source_policy"]["pair_rules"] = [
        {
            "rule_id": "bad-exemption",
            "entity_types": ["service"],
            "feature_overrides": {
                "embedding_floor_exempt_signals": ["private_source_fact"],
            },
        }
    ]

    with pytest.raises(ValueError, match="Unsupported embedding floor exemption"):
        _ = EntityResolutionRunConfig.model_validate(payload)


def test_source_policy_rejects_duplicate_floor_below_review_floor() -> None:
    """Duplicate embedding floor cannot be less strict than review embedding floor."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="service",
    ).model_dump()
    payload["source_policy"]["pair_rules"] = [
        {
            "rule_id": "bad-floor-order",
            "entity_types": ["service"],
            "feature_overrides": {
                "min_review_embedding_similarity": 0.76,
                "min_duplicate_embedding_similarity": 0.70,
            },
        }
    ]

    with pytest.raises(ValueError, match="min_duplicate_embedding_similarity"):
        _ = EntityResolutionRunConfig.model_validate(payload)


def test_source_policy_accepts_cross_source_same_profile_relation() -> None:
    """Source policy supports cross-source admission within a shared abstract profile."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="service",
    ).model_dump()
    payload["source_policy"]["source_profiles"] = {
        "PROFILE_SHARED": {"source_schemas": ["SOURCE_A", "SOURCE_B"]}
    }
    payload["source_policy"]["admission_rules"] = [
        {
            "rule_id": "shared-profile-cross-source-address",
            "entity_types": ["service"],
            "source_relation": "cross_source_same_profile",
            "source_profiles": ["PROFILE_SHARED"],
            "all_of": ["address_exact"],
        }
    ]
    payload["source_policy"]["pair_rules"] = [
        {
            "rule_id": "shared-profile-cross-source-scoring",
            "entity_types": ["service"],
            "source_relation": "cross_source_same_profile",
            "source_profiles": ["PROFILE_SHARED"],
            "feature_overrides": {"nlp_enabled": False},
        }
    ]

    config = EntityResolutionRunConfig.model_validate(payload)

    assert config.source_policy.admission_rules[0].source_relation == "cross_source_same_profile"
    assert config.source_policy.pair_rules[0].source_relation == "cross_source_same_profile"
