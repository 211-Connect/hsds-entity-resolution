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


def test_chunking_defaults_preserve_unbounded_behavior() -> None:
    """Default chunking config should leave all knobs unset."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="service",
    )
    assert config.chunking.generate_anchor_chunk_size is None
    assert config.chunking.score_candidate_chunk_size is None
    assert config.chunking.max_contact_overlap_pairs_per_anchor is None
    assert config.chunking.max_contact_index_fanout is None


def test_chunking_rejects_zero_sizes() -> None:
    """Chunk sizes and caps must be strictly positive when provided."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="organization",
    ).model_dump()
    payload["chunking"] = {"generate_anchor_chunk_size": 0}

    with pytest.raises(ValueError):
        _ = EntityResolutionRunConfig.model_validate(payload)


def _org_payload() -> dict[str, object]:
    """Return a mutable default organization config payload."""
    return EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="organization",
    ).model_dump()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ml", {"ml_enabled": False}),
        ("calibration", {"enabled": True}),
        ("ml_section_weight", 0.0),
    ],
)
def test_scoring_rejects_removed_ml_keys_with_migration_message(key: str, value: object) -> None:
    """Configs written for the removed ML section fail with a message naming the change."""
    payload = _org_payload()
    scoring = payload["scoring"]
    assert isinstance(scoring, dict)
    scoring[key] = value

    with pytest.raises(ValueError, match=r"ML scoring section was removed in .*1\.2\.0") as info:
        _ = EntityResolutionRunConfig.model_validate(payload)
    assert key in str(info.value)


@pytest.mark.parametrize("key", ["ml_section_weight", "ml_gate_threshold"])
def test_pair_rule_overrides_reject_removed_ml_keys(key: str) -> None:
    """Pair-rule feature overrides carrying ML keys fail with the migration message."""
    payload = _org_payload()
    payload["source_policy"] = {
        "source_profiles": {"P": {"source_schemas": ["S"]}},
        "pair_rules": [
            {
                "rule_id": "r",
                "source_profiles": ["P"],
                "feature_overrides": {
                    "deterministic_section_weight": 0.5,
                    "nlp_section_weight": 0.5,
                    key: 0.0,
                },
            }
        ],
    }

    with pytest.raises(ValueError, match=r"ML scoring section was removed"):
        _ = EntityResolutionRunConfig.model_validate(payload)


def test_pair_rule_section_weights_must_sum_to_one() -> None:
    """Two-section overrides are validated together and must sum to one."""
    payload = _org_payload()
    payload["source_policy"] = {
        "source_profiles": {"P": {"source_schemas": ["S"]}},
        "pair_rules": [
            {
                "rule_id": "r",
                "source_profiles": ["P"],
                "feature_overrides": {
                    "deterministic_section_weight": 0.6,
                    "nlp_section_weight": 0.3,
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="Section weights must sum to 1.0"):
        _ = EntityResolutionRunConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("entity_type", "previous_det", "previous_nlp"),
    [("organization", 0.45, 0.35), ("service", 0.40, 0.40)],
)
def test_default_section_weights_renormalise_previous_active_weights(
    entity_type: str, previous_det: float, previous_nlp: float
) -> None:
    """Defaults equal the pre-1.2.0 det/NLP weights renormalised without ML."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type=entity_type,  # type: ignore[arg-type]
    )
    active = previous_det + previous_nlp
    assert config.scoring.deterministic_section_weight == pytest.approx(previous_det / active)
    assert config.scoring.nlp_section_weight == pytest.approx(previous_nlp / active)
