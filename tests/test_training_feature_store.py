"""Tests for offline training feature materialization helpers."""

from __future__ import annotations

from hsds_entity_resolution.core.training_feature_store import _build_feature_row
from hsds_entity_resolution.core.training_features import FEATURE_SCHEMA_VERSION


class _Extractor:
    """Minimal extractor stub for materialization tests."""

    def extract_features(
        self, entity_a: dict[str, object], entity_b: dict[str, object]
    ) -> dict[str, float]:
        return {
            "name_token_sort": 0.88,
            "bigram_overlap": 0.51,
            "description_length_ratio": 0.42,
        }


def test_build_feature_row_persists_service_override_features() -> None:
    """Materialized service features should preserve override-derived ML inputs."""
    row = {
        "TRAINING_PAIR_ID": "tp-1",
        "PAIR_CANONICAL_KEY": "a|b",
        "TEAM_ID": "IL211",
        "SCOPE_ID": "regional",
        "SOURCE_RUN_ID": "run-1",
        "BASELINE_CONFIDENCE_SCORE": 0.74,
        "BASELINE_PREDICTED_DUPLICATE": False,
        "BASELINE_PREDICTED_MAYBE": True,
        "DETERMINISTIC_SECTION_SCORE": 0.33,
        "NLP_SECTION_SCORE": 0.81,
        "WEIGHT_DETERMINISTIC_SECTION": 1.0,
        "WEIGHT_NLP_SECTION": 1.0,
        "DUPLICATE_THRESHOLD": 0.82,
        "MAYBE_THRESHOLD": 0.5,
        "EMBEDDING_SIMILARITY": 0.9,
        "PIPELINE_SIGNALS": [
            {"MATCH_TYPE": "shared_phone", "RAW_CONTRIBUTION": 1.0},
            {"MATCH_TYPE": "shared_address", "RAW_CONTRIBUTION": 0.5},
            {"MATCH_TYPE": "name_similarity", "RAW_CONTRIBUTION": 0.93},
        ],
        "ENTITY_A_ID": "a",
        "ENTITY_A_NAME": "Alpha Services",
        "ENTITY_A_DESCRIPTION": "Food pantry",
        "ENTITY_A_EMAIL": None,
        "ENTITY_A_PHONES": ["111"],
        "ENTITY_A_WEBSITES": [],
        "ENTITY_A_LOCATIONS": [],
        "ENTITY_A_TAXONOMIES": [],
        "ENTITY_A_IDENTIFIERS": [],
        "ENTITY_A_SERVICES_ROLLUP": [],
        "ENTITY_A_ORGANIZATION_ID": "org-1",
        "ENTITY_A_ORGANIZATION_NAME": "Alpha Org",
        "ENTITY_A_EMBEDDING_VECTOR": [0.1, 0.2],
        "ENTITY_B_ID": "b",
        "ENTITY_B_NAME": "Alpha Service Center",
        "ENTITY_B_DESCRIPTION": "Food pantry and meals",
        "ENTITY_B_EMAIL": None,
        "ENTITY_B_PHONES": ["111"],
        "ENTITY_B_WEBSITES": [],
        "ENTITY_B_LOCATIONS": [],
        "ENTITY_B_TAXONOMIES": [],
        "ENTITY_B_IDENTIFIERS": [],
        "ENTITY_B_SERVICES_ROLLUP": [],
        "ENTITY_B_ORGANIZATION_ID": "org-1",
        "ENTITY_B_ORGANIZATION_NAME": "Alpha Org",
        "ENTITY_B_EMBEDDING_VECTOR": [0.2, 0.3],
    }

    feature_row = _build_feature_row(
        row=row,
        entity_type="service",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        extractor=_Extractor(),
    )

    assert feature_row["FEATURE_SCHEMA_VERSION"] == FEATURE_SCHEMA_VERSION
    assert feature_row["BASELINE_SCORE_BAND"] == "maybe"
    assert feature_row["FUZZY_NAME"] == 0.93
    assert feature_row["SHARED_PHONE"] == 1.0
    assert feature_row["SHARED_ADDRESS"] == 0.5
    assert feature_row["EMBEDDING_SIMILARITY"] == 0.9
    assert feature_row["NAME_TOKEN_SORT"] == 0.88
