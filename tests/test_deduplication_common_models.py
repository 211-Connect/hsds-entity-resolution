"""Tests for semantic dedupe table Pydantic models."""

import json
from pathlib import Path

from consumer.consumer_types.deduplication_common import (
    COMMON_SCHEMA_TABLE_MODELS,
    DataQualityFlagRow,
    DenormalizedOrganizationCacheRow,
)


def test_common_schema_model_registry_has_all_expected_tables() -> None:
    """The registry should mirror the TypeScript `Database` table map."""

    assert len(COMMON_SCHEMA_TABLE_MODELS) == 17
    assert "DUPLICATE_PAIRS" in COMMON_SCHEMA_TABLE_MODELS
    assert "DENORMALIZED_ORGANIZATION_CACHE" in COMMON_SCHEMA_TABLE_MODELS


def test_denormalized_organization_cache_row_accepts_nested_payloads() -> None:
    """Nested JSON payloads should be validated into typed Pydantic models."""

    row = DenormalizedOrganizationCacheRow.model_validate(
        {
            "ID": "org-1",
            "NAME": "Org",
            "ALTERNATE_NAME": None,
            "DESCRIPTION": None,
            "SHORT_DESCRIPTION": None,
            "EMAIL": None,
            "WEBSITES": ["https://example.org"],
            "SOURCE_SCHEMA": "TENANT_A",
            "ORIGINAL_ID": "123",
            "RESOURCE_WRITER_NAME": None,
            "ASSURED_DATE": None,
            "ASSURER_EMAIL": None,
            "IDENTIFIERS": [{"identifier_type": "EIN", "identifier": "12-3456789"}],
            "LOCATIONS": [
                {
                    "location_id": "loc-1",
                    "name": "Main",
                    "address_1": None,
                    "city": "San Francisco",
                    "state": "CA",
                    "postal_code": "94107",
                    "latitude": 37.77,
                    "longitude": -122.41,
                    "url": None,
                }
            ],
            "PHONES": ["4155551212"],
            "TAXONOMIES": [
                {
                    "taxonomy_term_id": "tax-1",
                    "code": "X",
                    "name": "Support",
                    "description": None,
                    "taxonomy_system_name": "211 LA Taxonomy",
                }
            ],
            "SERVICES": [
                {
                    "id": "svc-1",
                    "name": "Food Assistance",
                    "description": None,
                    "taxonomy_codes": [
                        {
                            "taxonomy_term_id": "st-1",
                            "code": "Y",
                            "name": "Food",
                            "description": None,
                            "taxonomy_system_name": "211 LA Taxonomy",
                        }
                    ],
                }
            ],
            "LAST_UPDATED": None,
        }
    )

    assert row.ID == "org-1"
    assert row.LOCATIONS[0].city == "San Francisco"
    assert row.SERVICES[0].taxonomy_codes[0].taxonomy_term_id == "st-1"
    assert row.TAXONOMIES[0].taxonomy_system_name == "211 LA Taxonomy"
    assert row.SERVICES[0].taxonomy_codes[0].taxonomy_system_name == "211 LA Taxonomy"


def test_data_quality_flag_row_validates_target_side_literal() -> None:
    """Target side should accept only A/B values."""

    row = DataQualityFlagRow.model_validate(
        {
            "ID": "flag-1",
            "DUPLICATE_PAIR_ID": "pair-1",
            "DUPLICATE_PAIR_SCORE_ID": None,
            "TEAM_ID": "team-1",
            "DEDUPLICATION_RUN_ID": "run-1",
            "ENTITY_TYPE": "organization",
            "NOTE": None,
            "TARGET_ENTITY_ID": None,
            "TARGET_SIDE": "A",
            "FLAGGED_BY": "reviewer-1",
            "CREATED_AT": "2026-03-05T00:00:00Z",
        }
    )

    assert row.TARGET_SIDE == "A"


def test_model_fields_align_with_common_experiment_schema_contract() -> None:
    """Critical dedupe table models should match the checked-in schema contract."""
    schema_contract_path = Path("tests/contract/common_experiment_schema_contract.json")
    schema_contract = json.loads(schema_contract_path.read_text(encoding="utf-8"))
    tables_to_check = (
        "DUPLICATE_PAIRS",
        "DUPLICATE_PAIR_SCORES",
        "DUPLICATE_REASONS",
        "MITIGATED_PAIRS",
        "DUPLICATE_CLUSTERS",
        "DUPLICATE_CLUSTER_PAIRS",
    )
    for table_name in tables_to_check:
        model = COMMON_SCHEMA_TABLE_MODELS[table_name]
        model_columns = set(model.model_fields.keys())
        expected_columns = set(schema_contract[table_name])
        assert model_columns == expected_columns, table_name
