"""Unit tests for source loader SQL helper functions and staged row contracts."""

from __future__ import annotations

import json
from typing import cast

import polars as pl
from consumer.consumer_adapter.source_loader import (
    _ORGANIZATION_FRAME_SCHEMA,
    _as_location_list,
    _literal_list,
    _load_organizations,
    _load_services,
    _quote_ident,
)


def test_quote_ident_escapes_double_quotes() -> None:
    """Identifier quoting should preserve embedded quote characters."""
    assert _quote_ident('COMMON"X') == '"COMMON""X"'


def test_literal_list_escapes_single_quotes() -> None:
    """Literal list renderer should escape single-quote characters."""
    rendered = _literal_list(["A", "B'C"])
    assert rendered == "'A', 'B''C'"


def test_as_location_list_coerces_integer_coordinates_to_float() -> None:
    """Whole-number lat/lon from JSON must become float so Polars Float64 structs build cleanly."""
    ints = [{"location_id": "1", "latitude": -103, "longitude": 35}]
    normalized = _as_location_list(ints)
    assert normalized[0]["latitude"] == -103.0
    assert normalized[0]["longitude"] == 35.0

    mixed = [{"latitude": -103, "longitude": 35.5}, {"latitude": 40.1, "longitude": -74}]
    from_json = _as_location_list(json.dumps(mixed))
    assert from_json[0]["longitude"] == 35.5
    assert from_json[1]["longitude"] == -74.0

    invalid = [{"latitude": "n/a", "longitude": 1}]
    cleared = _as_location_list(invalid)
    assert cleared[0]["latitude"] is None
    assert cleared[0]["longitude"] == 1.0

    batch = [
        {
            "entity_id": "e1",
            "source_schema": "S",
            "name": "n",
            "description": "",
            "emails": [],
            "phones": [],
            "websites": [],
            "locations": _as_location_list([{"latitude": -103, "longitude": 35}]),
            "taxonomies": [],
            "identifiers": [],
            "services_rollup": [],
            "display_name": None,
            "display_description": None,
            "alternate_name": None,
            "short_description": None,
            "resource_writer_name": None,
            "assured_date": None,
            "assurer_email": None,
            "original_id": None,
        }
    ]
    frame = pl.DataFrame(batch, schema=_ORGANIZATION_FRAME_SCHEMA)
    assert frame.height == 1


class _FakeCursor:
    """Minimal cursor stub for staged source-loader tests."""

    def __init__(self, *, columns: list[str], records: list[tuple[object, ...]]) -> None:
        self.description: list[tuple[str, object, object, object, object, object, object]] = [
            cast(
                tuple[str, object, object, object, object, object, object],
                (column, None, None, None, None, None, None),
            )
            for column in columns
        ]
        self._records = records
        self._consumed = False

    def execute(self, command: str) -> object:
        _ = command
        return None

    def fetchall(self) -> list[tuple[object, ...]]:
        if self._consumed:
            return []
        self._consumed = True
        return list(self._records)

    def fetchmany(self, size: int) -> list[tuple[object, ...]]:
        _ = size
        if self._consumed:
            return []
        self._consumed = True
        return list(self._records)

    def close(self) -> None:
        return None


def test_dbt_staged_loaders_only_organizations_supply_identifiers() -> None:
    """DBT-fed organization rows include identifiers while service rows do not."""
    org_cursor = _FakeCursor(
        columns=[
            "ID",
            "SOURCE_SCHEMA",
            "ORIGINAL_ID",
            "NAME",
            "DESCRIPTION",
            "EMAIL",
            "WEBSITE",
            "RESOURCE_WRITER_NAME",
            "ASSURED_DATE",
            "ASSURER_EMAIL",
            "PHONES",
            "LOCATIONS",
            "TAXONOMIES",
            "IDENTIFIERS",
            "SERVICES",
            "SERVICE_EMAILS",
            "SERVICE_WEBSITES",
        ],
        records=[
            (
                "org-1",
                "IL211",
                "org-original-1",
                "Alpha Org",
                "Primary care",
                "hello@alpha.org",
                "alpha.org",
                None,
                None,
                None,
                ["555-0100"],
                [],
                [],
                [{"identifier_type": "npi", "identifier": "123"}],
                [],
                ["hello@alpha.org"],
                ["alpha.org"],
            )
        ],
    )
    service_cursor = _FakeCursor(
        columns=[
            "ID",
            "SOURCE_SCHEMA",
            "ORIGINAL_ID",
            "ORGANIZATION_ID",
            "ORGANIZATION_NAME",
            "NAME",
            "ALTERNATE_NAME",
            "DESCRIPTION",
            "SHORT_DESCRIPTION",
            "APPLICATION_PROCESS",
            "FEES_DESCRIPTION",
            "ELIGIBILITY_DESCRIPTION",
            "EMAIL",
            "ORGANIZATION_EMAIL",
            "URL",
            "ORGANIZATION_WEBSITE",
            "PHONES",
            "LOCATIONS",
            "TAXONOMIES",
            "RESOURCE_WRITER_NAME",
            "ASSURED_DATE",
            "ASSURER_EMAIL",
        ],
        records=[
            (
                "svc-1",
                "IL211",
                "svc-original-1",
                "org-1",
                "Alpha Org",
                "Case Management",
                None,
                "Care coordination",
                None,
                None,
                None,
                None,
                None,
                "hello@alpha.org",
                None,
                "alpha.org",
                ["555-0100"],
                [],
                [],
                None,
                None,
                None,
            )
        ],
    )

    organizations = _load_organizations(org_cursor, '"DB"."SCHEMA"', ["IL211"])
    services = _load_services(service_cursor, '"DB"."SCHEMA"', ["IL211"])

    org_row = organizations.row(0, named=True)
    svc_row = services.row(0, named=True)
    assert "identifiers" in organizations.columns
    assert "identifiers" not in services.columns
    assert org_row["identifiers"] == [{"identifier_type": "npi", "identifier": "123"}]
    assert svc_row["organization_id"] == "org-1"
