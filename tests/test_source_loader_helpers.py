"""Unit tests for source loader SQL helper functions and staged row contracts."""

from __future__ import annotations

from typing import cast

from consumer.consumer_adapter.source_loader import (
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
