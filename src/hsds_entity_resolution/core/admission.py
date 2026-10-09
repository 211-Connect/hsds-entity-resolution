"""Candidate admission: one generic rule on Informative Keys plus a Structural Exclusion hook.

A pair becomes a candidate when

* its two records share at least one value of an **Informative Key** field, or
* its embedding similarity is at or above one floor (``blocking.similarity_threshold``),

and the caller's **Structural Exclusion**, if any, does not exclude it.

Whether a field is informative is decided per source schema: a field shared by many
records of one schema (a service name that is really a category label repeated across
hundreds of rows) proves nothing when two records share it. The caller measures that,
usually as a distinct-value ratio, and hands the engine an :data:`InformativeKeyTable`;
:func:`build_informative_key_table` builds one from ratios, a cutoff and overrides. A key
admits a pair only when the field is informative in both records' schemas.

Structural Exclusions are pairs that are different by construction (for example two
records of one site that a source system always splits by category). The engine does
not know any source system: the caller passes a callable, and every pair it excludes is
returned with its reason instead of being dropped silently.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeAlias, get_args

from hsds_entity_resolution.core.dataframe_utils import clean_string_list, clean_text_scalar
from hsds_entity_resolution.core.score_candidates import (
    _first_present,  # pyright: ignore[reportPrivateUsage]
    _normalize_address_component,  # pyright: ignore[reportPrivateUsage]
)

KeyField: TypeAlias = Literal["name", "phone", "email", "website", "address"]
KEY_FIELDS: tuple[KeyField, ...] = get_args(KeyField)

InformativeKeyTable: TypeAlias = Mapping[str, Mapping[KeyField, bool]]
"""Source schema (any case) → key field → whether the field is informative there.

A schema missing from the table, or a field missing for a schema, counts as informative.
"""

StructuralExclusion: TypeAlias = Callable[[Mapping[str, Any], Mapping[str, Any]], str | None]
"""``(entity_a, entity_b) -> reason`` when the pair is different by construction, else ``None``."""

EMBEDDING_FLOOR_RULE_ID = "embedding_floor"
EMBEDDING_FLOOR_REASON_CODE = "embedding_floor"
INFORMATIVE_KEY_RULE_PREFIX = "informative_key:"


def informative_key_reason_code(field: KeyField) -> str:
    """Return the candidate reason code naming the Informative Key that admitted a pair.

    Args:
        field: The key field whose shared value admitted the pair.

    Returns:
        For example ``"informative_key_phone"``.
    """
    return f"informative_key_{field}"


def informative_key_rule_id(fields: set[KeyField]) -> str:
    """Return the ``blocking_rule_id`` for a pair admitted by shared Informative Keys.

    Args:
        fields: Every key field the pair shares a value of.

    Returns:
        For example ``"informative_key:email+phone"``.
    """
    return INFORMATIVE_KEY_RULE_PREFIX + "+".join(sorted(fields))


def build_informative_key_table(
    *,
    ratios: Mapping[str, Mapping[KeyField, float | None]],
    cutoff: float,
    overrides: Mapping[str, Mapping[KeyField, bool]],
) -> dict[str, dict[KeyField, bool]]:
    """Decide per schema which key fields are informative.

    A field is informative when its distinct-value ratio in the schema is strictly above
    ``cutoff``; a field with no measured ratio (no values) is not. An override for the
    schema and field wins over the ratio.

    Args:
        ratios: Schema → field → distinct-value ratio in ``[0, 1]`` or ``None``.
        cutoff: The global cutoff.
        overrides: Schema → field → forced informative yes/no.

    Returns:
        A table keyed by upper-cased schema with every :data:`KEY_FIELDS` entry set.
    """
    upper_overrides = {schema.strip().upper(): fields for schema, fields in overrides.items()}
    table: dict[str, dict[KeyField, bool]] = {}
    for schema, schema_ratios in ratios.items():
        key = schema.strip().upper()
        schema_overrides = upper_overrides.get(key, {})
        decided: dict[KeyField, bool] = {}
        for field in KEY_FIELDS:
            if field in schema_overrides:
                decided[field] = schema_overrides[field]
                continue
            ratio = schema_ratios.get(field)
            decided[field] = ratio is not None and ratio > cutoff
        table[key] = decided
    return table


def is_informative(table: InformativeKeyTable | None, *, schema: str, field: KeyField) -> bool:
    """Return whether ``field`` is an Informative Key in ``schema``.

    Args:
        table: The run's table, or ``None`` when the caller supplied none.
        schema: Source schema of the record.
        field: Key field.

    Returns:
        ``True`` unless the table says the field is not informative in that schema.
    """
    if table is None:
        return True
    schema_table = _schema_entry(table, schema)
    if schema_table is None or field not in schema_table:
        return True
    return bool(schema_table[field])


def _schema_entry(table: InformativeKeyTable, schema: str) -> Mapping[KeyField, bool] | None:
    """Look up a schema case-insensitively."""
    if schema in table:
        return table[schema]
    upper = schema.strip().upper()
    for candidate, entry in table.items():
        if candidate.strip().upper() == upper:
            return entry
    return None


def key_values(entity: Mapping[str, Any], field: KeyField) -> set[str]:
    """Return the comparable values of one key field for one record.

    Values are compared exactly after light cleaning (lowercase, trimmed, whitespace
    collapsed); an address needs a street line and becomes ``street|city|state|zip``.

    Args:
        entity: A cleaned entity row.
        field: Key field.

    Returns:
        The record's values for the field; empty when it has none.
    """
    if field == "name":
        name = clean_text_scalar(entity.get("name"))
        return {name} if name else set()
    if field == "phone":
        return set(clean_string_list(entity.get("phones")))
    if field == "email":
        return set(clean_string_list(entity.get("emails")))
    if field == "website":
        return set(clean_string_list(entity.get("websites")))
    return set(_address_values(entity.get("locations")))


def _address_values(locations_value: Any) -> list[str]:
    """Build exact-address tokens, requiring a street line."""
    if not isinstance(locations_value, list):
        return []
    output: list[str] = []
    for location in locations_value:
        if not isinstance(location, dict):
            continue
        street = _normalize_address_component(
            _first_present(location, ("address_1", "address1", "line1", "street", "address"))
        )
        if not street:
            continue
        parts = [
            street,
            _normalize_address_component(_first_present(location, ("city",))),
            _normalize_address_component(_first_present(location, ("state",))),
            _normalize_address_component(
                _first_present(location, ("postal_code", "postal", "zip", "zipcode"))
            ),
        ]
        output.append("|".join(part for part in parts if part))
    return clean_string_list(output)
