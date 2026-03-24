"""Canonical taxonomy/service-rollup normalization and extraction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from hsds_entity_resolution.core.dataframe_utils import clean_text_scalar
from hsds_entity_resolution.types.rows import JsonObject

_TAXONOMY_CODE_KEYS: Final[tuple[str, ...]] = (
    "code",
    "CODE",
    "taxonomy_code",
    "taxonomyCode",
    "taxonomy_term_id",
    "taxonomyTermId",
)
_SERVICE_TAXONOMY_KEYS: Final[tuple[str, ...]] = (
    "taxonomies",
    "TAXONOMIES",
    "taxonomy_codes",
    "taxonomyCodes",
)
_SERVICE_NAME_KEYS: Final[tuple[str, ...]] = ("name", "NAME")


def clean_taxonomy_objects(value: object) -> list[JsonObject]:
    """Normalise taxonomy payloads into canonical objects with a lowercase ``code`` key.

    The canonical ``code`` field is always present and normalised to lowercase.
    Alias code keys (``CODE``, ``taxonomy_code``, ``taxonomyCode``, etc.) are
    removed — only the canonical ``code`` key is retained.  All other source
    fields (``name``, ``description``, etc.) are preserved.  Objects are
    deduplicated by normalised code and sorted by ``code`` for deterministic
    content hashing.
    """
    if not isinstance(value, list):
        return []
    seen_codes: set[str] = set()
    normalized: list[JsonObject] = []
    for item in value:
        if isinstance(item, str):
            code = clean_text_scalar(item)
            if code and code not in seen_codes:
                seen_codes.add(code)
                normalized.append({"code": code})
        elif isinstance(item, Mapping):
            code = _first_non_empty(item=item, keys=_TAXONOMY_CODE_KEYS)
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            # Preserve non-code source fields (e.g. name, description); drop alias
            # keys (CODE, taxonomy_code, etc.) so only the canonical lowercase
            # `code` key remains in the output object.
            entry: JsonObject = {k: v for k, v in item.items() if k not in _TAXONOMY_CODE_KEYS}
            entry["code"] = code
            normalized.append(entry)
    return sorted(normalized, key=lambda x: str(x.get("code", "")))


def clean_services_rollup(value: object) -> list[JsonObject]:
    """Normalize mixed service rollup payloads into canonical service objects."""
    if not isinstance(value, list):
        return []
    normalized: list[JsonObject] = []
    for item in value:
        if isinstance(item, str):
            name = clean_text_scalar(item)
            if name:
                normalized.append({"name": name, "taxonomies": []})
            continue
        if not isinstance(item, Mapping):
            continue
        taxonomies: set[str] = set()
        for key in _SERVICE_TAXONOMY_KEYS:
            taxonomies.update(extract_taxonomy_codes(item.get(key)))
        name = _first_non_empty(item=item, keys=_SERVICE_NAME_KEYS)
        normalized.append(
            {
                "name": name,
                "taxonomies": [{"code": code} for code in sorted(taxonomies)],
            }
        )
    return normalized


def extract_entity_taxonomy_codes(
    *, entity: Mapping[str, Any], include_parent_codes: bool = False
) -> set[str]:
    """Extract normalized taxonomy codes from entity taxonomies and services."""
    direct = extract_taxonomy_codes(entity.get("taxonomies", entity.get("TAXONOMIES")))
    services = extract_taxonomy_codes_from_services(
        entity.get("services_rollup", entity.get("SERVICES"))
    )
    codes = direct.union(services)
    if not include_parent_codes:
        return codes
    hierarchy_codes: set[str] = set()
    for code in codes:
        hierarchy_codes.add(code)
        hierarchy_codes.update(taxonomy_parent_codes(code))
    return hierarchy_codes


def extract_taxonomy_codes(value: object) -> set[str]:
    """Extract taxonomy codes from mixed list values and nested objects."""
    if not isinstance(value, list):
        return set()
    codes: set[str] = set()
    for item in value:
        if isinstance(item, str):
            normalized = clean_text_scalar(item)
            if normalized:
                codes.add(normalized)
            continue
        if isinstance(item, Mapping):
            code = _first_non_empty(item=item, keys=_TAXONOMY_CODE_KEYS)
            if code:
                codes.add(code)
    return codes


def extract_taxonomy_codes_from_services(value: object) -> set[str]:
    """Extract taxonomy codes from mixed service-rollup shapes."""
    if not isinstance(value, list):
        return set()
    codes: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            continue
        for key in _SERVICE_TAXONOMY_KEYS:
            codes.update(extract_taxonomy_codes(item.get(key)))
    return codes


def to_legacy_taxonomies(value: object) -> list[dict[str, str]]:
    """Map mixed taxonomy payloads to legacy list-of-code-objects format."""
    return [{"code": code} for code in sorted(extract_taxonomy_codes(value))]


def to_legacy_services(value: object) -> list[dict[str, Any]]:
    """Map mixed service payloads to legacy service shape for feature extraction."""
    if not isinstance(value, list):
        return []
    services: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        taxonomy_codes = sorted(extract_taxonomy_codes_from_services([item]))
        services.append(
            {
                "name": _first_non_empty(item=item, keys=_SERVICE_NAME_KEYS),
                "taxonomies": taxonomy_codes,
            }
        )
    return services


def taxonomy_parent_codes(code: str) -> set[str]:
    """Expand dashed taxonomy code prefixes for hierarchical matching."""
    if "-" not in code:
        return set()
    parts = code.split("-")
    return {"-".join(parts[:index]) for index in range(1, len(parts))}


def _first_non_empty(*, item: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return first present and normalized scalar value for known aliases."""
    for key in keys:
        if key in item:
            value = clean_text_scalar(item.get(key))
            if value:
                return value
    return ""
