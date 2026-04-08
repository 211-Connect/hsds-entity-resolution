"""Canonical taxonomy/service-rollup normalization and extraction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, cast

from hsds_entity_resolution.core.dataframe_utils import clean_text_scalar
from hsds_entity_resolution.types.rows import JsonObject, JsonValue

_TAXONOMY_CODE_KEYS: Final[tuple[str, ...]] = (
    "code",
    "CODE",
    "taxonomy_code",
    "taxonomyCode",
    "taxonomy_term_id",
    "taxonomyTermId",
)
_TAXONOMY_ID_KEYS: Final[tuple[str, ...]] = ("taxonomy_term_id", "taxonomyTermId")
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
    removed — only the canonical lowercase ``code`` key is retained. When a
    taxonomy-term identifier is present, it is preserved under the canonical
    ``taxonomy_term_id`` key so downstream persistence can reconstruct the
    denormalized cache rows. All other source fields (``name``,
    ``description``, ``taxonomy_system_name``, etc.) are preserved. Objects are deduplicated by
    normalised code and sorted by ``code`` for deterministic content hashing.
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
            entry: JsonObject = {
                k: v
                for k, v in item.items()
                if k not in _TAXONOMY_CODE_KEYS and k != "taxonomy_term_id"
            }
            entry["code"] = code
            taxonomy_term_id = _first_present_preserving_case(item=item, keys=_TAXONOMY_ID_KEYS)
            if taxonomy_term_id:
                entry["taxonomy_term_id"] = taxonomy_term_id
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
        raw_taxonomies: list[object] = []
        for key in _SERVICE_TAXONOMY_KEYS:
            field_value = item.get(key)
            if isinstance(field_value, list):
                raw_taxonomies.extend(field_value)
        name = _first_non_empty(item=item, keys=_SERVICE_NAME_KEYS)
        service: JsonObject = {
            "name": name,
            "taxonomies": cast(JsonValue, clean_taxonomy_objects(raw_taxonomies)),
        }
        service_id = _first_present_preserving_case(item=item, keys=("id", "ID"))
        if service_id:
            service["id"] = service_id
        description = _first_non_empty(item=item, keys=("description", "DESCRIPTION"))
        if description:
            service["description"] = description
        normalized.append(service)
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


def taxonomy_hierarchy_levels(code: str) -> tuple[str, ...]:
    """Return ordered HSIS hierarchy levels from root to exact code."""
    normalized = clean_text_scalar(code)
    if not normalized:
        return ()
    if "-" not in normalized:
        if normalized.isalpha() and len(normalized) > 1:
            return tuple(normalized[:index] for index in range(1, len(normalized) + 1))
        return (normalized,)

    head, *tail_segments = normalized.split("-")
    levels: list[str] = []
    if head.isalpha() and len(head) > 1:
        levels.extend(head[:index] for index in range(1, len(head) + 1))
        current = head
    else:
        levels.append(head)
        current = head

    for segment in tail_segments:
        dot_parts = segment.split(".")
        current = f"{current}-{dot_parts[0]}"
        levels.append(current)
        for dot_part in dot_parts[1:]:
            current = f"{current}.{dot_part}"
            levels.append(current)
    return tuple(dict.fromkeys(levels))


def taxonomy_parent_codes(code: str) -> set[str]:
    """Return all HSIS parent levels above the exact code."""
    hierarchy = taxonomy_hierarchy_levels(code)
    if len(hierarchy) <= 1:
        return set()
    return set(hierarchy[:-1])


def taxonomy_codes_match_or_parent_child(*, left_code: str, right_code: str) -> bool:
    """Return true for exact HSIS matches or direct parent-child relationships only."""
    left_hierarchy = taxonomy_hierarchy_levels(left_code)
    right_hierarchy = taxonomy_hierarchy_levels(right_code)
    if not left_hierarchy or not right_hierarchy:
        return False
    if left_hierarchy[-1] == right_hierarchy[-1]:
        return True
    if len(left_hierarchy) == len(right_hierarchy) + 1:
        return left_hierarchy[:-1] == right_hierarchy
    if len(right_hierarchy) == len(left_hierarchy) + 1:
        return right_hierarchy[:-1] == left_hierarchy
    return False


def _first_non_empty(*, item: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return first present and normalized scalar value for known aliases."""
    for key in keys:
        if key in item:
            value = clean_text_scalar(item.get(key))
            if value:
                return value
    return ""


def _first_present_preserving_case(*, item: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return first present scalar with whitespace normalized but original case preserved."""
    for key in keys:
        if key in item:
            value = item.get(key)
            if value is None:
                continue
            text_value = " ".join(str(value).split())
            if text_value:
                return text_value
    return ""
