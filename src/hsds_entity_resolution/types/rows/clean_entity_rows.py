"""Typed row payloads for entity-cleaning stage inputs and outputs."""

from __future__ import annotations

from typing import TypeAlias, TypedDict

from hsds_entity_resolution.types.domain import EntityType

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | dict[str, "JsonValue"] | list["JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class RawEntityRowInput(TypedDict, total=False):
    """Input row shape accepted by entity-cleaning helpers."""

    entity_id: object
    source_schema: object
    name: object
    description: object
    emails: object
    phones: object
    websites: object
    locations: object
    taxonomies: object
    identifiers: object
    services_rollup: object
    organization_name: object
    organization_id: object
    embedding_vector: object
    embedding: object
    # Display-quality passthrough fields: original casing, not used for hashing or embeddings.
    display_name: object
    display_description: object
    alternate_name: object
    short_description: object
    application_process: object
    fees_description: object
    eligibility_description: object
    resource_writer_name: object
    assured_date: object
    assurer_email: object
    original_id: object


class CleanPayloadValues(TypedDict):
    """Canonical cleaned payload fields used for dedupe semantics."""

    name: str
    description: str
    emails: list[str]
    phones: list[str]
    websites: list[str]
    locations: list[JsonObject]
    taxonomies: list[JsonObject]
    identifiers: list[JsonObject]
    services_rollup: list[JsonObject]
    organization_name: str
    organization_id: str


class CleanEntityRow(CleanPayloadValues):
    """Canonical cleaned entity row emitted by `clean_entities`."""

    entity_id: str
    entity_type: EntityType
    source_schema: str
    embedding_vector: list[float]
    content_hash: str
    # Display-quality passthrough fields: original casing preserved for cache presentation.
    display_name: str | None
    display_description: str | None
    alternate_name: str | None
    short_description: str | None
    application_process: str | None
    fees_description: str | None
    eligibility_description: str | None
    resource_writer_name: str | None
    assured_date: str | None
    assurer_email: str | None
    original_id: str | None
