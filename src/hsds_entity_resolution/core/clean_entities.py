"""Entity-cleaning stage for incremental HSDS entity resolution.

This stage reads raw source rows, cleans and canonicalises their field values
(lowercasing, whitespace stripping, format standardisation), attaches content
hashes, and assembles one wide denormalized record per entity for downstream
candidate generation and scoring.
"""

from __future__ import annotations

import json
from typing import Any, cast

import polars as pl
from dagster import get_dagster_logger

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.dataframe_utils import (
    clean_string_list,
    clean_text_scalar,
    ensure_columns,
    frame_with_schema,
    hash_values,
    to_dataframe,
)
from hsds_entity_resolution.core.taxonomy_utils import (
    clean_services_rollup,
    clean_taxonomy_objects,
)
from hsds_entity_resolution.observability import FrameTracer, IncrementalProgressLogger
from hsds_entity_resolution.types.contracts import CleanEntitiesResult
from hsds_entity_resolution.types.domain import EntityType
from hsds_entity_resolution.types.rows import (
    CleanEntityRow,
    CleanPayloadValues,
    JsonObject,
    JsonValue,
    RawEntityRowInput,
)

_ORGANIZATION_PAYLOAD_COLUMNS = [
    "name",
    "description",
    "emails",
    "phones",
    "websites",
    "locations",
    "taxonomies",
    "identifiers",
    "services_rollup",
]
_SERVICE_PAYLOAD_COLUMNS = [
    "name",
    "description",
    "emails",
    "phones",
    "websites",
    "locations",
    "taxonomies",
    "organization_name",
]

# Columns whose struct shape varies by source schema — typed as Object so that
# Polars stores the raw Python list-of-dict values without running struct
# inference.  Any inconsistency in the struct field count across rows (e.g. one
# service location has a "url" key, another does not) would cause a
# ComputeError in from_dicts if these columns were typed as List(Struct(...))
# or inferred from the data.  pl.Object bypasses that path entirely and the
# values are returned as-is by to_dicts(), which is what downstream stages
# (generate_candidates, score_candidates) rely on.
_NESTED_LIST_COLUMNS = {"locations", "taxonomies", "identifiers", "services_rollup"}

_CLEAN_ENTITY_SCHEMA: dict[str, Any] = {
    "entity_id": pl.String,
    "entity_type": pl.String,
    "source_schema": pl.String,
    "embedding_vector": pl.List(pl.Float64),
    "content_hash": pl.String,
    "name": pl.String,
    "description": pl.String,
    "emails": pl.List(pl.String),
    "phones": pl.List(pl.String),
    "websites": pl.List(pl.String),
    "locations": pl.Object,
    "taxonomies": pl.Object,
    "identifiers": pl.Object,
    "services_rollup": pl.Object,
    "organization_name": pl.String,
    "organization_id": pl.String,
    # Display-quality passthrough fields: original casing, passed through unchanged.
    "display_name": pl.String,
    "display_description": pl.String,
    "alternate_name": pl.String,
    "short_description": pl.String,
    "application_process": pl.String,
    "fees_description": pl.String,
    "eligibility_description": pl.String,
    "resource_writer_name": pl.String,
    "assured_date": pl.String,
    "assurer_email": pl.String,
    "original_id": pl.String,
}

_ENTITY_INDEX_SCHEMA: dict[str, Any] = {
    "entity_id": pl.String,
    "entity_type": pl.String,
    "content_hash": pl.String,
    "active_flag": pl.Boolean,
}


def clean_entities(
    *,
    organization_entities: pl.DataFrame | pl.LazyFrame,
    service_entities: pl.DataFrame | pl.LazyFrame,
    previous_entity_index: pl.DataFrame | pl.LazyFrame,
    config: EntityResolutionRunConfig,
    progress_logger: IncrementalProgressLogger | None = None,
    tracer: FrameTracer | None = None,
) -> CleanEntitiesResult:
    """Clean raw entities, validate embeddings, and classify incremental deltas."""
    organization_df = _clean_entity_frame(
        frame=to_dataframe(organization_entities),
        entity_type="organization",
        payload_columns=_ORGANIZATION_PAYLOAD_COLUMNS,
        strict=config.execution.strict_validation_mode,
        progress_logger=progress_logger,
    )
    if tracer is not None:
        tracer.log_frame(organization_df, "clean_entities.organization_df")

    service_df = _clean_entity_frame(
        frame=to_dataframe(service_entities),
        entity_type="service",
        payload_columns=_SERVICE_PAYLOAD_COLUMNS,
        strict=config.execution.strict_validation_mode,
        progress_logger=progress_logger,
    )
    if tracer is not None:
        tracer.log_frame(service_df, "clean_entities.service_df")

    current_index = _build_entity_index(organization_df=organization_df, service_df=service_df)
    if tracer is not None:
        tracer.log_frame(current_index, "clean_entities.entity_index")

    prior_index = _coerce_previous_entity_index(to_dataframe(previous_entity_index))
    if tracer is not None:
        tracer.log_frame(prior_index, "clean_entities.prior_index")

    changed_entities, entity_delta_summary, removed_entity_ids = _classify_deltas(
        current_index=current_index,
        prior_index=prior_index,
    )
    if tracer is not None:
        tracer.log_frame(changed_entities, "clean_entities.changed_entities")
        tracer.log_frame(removed_entity_ids, "clean_entities.removed_entity_ids")
        tracer.log_frame(entity_delta_summary, "clean_entities.entity_delta_summary")

    summary_row = entity_delta_summary.row(0, named=True)
    no_change = (
        int(summary_row["added_count"]) == 0
        and int(summary_row["changed_count"]) == 0
        and int(summary_row["removed_count"]) == 0
    )
    return CleanEntitiesResult(
        denormalized_organization=organization_df,
        denormalized_service=service_df,
        entity_index=current_index,
        entity_delta_summary=entity_delta_summary,
        removed_entity_ids=removed_entity_ids,
        changed_entities=changed_entities,
        no_change=no_change,
    )


def _clean_entity_frame(
    *,
    frame: pl.DataFrame,
    entity_type: EntityType,
    payload_columns: list[str],
    strict: bool,
    progress_logger: IncrementalProgressLogger | None = None,
) -> pl.DataFrame:
    """Clean one entity frame and attach a deterministic content hash."""
    required_columns = ["entity_id", "source_schema", *payload_columns]
    with_required = ensure_columns(frame=frame, columns=required_columns)
    if with_required.is_empty():
        return _empty_clean_entity_frame()
    input_rows = cast(list[RawEntityRowInput], with_required.to_dicts())
    stage_name = f"clean_entities.{entity_type}.rows"
    if progress_logger is not None:
        progress_logger.stage_started(stage=stage_name, total=len(input_rows))
    clean_rows: list[CleanEntityRow] = []
    for index, row in enumerate(input_rows, start=1):
        clean_rows.append(_clean_entity_row(row=row, entity_type=entity_type))
        if progress_logger is not None:
            progress_logger.stage_advanced(
                stage=stage_name,
                processed=index,
                total=len(input_rows),
            )
    if progress_logger is not None:
        progress_logger.stage_completed(
            stage=stage_name,
            detail={"clean_rows": len(clean_rows)},
        )
    clean_frame = frame_with_schema(clean_rows, _CLEAN_ENTITY_SCHEMA)
    validated_frame = _validate_embedding_column(frame=clean_frame, strict=strict)
    _log_clean_sample(rows=clean_rows, entity_type=entity_type)
    return validated_frame


def _log_clean_sample(*, rows: list[CleanEntityRow], entity_type: str) -> None:
    """Emit a DEBUG entry showing what the entity-cleaning stage produced.

    Covers the handoff between raw source rows and the canonical pipeline
    frames — specifically confirms that taxonomy codes were extracted and
    location objects carry the city/state keys the overlap evaluator expects.
    """
    if not rows:
        return
    _log = get_dagster_logger()
    sample_lines: list[str] = []
    for row in rows[:3]:
        tax_list = row.get("taxonomies") or []
        loc_list = row.get("locations") or []
        phones_list = row.get("phones") or []
        first_tax = tax_list[0].get("code", "?") if tax_list else "—"
        if loc_list and isinstance(loc_list[0], dict):
            first_loc_keys = sorted(loc_list[0].keys())
            first_city = loc_list[0].get("city", "—")
            first_state = loc_list[0].get("state", "—")
        else:
            first_loc_keys = []
            first_city = first_state = "—"
        sample_lines.append(
            f"  [{len(sample_lines)}] schema={row.get('source_schema') or '?'}"
            f" tax={len(tax_list)} first_code={first_tax!r}"
            f" loc={len(loc_list)} loc_keys={first_loc_keys}"
            f" city={first_city!r} state={first_state!r}"
            f" phones={len(phones_list)}"
        )
    _log.debug(
        "🔧 clean_sample entity_type=%s total=%d\n%s",
        entity_type,
        len(rows),
        "\n".join(sample_lines),
    )


def _clean_entity_row(*, row: RawEntityRowInput, entity_type: EntityType) -> CleanEntityRow:
    """Clean one row into stable canonical values."""
    entity_id = clean_text_scalar(row.get("entity_id"))
    source_schema = clean_text_scalar(row.get("source_schema"))
    payload_values, hash_inputs = _clean_payload_fields(row=row)
    content_hash = hash_values([entity_id, entity_type, source_schema, *hash_inputs])
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "source_schema": source_schema,
        "embedding_vector": _clean_embedding_value(row=row),
        "content_hash": content_hash,
        **payload_values,
        # Preserve original casing for display; not used in hashing or embeddings.
        "display_name": _passthrough_optional(row.get("display_name")),
        "display_description": _passthrough_optional(row.get("display_description")),
        "alternate_name": _passthrough_optional(row.get("alternate_name")),
        "short_description": _passthrough_optional(row.get("short_description")),
        "application_process": _passthrough_optional(row.get("application_process")),
        "fees_description": _passthrough_optional(row.get("fees_description")),
        "eligibility_description": _passthrough_optional(row.get("eligibility_description")),
        "resource_writer_name": _passthrough_optional(row.get("resource_writer_name")),
        "assured_date": _passthrough_optional(row.get("assured_date")),
        "assurer_email": _passthrough_optional(row.get("assurer_email")),
        "original_id": _passthrough_optional(row.get("original_id")),
    }


def _clean_payload_fields(*, row: RawEntityRowInput) -> tuple[CleanPayloadValues, list[str]]:
    """Clean payload fields and return hash input values."""
    name = clean_text_scalar(row.get("name"))
    description = clean_text_scalar(row.get("description"))
    emails = clean_string_list(row.get("emails"))
    phones = clean_string_list(row.get("phones"))
    websites = clean_string_list(row.get("websites"))
    locations = _clean_object_list(row.get("locations"))
    taxonomies = clean_taxonomy_objects(row.get("taxonomies"))
    identifiers = _clean_object_list(row.get("identifiers"))
    services_rollup = clean_services_rollup(row.get("services_rollup"))
    organization_name = clean_text_scalar(row.get("organization_name"))
    organization_id = clean_text_scalar(row.get("organization_id"))
    values: CleanPayloadValues = {
        "name": name,
        "description": description,
        "emails": emails,
        "phones": phones,
        "websites": websites,
        "locations": locations,
        "taxonomies": taxonomies,
        "identifiers": identifiers,
        "services_rollup": services_rollup,
        "organization_name": organization_name,
        "organization_id": organization_id,
    }
    hash_inputs = [
        f"name={name}",
        f"description={description}",
        f"emails={_to_hash_token(emails)}",
        f"phones={_to_hash_token(phones)}",
        f"websites={_to_hash_token(websites)}",
        f"locations={_to_hash_token(locations)}",
        f"taxonomies={_to_hash_token(taxonomies)}",
        f"identifiers={_to_hash_token(identifiers)}",
        f"services_rollup={_to_hash_token(services_rollup)}",
        f"organization_name={organization_name}",
        f"organization_id={organization_id}",
    ]
    return values, hash_inputs


def _passthrough_optional(value: object) -> str | None:
    """Normalize whitespace without altering case; used for display-quality fields."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _clean_embedding_value(*, row: RawEntityRowInput) -> list[float]:
    """Clean embedding column aliases into `embedding_vector`."""
    embedding = row.get("embedding_vector", row.get("embedding"))
    if not isinstance(embedding, list):
        return []
    return [float(value) for value in embedding]


def _clean_object_list(value: object) -> list[JsonObject]:
    """Clean unknown value to a list of object dictionaries."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _to_hash_token(value: object) -> str:
    """Serialise nested payload values into deterministic tokens for content hashing."""
    normalized = _clean_for_hash(value)
    return json.dumps(normalized, separators=(",", ":"), sort_keys=True)


def _clean_for_hash(value: object) -> JsonValue:
    """Recursively normalise nested values to stable hash-ready structures."""
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_text_scalar(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, dict):
        output: dict[str, JsonValue] = {}
        for key in sorted(value.keys()):
            output[clean_text_scalar(key)] = _clean_for_hash(value[key])
        return output
    if isinstance(value, list | tuple):
        items = [_clean_for_hash(item) for item in value]
        tokens = [json.dumps(item, separators=(",", ":"), sort_keys=True) for item in items]
        return [json.loads(token) for token in sorted(tokens)]
    return clean_text_scalar(str(value))


def _validate_embedding_column(*, frame: pl.DataFrame, strict: bool) -> pl.DataFrame:
    """Validate embedding shape/dtype contract and coerce if permitted."""
    vectors = frame.get_column("embedding_vector").to_list()
    lengths = {len(vector) for vector in vectors if isinstance(vector, list)}
    has_invalid = any(not isinstance(vector, list) or len(vector) == 0 for vector in vectors)
    if strict and (has_invalid or len(lengths) > 1):
        message = "Embedding structural validation failed: expected non-empty equal-length vectors"
        raise ValueError(message)
    if len(lengths) <= 1:
        return frame
    max_len = max(lengths)
    # Non-strict mode pads shorter vectors to maintain consistent matrix dimensions.
    padded = [vector + [0.0] * (max_len - len(vector)) for vector in vectors]
    return frame.with_columns(pl.Series(name="embedding_vector", values=padded))


def _build_entity_index(*, organization_df: pl.DataFrame, service_df: pl.DataFrame) -> pl.DataFrame:
    """Build current active entity index from cleaned entities."""
    combined = pl.concat([organization_df, service_df], how="diagonal_relaxed")
    return combined.select(["entity_id", "entity_type", "content_hash"]).with_columns(
        pl.lit(True).alias("active_flag")
    )


def _coerce_previous_entity_index(frame: pl.DataFrame) -> pl.DataFrame:
    """Coerce prior index to the required minimal column set for reconciliation."""
    required = ["entity_id", "entity_type", "content_hash", "active_flag"]
    coerced = ensure_columns(frame=frame, columns=required)
    return coerced.select(required).with_columns(
        pl.col("entity_id").map_elements(clean_text_scalar, return_dtype=pl.String),
        pl.col("entity_type").map_elements(clean_text_scalar, return_dtype=pl.String),
        pl.col("content_hash").map_elements(clean_text_scalar, return_dtype=pl.String),
        pl.when(pl.col("active_flag").is_null())
        .then(pl.lit(True))
        .otherwise(pl.col("active_flag"))
        .alias("active_flag"),
    )


def _classify_deltas(
    *,
    current_index: pl.DataFrame,
    prior_index: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Classify added/changed/unchanged/removed entities for incremental execution."""
    active_prior = prior_index.filter(pl.col("active_flag"))
    joined = current_index.join(
        active_prior.select(["entity_id", "entity_type", "content_hash"]).rename(
            {"content_hash": "prior_content_hash"}
        ),
        on=["entity_id", "entity_type"],
        how="left",
    )
    changed_entities = joined.with_columns(
        pl.when(pl.col("prior_content_hash").is_null())
        .then(pl.lit("added"))
        .when(pl.col("prior_content_hash") != pl.col("content_hash"))
        .then(pl.lit("changed"))
        .otherwise(pl.lit("unchanged"))
        .alias("delta_class")
    )
    removed_entity_ids = (
        active_prior.join(current_index, on=["entity_id", "entity_type"], how="anti")
        .select(["entity_id", "entity_type"])
        .with_columns(pl.lit("entity_deleted").alias("cleanup_reason"))
    )
    entity_delta_summary = _build_delta_summary(
        changed_entities=changed_entities, removed_entity_ids=removed_entity_ids
    )
    return changed_entities, entity_delta_summary, removed_entity_ids


def _build_delta_summary(
    *, changed_entities: pl.DataFrame, removed_entity_ids: pl.DataFrame
) -> pl.DataFrame:
    """Build a one-row delta summary frame with deterministic counts."""
    added_count = changed_entities.filter(pl.col("delta_class") == "added").height
    changed_count = changed_entities.filter(pl.col("delta_class") == "changed").height
    unchanged_count = changed_entities.filter(pl.col("delta_class") == "unchanged").height
    removed_count = removed_entity_ids.height
    return pl.DataFrame(
        {
            "added_count": [added_count],
            "changed_count": [changed_count],
            "unchanged_count": [unchanged_count],
            "removed_count": [removed_count],
        }
    )


def _empty_clean_entity_frame() -> pl.DataFrame:
    """Return canonical empty cleaned entity frame."""
    return pl.DataFrame(schema=_CLEAN_ENTITY_SCHEMA)
