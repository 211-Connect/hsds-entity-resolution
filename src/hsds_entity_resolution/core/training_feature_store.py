"""Offline materialization helpers for typed training feature rows."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from hsds_entity_resolution.core.clean_entities import _clean_entity_row
from hsds_entity_resolution.core.ml_inference import build_feature_extractor, to_legacy_entity
from hsds_entity_resolution.core.training_features import (
    FEATURE_SCHEMA_VERSION,
    build_api_feature_payload,
    build_signal_overrides_from_pipeline_signals,
    feature_names_for_entity_type,
)
from hsds_entity_resolution.types.domain import EntityType
from hsds_entity_resolution.types.rows import RawEntityRowInput

RunSelectionPolicy = Literal["latest", "all"]
_REVIEW_SOURCE = "ER_RUNTIME.DUPLICATE_PAIR_SCORES"
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaterializedTrainingFeatures:
    """Summary of one offline feature materialization run."""

    entity_type: EntityType
    feature_schema_version: str
    selected_pairs: int
    inserted_pairs: int
    skipped_pairs: int
    source_run_id: str | None


def materialize_training_features(
    *,
    cursor: Any,
    database: str,
    schema: str,
    source_database: str,
    source_schema: str,
    team_id: str | None,
    scope_id: str | None,
    entity_type: EntityType,
    run_selection: RunSelectionPolicy = "latest",
    source_run_id: str | None = None,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
    taxonomy_embeddings: dict[str, list[float]] | None = None,
    limit: int | None = None,
) -> MaterializedTrainingFeatures:
    """Materialize reviewed typed pair features for one entity type."""
    rows = _load_reviewed_pairs_from_runtime(
        cursor=cursor,
        database=database,
        schema=schema,
        source_database=source_database,
        source_schema=source_schema,
        team_id=team_id,
        entity_type=entity_type,
        run_selection=run_selection,
        source_run_id=source_run_id,
        limit=limit,
    )
    if not rows:
        return MaterializedTrainingFeatures(
            entity_type=entity_type,
            feature_schema_version=feature_schema_version,
            selected_pairs=0,
            inserted_pairs=0,
            skipped_pairs=0,
            source_run_id=source_run_id,
        )
    _log_reviewed_score_diagnostics(rows=rows, entity_type=entity_type)

    reason_map = _load_reason_map(
        cursor=cursor,
        source_database=source_database,
        source_schema=source_schema,
        source_score_ids=[str(row["SOURCE_SCORE_ID"]) for row in rows],
    )
    extractor = build_feature_extractor(
        entity_type=entity_type,
        taxonomy_embeddings=taxonomy_embeddings,
    )
    inserted_pairs = 0
    skipped_pairs = 0
    resolved_run_id = source_run_id or _resolve_latest_source_run_id(
        rows=rows, run_selection=run_selection
    )
    for row in rows:
        row["PIPELINE_SIGNALS"] = reason_map.get(str(row["SOURCE_SCORE_ID"]), [])
        entity_a_snapshot_id = _ensure_entity_snapshot(
            cursor=cursor,
            database=database,
            schema=schema,
            entity_type=entity_type,
            source_run_id=_safe_optional_str(row.get("SOURCE_RUN_ID")),
            row=row,
            prefix="ENTITY_A",
        )
        entity_b_snapshot_id = _ensure_entity_snapshot(
            cursor=cursor,
            database=database,
            schema=schema,
            entity_type=entity_type,
            source_run_id=_safe_optional_str(row.get("SOURCE_RUN_ID")),
            row=row,
            prefix="ENTITY_B",
        )
        training_pair_id = _ensure_training_pair(
            cursor=cursor,
            database=database,
            schema=schema,
            row=row,
            scope_id=scope_id,
            entity_a_snapshot_id=entity_a_snapshot_id,
            entity_b_snapshot_id=entity_b_snapshot_id,
        )
        _sync_pair_review(
            cursor=cursor,
            database=database,
            schema=schema,
            training_pair_id=training_pair_id,
            row=row,
        )
        if _feature_row_exists(
            cursor=cursor,
            database=database,
            schema=schema,
            entity_type=entity_type,
            training_pair_id=training_pair_id,
            feature_schema_version=feature_schema_version,
        ):
            skipped_pairs += 1
            continue
        feature_row = _build_feature_row(
            row={**row, "TRAINING_PAIR_ID": training_pair_id, "SCOPE_ID": scope_id},
            entity_type=entity_type,
            feature_schema_version=feature_schema_version,
            extractor=extractor,
        )
        _insert_feature_rows(
            cursor=cursor,
            database=database,
            schema=schema,
            entity_type=entity_type,
            feature_rows=[feature_row],
        )
        inserted_pairs += 1
    return MaterializedTrainingFeatures(
        entity_type=entity_type,
        feature_schema_version=feature_schema_version,
        selected_pairs=len(rows),
        inserted_pairs=inserted_pairs,
        skipped_pairs=skipped_pairs,
        source_run_id=resolved_run_id,
    )


def _load_reviewed_pairs_from_runtime(
    *,
    cursor: Any,
    database: str,
    schema: str,
    source_database: str,
    source_schema: str,
    team_id: str | None,
    entity_type: EntityType,
    run_selection: RunSelectionPolicy,
    source_run_id: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Load reviewed pair rows from ER_RUNTIME and any existing training mapping."""
    cache_table = (
        "DENORMALIZED_ORGANIZATION_CACHE"
        if entity_type == "organization"
        else "DENORMALIZED_SERVICE_CACHE"
    )
    entity_a_selects = _typed_cache_selects(
        entity_type=entity_type,
        cache_alias="cache_a",
        prefix="ENTITY_A",
    )
    entity_b_selects = _typed_cache_selects(
        entity_type=entity_type,
        cache_alias="cache_b",
        prefix="ENTITY_B",
    )
    ctes: list[str] = []
    params: list[Any] = []
    latest_run_clause = ""
    if run_selection == "latest" and source_run_id is None:
        params.append(entity_type)
        if team_id is not None:
            params.append(team_id)
        ctes.append(
            f"""
        latest_run AS (
            SELECT DEDUPLICATION_RUN_ID
            FROM (
                SELECT
                    dps.DEDUPLICATION_RUN_ID,
                    MAX(COALESCE(dps.REVIEWED_AT, dps.CREATED_AT)) AS latest_activity
                FROM {source_database}.{source_schema}.DUPLICATE_PAIR_SCORES dps
                WHERE dps.ENTITY_TYPE = %s
                  AND dps.IS_DUPLICATE IS NOT NULL
                  {"AND dps.TEAM_ID = %s" if team_id is not None else ""}
                GROUP BY dps.DEDUPLICATION_RUN_ID
                QUALIFY ROW_NUMBER() OVER (
                    ORDER BY latest_activity DESC, DEDUPLICATION_RUN_ID DESC
                ) = 1
            )
        )
        """.strip()
        )
        latest_run_clause = (
            "AND dps.DEDUPLICATION_RUN_ID = (SELECT DEDUPLICATION_RUN_ID FROM latest_run)"
        )
    run_clause = "AND dps.DEDUPLICATION_RUN_ID = %s" if source_run_id is not None else ""
    limit_clause = f"LIMIT {limit}" if limit is not None and limit > 0 else ""
    params.append(entity_type)
    if team_id is not None:
        params.append(team_id)
    ctes.append(
        f"""
        reviewed_scores AS (
            SELECT
                dps.ID AS SOURCE_SCORE_ID,
                dps.DUPLICATE_PAIR_ID AS SOURCE_PAIR_ID,
                dps.DEDUPLICATION_RUN_ID AS SOURCE_RUN_ID,
                dps.TEAM_ID,
                dps.ENTITY_TYPE,
                dps.PREDICTED_DUPLICATE,
                dps.CONFIDENCE_SCORE,
                dps.LEGACY_CONFIDENCE_SCORE,
                dps.SHADOW_CONFIDENCE_SCORE,
                dps.SHADOW_LOG_ODDS,
                dps.CALIBRATION_VERSION,
                dps.DETERMINISTIC_SECTION_SCORE,
                dps.NLP_SECTION_SCORE,
                dps.ML_SECTION_SCORE,
                dps.RAW_DETERMINISTIC_SCORE,
                dps.RAW_NLP_SCORE,
                dps.RAW_ML_SCORE,
                dps.EMBEDDING_SIMILARITY,
                dps.IS_DUPLICATE AS TEAM_REVIEW_LABEL,
                dps.REVIEWED_BY,
                dps.REVIEWED_AT,
                dps.CREATED_AT AS SCORE_CREATED_AT
            FROM {source_database}.{source_schema}.DUPLICATE_PAIR_SCORES dps
            WHERE dps.ENTITY_TYPE = %s
              AND dps.IS_DUPLICATE IS NOT NULL
              {"AND dps.TEAM_ID = %s" if team_id is not None else ""}
              {run_clause}
              {latest_run_clause}
        )
        """.strip()
    )
    ctes.append(
        f"""
        mitigation AS (
            SELECT
                mp.DUPLICATE_PAIR_ID,
                mp.DEDUPLICATION_RUN_ID,
                MAX(mp.MITIGATION_REASON) AS MITIGATION_REASON
            FROM {source_database}.{source_schema}.MITIGATED_PAIRS mp
            GROUP BY mp.DUPLICATE_PAIR_ID, mp.DEDUPLICATION_RUN_ID
        )
        """.strip()
    )
    sql = f"""
        WITH {", ".join(ctes)}
        SELECT
            rs.SOURCE_SCORE_ID,
            rs.SOURCE_PAIR_ID,
            rs.SOURCE_RUN_ID,
            rs.TEAM_ID,
            rs.ENTITY_TYPE,
            rs.PREDICTED_DUPLICATE,
            rs.CONFIDENCE_SCORE,
            rs.LEGACY_CONFIDENCE_SCORE,
            rs.SHADOW_CONFIDENCE_SCORE,
            rs.SHADOW_LOG_ODDS,
            rs.CALIBRATION_VERSION,
            rs.DETERMINISTIC_SECTION_SCORE,
            rs.NLP_SECTION_SCORE,
            rs.ML_SECTION_SCORE,
            rs.RAW_DETERMINISTIC_SCORE,
            rs.RAW_NLP_SCORE,
            rs.RAW_ML_SCORE,
            rs.EMBEDDING_SIMILARITY,
            rs.TEAM_REVIEW_LABEL,
            rs.REVIEWED_BY,
            rs.REVIEWED_AT,
            rs.SCORE_CREATED_AT,
            dp.ENTITY_A_ID,
            dp.ENTITY_B_ID,
            CASE
                WHEN dp.ENTITY_A_ID < dp.ENTITY_B_ID
                    THEN dp.ENTITY_A_ID || '|' || dp.ENTITY_B_ID
                ELSE dp.ENTITY_B_ID || '|' || dp.ENTITY_A_ID
            END AS PAIR_CANONICAL_KEY,
            dr.FULL_CONFIG AS PIPELINE_CONFIG_SNAPSHOT,
            dr.DUPLICATE_THRESHOLD,
            dr.MAYBE_THRESHOLD,
            dr.WEIGHT_DETERMINISTIC_SECTION,
            dr.WEIGHT_NLP_SECTION,
            dr.LIGHTGBM_MODEL_VERSION,
            dr.EMBEDDING_MODEL,
            dr.JOB_NAME,
            IFF(m.MITIGATION_REASON IS NULL, FALSE, TRUE) AS WAS_MITIGATED,
            m.MITIGATION_REASON,
            cache_a.ID AS ENTITY_A_ID_CONFIRMED,
            cache_a.SOURCE_SCHEMA AS ENTITY_A_SOURCE_SCHEMA,
            cache_a.ORIGINAL_ID AS ENTITY_A_ORIGINAL_ID,
            cache_a.RESOURCE_WRITER_NAME AS ENTITY_A_RESOURCE_WRITER_NAME,
            cache_a.ASSURED_DATE AS ENTITY_A_ASSURED_DATE,
            cache_a.ASSURER_EMAIL AS ENTITY_A_ASSURER_EMAIL,
            cache_a.LAST_UPDATED AS ENTITY_A_LAST_UPDATED,
            {entity_a_selects},
            cache_b.ID AS ENTITY_B_ID_CONFIRMED,
            cache_b.SOURCE_SCHEMA AS ENTITY_B_SOURCE_SCHEMA,
            cache_b.ORIGINAL_ID AS ENTITY_B_ORIGINAL_ID,
            cache_b.RESOURCE_WRITER_NAME AS ENTITY_B_RESOURCE_WRITER_NAME,
            cache_b.ASSURED_DATE AS ENTITY_B_ASSURED_DATE,
            cache_b.ASSURER_EMAIL AS ENTITY_B_ASSURER_EMAIL,
            cache_b.LAST_UPDATED AS ENTITY_B_LAST_UPDATED,
            {entity_b_selects},
            tp.TRAINING_PAIR_ID AS EXISTING_TRAINING_PAIR_ID
        FROM reviewed_scores rs
        JOIN {source_database}.{source_schema}.DUPLICATE_PAIRS dp
          ON rs.SOURCE_PAIR_ID = dp.ID
        JOIN {source_database}.{source_schema}.DEDUPLICATION_RUN dr
          ON rs.SOURCE_RUN_ID = dr.ID
        LEFT JOIN mitigation m
          ON rs.SOURCE_PAIR_ID = m.DUPLICATE_PAIR_ID
         AND rs.SOURCE_RUN_ID = m.DEDUPLICATION_RUN_ID
        JOIN {source_database}.{source_schema}.{cache_table} cache_a
          ON dp.ENTITY_A_ID = cache_a.ID
        JOIN {source_database}.{source_schema}.{cache_table} cache_b
          ON dp.ENTITY_B_ID = cache_b.ID
        LEFT JOIN {database}.{schema}.TRAINING_PAIR tp
          ON tp.SOURCE_SCORE_ID = rs.SOURCE_SCORE_ID
        ORDER BY rs.REVIEWED_AT DESC, rs.SOURCE_SCORE_ID
        {limit_clause}
    """
    if source_run_id is not None:
        params.append(source_run_id)
    cursor.execute(sql, tuple(params))
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def _typed_cache_selects(
    *,
    entity_type: EntityType,
    cache_alias: str,
    prefix: str,
) -> str:
    """Build the typed cache select list for one entity side."""
    if entity_type == "organization":
        type_specific_columns = (
            (f"{cache_alias}.EMAIL", f"{prefix}_EMAIL"),
            (f"{cache_alias}.IDENTIFIERS", f"{prefix}_IDENTIFIERS"),
            (f"{cache_alias}.SERVICES", f"{prefix}_SERVICES_ROLLUP"),
            ("NULL", f"{prefix}_ORGANIZATION_ID"),
            ("NULL", f"{prefix}_ORGANIZATION_NAME"),
            ("NULL", f"{prefix}_APPLICATION_PROCESS"),
            ("NULL", f"{prefix}_FEES_DESCRIPTION"),
            ("NULL", f"{prefix}_ELIGIBILITY_DESCRIPTION"),
        )
    else:
        type_specific_columns = (
            ("NULL", f"{prefix}_EMAIL"),
            ("NULL", f"{prefix}_IDENTIFIERS"),
            ("NULL", f"{prefix}_SERVICES_ROLLUP"),
            (f"{cache_alias}.ORGANIZATION_ID", f"{prefix}_ORGANIZATION_ID"),
            (f"{cache_alias}.ORGANIZATION_NAME", f"{prefix}_ORGANIZATION_NAME"),
            (f"{cache_alias}.APPLICATION_PROCESS", f"{prefix}_APPLICATION_PROCESS"),
            (f"{cache_alias}.FEES_DESCRIPTION", f"{prefix}_FEES_DESCRIPTION"),
            (
                f"{cache_alias}.ELIGIBILITY_DESCRIPTION",
                f"{prefix}_ELIGIBILITY_DESCRIPTION",
            ),
        )
    select_columns = [
        f"{cache_alias}.NAME AS {prefix}_NAME",
        f"{cache_alias}.ALTERNATE_NAME AS {prefix}_ALTERNATE_NAME",
        f"{cache_alias}.DESCRIPTION AS {prefix}_DESCRIPTION",
        f"{cache_alias}.SHORT_DESCRIPTION AS {prefix}_SHORT_DESCRIPTION",
        f"{cache_alias}.PHONES AS {prefix}_PHONES",
        f"{cache_alias}.WEBSITES AS {prefix}_WEBSITES",
        f"{cache_alias}.LOCATIONS AS {prefix}_LOCATIONS",
        f"{cache_alias}.TAXONOMIES AS {prefix}_TAXONOMIES",
        *(f"{value} AS {alias}" for value, alias in type_specific_columns),
        f"NULL AS {prefix}_EMBEDDING_VECTOR",
    ]
    return ",\n            ".join(select_columns)


def _load_reason_map(
    *,
    cursor: Any,
    source_database: str,
    source_schema: str,
    source_score_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Load per-score reason rows for the reviewed source scores."""
    if not source_score_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(source_score_ids))
    cursor.execute(
        f"""
        SELECT
            DUPLICATE_PAIR_SCORE_ID,
            DUPLICATE_PAIR_ID,
            DEDUPLICATION_RUN_ID,
            MATCH_TYPE,
            ENTITY_TYPE,
            RAW_CONTRIBUTION,
            WEIGHTED_CONTRIBUTION,
            SIGNAL_WEIGHT,
            MATCHED_VALUE,
            ENTITY_A_VALUE,
            ENTITY_B_VALUE,
            SIMILARITY_SCORE,
            CREATED_AT
        FROM {source_database}.{source_schema}.DUPLICATE_REASONS
        WHERE DUPLICATE_PAIR_SCORE_ID IN ({placeholders})
        ORDER BY DUPLICATE_PAIR_SCORE_ID, CREATED_AT, MATCH_TYPE
        """,
        tuple(source_score_ids),
    )
    reason_map: dict[str, list[dict[str, Any]]] = {}
    for row in cursor.fetchall():
        score_id = str(row[0])
        reason_map.setdefault(score_id, []).append(
            {
                "DUPLICATE_PAIR_ID": row[1],
                "DEDUPLICATION_RUN_ID": row[2],
                "MATCH_TYPE": row[3],
                "ENTITY_TYPE": row[4],
                "RAW_CONTRIBUTION": row[5],
                "WEIGHTED_CONTRIBUTION": row[6],
                "SIGNAL_WEIGHT": row[7],
                "MATCHED_VALUE": row[8],
                "ENTITY_A_VALUE": row[9],
                "ENTITY_B_VALUE": row[10],
                "SIMILARITY_SCORE": row[11],
                "CREATED_AT": row[12],
            }
        )
    return reason_map


def _ensure_entity_snapshot(
    *,
    cursor: Any,
    database: str,
    schema: str,
    entity_type: EntityType,
    source_run_id: str | None,
    row: dict[str, Any],
    prefix: str,
) -> str:
    """Insert or reuse one immutable entity snapshot bundle."""
    source_entity = _source_entity_row(row=row, prefix=prefix, entity_type=entity_type)
    cleaned = _clean_entity_row(
        row=_clean_entity_input(source_entity=source_entity),
        entity_type=entity_type,
    )
    cursor.execute(
        f"""
        SELECT SNAPSHOT_ID
        FROM {database}.{schema}.ENTITY_SNAPSHOT
        WHERE ENTITY_ID = %s AND CONTENT_HASH = %s
        """,
        (cleaned["entity_id"], cleaned["content_hash"]),
    )
    existing = cursor.fetchone()
    if existing:
        return str(existing[0])

    snapshot_id = str(uuid.uuid4())
    cursor.execute(
        f"""
        INSERT INTO {database}.{schema}.ENTITY_SNAPSHOT (
            SNAPSHOT_ID, ENTITY_ID, ENTITY_TYPE, SOURCE_SCHEMA, ORIGINAL_ID,
            CONTENT_HASH, SOURCE_RUN_ID, RESOURCE_WRITER_NAME, ASSURED_DATE,
            ASSURER_EMAIL, LAST_UPDATED
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            snapshot_id,
            cleaned["entity_id"],
            entity_type,
            cleaned["source_schema"],
            source_entity.get("original_id"),
            cleaned["content_hash"],
            source_run_id,
            source_entity.get("resource_writer_name"),
            source_entity.get("assured_date"),
            source_entity.get("assurer_email"),
            source_entity.get("last_updated"),
        ),
    )
    child_table = "ORGANIZATION_SNAPSHOT" if entity_type == "organization" else "SERVICE_SNAPSHOT"
    cursor.execute(
        f"""
        INSERT INTO {database}.{schema}.{child_table} (
            SNAPSHOT_ID, NAME, ALTERNATE_NAME, DESCRIPTION, SHORT_DESCRIPTION, EMAIL,
            PHONES, WEBSITES, LOCATIONS, TAXONOMIES, IDENTIFIERS, SERVICES,
            ORGANIZATION_ID, ORGANIZATION_NAME, APPLICATION_PROCESS, FEES_DESCRIPTION,
            ELIGIBILITY_DESCRIPTION, EMBEDDING_VECTOR, EMBEDDING_MODEL_VERSION
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            snapshot_id,
            source_entity.get("name"),
            source_entity.get("alternate_name"),
            source_entity.get("description"),
            source_entity.get("short_description"),
            source_entity.get("email"),
            _json_or_none(source_entity.get("phones")),
            _json_or_none(source_entity.get("websites")),
            _json_or_none(source_entity.get("locations")),
            _json_or_none(source_entity.get("taxonomies")),
            _json_or_none(source_entity.get("identifiers")),
            _json_or_none(source_entity.get("services_rollup")),
            source_entity.get("organization_id"),
            source_entity.get("organization_name"),
            source_entity.get("application_process"),
            source_entity.get("fees_description"),
            source_entity.get("eligibility_description"),
            _json_or_none(source_entity.get("embedding_vector")),
            None,
        ),
    )
    return snapshot_id


def _ensure_training_pair(
    *,
    cursor: Any,
    database: str,
    schema: str,
    row: dict[str, Any],
    scope_id: str | None,
    entity_a_snapshot_id: str,
    entity_b_snapshot_id: str,
) -> str:
    """Insert or reuse one immutable training pair derived from a reviewed score row."""
    source_score_id = _safe_optional_str(row.get("SOURCE_SCORE_ID"))
    if source_score_id is None:
        message = "Missing SOURCE_SCORE_ID for training pair materialization."
        raise ValueError(message)
    cursor.execute(
        f"""
        SELECT TRAINING_PAIR_ID
        FROM {database}.{schema}.TRAINING_PAIR
        WHERE SOURCE_SCORE_ID = %s
        """,
        (source_score_id,),
    )
    existing = cursor.fetchone()
    if existing:
        return str(existing[0])

    training_pair_id = str(uuid.uuid4())
    baseline_confidence_score = _baseline_confidence_score(row)
    duplicate_threshold = _safe_float(row.get("DUPLICATE_THRESHOLD"), default=1.0)
    maybe_threshold = _safe_float(row.get("MAYBE_THRESHOLD"), default=1.0)
    pipeline_config_snapshot = _coerce_variant(row.get("PIPELINE_CONFIG_SNAPSHOT"))
    scalar_placeholders = ", ".join(["%s"] * 26)
    cursor.execute(
        f"""
        INSERT INTO {database}.{schema}.TRAINING_PAIR (
            TRAINING_PAIR_ID, ENTITY_A_SNAPSHOT_ID, ENTITY_B_SNAPSHOT_ID,
            ENTITY_A_ID, ENTITY_B_ID, ENTITY_TYPE, TEAM_ID, SCOPE_ID,
            SOURCE_RUN_ID, SOURCE_PAIR_ID, SOURCE_SCORE_ID,
            BASELINE_CONFIDENCE_SCORE, BASELINE_PREDICTED_DUPLICATE, BASELINE_PREDICTED_MAYBE,
            CONFIDENCE_SCORE, DETERMINISTIC_SECTION_SCORE, NLP_SECTION_SCORE, ML_SECTION_SCORE,
            RAW_DETERMINISTIC_SCORE, RAW_NLP_SCORE, RAW_ML_SCORE, EMBEDDING_SIMILARITY,
            PREDICTED_DUPLICATE, PREDICTED_MAYBE, WAS_MITIGATED, MITIGATION_REASON,
            PIPELINE_SIGNALS, PIPELINE_CONFIG_SNAPSHOT, PIPELINE_POLICY_VERSION,
            PIPELINE_MODEL_VERSION, PAIR_CANONICAL_KEY
        ) VALUES (
            {scalar_placeholders},
            PARSE_JSON(%s),
            PARSE_JSON(%s),
            %s, %s, %s
        )
        """,
        (
            training_pair_id,
            entity_a_snapshot_id,
            entity_b_snapshot_id,
            row.get("ENTITY_A_ID"),
            row.get("ENTITY_B_ID"),
            row.get("ENTITY_TYPE"),
            row.get("TEAM_ID"),
            scope_id or _safe_optional_str(row.get("JOB_NAME")),
            row.get("SOURCE_RUN_ID"),
            row.get("SOURCE_PAIR_ID"),
            source_score_id,
            baseline_confidence_score,
            baseline_confidence_score >= duplicate_threshold,
            baseline_confidence_score >= maybe_threshold,
            _safe_float(row.get("CONFIDENCE_SCORE"), default=0.0),
            _safe_float(row.get("DETERMINISTIC_SECTION_SCORE"), default=0.0),
            _safe_float(row.get("NLP_SECTION_SCORE"), default=0.0),
            _safe_float(row.get("ML_SECTION_SCORE"), default=0.0),
            _safe_float(row.get("RAW_DETERMINISTIC_SCORE"), default=0.0),
            _safe_float(row.get("RAW_NLP_SCORE"), default=0.0),
            _safe_float(row.get("RAW_ML_SCORE"), default=0.0),
            _safe_float(row.get("EMBEDDING_SIMILARITY"), default=0.0),
            bool(row.get("PREDICTED_DUPLICATE")),
            _safe_float(row.get("CONFIDENCE_SCORE"), default=0.0) >= maybe_threshold,
            bool(row.get("WAS_MITIGATED")),
            _safe_optional_str(row.get("MITIGATION_REASON")),
            json.dumps(row.get("PIPELINE_SIGNALS") or []),
            json.dumps(pipeline_config_snapshot or {}),
            _extract_policy_version(pipeline_config_snapshot),
            _extract_model_version(row=row, pipeline_config_snapshot=pipeline_config_snapshot),
            row.get("PAIR_CANONICAL_KEY"),
        ),
    )
    return training_pair_id


def _sync_pair_review(
    *,
    cursor: Any,
    database: str,
    schema: str,
    training_pair_id: str,
    row: dict[str, Any],
) -> None:
    """Synchronize derived PAIR_REVIEW rows from reviewed DUPLICATE_PAIR_SCORES rows."""
    source_score_id = _safe_optional_str(row.get("SOURCE_SCORE_ID"))
    reviewed_at = row.get("REVIEWED_AT")
    reviewed_by = _safe_optional_str(row.get("REVIEWED_BY"))
    review_decision = _review_decision_from_bool(row.get("TEAM_REVIEW_LABEL"))
    if (
        source_score_id is None
        or reviewed_at is None
        or reviewed_by is None
        or review_decision is None
    ):
        return
    cursor.execute(
        f"""
        SELECT REVIEW_ID
        FROM {database}.{schema}.PAIR_REVIEW
        WHERE SOURCE_SCORE_ID = %s
          AND REVIEWED_AT = %s
          AND REVIEW_DECISION = %s
          AND REVIEWED_BY = %s
        """,
        (source_score_id, reviewed_at, review_decision, reviewed_by),
    )
    if cursor.fetchone():
        return
    cursor.execute(
        f"""
        SELECT REVIEW_ID
        FROM {database}.{schema}.PAIR_REVIEW
        WHERE SOURCE_SCORE_ID = %s
          AND IS_ACTIVE = TRUE
        ORDER BY REVIEWED_AT DESC
        """,
        (source_score_id,),
    )
    active_review_ids = [str(result[0]) for result in cursor.fetchall()]
    review_id = str(uuid.uuid4())
    if active_review_ids:
        cursor.execute(
            f"""
            UPDATE {database}.{schema}.PAIR_REVIEW
            SET IS_ACTIVE = FALSE, SUPERSEDED_BY = %s, SUPERSEDED_AT = CURRENT_TIMESTAMP()
            WHERE SOURCE_SCORE_ID = %s
              AND IS_ACTIVE = TRUE
            """,
            (review_id, source_score_id),
        )
    cursor.execute(
        f"""
        INSERT INTO {database}.{schema}.PAIR_REVIEW (
            REVIEW_ID, TRAINING_PAIR_ID, SOURCE_SCORE_ID, REVIEW_SOURCE,
            REVIEWED_BY, REVIEW_DECISION, IS_ACTIVE, REVIEWED_AT
        ) VALUES (%s, %s, %s, %s, %s, %s, TRUE, %s)
        """,
        (
            review_id,
            training_pair_id,
            source_score_id,
            _REVIEW_SOURCE,
            reviewed_by,
            review_decision,
            reviewed_at,
        ),
    )


def _feature_row_exists(
    *,
    cursor: Any,
    database: str,
    schema: str,
    entity_type: EntityType,
    training_pair_id: str,
    feature_schema_version: str,
) -> bool:
    """Return whether a typed pair-feature row already exists."""
    feature_table = (
        "ORGANIZATION_PAIR_FEATURES" if entity_type == "organization" else "SERVICE_PAIR_FEATURES"
    )
    cursor.execute(
        f"""
        SELECT 1
        FROM {database}.{schema}.{feature_table}
        WHERE TRAINING_PAIR_ID = %s
          AND FEATURE_SCHEMA_VERSION = %s
        """,
        (training_pair_id, feature_schema_version),
    )
    return cursor.fetchone() is not None


def _build_feature_row(
    *,
    row: dict[str, Any],
    entity_type: EntityType,
    feature_schema_version: str,
    extractor: Any,
) -> dict[str, Any]:
    """Convert one reviewed training pair into one typed feature row."""
    pair = {
        "pair_key": row["PAIR_CANONICAL_KEY"],
        "embedding_similarity": _safe_float(row.get("EMBEDDING_SIMILARITY"), default=0.0),
        "entity_a": _legacy_entity_from_prefixed_row(row=row, prefix="ENTITY_A"),
        "entity_b": _legacy_entity_from_prefixed_row(row=row, prefix="ENTITY_B"),
        "signal_overrides": build_signal_overrides_from_pipeline_signals(
            pipeline_signals=_coerce_variant(row.get("PIPELINE_SIGNALS")),
            nlp_score=_safe_float(row.get("NLP_SECTION_SCORE"), default=0.0),
        ),
    }
    features = build_api_feature_payload(
        pair=pair,
        extractor=extractor,
        entity_type=entity_type,
    )
    return {
        "PAIR_FEATURE_ID": str(uuid.uuid4()),
        "TRAINING_PAIR_ID": str(row["TRAINING_PAIR_ID"]),
        "ENTITY_TYPE": entity_type,
        "TEAM_ID": _safe_optional_str(row.get("TEAM_ID")),
        "SCOPE_ID": _safe_optional_str(row.get("SCOPE_ID")),
        "SOURCE_RUN_ID": _safe_optional_str(row.get("SOURCE_RUN_ID")),
        "FEATURE_SCHEMA_VERSION": feature_schema_version,
        "BASELINE_SCORE_BAND": _score_band_for_row(row),
        "BASELINE_CONFIDENCE_SCORE": _baseline_confidence_score(row),
        **{key.upper(): value for key, value in features.items()},
    }


def _insert_feature_rows(
    *,
    cursor: Any,
    database: str,
    schema: str,
    entity_type: EntityType,
    feature_rows: list[dict[str, Any]],
) -> None:
    """Bulk-insert typed pair-feature rows."""
    if not feature_rows:
        return
    feature_table = (
        "ORGANIZATION_PAIR_FEATURES" if entity_type == "organization" else "SERVICE_PAIR_FEATURES"
    )
    columns = (
        "PAIR_FEATURE_ID",
        "TRAINING_PAIR_ID",
        "ENTITY_TYPE",
        "TEAM_ID",
        "SCOPE_ID",
        "SOURCE_RUN_ID",
        "FEATURE_SCHEMA_VERSION",
        "BASELINE_SCORE_BAND",
        "BASELINE_CONFIDENCE_SCORE",
        *(feature.upper() for feature in feature_names_for_entity_type(entity_type)),
    )
    placeholders = ", ".join(["%s"] * len(columns))
    cursor.executemany(
        f"""
        INSERT INTO {database}.{schema}.{feature_table}
        ({", ".join(columns)})
        VALUES ({placeholders})
        """,
        [tuple(row.get(column) for column in columns) for row in feature_rows],
    )


def _source_entity_row(
    *,
    row: dict[str, Any],
    prefix: str,
    entity_type: EntityType,
) -> dict[str, Any]:
    """Build a raw entity row suitable for clean-entity hashing and snapshot storage."""
    email = _safe_optional_str(row.get(f"{prefix}_EMAIL"))
    return {
        "entity_id": row.get(f"{prefix}_ID"),
        "source_schema": row.get(f"{prefix}_SOURCE_SCHEMA"),
        "name": row.get(f"{prefix}_NAME"),
        "alternate_name": row.get(f"{prefix}_ALTERNATE_NAME"),
        "description": row.get(f"{prefix}_DESCRIPTION"),
        "short_description": row.get(f"{prefix}_SHORT_DESCRIPTION"),
        "emails": [email] if email else [],
        "phones": _coerce_variant(row.get(f"{prefix}_PHONES")),
        "websites": _coerce_variant(row.get(f"{prefix}_WEBSITES")),
        "locations": _coerce_variant(row.get(f"{prefix}_LOCATIONS")),
        "taxonomies": _coerce_variant(row.get(f"{prefix}_TAXONOMIES")),
        "identifiers": _coerce_variant(row.get(f"{prefix}_IDENTIFIERS")),
        "services_rollup": _coerce_variant(row.get(f"{prefix}_SERVICES_ROLLUP")),
        "organization_name": row.get(f"{prefix}_ORGANIZATION_NAME"),
        "organization_id": row.get(f"{prefix}_ORGANIZATION_ID"),
        "embedding_vector": _coerce_variant(row.get(f"{prefix}_EMBEDDING_VECTOR")),
        "application_process": row.get(f"{prefix}_APPLICATION_PROCESS"),
        "fees_description": row.get(f"{prefix}_FEES_DESCRIPTION"),
        "eligibility_description": row.get(f"{prefix}_ELIGIBILITY_DESCRIPTION"),
        "resource_writer_name": row.get(f"{prefix}_RESOURCE_WRITER_NAME"),
        "assured_date": row.get(f"{prefix}_ASSURED_DATE"),
        "assurer_email": row.get(f"{prefix}_ASSURER_EMAIL"),
        "original_id": row.get(f"{prefix}_ORIGINAL_ID"),
        "last_updated": row.get(f"{prefix}_LAST_UPDATED"),
        "entity_type": entity_type,
        "email": email,
    }


def _clean_entity_input(*, source_entity: dict[str, Any]) -> RawEntityRowInput:
    """Project a source-entity snapshot into the typed cleaner input contract."""
    return {
        "entity_id": source_entity.get("entity_id"),
        "source_schema": source_entity.get("source_schema"),
        "name": source_entity.get("name"),
        "alternate_name": source_entity.get("alternate_name"),
        "description": source_entity.get("description"),
        "short_description": source_entity.get("short_description"),
        "emails": source_entity.get("emails"),
        "phones": source_entity.get("phones"),
        "websites": source_entity.get("websites"),
        "locations": source_entity.get("locations"),
        "taxonomies": source_entity.get("taxonomies"),
        "identifiers": source_entity.get("identifiers"),
        "services_rollup": source_entity.get("services_rollup"),
        "organization_name": source_entity.get("organization_name"),
        "organization_id": source_entity.get("organization_id"),
        "embedding_vector": source_entity.get("embedding_vector"),
        "application_process": source_entity.get("application_process"),
        "fees_description": source_entity.get("fees_description"),
        "eligibility_description": source_entity.get("eligibility_description"),
        "resource_writer_name": source_entity.get("resource_writer_name"),
        "assured_date": source_entity.get("assured_date"),
        "assurer_email": source_entity.get("assurer_email"),
        "original_id": source_entity.get("original_id"),
    }


def _legacy_entity_from_prefixed_row(*, row: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Convert one typed snapshot row into the legacy extractor input shape."""
    email = _safe_optional_str(row.get(f"{prefix}_EMAIL"))
    return to_legacy_entity(
        row={
            "entity_id": row.get(f"{prefix}_ID"),
            "name": row.get(f"{prefix}_NAME"),
            "description": row.get(f"{prefix}_DESCRIPTION"),
            "emails": [email] if email else [],
            "phones": _coerce_variant(row.get(f"{prefix}_PHONES")),
            "websites": _coerce_variant(row.get(f"{prefix}_WEBSITES")),
            "locations": _coerce_variant(row.get(f"{prefix}_LOCATIONS")),
            "taxonomies": _coerce_variant(row.get(f"{prefix}_TAXONOMIES")),
            "identifiers": _coerce_variant(row.get(f"{prefix}_IDENTIFIERS")),
            "services_rollup": _coerce_variant(row.get(f"{prefix}_SERVICES_ROLLUP")),
            "organization_name": row.get(f"{prefix}_ORGANIZATION_NAME"),
            "organization_id": row.get(f"{prefix}_ORGANIZATION_ID"),
            "embedding_vector": _coerce_variant(row.get(f"{prefix}_EMBEDDING_VECTOR")),
        }
    )


def _coerce_variant(value: Any, *, wrap_scalar: bool = False) -> Any:
    """Best-effort normalize Snowflake VARIANT values into Python containers."""
    if value is None:
        return [] if wrap_scalar else []
    if isinstance(value, (list, dict)):
        return value
    if wrap_scalar and isinstance(value, str):
        return [value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return [] if wrap_scalar else []
        if wrap_scalar and not stripped.startswith(("[", "{", '"')):
            return [stripped]
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped] if wrap_scalar else stripped
    return [value] if wrap_scalar else value


def _baseline_confidence_score(row: dict[str, Any]) -> float:
    """Compute the ML-free baseline score from section scores and run weights."""
    det_weight = _safe_float(row.get("WEIGHT_DETERMINISTIC_SECTION"), default=0.0)
    nlp_weight = _safe_float(row.get("WEIGHT_NLP_SECTION"), default=0.0)
    weight_total = det_weight + nlp_weight
    if weight_total <= 0.0:
        return 0.0
    return (
        _safe_float(row.get("DETERMINISTIC_SECTION_SCORE"), default=0.0)
        + _safe_float(row.get("NLP_SECTION_SCORE"), default=0.0)
    ) / weight_total


def _score_band_for_row(row: dict[str, Any]) -> str:
    """Resolve the baseline score band persisted with a feature row."""
    baseline_confidence_score = _baseline_confidence_score(row)
    duplicate_threshold = _safe_float(row.get("DUPLICATE_THRESHOLD"), default=1.0)
    maybe_threshold = _safe_float(row.get("MAYBE_THRESHOLD"), default=1.0)
    if baseline_confidence_score >= duplicate_threshold:
        return "duplicate"
    if baseline_confidence_score >= maybe_threshold:
        return "maybe"
    return "below_maybe"


def _resolve_latest_source_run_id(
    *,
    rows: list[dict[str, Any]],
    run_selection: RunSelectionPolicy,
) -> str | None:
    """Return a representative source run ID for summary display."""
    if run_selection != "latest":
        return None
    for row in rows:
        source_run_id = _safe_optional_str(row.get("SOURCE_RUN_ID"))
        if source_run_id:
            return source_run_id
    return None


def _review_decision_from_bool(value: Any) -> str | None:
    """Map UI review booleans to training review-decision vocabulary."""
    if value is True:
        return "TRUE_DUPLICATE"
    if value is False:
        return "FALSE_POSITIVE"
    return None


def _log_reviewed_score_diagnostics(
    *,
    rows: list[dict[str, Any]],
    entity_type: EntityType,
) -> None:
    """Log lightweight reviewed-label diagnostics for legacy vs shadow scores."""
    if not rows:
        return
    shadow_present = any(row.get("SHADOW_CONFIDENCE_SCORE") is not None for row in rows)
    if not shadow_present:
        return
    positives = [row for row in rows if row.get("TEAM_REVIEW_LABEL") is True]
    negatives = [row for row in rows if row.get("TEAM_REVIEW_LABEL") is False]
    if not positives or not negatives:
        return
    legacy_top_rate = _top_decile_duplicate_rate(rows=rows, score_field="CONFIDENCE_SCORE")
    shadow_top_rate = _top_decile_duplicate_rate(rows=rows, score_field="SHADOW_CONFIDENCE_SCORE")
    _log.info(
        "training_feature_store reviewed score shadow | entity_type=%s reviewed=%d"
        " legacy_pos_mean=%.3f legacy_neg_mean=%.3f shadow_pos_mean=%.3f shadow_neg_mean=%.3f"
        " legacy_top_decile_dup_rate=%.3f shadow_top_decile_dup_rate=%.3f",
        entity_type,
        len(rows),
        _mean_score(positives, "CONFIDENCE_SCORE"),
        _mean_score(negatives, "CONFIDENCE_SCORE"),
        _mean_score(positives, "SHADOW_CONFIDENCE_SCORE"),
        _mean_score(negatives, "SHADOW_CONFIDENCE_SCORE"),
        legacy_top_rate,
        shadow_top_rate,
    )


def _mean_score(rows: list[dict[str, Any]], score_field: str) -> float:
    scores = [_safe_float(row.get(score_field), default=0.0) for row in rows]
    return sum(scores) / max(len(scores), 1)


def _top_decile_duplicate_rate(*, rows: list[dict[str, Any]], score_field: str) -> float:
    ordered = sorted(
        rows,
        key=lambda row: _safe_float(row.get(score_field), default=0.0),
        reverse=True,
    )
    cutoff = max(len(ordered) // 10, 1)
    top = ordered[:cutoff]
    positives = sum(1 for row in top if row.get("TEAM_REVIEW_LABEL") is True)
    return positives / max(len(top), 1)


def _extract_policy_version(pipeline_config_snapshot: Any) -> str | None:
    """Extract policy version from run full-config payload when present."""
    config = pipeline_config_snapshot if isinstance(pipeline_config_snapshot, dict) else {}
    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        return _safe_optional_str(metadata.get("policy_version"))
    return _safe_optional_str(config.get("policy_version"))


def _extract_model_version(*, row: dict[str, Any], pipeline_config_snapshot: Any) -> str | None:
    """Extract a stable model version for auditability."""
    config = pipeline_config_snapshot if isinstance(pipeline_config_snapshot, dict) else {}
    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        model_version = _safe_optional_str(metadata.get("model_version"))
        if model_version is not None:
            return model_version
    return _safe_optional_str(row.get("LIGHTGBM_MODEL_VERSION")) or _safe_optional_str(
        row.get("EMBEDDING_MODEL")
    )


def _json_or_none(value: Any) -> str | None:
    """Serialize JSON-ish values for Snowflake VARIANT inserts."""
    if value is None:
        return None
    return json.dumps(value)


def _safe_float(value: Any, *, default: float) -> float:
    """Convert unknown numeric-ish values to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_optional_str(value: Any) -> str | None:
    """Convert unknown value to stripped string or None."""
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None
