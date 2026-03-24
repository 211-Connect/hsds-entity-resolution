"""Training schema bootstrap and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hsds_entity_resolution.core.training_features import (
    ORGANIZATION_FEATURES,
    SERVICE_FEATURES,
)


def training_schema_contract() -> dict[str, tuple[str, ...]]:
    """Return required training-data tables and columns."""
    shared_snapshot_columns = (
        "SNAPSHOT_ID",
        "ENTITY_ID",
        "ENTITY_TYPE",
        "SOURCE_SCHEMA",
        "ORIGINAL_ID",
        "CONTENT_HASH",
        "CAPTURED_AT",
        "SOURCE_RUN_ID",
        "RESOURCE_WRITER_NAME",
        "ASSURED_DATE",
        "ASSURER_EMAIL",
        "LAST_UPDATED",
    )
    typed_snapshot_columns = (
        "SNAPSHOT_ID",
        "NAME",
        "ALTERNATE_NAME",
        "DESCRIPTION",
        "SHORT_DESCRIPTION",
        "EMAIL",
        "PHONES",
        "WEBSITES",
        "LOCATIONS",
        "TAXONOMIES",
        "IDENTIFIERS",
        "SERVICES",
        "ORGANIZATION_ID",
        "ORGANIZATION_NAME",
        "APPLICATION_PROCESS",
        "FEES_DESCRIPTION",
        "ELIGIBILITY_DESCRIPTION",
        "EMBEDDING_VECTOR",
        "EMBEDDING_MODEL_VERSION",
    )
    return {
        "ENTITY_SNAPSHOT": shared_snapshot_columns,
        "ORGANIZATION_SNAPSHOT": typed_snapshot_columns,
        "SERVICE_SNAPSHOT": typed_snapshot_columns,
        "TRAINING_PAIR": (
            "TRAINING_PAIR_ID",
            "ENTITY_A_SNAPSHOT_ID",
            "ENTITY_B_SNAPSHOT_ID",
            "ENTITY_A_ID",
            "ENTITY_B_ID",
            "ENTITY_TYPE",
            "TEAM_ID",
            "SCOPE_ID",
            "SOURCE_RUN_ID",
            "SOURCE_PAIR_ID",
            "SOURCE_SCORE_ID",
            "BASELINE_CONFIDENCE_SCORE",
            "BASELINE_PREDICTED_DUPLICATE",
            "BASELINE_PREDICTED_MAYBE",
            "CONFIDENCE_SCORE",
            "DETERMINISTIC_SECTION_SCORE",
            "NLP_SECTION_SCORE",
            "ML_SECTION_SCORE",
            "RAW_DETERMINISTIC_SCORE",
            "RAW_NLP_SCORE",
            "RAW_ML_SCORE",
            "EMBEDDING_SIMILARITY",
            "PREDICTED_DUPLICATE",
            "PREDICTED_MAYBE",
            "WAS_MITIGATED",
            "MITIGATION_REASON",
            "PIPELINE_SIGNALS",
            "PIPELINE_CONFIG_SNAPSHOT",
            "PIPELINE_POLICY_VERSION",
            "PIPELINE_MODEL_VERSION",
            "PAIR_CANONICAL_KEY",
            "CREATED_AT",
        ),
        "PAIR_REVIEW": (
            "REVIEW_ID",
            "TRAINING_PAIR_ID",
            "SOURCE_SCORE_ID",
            "REVIEW_SOURCE",
            "REVIEWED_BY",
            "REVIEW_SESSION_ID",
            "REVIEW_DECISION",
            "REVIEWER_CONFIDENCE",
            "REVIEW_NOTES",
            "IS_ACTIVE",
            "SUPERSEDED_BY",
            "SUPERSEDED_AT",
            "REVIEWED_AT",
        ),
        "TRAINING_DATASET": (
            "DATASET_ID",
            "DATASET_NAME",
            "DATASET_VERSION",
            "DESCRIPTION",
            "ENTITY_TYPE",
            "IS_ACTIVE",
            "CREATED_AT",
            "CREATED_BY",
            "TOTAL_PAIRS",
            "TRUE_DUPLICATE_COUNT",
            "FALSE_POSITIVE_COUNT",
            "UNSURE_COUNT",
        ),
        "TRAINING_DATASET_MEMBER": (
            "DATASET_ID",
            "TRAINING_PAIR_ID",
            "LABEL",
            "LABEL_SOURCE",
            "REVIEW_ID",
            "ADDED_AT",
            "ADDED_BY",
        ),
        "ORGANIZATION_PAIR_FEATURES": (
            "PAIR_FEATURE_ID",
            "TRAINING_PAIR_ID",
            "ENTITY_TYPE",
            "TEAM_ID",
            "SCOPE_ID",
            "SOURCE_RUN_ID",
            "FEATURE_SCHEMA_VERSION",
            "BASELINE_SCORE_BAND",
            "BASELINE_CONFIDENCE_SCORE",
            *(feature.upper() for feature in ORGANIZATION_FEATURES),
            "CREATED_AT",
        ),
        "SERVICE_PAIR_FEATURES": (
            "PAIR_FEATURE_ID",
            "TRAINING_PAIR_ID",
            "ENTITY_TYPE",
            "TEAM_ID",
            "SCOPE_ID",
            "SOURCE_RUN_ID",
            "FEATURE_SCHEMA_VERSION",
            "BASELINE_SCORE_BAND",
            "BASELINE_CONFIDENCE_SCORE",
            *(feature.upper() for feature in SERVICE_FEATURES),
            "CREATED_AT",
        ),
    }


def ensure_training_schema(connection: Any, *, database: str, schema: str) -> None:
    """Create and validate the training schema before materialization begins."""
    if _schema_is_valid(connection=connection, database=database, schema=schema):
        return
    _apply_training_schema_ddl(connection=connection, database=database, schema=schema)
    if not _schema_is_valid(connection=connection, database=database, schema=schema):
        message = (
            f"Training schema {database}.{schema} is still missing required tables or columns "
            "after bootstrap."
        )
        raise RuntimeError(message)


def _schema_is_valid(*, connection: Any, database: str, schema: str) -> bool:
    """Return True when every required training-data table and column exists."""
    expected = training_schema_contract()
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"""
            SELECT TABLE_NAME, COLUMN_NAME
            FROM {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s
            """,
            (schema.upper(),),
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()
    if not rows:
        return False
    actual: dict[str, set[str]] = {}
    for table_name, column_name in rows:
        actual.setdefault(str(table_name), set()).add(str(column_name))
    for table_name, required_columns in expected.items():
        actual_columns = actual.get(table_name)
        if actual_columns is None:
            return False
        if not set(required_columns).issubset(actual_columns):
            return False
    return True


def _apply_training_schema_ddl(*, connection: Any, database: str, schema: str) -> None:
    """Execute the checked-in training schema DDL with configured database/schema."""
    ddl_path = Path(__file__).resolve().parents[3] / "scripts" / "create_training_data_schema.sql"
    ddl = ddl_path.read_text(encoding="utf-8").format(database=database, schema=schema)
    connection.execute_string(ddl)
