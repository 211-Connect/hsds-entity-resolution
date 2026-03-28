"""Contract tests for required consumer SQL templates."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from consumer.consumer_types.deduplication_common import (
    COMMON_SCHEMA_TABLE_MODELS,
    DatabaseTableName,
    column_names,
)

REQUIRED_SQL_TEMPLATES = [
    "merge_deduplication_run.sql",
    "merge_denormalized_organization_cache.sql",
    "merge_denormalized_service_cache.sql",
    "merge_duplicate_pairs.sql",
    "merge_duplicate_pair_scores.sql",
    "merge_duplicate_reasons.sql",
    "merge_mitigated_pairs.sql",
    "merge_duplicate_clusters.sql",
    "merge_duplicate_cluster_pairs.sql",
    "delete_cascade_removed_pairs.sql",
    "delete_orphan_duplicate_reasons.sql",
    "recompute_cluster_aggregates.sql",
]

# Tables that have a MERGE template and a corresponding Pydantic row model.
# Ordered to match PersistenceExecutor._MERGE_TEMPLATE_BY_TABLE for readability.
_MERGE_COVERAGE_CASES: list[tuple[DatabaseTableName, str]] = [
    ("DEDUPLICATION_RUN", "merge_deduplication_run.sql"),
    ("DENORMALIZED_ORGANIZATION_CACHE", "merge_denormalized_organization_cache.sql"),
    ("DENORMALIZED_SERVICE_CACHE", "merge_denormalized_service_cache.sql"),
    ("DUPLICATE_PAIRS", "merge_duplicate_pairs.sql"),
    ("DUPLICATE_PAIR_SCORES", "merge_duplicate_pair_scores.sql"),
    ("DUPLICATE_REASONS", "merge_duplicate_reasons.sql"),
    ("MITIGATED_PAIRS", "merge_mitigated_pairs.sql"),
    ("DUPLICATE_CLUSTERS", "merge_duplicate_clusters.sql"),
    ("DUPLICATE_CLUSTER_PAIRS", "merge_duplicate_cluster_pairs.sql"),
]


def _parse_insert_columns(sql: str) -> frozenset[str]:
    """Return column names declared in the MERGE WHEN NOT MATCHED INSERT (...) block."""
    match = re.search(
        r"WHEN NOT MATCHED THEN INSERT\s*\(([^)]+)\)",
        sql,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return frozenset()
    block = match.group(1)
    return frozenset(col.strip() for col in block.split(",") if col.strip())


def test_required_sql_templates_exist_and_compile() -> None:
    """Required SQL templates should exist and support format interpolation."""
    sql_dir = Path("consumer/consumer_adapter/sql")
    for template_name in REQUIRED_SQL_TEMPLATES:
        template_path = sql_dir / template_name
        assert template_path.exists(), f"Missing SQL template: {template_name}"
        rendered = template_path.read_text(encoding="utf-8").format(
            database="DEDUPLICATION",
            schema="ER_RUNTIME",
            target_table="DUPLICATE_PAIRS",
            stage_table="STG_DUPLICATE_PAIRS",
            run_id="run-1",
            removed_pairs_stage_table="STG_REMOVED_PAIR_IDS",
        )
        assert rendered.strip()


@pytest.mark.parametrize(
    "table_name,template_name",
    _MERGE_COVERAGE_CASES,
    ids=[template for _, template in _MERGE_COVERAGE_CASES],
)
def test_merge_template_insert_covers_all_model_fields(
    table_name: DatabaseTableName,
    template_name: str,
) -> None:
    """INSERT column list in each MERGE template must include every field in its Pydantic model.

    When a field is added to a row model in deduplication_common.py it must also be added to
    the corresponding WHEN NOT MATCHED THEN INSERT (...) block. This test catches that drift
    before it reaches Snowflake, where a missing column produces a silent NULL or an error.
    """
    sql_dir = Path("consumer/consumer_adapter/sql")
    sql = (sql_dir / template_name).read_text(encoding="utf-8")
    model = COMMON_SCHEMA_TABLE_MODELS[table_name]
    insert_columns = _parse_insert_columns(sql)
    model_fields = column_names(model)
    missing = [f for f in model_fields if f not in insert_columns]
    assert not missing, (
        f"{template_name} INSERT block is missing {len(missing)} field(s) "
        f"from {model.__name__}: {missing}\n"
        f"Add each missing field to the WHEN NOT MATCHED THEN INSERT (...) and VALUES (...) "
        f"clauses, and to WHEN MATCHED THEN UPDATE SET if appropriate."
    )
