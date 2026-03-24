"""Contract tests that guard SQL/mapping against COMMON_EXPERIMENT schema drift."""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
from consumer.consumer_adapter.artifact_contracts import assert_valid_artifacts
from consumer.consumer_adapter.mapper import ConsumerRunContext, map_artifacts_to_tables
from consumer.consumer_types.deduplication_common import DatabaseTableName

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.pipeline import run_incremental

_SCHEMA_CONTRACT_PATH = Path("tests/contract/common_experiment_schema_contract.json")
_SQL_DIR = Path("consumer/consumer_adapter/sql")
_IDENTIFIER_PATTERN = re.compile(r"\b(?:target|source)\.([A-Z_]+)\b")
_INSERT_COLUMNS_PATTERN = re.compile(
    r"WHEN NOT MATCHED THEN INSERT\s*\((.*?)\)\s*VALUES",
    flags=re.DOTALL,
)
_DELETE_PATTERN = re.compile(
    r"DELETE FROM\s+[A-Z0-9_]+\.[A-Z0-9_]+\.([A-Z_]+)\s+WHERE\s+([A-Z_]+)\s+IN",
    flags=re.DOTALL,
)


def _load_schema_contract() -> dict[str, set[str]]:
    """Load expected COMMON_EXPERIMENT table columns keyed by table name."""
    payload = json.loads(_SCHEMA_CONTRACT_PATH.read_text(encoding="utf-8"))
    return {table: set(columns) for table, columns in payload.items()}


def _render_template(template_name: str) -> str:
    """Render one SQL template with placeholder values."""
    return (
        (_SQL_DIR / template_name)
        .read_text(encoding="utf-8")
        .format(
            database="DEDUPLICATION",
            schema="COMMON_EXPERIMENT",
            target_table="DUPLICATE_PAIRS",
            stage_table="STG_GENERIC",
            run_id="run-1",
            removed_pairs_stage_table="STG_REMOVED_PAIR_IDS",
        )
    )


def _columns_from_merge_template(rendered_sql: str) -> set[str]:
    """Extract referenced table columns from merge SQL."""
    columns = set(_IDENTIFIER_PATTERN.findall(rendered_sql))
    insert_match = _INSERT_COLUMNS_PATTERN.search(rendered_sql)
    if insert_match is None:
        return columns
    insert_columns = {
        column.strip() for column in insert_match.group(1).split(",") if column.strip()
    }
    return columns | insert_columns


def _sample_artifacts() -> dict[str, pl.DataFrame]:
    """Build one deterministic sample artifact bundle for mapping validation."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1",
        scope_id="scope-1",
        entity_type="organization",
    )
    org = pl.DataFrame(
        {
            "entity_id": ["org-a", "org-b"],
            "source_schema": ["COMMON_EXPERIMENT", "COMMON_EXPERIMENT"],
            "name": ["Alpha Clinic", "Alpha Clinic"],
            "description": ["A", "A"],
            "emails": [["one@example.org"], ["one@example.org"]],
            "phones": [["555-0000"], ["555-0000"]],
            "websites": [["example.org"], ["example.org"]],
            "locations": [[], []],
            "taxonomies": [[], []],
            "identifiers": [[], []],
            "services_rollup": [[], []],
            "embedding_vector": [[1.0, 0.0], [0.99, 0.01]],
        }
    )
    result = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    return {
        "denormalized_organization": result.denormalized_organization,
        "denormalized_service": result.denormalized_service,
        "entity_delta_summary": result.entity_delta_summary,
        "removed_entity_ids": result.removed_entity_ids,
        "candidate_pairs": result.candidate_pairs,
        "scored_pairs": result.scored_pairs,
        "pair_reasons": result.pair_reasons,
        "mitigation_events": result.mitigation_events,
        "removed_pair_ids": result.removed_pair_ids,
        "pair_id_remap": result.pair_id_remap,
        "clusters": result.clusters,
        "cluster_pairs": result.cluster_pairs,
        "pair_state_index": result.pair_state_index,
        "review_queue_items": result.review_queue_items,
        "run_summary": result.run_summary,
    }


def test_merge_templates_reference_only_contract_columns() -> None:
    """Merge templates must only read/write columns declared in schema contract."""
    schema_contract = _load_schema_contract()
    template_table_map = {
        "merge_duplicate_pairs.sql": "DUPLICATE_PAIRS",
        "merge_duplicate_pair_scores.sql": "DUPLICATE_PAIR_SCORES",
        "merge_duplicate_reasons.sql": "DUPLICATE_REASONS",
        "merge_mitigated_pairs.sql": "MITIGATED_PAIRS",
        "merge_duplicate_clusters.sql": "DUPLICATE_CLUSTERS",
        "merge_duplicate_cluster_pairs.sql": "DUPLICATE_CLUSTER_PAIRS",
    }
    for template_name, table_name in template_table_map.items():
        rendered_sql = _render_template(template_name)
        referenced_columns = _columns_from_merge_template(rendered_sql)
        unexpected = referenced_columns - schema_contract[table_name]
        assert not unexpected, f"{template_name} references unknown columns: {sorted(unexpected)}"


def test_delete_cascade_filters_use_contract_columns() -> None:
    """Cascade delete template must filter each table by an expected pair-key column."""
    schema_contract = _load_schema_contract()
    rendered_sql = _render_template("delete_cascade_removed_pairs.sql")
    matches = _DELETE_PATTERN.findall(rendered_sql)
    assert len(matches) == 5
    assert rendered_sql.count("SELECT ID FROM STG_REMOVED_PAIR_IDS") == 5
    for table_name, filter_column in matches:
        assert table_name in schema_contract
        assert filter_column in schema_contract[table_name]


def test_mapped_frames_match_schema_contract_columns() -> None:
    """Mapped persistence frames should exactly match declared schema contract columns."""
    artifacts = _sample_artifacts()
    assert_valid_artifacts(artifacts)
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-1",
        scope_id="scope-1",
        entity_type="organization",
    )
    mapped = map_artifacts_to_tables(
        artifacts=artifacts,
        context=ConsumerRunContext(
            run_id="run-1",
            team_id="team-1",
            scope_id="scope-1",
            entity_type="organization",
            config=config,
        ),
    )
    schema_contract = _load_schema_contract()
    tables_to_check: tuple[DatabaseTableName, ...] = (
        "DUPLICATE_PAIRS",
        "DUPLICATE_PAIR_SCORES",
        "DUPLICATE_REASONS",
        "MITIGATED_PAIRS",
        "DUPLICATE_CLUSTERS",
        "DUPLICATE_CLUSTER_PAIRS",
    )
    for table_name in tables_to_check:
        expected_columns = schema_contract[table_name]
        frame = mapped.table_frames[table_name]
        assert set(frame.columns) == expected_columns
