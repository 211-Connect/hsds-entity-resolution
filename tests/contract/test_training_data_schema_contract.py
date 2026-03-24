"""Contract tests for TRAINING_DATA schema DDL alignment."""

from __future__ import annotations

import json
import re
from pathlib import Path

from hsds_entity_resolution.core.training_schema import training_schema_contract

_SCHEMA_CONTRACT_PATH = Path("tests/contract/training_data_schema_contract.json")
_DDL_PATH = Path("scripts/create_training_data_schema.sql")
_CREATE_TABLE_PATTERN = re.compile(
    r"CREATE TABLE IF NOT EXISTS \{database\}\.\{schema\}\.([A-Z_]+)\s*\((.*?)\)\s*COMMENT",
    flags=re.DOTALL,
)


def _load_json_contract() -> dict[str, tuple[str, ...]]:
    """Load checked-in training schema contract."""
    payload = json.loads(_SCHEMA_CONTRACT_PATH.read_text(encoding="utf-8"))
    return {table: tuple(columns) for table, columns in payload.items()}


def _ddl_table_columns() -> dict[str, set[str]]:
    """Extract declared columns from the training schema DDL."""
    ddl = _DDL_PATH.read_text(encoding="utf-8")
    tables: dict[str, set[str]] = {}
    for table_name, body in _CREATE_TABLE_PATTERN.findall(ddl):
        columns: set[str] = set()
        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line or line.startswith("--") or line.startswith("CONSTRAINT"):
                continue
            column_name = line.split()[0]
            if column_name.isupper():
                columns.add(column_name)
        tables[table_name] = columns
    return tables


def test_training_schema_contract_json_matches_code_contract() -> None:
    """Checked-in training schema contract should match the runtime expectation."""
    assert _load_json_contract() == training_schema_contract()


def test_training_schema_ddl_covers_contract_columns() -> None:
    """Training schema DDL must declare every required table and column."""
    ddl_columns = _ddl_table_columns()
    for table_name, required_columns in training_schema_contract().items():
        assert table_name in ddl_columns, table_name
        missing = set(required_columns) - ddl_columns[table_name]
        assert not missing, f"{table_name} missing columns in DDL: {sorted(missing)}"
