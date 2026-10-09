"""Contract test for the checked-in training schema contract."""

from __future__ import annotations

import json
from pathlib import Path

from hsds_entity_resolution.core.training_schema import training_schema_contract

_SCHEMA_CONTRACT_PATH = Path("tests/contract/training_data_schema_contract.json")


def _load_json_contract() -> dict[str, tuple[str, ...]]:
    """Load checked-in training schema contract."""
    payload = json.loads(_SCHEMA_CONTRACT_PATH.read_text(encoding="utf-8"))
    return {table: tuple(columns) for table, columns in payload.items()}


def test_training_schema_contract_json_matches_code_contract() -> None:
    """Checked-in training schema contract should match the runtime expectation."""
    assert _load_json_contract() == training_schema_contract()
