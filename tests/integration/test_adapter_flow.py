"""Integration-style tests for consumer adapter orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from consumer.consumer_adapter.consumer import (
    ConsumerAdapter,
    ConsumerRunRequest,
    _requires_version_rescore,
)


@dataclass(frozen=True)
class _FakeConfig:
    """Minimal config object used by fake session manager."""

    database: str = "DEDUPLICATION"
    schema: str = "COMMON_EXPERIMENT"


class _FakeCursor:
    """Cursor that records SQL commands for assertions."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self._rowcount = 1

    def execute(self, command: str) -> object:
        self.commands.append(command)
        return None

    @property
    def rowcount(self) -> int:
        return self._rowcount

    def close(self) -> None:
        return None


class _FakeConnection:
    """No-op connection object for transaction wrapper."""

    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeSessionManager:
    """Session manager replacement for pure in-memory adapter tests."""

    def __init__(self) -> None:
        self._config = _FakeConfig()
        self.cursor = _FakeCursor()

    @property
    def database(self) -> str:
        return self._config.database

    @property
    def schema(self) -> str:
        return self._config.schema

    @contextmanager
    def connection(self) -> Iterator[_FakeConnection]:
        yield _FakeConnection(self.cursor)

    @contextmanager
    def transaction(self, connection: object) -> Iterator[_FakeCursor]:
        del connection
        yield self.cursor


class _FakeEmbeddingAdapter:
    """Embedding adapter with deterministic generated vectors."""

    def embed_batch_sync(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        del batch_size
        return [[float(len(text)), 1.0] for text in texts]


def test_consumer_adapter_runs_end_to_end_with_fake_persistence(tmp_path: Path) -> None:
    """Consumer adapter should execute pipeline, mapping, staging, and SQL flow."""
    session_manager = _FakeSessionManager()
    adapter = ConsumerAdapter(
        session_manager=session_manager,
        embedding_adapter=_FakeEmbeddingAdapter(),
        sql_directory=Path("consumer/consumer_adapter/sql"),
        local_stage_dir=tmp_path,
        strict_validation_mode=True,
    )
    request = ConsumerRunRequest(
        team_id="team-1",
        scope_id="scope-1",
        entity_type="organization",
        organization_entities=pl.DataFrame(
            {
                "entity_id": ["org-a", "org-b"],
                "source_schema": ["COMMON_EXPERIMENT", "COMMON_EXPERIMENT"],
                "name": ["Alpha", "Alpha Clinic"],
                "description": ["A", "A clinic"],
                "emails": [["one@example.org"], ["one@example.org"]],
                "phones": [["555-0000"], ["555-0000"]],
                "websites": [["example.org"], ["example.org"]],
                "locations": [[], []],
                "taxonomies": [[], []],
                "identifiers": [[], []],
                "services_rollup": [[], []],
                "embedding_vector": [[1.0, 0.0], [0.99, 0.01]],
            }
        ),
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        run_id="run-integration-1",
    )
    result = adapter.run(request)
    assert result.run_id == "run-integration-1"
    assert "DEDUPLICATION_RUN" in result.persisted_tables
    assert result.reconciliation.pair_state_count >= 0
    assert result.reconciliation.run_state_count == 1
    assert any(
        "MERGE INTO DEDUPLICATION.COMMON_EXPERIMENT.DUPLICATE_PAIRS" in cmd
        for cmd in session_manager.cursor.commands
    )


def test_requires_version_rescore_only_when_previous_state_exists() -> None:
    """Version rescore trigger should compare against previously persisted run state."""
    assert (
        _requires_version_rescore(
            current_policy_version="hsds-er-v1",
            current_model_version="m1",
            previous_policy_version=None,
            previous_model_version=None,
        )
        is False
    )
    assert (
        _requires_version_rescore(
            current_policy_version="hsds-er-v2",
            current_model_version="m1",
            previous_policy_version="hsds-er-v1",
            previous_model_version="m1",
        )
        is True
    )
