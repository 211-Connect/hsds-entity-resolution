"""Tests for persistence executor SQL statement execution behavior."""

from __future__ import annotations

from pathlib import Path

from consumer.consumer_adapter.persistence_executor import PersistenceExecutor


class _FakeCursor:
    """Minimal fake cursor capturing execute calls and rowcount."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self._rowcount = 1

    def execute(self, command: str) -> object:
        self.commands.append(command)
        self._rowcount = 1
        return object()

    @property
    def rowcount(self) -> int:
        return self._rowcount


def test_shared_templates_execute_as_individual_statements() -> None:
    """Shared templates with multiple statements should execute one statement at a time."""
    executor = PersistenceExecutor(Path("consumer/consumer_adapter/sql"))
    cursor = _FakeCursor()

    results = executor._execute_shared_templates(
        cursor=cursor,
        database="DEDUPLICATION",
        schema="COMMON_EXPERIMENT",
        run_id="run-1",
        stage_tables={"removed_pair_ids": "STG_REMOVED_PAIR_IDS"},
    )

    delete_commands = [command for command in cursor.commands if command.startswith("DELETE FROM")]
    assert len(delete_commands) == 5
    assert all(command.count("DELETE FROM") == 1 for command in delete_commands)
    assert [result.template for result in results] == [
        "delete_cascade_removed_pairs.sql",
        "recompute_cluster_aggregates.sql",
    ]
    assert results[0].rowcount == 5
    assert results[1].rowcount == 1
