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


def test_execute_cleanup_runs_four_templates_in_order() -> None:
    """execute_cleanup should run delete_cascade, purge_stale, recompute, and orphan cleanup."""
    executor = PersistenceExecutor(Path("consumer/consumer_adapter/sql"))
    cursor = _FakeCursor()

    results = executor.execute_cleanup(
        cursor=cursor,
        database="DEDUPLICATION",
        schema="ER_RUNTIME",
        run_id="run-1",
        scope_id="scope-1",
        entity_type="organization",
        stage_tables={"removed_pair_ids": "STG_REMOVED_PAIR_IDS"},
    )

    assert [result.template for result in results] == [
        "delete_cascade_removed_pairs.sql",
        "purge_stale_pairs.sql",
        "recompute_cluster_aggregates.sql",
        "delete_orphan_duplicate_reasons.sql",
    ]


def test_execute_cleanup_splits_multi_statement_templates() -> None:
    """Each SQL statement within a template is executed as a separate cursor call."""
    executor = PersistenceExecutor(Path("consumer/consumer_adapter/sql"))
    cursor = _FakeCursor()

    executor.execute_cleanup(
        cursor=cursor,
        database="DEDUPLICATION",
        schema="ER_RUNTIME",
        run_id="run-1",
        scope_id="scope-1",
        entity_type="organization",
        stage_tables={"removed_pair_ids": "STG_REMOVED_PAIR_IDS"},
    )

    # Every command passed to the cursor must contain at most one statement.
    assert all(cmd.count(";") == 1 for cmd in cursor.commands)


def test_execute_cleanup_delete_statements_are_individual() -> None:
    """DELETE statements must not be batched — each DELETE is one cursor.execute call."""
    executor = PersistenceExecutor(Path("consumer/consumer_adapter/sql"))
    cursor = _FakeCursor()

    executor.execute_cleanup(
        cursor=cursor,
        database="DEDUPLICATION",
        schema="ER_RUNTIME",
        run_id="run-1",
        scope_id="scope-1",
        entity_type="organization",
        stage_tables={"removed_pair_ids": "STG_REMOVED_PAIR_IDS"},
    )

    delete_commands = [cmd for cmd in cursor.commands if cmd.startswith("DELETE FROM")]
    # delete_cascade (5) + purge_stale (4 top-level DELETEs; the fifth is inside a WITH CTE)
    # + orphan cleanup (1) = 10
    assert len(delete_commands) == 10
    assert all(cmd.count("DELETE FROM") == 1 for cmd in delete_commands)
