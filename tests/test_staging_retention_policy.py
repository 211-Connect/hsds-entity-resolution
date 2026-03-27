"""Tests for ER_STAGING retention behavior."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from consumer.consumer_adapter.consumer import (
    ConsumerAdapter,
    _should_truncate_staging_after_success,
)


class _RecordingCursor:
    """Cursor fake that records executed SQL commands."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def execute(self, command: str) -> object:
        self.commands.append(command)
        return object()


class _RecordingSessionManager:
    """Session manager fake that yields a recording cursor for transactions."""

    def __init__(self) -> None:
        self.cursor = _RecordingCursor()

    @property
    def database(self) -> str:
        return "DEDUPLICATION"

    @property
    def schema(self) -> str:
        return "ER_RUNTIME"

    @contextmanager
    def connection(self) -> Iterator[object]:
        yield object()

    @contextmanager
    def transaction(self, connection: object) -> Iterator[_RecordingCursor]:
        del connection
        yield self.cursor


class _NoopEmbeddingAdapter:
    """Embedding adapter fake for adapter construction."""

    def embed_batch_sync(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        del texts, batch_size
        return []


def _make_adapter(session_manager: _RecordingSessionManager, tmp_path: Path) -> ConsumerAdapter:
    return ConsumerAdapter(
        session_manager=session_manager,
        embedding_adapter=_NoopEmbeddingAdapter(),
        sql_directory=Path("consumer/consumer_adapter/sql"),
        local_stage_dir=tmp_path,
        strict_validation_mode=True,
    )


def test_should_truncate_staging_after_success_defaults_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staging cleanup should be opt-in."""
    monkeypatch.delenv("ER_STAGING_TRUNCATE_AFTER_SUCCESS", raising=False)
    assert _should_truncate_staging_after_success() is False


def test_cleanup_staging_tables_truncates_all_stage_tables_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Successful runs should truncate staging tables only when explicitly enabled."""
    monkeypatch.setenv("ER_STAGING_TRUNCATE_AFTER_SUCCESS", "true")
    session_manager = _RecordingSessionManager()
    adapter = _make_adapter(session_manager, tmp_path)

    adapter._cleanup_staging_tables(
        {
            "DEDUPLICATION_RUN": "DEDUPLICATION.ER_STAGING.STG_DEDUPLICATION_RUN",
            "DUPLICATE_PAIRS": "DEDUPLICATION.ER_STAGING.STG_DUPLICATE_PAIRS",
            "removed_pair_ids": "DEDUPLICATION.ER_STAGING.STG_REMOVED_PAIR_IDS",
        }
    )

    assert session_manager.cursor.commands == [
        "TRUNCATE TABLE DEDUPLICATION.ER_STAGING.STG_DEDUPLICATION_RUN",
        "TRUNCATE TABLE DEDUPLICATION.ER_STAGING.STG_DUPLICATE_PAIRS",
        "TRUNCATE TABLE DEDUPLICATION.ER_STAGING.STG_REMOVED_PAIR_IDS",
    ]


def test_cleanup_staging_tables_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Latest-run staging data should remain available unless cleanup is enabled."""
    monkeypatch.delenv("ER_STAGING_TRUNCATE_AFTER_SUCCESS", raising=False)
    session_manager = _RecordingSessionManager()
    adapter = _make_adapter(session_manager, tmp_path)

    adapter._cleanup_staging_tables(
        {"DEDUPLICATION_RUN": "DEDUPLICATION.ER_STAGING.STG_DEDUPLICATION_RUN"}
    )

    assert session_manager.cursor.commands == []
