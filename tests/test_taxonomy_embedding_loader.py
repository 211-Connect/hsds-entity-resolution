"""Tests for HSIS taxonomy embedding loader and consumer adapter integration."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from consumer.embeddings.hsis.taxonomy_loader import (
    generate_taxonomy_schema_name,
    load_taxonomy_embeddings,
)

# ---------------------------------------------------------------------------
# generate_taxonomy_schema_name
# ---------------------------------------------------------------------------


def test_schema_name_standard_model() -> None:
    """Model name slashes/hyphens are replaced and output is upper-cased."""
    assert generate_taxonomy_schema_name("BAAI/bge-m3", 1024, "F32") == "BAAI_BGE_M3_1024_F32"


def test_schema_name_dots_replaced() -> None:
    """Dots in model names are converted to underscores."""
    assert (
        generate_taxonomy_schema_name("text.embedding.3", 512, "F16") == "TEXT_EMBEDDING_3_512_F16"
    )


def test_schema_name_precision_uppercased() -> None:
    """Precision is always upper-cased in the output."""
    assert generate_taxonomy_schema_name("model", 256, "f32") == "MODEL_256_F32"


# ---------------------------------------------------------------------------
# Fake connection helpers
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Minimal cursor fake that returns configurable rows."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def execute(self, query: str) -> None:  # noqa: ARG002
        """Accept any query string."""

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Return pre-configured rows."""
        return list(self._rows)

    def close(self) -> None:
        """No-op close."""


class _FakeConnection:
    """Minimal Snowflake connection fake."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def cursor(self) -> _FakeCursor:
        """Return a cursor that yields configured rows."""
        return _FakeCursor(self._rows)


class _ErrorCursor:
    """Cursor that raises on execute to simulate query failure."""

    def execute(self, query: str) -> None:  # noqa: ARG002
        """Raise to simulate a Snowflake error."""
        msg = "Table does not exist"
        raise RuntimeError(msg)

    def fetchall(self) -> list[tuple[Any, ...]]:  # pragma: no cover
        """Never reached when execute raises."""
        return []

    def close(self) -> None:
        """No-op close."""


class _ErrorConnection:
    """Connection whose cursor always raises on execute."""

    def cursor(self) -> _ErrorCursor:
        """Return a cursor that raises."""
        return _ErrorCursor()


# ---------------------------------------------------------------------------
# load_taxonomy_embeddings — happy-path
# ---------------------------------------------------------------------------


def test_load_returns_code_to_vector_mapping() -> None:
    """Rows with list vectors are returned keyed by CODE."""
    rows = [
        ("BD-1800", [0.1, 0.2, 0.3]),
        ("LR-8000", [0.4, 0.5, 0.6]),
    ]
    conn = _FakeConnection(rows)
    result = load_taxonomy_embeddings(conn)

    assert set(result.keys()) == {"BD-1800", "LR-8000"}
    assert result["BD-1800"] == pytest.approx([0.1, 0.2, 0.3])
    assert result["LR-8000"] == pytest.approx([0.4, 0.5, 0.6])


def test_load_parses_json_string_vectors() -> None:
    """Rows whose vectors are JSON-encoded strings are parsed correctly."""
    vector = [0.1] * 10
    rows = [("BD-1800", json.dumps(vector))]
    conn = _FakeConnection(rows)
    result = load_taxonomy_embeddings(conn)

    assert "BD-1800" in result
    assert len(result["BD-1800"]) == 10
    assert result["BD-1800"][0] == pytest.approx(0.1)


def test_load_skips_rows_with_empty_code() -> None:
    """Rows with falsy CODE values are silently dropped."""
    rows: list[tuple[Any, ...]] = [
        ("", [0.1, 0.2]),
        (None, [0.3, 0.4]),
        ("VALID-CODE", [0.5, 0.6]),
    ]
    conn = _FakeConnection(rows)
    result = load_taxonomy_embeddings(conn)

    assert list(result.keys()) == ["VALID-CODE"]


def test_load_skips_unparseable_string_vectors(caplog: pytest.LogCaptureFixture) -> None:
    """Rows with malformed string vectors are skipped without raising."""
    rows: list[tuple[Any, ...]] = [
        ("CODE-GOOD", [1.0, 2.0]),
        ("CODE-BAD", "not-valid-json"),
    ]
    conn = _FakeConnection(rows)
    with caplog.at_level(logging.INFO):
        result = load_taxonomy_embeddings(conn)

    assert "CODE-GOOD" in result
    assert "CODE-BAD" not in result


def test_load_returns_empty_on_query_error(caplog: pytest.LogCaptureFixture) -> None:
    """Query failures produce an empty dict with a warning log, not an exception."""
    conn = _ErrorConnection()
    with caplog.at_level(logging.WARNING):
        result = load_taxonomy_embeddings(conn)

    assert result == {}
    assert "Taxonomy embedding load failed" in caplog.text


def test_load_empty_table_returns_empty_dict() -> None:
    """Empty result set returns an empty mapping."""
    conn = _FakeConnection([])
    result = load_taxonomy_embeddings(conn)
    assert result == {}


# ---------------------------------------------------------------------------
# load_taxonomy_embeddings — schema name forwarding
# ---------------------------------------------------------------------------


def test_load_uses_custom_model_name() -> None:
    """Custom model_name parameter reaches the SQL query via schema name."""
    executed_queries: list[str] = []

    class _TrackingCursor(_FakeCursor):
        def execute(self, query: str) -> None:
            executed_queries.append(query)

    class _TrackingConnection:
        def cursor(self) -> _TrackingCursor:
            return _TrackingCursor([])

    load_taxonomy_embeddings(
        _TrackingConnection(),
        model_name="custom/model-v2",
        dimensions=768,
        precision="F16",
    )

    assert len(executed_queries) == 1
    assert "CUSTOM_MODEL_V2_768_F16" in executed_queries[0]
    assert "TAXONOMY_EMBEDDINGS" in executed_queries[0]


# ---------------------------------------------------------------------------
# ConsumerAdapter._load_taxonomy_embeddings integration
# ---------------------------------------------------------------------------


class _FakeSessionManager:
    """Minimal session manager whose connection yields a fake Snowflake connection."""

    def __init__(self, taxonomy_rows: list[tuple[Any, ...]]) -> None:
        self._rows = taxonomy_rows

    @property
    def database(self) -> str:
        """Return fake database name."""
        return "DEDUPLICATION"

    @property
    def schema(self) -> str:
        """Return fake schema name."""
        return "COMMON_EXPERIMENT"

    @contextmanager
    def connection(self) -> Iterator[_FakeConnection]:
        """Yield a fake Snowflake connection."""
        yield _FakeConnection(self._rows)

    @contextmanager
    def transaction(self, connection: object) -> Iterator[object]:
        """Yield a placeholder cursor."""
        del connection
        yield object()


def test_consumer_adapter_load_taxonomy_embeddings() -> None:
    """ConsumerAdapter._load_taxonomy_embeddings returns the loaded embedding dict."""
    from consumer.consumer_adapter.consumer import ConsumerAdapter

    class _NoOpEmbeddingAdapter:
        def embed_batch_sync(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
            return [[0.0] for _ in texts]

    rows = [("BD-1800", [0.1, 0.2, 0.3])]
    adapter = ConsumerAdapter(
        session_manager=_FakeSessionManager(rows),
        embedding_adapter=_NoOpEmbeddingAdapter(),
        sql_directory=Path("consumer/consumer_adapter/sql"),
        local_stage_dir=Path("/tmp/er_stage_test"),
    )
    result = adapter._load_taxonomy_embeddings()

    assert "BD-1800" in result
    assert result["BD-1800"] == pytest.approx([0.1, 0.2, 0.3])


def test_consumer_adapter_load_taxonomy_embeddings_graceful_failure() -> None:
    """ConsumerAdapter._load_taxonomy_embeddings returns {} when Snowflake fails."""
    from consumer.consumer_adapter.consumer import ConsumerAdapter

    class _NoOpEmbeddingAdapter:
        def embed_batch_sync(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
            return [[0.0] for _ in texts]

    class _ErrorSessionManager(_FakeSessionManager):
        @contextmanager
        def connection(self) -> Iterator[object]:
            msg = "Connection refused"
            raise RuntimeError(msg)
            yield  # type: ignore[misc]  # make generator

    adapter = ConsumerAdapter(
        session_manager=_ErrorSessionManager([]),
        embedding_adapter=_NoOpEmbeddingAdapter(),
        sql_directory=Path("consumer/consumer_adapter/sql"),
        local_stage_dir=Path("/tmp/er_stage_test"),
    )
    result = adapter._load_taxonomy_embeddings()
    assert result == {}
