"""Query-construction tests for Snowflake embedding store helpers."""

from __future__ import annotations

from consumer.embeddings.stores.snowflake_embedding_store import (
    _table_columns,
    ensure_embedding_cache_tables,
    fetch_embeddings_by_reference_ids,
)


class _FakeCursor:
    """Minimal cursor fake for asserting issued SQL statements."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self._fetchone_calls = 0
        self._last_query = ""

    def execute(self, query: str) -> None:
        self.queries.append(query)
        self._last_query = query

    def fetchone(self) -> tuple[int]:
        self._fetchone_calls += 1
        if "INFORMATION_SCHEMA.TABLES" in self._last_query and "TABLE_SCHEMA = 'NE211_V2'" in self._last_query:
            return (0,)
        if "SELECT TABLE_SCHEMA" in self._last_query:
            return ("NE211",)
        return (1,)

    def fetchall(self) -> list[tuple[object, ...]]:
        if "SELECT COLUMN_NAME" in self.queries[-1]:
            return [("SERVICE_ID",), ("EMBEDDING_TYPE",)]
        return [("svc-1", [1.0, 2.0], "hash-1")]

    def close(self) -> None:
        return None


class _FakeConnection:
    """Connection fake that returns one reusable fake cursor."""

    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor


def test_fetch_embeddings_uses_norse_staging_information_schema() -> None:
    """Fetch helper should resolve metadata from NORSE_STAGING information schema."""
    cursor = _FakeCursor()
    connection = _FakeConnection(cursor)

    result = fetch_embeddings_by_reference_ids(
        snowflake_conn=connection,
        tenant_id="DUPAGEC211",
        reference_ids=["svc-1"],
        reference_type="service",
    )

    assert any(
        "NORSE_STAGING.INFORMATION_SCHEMA.COLUMNS" in query
        for query in cursor.queries
    )
    assert result == {"svc-1": {"embedding": [1.0, 2.0], "hash": "hash-1"}}


def test_table_columns_uses_norse_staging_information_schema() -> None:
    """Column helper should query NORSE_STAGING information schema explicitly."""
    cursor = _FakeCursor()

    columns = _table_columns(
        cursor=cursor,
        tenant_id="DUPAGEC211",
        table_name="BAAI_BGE_M3_1024_F32_EMBEDDING_CACHE",
    )

    assert "NORSE_STAGING.INFORMATION_SCHEMA.COLUMNS" in cursor.queries[0]
    assert columns == {"SERVICE_ID", "EMBEDDING_TYPE"}


def test_ensure_embedding_cache_tables_creates_missing_doc_and_taxonomy_tables() -> None:
    """Missing tenant cache tables should be cloned from an existing tenant schema."""
    cursor = _FakeCursor()
    connection = _FakeConnection(cursor)

    ensure_embedding_cache_tables(
        connection,
        tenant_id="NE211_V2",
        table_prefix="BAAI_BGE_M3_1024_F32",
    )

    assert any(
        "CREATE SCHEMA IF NOT EXISTS NORSE_STAGING.NE211_V2" in query
        for query in cursor.queries
    )
    assert any(
        "CREATE TABLE IF NOT EXISTS NORSE_STAGING.NE211_V2.BAAI_BGE_M3_1024_F32_EMBEDDING_CACHE "
        "LIKE NORSE_STAGING.NE211.BAAI_BGE_M3_1024_F32_EMBEDDING_CACHE"
        in query.replace("\n", " ")
        for query in cursor.queries
    )
    assert any(
        "CREATE TABLE IF NOT EXISTS NORSE_STAGING.NE211_V2.BAAI_BGE_M3_1024_F32_TAXONOMY_EMBEDDING_CACHE "
        "LIKE NORSE_STAGING.NE211.BAAI_BGE_M3_1024_F32_TAXONOMY_EMBEDDING_CACHE"
        in query.replace("\n", " ")
        for query in cursor.queries
    )
