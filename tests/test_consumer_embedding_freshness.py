"""Tests for consumer embedding freshness behavior."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from consumer.consumer_adapter.consumer import (
    ConsumerAdapter,
    _calculate_semantic_content_hash,
)


class _FakeSessionManager:
    """Minimal session manager for embedding freshness unit tests."""

    @property
    def database(self) -> str:
        """Return fake database name."""
        return "DEDUPLICATION"

    @property
    def schema(self) -> str:
        """Return fake schema name."""
        return "ER_RUNTIME"

    @contextmanager
    def connection(self) -> Iterator[object]:
        """Yield a placeholder connection object."""
        yield object()

    @contextmanager
    def transaction(self, connection: object) -> Iterator[object]:
        """Yield a placeholder transaction cursor."""
        del connection
        yield object()


class _RecordingEmbeddingAdapter:
    """Embedding adapter fake that records generation requests."""

    def __init__(self, vector: list[float]) -> None:
        self.calls: list[list[str]] = []
        self._vector = vector

    def embed_batch_sync(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        """Record input texts and return fixed vectors."""
        del batch_size
        self.calls.append(texts)
        return [self._vector for _ in texts]


def _make_adapter(*, tmp_path: Path, embedding_vector: list[float]) -> ConsumerAdapter:
    """Build a consumer adapter with fake dependencies for embedding-only tests."""
    return ConsumerAdapter(
        session_manager=_FakeSessionManager(),
        embedding_adapter=_RecordingEmbeddingAdapter(embedding_vector),
        sql_directory=Path("consumer/consumer_adapter/sql"),
        local_stage_dir=tmp_path,
        strict_validation_mode=True,
    )


def _single_row_frame(*, vector: list[float]) -> pl.DataFrame:
    """Return one-row entity frame used by freshness tests."""
    return pl.DataFrame(
        {
            "entity_id": ["org-1"],
            "source_schema": ["TENANT_A"],
            "name": ["Alpha"],
            "description": ["Updated profile"],
            "embedding_vector": [vector],
        }
    )


def test_ensure_embeddings_regenerates_stale_non_null_vectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Consumer should regenerate non-null embeddings when cache hash is stale."""
    adapter = _make_adapter(tmp_path=tmp_path, embedding_vector=[9.0, 8.0])
    captured_fetch: dict[str, Any] = {}
    captured_upserts: list[dict[str, dict[str, object]]] = []

    def _fake_fetch(**kwargs: object) -> dict[str, dict[str, object]]:
        captured_fetch.update(kwargs)
        return {"org-1": {"embedding": [1.0, 1.0], "hash": "stale-hash"}}

    def _fake_upsert(**kwargs: object) -> int:
        embeddings_map = kwargs.get("embeddings_map")
        if isinstance(embeddings_map, dict):
            captured_upserts.append(embeddings_map)
        return 1

    monkeypatch.setattr(
        "consumer.consumer_adapter.consumer.fetch_embeddings_by_reference_ids",
        _fake_fetch,
    )
    monkeypatch.setattr(
        "consumer.consumer_adapter.consumer.upsert_embeddings_by_reference_id",
        _fake_upsert,
    )

    result = adapter._ensure_embeddings(
        _single_row_frame(vector=[0.5, 0.5]), entity_type="organization"
    )
    expected_text = "ORGANIZATION: Alpha - Updated profile"
    expected_hash = _calculate_semantic_content_hash(expected_text)
    embedding_adapter = adapter._embedding_adapter
    assert isinstance(embedding_adapter, _RecordingEmbeddingAdapter)
    assert embedding_adapter.calls == [[expected_text]]
    assert result.to_dicts()[0]["embedding_vector"] == [9.0, 8.0]
    assert captured_fetch["reference_ids"] == ["org-1"]
    assert captured_upserts[0]["org-1"]["hash"] == expected_hash


def test_ensure_embeddings_reuses_fresh_cache_for_non_null_vectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Consumer should reuse cache for non-null embeddings when semantic hash matches."""
    adapter = _make_adapter(tmp_path=tmp_path, embedding_vector=[3.0, 3.0])
    expected_text = "ORGANIZATION: Alpha - Updated profile"
    expected_hash = _calculate_semantic_content_hash(expected_text)
    upsert_called = False

    def _fake_fetch(**kwargs: object) -> dict[str, dict[str, object]]:
        del kwargs
        return {"org-1": {"embedding": [7.0, 6.0], "hash": expected_hash}}

    def _fake_upsert(**kwargs: object) -> int:
        del kwargs
        nonlocal upsert_called
        upsert_called = True
        return 1

    monkeypatch.setattr(
        "consumer.consumer_adapter.consumer.fetch_embeddings_by_reference_ids",
        _fake_fetch,
    )
    monkeypatch.setattr(
        "consumer.consumer_adapter.consumer.upsert_embeddings_by_reference_id",
        _fake_upsert,
    )

    result = adapter._ensure_embeddings(
        _single_row_frame(vector=[0.5, 0.5]), entity_type="organization"
    )
    embedding_adapter = adapter._embedding_adapter
    assert isinstance(embedding_adapter, _RecordingEmbeddingAdapter)
    assert embedding_adapter.calls == []
    assert result.to_dicts()[0]["embedding_vector"] == [7.0, 6.0]
    assert upsert_called is False


def test_ensure_embeddings_logs_hash_mismatch_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Consumer should log mismatch classification and sample details for stale cache hashes."""
    adapter = _make_adapter(tmp_path=tmp_path, embedding_vector=[5.0, 5.0])

    def _fake_fetch(**kwargs: object) -> dict[str, dict[str, object]]:
        del kwargs
        return {"org-1": {"embedding": [7.0, 6.0], "hash": "stale-hash"}}

    def _fake_upsert(**kwargs: object) -> int:
        del kwargs
        return 1

    monkeypatch.setattr(
        "consumer.consumer_adapter.consumer.fetch_embeddings_by_reference_ids",
        _fake_fetch,
    )
    monkeypatch.setattr(
        "consumer.consumer_adapter.consumer.upsert_embeddings_by_reference_id",
        _fake_upsert,
    )

    with caplog.at_level(logging.INFO):
        adapter._ensure_embeddings(_single_row_frame(vector=[0.5, 0.5]), entity_type="organization")

    assert (
        "Embedding cache lookup summary: tenant=TENANT_A entity_type=organization requested=1 "
        "cache_rows=1 reused=0 generated=1 entry_miss=0 hash_missing=0 hash_malformed=1 "
        "hash_case_only=0 hash_mismatch=0 invalid_vectors=0"
    ) in caplog.text
    assert (
        "Embedding cache hash mismatch sample: tenant=TENANT_A entity_type=organization "
        "entity_id=org-1 reason=malformed_cached_hash"
    ) in caplog.text
