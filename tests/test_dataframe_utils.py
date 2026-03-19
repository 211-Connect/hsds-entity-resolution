"""Unit tests for shared dataframe utility helpers."""

from __future__ import annotations

import polars as pl

from hsds_entity_resolution.core.dataframe_utils import (
    clean_string_list,
    clean_text_scalar,
    ensure_columns,
    hash_values,
    to_dataframe,
)


def test_to_dataframe_materializes_lazy_frame() -> None:
    """Lazy frames should be materialized to DataFrame."""
    lazy = pl.DataFrame({"a": [1]}).lazy()
    materialized = to_dataframe(lazy)
    assert isinstance(materialized, pl.DataFrame)
    assert materialized.height == 1


def test_ensure_columns_handles_empty_and_non_empty_frames() -> None:
    """ensure_columns should add missing columns without adding fake rows."""
    empty = ensure_columns(pl.DataFrame(), ["id", "name"])
    assert empty.height == 0
    assert set(empty.columns) == {"id", "name"}

    non_empty = ensure_columns(pl.DataFrame({"id": [1]}), ["id", "name"])
    assert non_empty.height == 1
    assert non_empty.row(0, named=True)["name"] is None


def test_normalizers_and_hash_are_deterministic() -> None:
    """Normalization and hashing should provide stable outputs."""
    assert clean_text_scalar("  A  B  ") == "a b"
    assert clean_text_scalar(None) == ""
    assert clean_string_list([" A ", "a", "B "]) == ["a", "b"]
    assert clean_string_list("not-list") == []
    assert hash_values(["A", "B"]) == hash_values([" a ", "b"])
