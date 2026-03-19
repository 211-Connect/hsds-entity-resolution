"""Shared Polars frame helpers for stage implementations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

import polars as pl


def frame_with_schema(rows: Sequence[Any], schema: dict[str, Any]) -> pl.DataFrame:
    """Build a DataFrame using *schema* as explicit type overrides.

    For the empty case, uses ``schema=`` so column types are always known.

    For the non-empty case, uses ``schema_overrides=`` for scalar and list
    columns.  Columns typed as ``pl.Object`` are injected **after** the base
    frame is built via ``pl.Series(dtype=pl.Object)`` — this completely
    bypasses Polars' ``from_dicts`` struct-inference pass for those columns.
    That inference pass crashes when different rows carry list-of-dict payloads
    whose dict schemas are inconsistent (e.g. one location object has a "url"
    key, another does not), so the bypass is required for any column that
    stores heterogeneous list-of-struct data.
    """
    if not rows:
        return pl.DataFrame(schema=schema)

    object_cols = {col for col, dt in schema.items() if dt is pl.Object}
    if not object_cols:
        return pl.DataFrame(rows, schema_overrides=schema)

    # Build the non-object columns first so from_dicts never sees the structs.
    base_schema = {k: v for k, v in schema.items() if k not in object_cols}
    base_rows: list[Any] = [{k: v for k, v in row.items() if k not in object_cols} for row in rows]
    df = pl.DataFrame(base_rows, schema_overrides=base_schema)

    # Append each object column as a typed Series in schema-declared order.
    for col in schema:
        if col in object_cols:
            df = df.with_columns(
                pl.Series(name=col, values=[row.get(col) for row in rows], dtype=pl.Object)
            )
    return df


def to_dataframe(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Materialize a frame to `DataFrame` while preserving schema."""
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    return frame


def ensure_columns(frame: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    """Ensure all expected columns exist, defaulting to null values."""
    missing = [column for column in columns if column not in frame.columns]
    if not missing:
        return frame
    if frame.is_empty():
        expanded = frame
        for column in missing:
            expanded = expanded.with_columns(pl.Series(name=column, values=[], dtype=pl.Null))
        return expanded
    additions = [pl.lit(None).alias(column) for column in missing]
    return frame.with_columns(additions)


def clean_text_scalar(value: object) -> str:
    """Clean scalar text values for stable hash and key behavior."""
    if value is None:
        return ""
    text_value = str(value).strip().lower()
    return " ".join(text_value.split())


def clean_string_list(value: object) -> list[str]:
    """Clean list-like values to lower/trimmed unique strings."""
    if not isinstance(value, list):
        return []
    cleaned = [clean_text_scalar(item) for item in value]
    return sorted({item for item in cleaned if item})


def hash_values(values: list[object]) -> str:
    """Build deterministic hash digest from ordered scalar values."""
    cleaned = [clean_text_scalar(item) for item in values]
    payload = "||".join(cleaned)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
