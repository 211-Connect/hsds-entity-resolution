"""Frame-level tracing utilities for pipeline debug observability.

Usage pattern::

    tracer = FrameTracer(entity_id="some-entity-id")
    tracer.announce()                          # INFO: marks the entity for the run
    tracer.log_frame(frame, "stage:label")     # DEBUG: shape + tracer row + stats

The tracer is instantiated once in ``run_incremental`` from the first entity in
the organization input frame and threaded through every stage.  Each ``log_frame``
call emits a DEBUG log containing:

- Frame shape (rows × columns)
- Whether the tracer entity appears in this frame and its full row
- Compact summary statistics (null counts, score distributions)

Any stage where the tracer entity disappears is a candidate data-loss point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import polars as pl
from dagster import get_dagster_logger


@dataclass
class FrameTracer:
    """Track one entity through all pipeline stages for frame-level debugging.

    Instantiate once at the start of ``run_incremental`` using the first
    entity_id available in the organization frame.  Pass the instance as
    ``tracer: FrameTracer | None = None`` to each stage function and call
    ``log_frame`` immediately after any DataFrame is created or mutated.
    """

    entity_id: str

    def announce(self) -> None:
        """Emit one INFO log establishing which entity will be traced this run."""
        _log = get_dagster_logger()
        _log.info(
            "🔍 PIPELINE TRACER  entity_id=%r — row state logged at every stage",
            self.entity_id,
        )

    def log_frame(self, frame: pl.DataFrame, label: str) -> None:
        """Log frame shape, summary statistics, and the tracer entity row.

        Emitted at DEBUG level so it does not appear in normal production runs.
        When the tracer entity is absent from the frame the log explicitly flags
        it so you can identify the exact stage where data was lost or filtered.
        """
        _log = get_dagster_logger()
        if not _log.isEnabledFor(logging.DEBUG):
            return

        tracer_rows = self._find_tracer_rows(frame)

        if tracer_rows:
            row_preview = _format_row_preview(tracer_rows[0])
            tracer_line = f"✓ tracer present ({len(tracer_rows)} row(s)):\n    {row_preview}"
            if len(tracer_rows) > 1:
                tracer_line += f"\n    [+{len(tracer_rows) - 1} additional row(s) omitted]"
        else:
            tracer_line = "✗ tracer NOT found — possible data-loss point"

        stats = _frame_stats(frame)

        _log.debug(
            "🔍 FRAME[%s]  shape=%d×%d  entity=%r\n  %s\n  stats: %s",
            label,
            frame.height,
            frame.width,
            self.entity_id,
            tracer_line,
            stats,
        )

    def _find_tracer_rows(self, frame: pl.DataFrame) -> list[dict[str, Any]]:
        """Return up to 2 rows that reference the tracer entity."""
        if frame.is_empty():
            return []

        # Scalar entity columns — entity frames and pair frames
        for col in ("entity_id", "entity_id_a", "entity_id_b"):
            if col in frame.columns:
                try:
                    rows = frame.filter(pl.col(col) == self.entity_id).to_dicts()
                    if rows:
                        return rows[:2]
                except Exception:  # noqa: BLE001
                    pass

        # List column — cluster frames where entity_ids is list[str]
        if "entity_ids" in frame.columns:
            try:
                rows = frame.filter(pl.col("entity_ids").list.contains(self.entity_id)).to_dicts()
                if rows:
                    return rows[:1]
            except Exception:  # noqa: BLE001
                pass

        return []


def _format_row_preview(row: dict[str, Any]) -> str:
    """Format one row dict as a compact readable string, truncating long values."""
    parts: list[str] = []
    for key, val in row.items():
        if isinstance(val, list):
            parts.append(f"{key}=[{len(val)} items]")
        elif isinstance(val, float):
            parts.append(f"{key}={val:.4f}")
        elif isinstance(val, str) and len(val) > 60:
            parts.append(f"{key}={val[:60]!r}…")
        elif val is None:
            parts.append(f"{key}=∅")
        else:
            parts.append(f"{key}={val!r}")
    return "  ".join(parts)


def _frame_stats(frame: pl.DataFrame) -> str:
    """Return compact summary statistics as a single readable line."""
    if frame.is_empty():
        return "empty (0 rows)"

    parts: list[str] = [f"rows={frame.height}  cols={frame.width}"]

    # Null counts for any column that has at least one null
    null_counts = frame.null_count()
    for col_name in frame.columns:
        n = int(null_counts.get_column(col_name)[0])
        if n > 0:
            parts.append(f"nulls[{col_name}]={n}")

    # Score distribution for float columns whose name contains "score"
    for col, dt in zip(frame.columns, frame.dtypes, strict=True):
        if "score" in col.lower() and dt in (pl.Float32, pl.Float64):
            series = frame.get_column(col).drop_nulls()
            if series.len() > 0:
                min_val = series.min()
                max_val = series.max()
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    parts.append(
                        f"{col}=[{float(min_val):.3f}…{float(max_val):.3f}  n={series.len()}]"
                    )

    return "  ".join(parts)
