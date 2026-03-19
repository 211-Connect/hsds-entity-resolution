"""Shared incremental pipeline progress logging helpers."""

from __future__ import annotations

from collections.abc import Callable

_DEFAULT_PERCENT_STEP = 5


class IncrementalProgressLogger:
    """Emit stage and percent progress updates to an injected logger sink.

    When ``emit_debug`` is supplied the logger runs in two-tier mode:
    - INFO  → compact message containing only the stage-level fields
      (stage, status, percent, processed, total, event).  Repetitive context
      fields such as run_id / scope_id / source are omitted because they
      appear elsewhere in the Dagster log prefix.
    - DEBUG → full verbose message including all context fields.

    When ``emit_debug`` is *not* supplied (backward-compatible default) every
    message is emitted at INFO with the full context included.
    """

    def __init__(
        self,
        *,
        emit_info: Callable[[str], None],
        emit_debug: Callable[[str], None] | None = None,
        context: dict[str, str] | None = None,
        percent_step: int = _DEFAULT_PERCENT_STEP,
    ) -> None:
        """Initialize progress logger with context and emission strategy."""
        self._emit_info = emit_info
        self._emit_debug = emit_debug
        self._context = context or {}
        self._percent_step = max(1, min(100, percent_step))
        self._next_percent_by_stage: dict[str, int] = {}

    # Fields included in the compact INFO message (two-tier mode).
    _COMPACT_KEYS: frozenset[str] = frozenset(
        {"stage", "status", "event", "percent", "processed", "total"}
    )

    def stage_started(
        self,
        *,
        stage: str,
        total: int | None = None,
        detail: dict[str, object] | None = None,
    ) -> None:
        """Emit stage-start message and reset percentage thresholds."""
        self._next_percent_by_stage[stage] = self._percent_step
        payload: dict[str, object] = {"stage": stage, "status": "started"}
        if total is not None:
            payload["total"] = total
        if detail:
            payload.update(detail)
        self._emit(payload)

    def stage_advanced(
        self,
        *,
        stage: str,
        processed: int,
        total: int,
        detail: dict[str, object] | None = None,
    ) -> None:
        """Emit stage progress at configured percent checkpoints."""
        if total <= 0:
            return
        safe_processed = max(0, min(processed, total))
        percent = int((safe_processed * 100) / total)
        next_percent = self._next_percent_by_stage.get(stage, self._percent_step)
        should_emit = percent >= next_percent or safe_processed == total
        if not should_emit:
            return
        while next_percent <= percent:
            next_percent += self._percent_step
        self._next_percent_by_stage[stage] = next_percent
        payload: dict[str, object] = {
            "stage": stage,
            "status": "progress",
            "processed": safe_processed,
            "total": total,
            "percent": percent,
        }
        if detail:
            payload.update(detail)
        self._emit(payload)

    def stage_completed(
        self,
        *,
        stage: str,
        detail: dict[str, object] | None = None,
    ) -> None:
        """Emit stage-completed message and clear stage threshold state."""
        if stage in self._next_percent_by_stage:
            del self._next_percent_by_stage[stage]
        payload: dict[str, object] = {"stage": stage, "status": "completed"}
        if detail:
            payload.update(detail)
        self._emit(payload)

    def event(self, *, message: str, detail: dict[str, object] | None = None) -> None:
        """Emit one non-stage informational progress event."""
        payload: dict[str, object] = {"event": message}
        if detail:
            payload.update(detail)
        self._emit(payload)

    def _emit(self, detail: dict[str, object]) -> None:
        """Route the payload to INFO (compact) and optionally DEBUG (verbose)."""
        if self._emit_debug is not None:
            self._emit_info(self._format_message(detail, include_context=False))
            self._emit_debug(self._format_message(detail, include_context=True))
        else:
            self._emit_info(self._format_message(detail, include_context=True))

    def _format_message(self, detail: dict[str, object], *, include_context: bool) -> str:
        """Format context and detail fields into one deterministic log message."""
        pairs: list[tuple[str, object]] = []
        if include_context:
            for key in sorted(self._context.keys()):
                pairs.append((key, self._context[key]))
            for key in sorted(detail.keys()):
                pairs.append((key, detail[key]))
        else:
            for key in sorted(detail.keys()):
                if key in self._COMPACT_KEYS:
                    pairs.append((key, detail[key]))
        rendered = " ".join(f"{k}={v}" for k, v in pairs)
        return f"Incremental progress: {rendered}"
