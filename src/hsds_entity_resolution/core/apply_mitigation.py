"""Mitigation and finalization stage for scored entity pairs."""

from __future__ import annotations

from typing import Any

import polars as pl
from dagster import get_dagster_logger

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.dataframe_utils import (
    clean_text_scalar,
    ensure_columns,
    frame_with_schema,
    to_dataframe,
)
from hsds_entity_resolution.core.evidence_policy import count_contributing_reasons
from hsds_entity_resolution.core.pair_tiering import is_review_eligible_score
from hsds_entity_resolution.types.contracts import ApplyMitigationResult
from hsds_entity_resolution.types.frames import (
    FINALIZED_PAIRS_SCHEMA,
    MITIGATION_EVENTS_SCHEMA,
    PAIR_ID_REMAP_SCHEMA,
    PAIR_STATE_INDEX_SCHEMA,
    REMOVED_PAIR_IDS_SCHEMA,
)


def apply_mitigation(
    *,
    scored_pairs: pl.DataFrame | pl.LazyFrame,
    pair_reasons: pl.DataFrame | pl.LazyFrame,
    removed_entity_ids: pl.DataFrame | pl.LazyFrame,
    previous_pair_state_index: pl.DataFrame | pl.LazyFrame,
    config: EntityResolutionRunConfig,
    changed_entity_ids: set[str] | None = None,
    no_change: bool = False,
    scope_removed: bool = False,
) -> ApplyMitigationResult:
    """Apply mitigation overrides and emit reconciliation artifacts."""
    logger = get_dagster_logger()
    scored_pairs_df = to_dataframe(scored_pairs)
    pair_reasons_df = to_dataframe(pair_reasons)
    removed_entity_ids_df = ensure_columns(
        frame=to_dataframe(removed_entity_ids),
        columns=["entity_id"],
    )
    previous_pair_state_input_df = to_dataframe(previous_pair_state_index)
    previous_pair_state_df = _normalize_previous_pair_state(previous_pair_state_input_df)
    removed_entities = set(removed_entity_ids_df.get_column("entity_id").to_list())
    normalized_changed_entities = (
        None
        if changed_entity_ids is None
        else {
            clean_text_scalar(entity_id)
            for entity_id in changed_entity_ids
            if clean_text_scalar(entity_id)
        }
    )
    impacted_entities = (
        None
        if normalized_changed_entities is None
        else normalized_changed_entities
        | {entity_id for entity_id in removed_entities if entity_id}
    )
    _log_apply_mitigation_start(
        scored_pairs=scored_pairs_df,
        pair_reasons=pair_reasons_df,
        previous_pair_state=previous_pair_state_df,
        removed_entities=removed_entities,
        impacted_entities=impacted_entities,
        config=config,
        no_change=no_change,
        scope_removed=scope_removed,
    )
    if scope_removed:
        removed_pair_ids = _removed_pairs_for_scope_removed(
            previous_pair_state=previous_pair_state_df
        )
        result = ApplyMitigationResult(
            finalized_scored_pairs=_empty_finalized_pairs_frame(),
            mitigation_events=_empty_mitigation_events_frame(),
            removed_pair_ids=removed_pair_ids,
            pair_id_remap=_empty_pair_id_remap_frame(),
            pair_state_index=_empty_pair_state_index_frame(),
        )
        _log_apply_mitigation_summary(
            result=result,
            config=config,
            mode="scope_removed",
        )
        return result
    if scored_pairs_df.is_empty():
        if no_change:
            preserved_pair_state = _preserve_pair_state_for_no_change(
                previous_pair_state=previous_pair_state_df,
                scope_id=config.metadata.scope_id,
            )
            result = ApplyMitigationResult(
                finalized_scored_pairs=_empty_finalized_pairs_frame(),
                mitigation_events=_empty_mitigation_events_frame(),
                removed_pair_ids=_empty_removed_pair_ids_frame(),
                pair_id_remap=_empty_pair_id_remap_frame(),
                pair_state_index=preserved_pair_state,
            )
            _log_apply_mitigation_summary(
                result=result,
                config=config,
                mode="no_change_preserved",
            )
            return result
        removed_pair_ids = _removed_pairs_from_previous_only(
            previous_pair_state=previous_pair_state_df,
            removed_entities=removed_entities,
            changed_entity_ids=impacted_entities,
        )
        result = ApplyMitigationResult(
            finalized_scored_pairs=_empty_finalized_pairs_frame(),
            mitigation_events=_empty_mitigation_events_frame(),
            removed_pair_ids=removed_pair_ids,
            pair_id_remap=_empty_pair_id_remap_frame(),
            pair_state_index=_preserve_unaffected_pair_state(
                previous_pair_state=previous_pair_state_df,
                changed_entity_ids=impacted_entities,
                scope_id=config.metadata.scope_id,
            ),
        )
        _log_apply_mitigation_summary(
            result=result,
            config=config,
            mode="reconciliation_without_current_scores",
        )
        return result
    finalized_rows, mitigation_rows, removed_pair_rows = _mitigate_rows(
        scored_pairs=scored_pairs_df,
        pair_reasons=pair_reasons_df,
        previous_pair_state=previous_pair_state_df,
        removed_entities=removed_entities,
        changed_entity_ids=impacted_entities,
        config=config,
    )
    finalized = frame_with_schema(finalized_rows, FINALIZED_PAIRS_SCHEMA)
    mitigation_events = frame_with_schema(mitigation_rows, MITIGATION_EVENTS_SCHEMA)
    removed_pair_ids = frame_with_schema(removed_pair_rows, REMOVED_PAIR_IDS_SCHEMA)
    pair_state_index = _combine_pair_state_index(
        current_pair_state=_build_pair_state_index(finalized_scored_pairs=finalized, config=config),
        previous_pair_state=previous_pair_state_df,
        changed_entity_ids=impacted_entities,
        scope_id=config.metadata.scope_id,
    )
    result = ApplyMitigationResult(
        finalized_scored_pairs=finalized,
        mitigation_events=mitigation_events,
        removed_pair_ids=removed_pair_ids,
        pair_id_remap=_empty_pair_id_remap_frame(),
        pair_state_index=pair_state_index,
    )
    _log_apply_mitigation_summary(
        result=result,
        config=config,
        mode="reconciliation_only" if not config.mitigation.enabled else "mitigation_enabled",
    )
    if not config.mitigation.enabled:
        logger.info(
            "ℹ️ apply_mitigation mitigation policy disabled; reconciliation-only mode is active"
        )
    return result


def _mitigate_rows(
    *,
    scored_pairs: pl.DataFrame,
    pair_reasons: pl.DataFrame,
    previous_pair_state: pl.DataFrame,
    removed_entities: set[str],
    changed_entity_ids: set[str] | None,
    config: EntityResolutionRunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Mitigate one row at a time and accumulate reconciliation details."""
    reason_count = count_contributing_reasons(pair_reasons=pair_reasons)
    rows = (
        scored_pairs.join(reason_count, on="pair_key", how="left")
        .with_columns(pl.col("reason_count").fill_null(0))
        .to_dicts()
    )
    finalized_rows: list[dict[str, Any]] = []
    mitigation_rows: list[dict[str, Any]] = []
    removed_pair_rows: list[dict[str, Any]] = []
    for row in rows:
        result = _mitigate_row(row=row, removed_entities=removed_entities, config=config)
        finalized_rows.append(result["finalized_row"])
        mitigation_row = result["mitigation_row"]
        if mitigation_row is not None:
            mitigation_rows.append(mitigation_row)
            removed_pair_rows.append(
                {"pair_key": row["pair_key"], "cleanup_reason": mitigation_row["mitigation_reason"]}
            )
    removed_pair_rows = _append_previous_pair_removals(
        previous_pair_state=previous_pair_state,
        finalized_rows=finalized_rows,
        removed_pair_rows=removed_pair_rows,
        removed_entities=removed_entities,
        changed_entity_ids=changed_entity_ids,
        config=config,
    )
    return finalized_rows, mitigation_rows, removed_pair_rows


def _log_apply_mitigation_start(
    *,
    scored_pairs: pl.DataFrame,
    pair_reasons: pl.DataFrame,
    previous_pair_state: pl.DataFrame,
    removed_entities: set[str],
    impacted_entities: set[str] | None,
    config: EntityResolutionRunConfig,
    no_change: bool,
    scope_removed: bool,
) -> None:
    """Emit a compact start-of-stage snapshot for mitigation debugging."""
    logger = get_dagster_logger()
    logger.info(
        "ℹ️ apply_mitigation_start enabled=%s scope_removed=%s no_change=%s"
        " scored_pairs=%d pair_reasons=%d previous_pair_state=%d removed_entities=%d"
        " impacted_entities=%s min_embedding_similarity=%.2f require_reason_match=%s",
        config.mitigation.enabled,
        scope_removed,
        no_change,
        scored_pairs.height,
        pair_reasons.height,
        previous_pair_state.height,
        len(removed_entities),
        "all" if impacted_entities is None else len(impacted_entities),
        config.mitigation.min_embedding_similarity,
        config.mitigation.require_reason_match,
    )


def _log_apply_mitigation_summary(
    *,
    result: ApplyMitigationResult,
    config: EntityResolutionRunConfig,
    mode: str,
) -> None:
    """Emit one end-of-stage summary explaining mitigation output volume."""
    logger = get_dagster_logger()
    finalized = result.finalized_scored_pairs
    mitigated = result.mitigation_events
    removed = result.removed_pair_ids
    pair_state = result.pair_state_index
    predicted_duplicates = (
        finalized.filter(pl.col("predicted_duplicate").fill_null(False)).height
        if not finalized.is_empty()
        else 0
    )
    review_eligible = (
        finalized.filter(pl.col("review_eligible").fill_null(False)).height
        if not finalized.is_empty()
        else 0
    )
    cleanup_breakdown = (
        _format_cleanup_breakdown(removed_pair_ids=removed) if not removed.is_empty() else "none"
    )
    logger.info(
        "ℹ️ apply_mitigation_summary mode=%s enabled=%s finalized_pairs=%d"
        " predicted_duplicates=%d review_eligible=%d mitigation_events=%d"
        " removed_pair_ids=%d pair_state_index=%d cleanup_reasons=[%s]",
        mode,
        config.mitigation.enabled,
        finalized.height,
        predicted_duplicates,
        review_eligible,
        mitigated.height,
        removed.height,
        pair_state.height,
        cleanup_breakdown,
    )


def _format_cleanup_breakdown(*, removed_pair_ids: pl.DataFrame) -> str:
    """Render a stable cleanup-reason histogram for mitigation debugging."""
    counts = removed_pair_ids.group_by("cleanup_reason").len().sort("cleanup_reason").to_dicts()
    return ", ".join(
        f"{row['cleanup_reason']}={row['len']}"
        for row in counts
        if row.get("cleanup_reason") is not None
    )


def _append_previous_pair_removals(
    *,
    previous_pair_state: pl.DataFrame,
    finalized_rows: list[dict[str, Any]],
    removed_pair_rows: list[dict[str, Any]],
    removed_entities: set[str],
    changed_entity_ids: set[str] | None,
    config: EntityResolutionRunConfig,
) -> list[dict[str, Any]]:
    """Append cleanup rows for previously retained pairs no longer retained."""
    if previous_pair_state.is_empty():
        return removed_pair_rows
    existing: dict[str, str] = {
        str(row["pair_key"]): str(row["cleanup_reason"]) for row in removed_pair_rows
    }
    finalized_by_key = {str(row["pair_key"]): row for row in finalized_rows}
    current_retained = {
        str(row["pair_key"])
        for row in finalized_rows
        if _retained_for_state(row=row, config=config)
    }
    prior_retained = set(
        _affected_previous_pair_state(
            previous_pair_state=previous_pair_state,
            changed_entity_ids=changed_entity_ids,
        )
        .filter(pl.col("retained_flag"))
        .get_column("pair_key")
        .to_list()
    )
    for pair_key in sorted(prior_retained - current_retained):
        if pair_key in existing:
            continue
        reason = _prior_cleanup_reason(
            pair_key=pair_key,
            finalized_row=finalized_by_key.get(pair_key),
            removed_entities=removed_entities,
        )
        existing[pair_key] = reason
    return [{"pair_key": key, "cleanup_reason": existing[key]} for key in sorted(existing.keys())]


def _prior_cleanup_reason(
    *,
    pair_key: str,
    finalized_row: dict[str, Any] | None,
    removed_entities: set[str],
) -> str:
    """Classify why a previously retained pair should now be removed."""
    left, right = _pair_ids_from_key(pair_key)
    if left in removed_entities or right in removed_entities:
        return "entity_deleted"
    if finalized_row is None:
        return "candidate_lost"
    mitigation_reason = finalized_row.get("mitigation_reason")
    if mitigation_reason:
        return "mitigated_removed"
    return "score_dropped"


def _mitigate_row(
    *,
    row: dict[str, Any],
    removed_entities: set[str],
    config: EntityResolutionRunConfig,
) -> dict[str, Any]:
    """Apply mitigation policy to one scored pair."""
    predicted = bool(row["predicted_duplicate"])
    post_prediction = predicted
    mitigation_reason: str | None = None
    if row["entity_a_id"] in removed_entities or row["entity_b_id"] in removed_entities:
        post_prediction = False
        mitigation_reason = "entity_deleted"
    elif _needs_mitigation(row=row, config=config):
        post_prediction = False
        mitigation_reason = "low_evidence_override"
    review_eligible = _row_review_eligible(row=row, config=config)
    if mitigation_reason is not None:
        review_eligible = False
    finalized_row = {
        **row,
        "predicted_duplicate": post_prediction,
        "review_eligible": review_eligible,
        "mitigation_reason": mitigation_reason,
    }
    if mitigation_reason is None:
        return {"finalized_row": finalized_row, "mitigation_row": None}
    mitigation_row = {
        "pair_key": row["pair_key"],
        "mitigation_reason": mitigation_reason,
        "evidence": {
            "embedding_similarity": float(row["embedding_similarity"]),
            "reason_count": int(row["reason_count"]),
        },
        "pre_mitigation_prediction": predicted,
        "post_mitigation_prediction": post_prediction,
    }
    return {"finalized_row": finalized_row, "mitigation_row": mitigation_row}


def _needs_mitigation(*, row: dict[str, Any], config: EntityResolutionRunConfig) -> bool:
    """Evaluate mitigation predicates for one row."""
    if not config.mitigation.enabled:
        return False
    if not bool(row["predicted_duplicate"]):
        return False
    if float(row["embedding_similarity"]) >= config.mitigation.min_embedding_similarity:
        return False
    if not config.mitigation.require_reason_match:
        return True
    return int(row["reason_count"]) < config.scoring.min_reason_count_for_keep


def _empty_finalized_pairs_frame() -> pl.DataFrame:
    """Return canonical empty finalized score frame."""
    return pl.DataFrame(schema=FINALIZED_PAIRS_SCHEMA)


def _empty_mitigation_events_frame() -> pl.DataFrame:
    """Return canonical empty mitigation-events frame."""
    return pl.DataFrame(schema=MITIGATION_EVENTS_SCHEMA)


def _empty_removed_pair_ids_frame() -> pl.DataFrame:
    """Return canonical empty removed-pair artifact."""
    return pl.DataFrame(schema=REMOVED_PAIR_IDS_SCHEMA)


def _empty_pair_id_remap_frame() -> pl.DataFrame:
    """Return canonical empty pair-id-remap artifact."""
    return pl.DataFrame(schema=PAIR_ID_REMAP_SCHEMA)


def _empty_pair_state_index_frame() -> pl.DataFrame:
    """Return canonical empty pair-state index."""
    return pl.DataFrame(schema=PAIR_STATE_INDEX_SCHEMA)


def _build_pair_state_index(
    *,
    finalized_scored_pairs: pl.DataFrame,
    config: EntityResolutionRunConfig,
) -> pl.DataFrame:
    """Build minimal pair-state index for next incremental run.

    Only review-eligible pairs (duplicate / maybe band) are stored. below_maybe
    pairs are never read by the incremental reconciler — storing them wastes
    space and query cost without contributing to correctness.
    """
    if finalized_scored_pairs.is_empty():
        return _empty_pair_state_index_frame()
    indexed = ensure_columns(
        frame=finalized_scored_pairs,
        columns=["review_eligible", "predicted_duplicate"],
    )
    return (
        indexed.select(
            [
                "pair_key",
                "entity_a_id",
                "entity_b_id",
                "entity_type",
                "predicted_duplicate",
                "review_eligible",
            ]
        )
        .with_columns(
            pl.lit(config.metadata.scope_id).alias("scope_id"),
            (
                pl.when(pl.col("review_eligible").is_not_null())
                .then(pl.col("review_eligible").cast(pl.Boolean))
                .otherwise(pl.col("predicted_duplicate").cast(pl.Boolean))
            ).alias("retained_flag"),
        )
        .drop("predicted_duplicate", "review_eligible")
        .filter(pl.col("retained_flag"))
    )


def _retained_for_state(*, row: dict[str, Any], config: EntityResolutionRunConfig) -> bool:
    """Return whether a finalized row should be retained in pair-state history."""
    return _row_review_eligible(row=row, config=config)


def _row_review_eligible(*, row: dict[str, Any], config: EntityResolutionRunConfig) -> bool:
    """Read review eligibility from row when present, else derive from shared thresholds."""
    value = row.get("review_eligible")
    if isinstance(value, bool):
        return value
    final_score = row.get("final_score")
    if final_score is None:
        return bool(row.get("predicted_duplicate"))
    return is_review_eligible_score(
        final_score=float(final_score),
        duplicate_threshold=config.scoring.duplicate_threshold,
        maybe_threshold=config.scoring.maybe_threshold,
    )


def _normalize_previous_pair_state(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize previous pair-state index schema and keys for reconciliation comparisons."""
    with_required = ensure_columns(
        frame=frame,
        columns=["entity_a_id", "entity_b_id", "entity_type", "retained_flag"],
    )
    if with_required.is_empty():
        return _empty_pair_state_index_frame()
    normalized = with_required.select(
        ["entity_a_id", "entity_b_id", "entity_type", "retained_flag"]
    ).with_columns(
        pl.col("entity_a_id").map_elements(clean_text_scalar, return_dtype=pl.String),
        pl.col("entity_b_id").map_elements(clean_text_scalar, return_dtype=pl.String),
        pl.col("entity_type").map_elements(clean_text_scalar, return_dtype=pl.String),
        pl.when(pl.col("retained_flag").is_null())
        .then(pl.lit(True))
        .otherwise(pl.col("retained_flag").cast(pl.Boolean))
        .alias("retained_flag"),
    )
    return normalized.with_columns(
        pl.when(pl.col("entity_a_id") < pl.col("entity_b_id"))
        .then(pl.concat_str([pl.col("entity_a_id"), pl.lit("__"), pl.col("entity_b_id")]))
        .otherwise(pl.concat_str([pl.col("entity_b_id"), pl.lit("__"), pl.col("entity_a_id")]))
        .alias("pair_key"),
        pl.lit("").alias("scope_id"),
    )


def _removed_pairs_from_previous_only(
    *,
    previous_pair_state: pl.DataFrame,
    removed_entities: set[str],
    changed_entity_ids: set[str] | None,
) -> pl.DataFrame:
    """Build removed-pair reconciliation rows when current run produces zero scored pairs."""
    affected_previous = _affected_previous_pair_state(
        previous_pair_state=previous_pair_state,
        changed_entity_ids=changed_entity_ids,
    )
    if affected_previous.is_empty():
        return _empty_removed_pair_ids_frame()
    rows: list[dict[str, Any]] = []
    for row in affected_previous.filter(pl.col("retained_flag")).to_dicts():
        pair_key = str(row["pair_key"])
        left = str(row["entity_a_id"])
        right = str(row["entity_b_id"])
        reason = (
            "entity_deleted"
            if left in removed_entities or right in removed_entities
            else "candidate_lost"
        )
        rows.append({"pair_key": pair_key, "cleanup_reason": reason})
    return frame_with_schema(rows, REMOVED_PAIR_IDS_SCHEMA)


def _removed_pairs_for_scope_removed(*, previous_pair_state: pl.DataFrame) -> pl.DataFrame:
    """Build removed-pair rows for explicit scope decommission/remap cleanup runs."""
    if previous_pair_state.is_empty():
        return _empty_removed_pair_ids_frame()
    retained_pairs = sorted(
        set(previous_pair_state.filter(pl.col("retained_flag")).get_column("pair_key").to_list())
    )
    if not retained_pairs:
        return _empty_removed_pair_ids_frame()
    rows = [
        {"pair_key": pair_key, "cleanup_reason": "scope_removed"} for pair_key in retained_pairs
    ]
    return frame_with_schema(rows, REMOVED_PAIR_IDS_SCHEMA)


def _preserve_pair_state_for_no_change(
    *, previous_pair_state: pl.DataFrame, scope_id: str
) -> pl.DataFrame:
    """Preserve previous pair-state snapshot for idempotent no-change incremental runs."""
    if previous_pair_state.is_empty():
        return _empty_pair_state_index_frame()
    return previous_pair_state.select(
        ["pair_key", "entity_a_id", "entity_b_id", "entity_type", "retained_flag"]
    ).with_columns(pl.lit(scope_id).alias("scope_id"))


def _affected_previous_pair_state(
    *,
    previous_pair_state: pl.DataFrame,
    changed_entity_ids: set[str] | None,
) -> pl.DataFrame:
    """Return previously retained pairs whose membership intersects the changed entity set."""
    if previous_pair_state.is_empty():
        return _empty_pair_state_index_frame()
    if changed_entity_ids is None:
        return previous_pair_state
    if not changed_entity_ids:
        return _empty_pair_state_index_frame()
    return previous_pair_state.filter(
        pl.col("entity_a_id").is_in(changed_entity_ids)
        | pl.col("entity_b_id").is_in(changed_entity_ids)
    )


def _preserve_unaffected_pair_state(
    *,
    previous_pair_state: pl.DataFrame,
    changed_entity_ids: set[str] | None,
    scope_id: str,
) -> pl.DataFrame:
    """Preserve prior retained pairs whose entities were untouched by this incremental run."""
    if previous_pair_state.is_empty():
        return _empty_pair_state_index_frame()
    if changed_entity_ids is None:
        return _empty_pair_state_index_frame()
    if not changed_entity_ids:
        return _preserve_pair_state_for_no_change(
            previous_pair_state=previous_pair_state,
            scope_id=scope_id,
        )
    preserved = previous_pair_state.filter(
        ~(
            pl.col("entity_a_id").is_in(changed_entity_ids)
            | pl.col("entity_b_id").is_in(changed_entity_ids)
        )
    )
    if preserved.is_empty():
        return _empty_pair_state_index_frame()
    return preserved.select(
        ["pair_key", "entity_a_id", "entity_b_id", "entity_type", "retained_flag"]
    ).with_columns(pl.lit(scope_id).alias("scope_id"))


def _combine_pair_state_index(
    *,
    current_pair_state: pl.DataFrame,
    previous_pair_state: pl.DataFrame,
    changed_entity_ids: set[str] | None,
    scope_id: str,
) -> pl.DataFrame:
    """Merge current retained pairs with previously retained unaffected pairs."""
    preserved_previous = _preserve_unaffected_pair_state(
        previous_pair_state=previous_pair_state,
        changed_entity_ids=changed_entity_ids,
        scope_id=scope_id,
    )
    if current_pair_state.is_empty():
        return preserved_previous
    if preserved_previous.is_empty():
        return current_pair_state
    return (
        pl.concat([current_pair_state, preserved_previous], how="diagonal_relaxed")
        .unique(subset=["pair_key"], keep="first")
        .sort("pair_key")
    )


def _pair_ids_from_key(pair_key: str) -> tuple[str, str]:
    """Parse pair key into entity IDs while tolerating malformed values."""
    parts = pair_key.split("__", maxsplit=1)
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]
