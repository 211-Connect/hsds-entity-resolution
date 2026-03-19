"""Mitigation and finalization stage for scored entity pairs."""

from __future__ import annotations

from typing import Any

import polars as pl

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
    no_change: bool = False,
    scope_removed: bool = False,
) -> ApplyMitigationResult:
    """Apply mitigation overrides and emit reconciliation artifacts."""
    scored_pairs_df = to_dataframe(scored_pairs)
    pair_reasons_df = to_dataframe(pair_reasons)
    removed_entity_ids_df = ensure_columns(
        frame=to_dataframe(removed_entity_ids),
        columns=["entity_id"],
    )
    previous_pair_state_input_df = to_dataframe(previous_pair_state_index)
    previous_pair_state_df = _normalize_previous_pair_state(previous_pair_state_input_df)
    removed_entities = set(removed_entity_ids_df.get_column("entity_id").to_list())
    if scope_removed:
        removed_pair_ids = _removed_pairs_for_scope_removed(
            previous_pair_state=previous_pair_state_df
        )
        return ApplyMitigationResult(
            finalized_scored_pairs=_empty_finalized_pairs_frame(),
            mitigation_events=_empty_mitigation_events_frame(),
            removed_pair_ids=removed_pair_ids,
            pair_id_remap=_empty_pair_id_remap_frame(),
            pair_state_index=_empty_pair_state_index_frame(),
        )
    if scored_pairs_df.is_empty():
        if no_change:
            preserved_pair_state = _preserve_pair_state_for_no_change(
                previous_pair_state=previous_pair_state_df,
                scope_id=config.metadata.scope_id,
            )
            return ApplyMitigationResult(
                finalized_scored_pairs=_empty_finalized_pairs_frame(),
                mitigation_events=_empty_mitigation_events_frame(),
                removed_pair_ids=_empty_removed_pair_ids_frame(),
                pair_id_remap=_empty_pair_id_remap_frame(),
                pair_state_index=preserved_pair_state,
            )
        removed_pair_ids = _removed_pairs_from_previous_only(
            previous_pair_state=previous_pair_state_df,
            removed_entities=removed_entities,
        )
        return ApplyMitigationResult(
            finalized_scored_pairs=_empty_finalized_pairs_frame(),
            mitigation_events=_empty_mitigation_events_frame(),
            removed_pair_ids=removed_pair_ids,
            pair_id_remap=_empty_pair_id_remap_frame(),
            pair_state_index=_empty_pair_state_index_frame(),
        )
    finalized_rows, mitigation_rows, removed_pair_rows = _mitigate_rows(
        scored_pairs=scored_pairs_df,
        pair_reasons=pair_reasons_df,
        previous_pair_state=previous_pair_state_df,
        removed_entities=removed_entities,
        config=config,
    )
    finalized = frame_with_schema(finalized_rows, FINALIZED_PAIRS_SCHEMA)
    mitigation_events = frame_with_schema(mitigation_rows, MITIGATION_EVENTS_SCHEMA)
    removed_pair_ids = frame_with_schema(removed_pair_rows, REMOVED_PAIR_IDS_SCHEMA)
    pair_state_index = _build_pair_state_index(finalized_scored_pairs=finalized, config=config)
    return ApplyMitigationResult(
        finalized_scored_pairs=finalized,
        mitigation_events=mitigation_events,
        removed_pair_ids=removed_pair_ids,
        pair_id_remap=_empty_pair_id_remap_frame(),
        pair_state_index=pair_state_index,
    )


def _mitigate_rows(
    *,
    scored_pairs: pl.DataFrame,
    pair_reasons: pl.DataFrame,
    previous_pair_state: pl.DataFrame,
    removed_entities: set[str],
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
        config=config,
    )
    return finalized_rows, mitigation_rows, removed_pair_rows


def _append_previous_pair_removals(
    *,
    previous_pair_state: pl.DataFrame,
    finalized_rows: list[dict[str, Any]],
    removed_pair_rows: list[dict[str, Any]],
    removed_entities: set[str],
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
        previous_pair_state.filter(pl.col("retained_flag")).get_column("pair_key").to_list()
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
        .then(pl.concat_str([pl.col("entity_a_id"), pl.lit("::"), pl.col("entity_b_id")]))
        .otherwise(pl.concat_str([pl.col("entity_b_id"), pl.lit("::"), pl.col("entity_a_id")]))
        .alias("pair_key"),
        pl.lit("").alias("scope_id"),
    )


def _removed_pairs_from_previous_only(
    *,
    previous_pair_state: pl.DataFrame,
    removed_entities: set[str],
) -> pl.DataFrame:
    """Build removed-pair reconciliation rows when current run produces zero scored pairs."""
    if previous_pair_state.is_empty():
        return _empty_removed_pair_ids_frame()
    rows: list[dict[str, Any]] = []
    for row in previous_pair_state.filter(pl.col("retained_flag")).to_dicts():
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


def _pair_ids_from_key(pair_key: str) -> tuple[str, str]:
    """Parse pair key into entity IDs while tolerating malformed values."""
    parts = pair_key.split("::", maxsplit=1)
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]
