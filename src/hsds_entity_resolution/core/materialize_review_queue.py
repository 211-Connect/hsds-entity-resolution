"""Review queue materialization stage for steward-facing triage flows."""

from __future__ import annotations

import polars as pl

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.pair_tiering import (
    is_review_eligible_outcome,
    is_review_eligible_score,
)
from hsds_entity_resolution.types.contracts import MaterializeReviewQueueResult
from hsds_entity_resolution.types.frames import REVIEW_QUEUE_SCHEMA


def materialize_review_queue(
    *,
    finalized_scored_pairs: pl.DataFrame,
    removed_pair_ids: pl.DataFrame,
    clusters: pl.DataFrame,
    cluster_pairs: pl.DataFrame,
    config: EntityResolutionRunConfig,
) -> MaterializeReviewQueueResult:
    """Build ranked review queue from finalized eligible pairs."""
    if finalized_scored_pairs.is_empty():
        return MaterializeReviewQueueResult(review_queue_items=_empty_review_queue_frame())
    removed_keys = (
        set(removed_pair_ids.get_column("pair_key").to_list())
        if "pair_key" in removed_pair_ids.columns
        else set()
    )
    queue = finalized_scored_pairs.filter(
        _review_eligibility_expression(
            config=config,
            available_columns=set(finalized_scored_pairs.columns),
        )
    )
    if removed_keys:
        queue = queue.filter(~pl.col("pair_key").is_in(sorted(removed_keys)))
    if queue.is_empty():
        return MaterializeReviewQueueResult(review_queue_items=_empty_review_queue_frame())
    pair_to_risk = _pair_cluster_risk_lookup(clusters=clusters, cluster_pairs=cluster_pairs)
    queue = queue.with_columns(
        pl.col("pair_key")
        .map_elements(lambda key: pair_to_risk.get(str(key), 0.0), return_dtype=pl.Float64)
        .alias("cluster_risk_score")
    )
    review_items = queue.with_columns(
        (
            pl.col("final_score") * 0.55
            + pl.col("embedding_similarity") * 0.25
            + pl.col("cluster_risk_score") * 0.20
        ).alias("priority_score"),
        pl.lit(config.metadata.team_id).alias("team_id"),
        pl.lit(config.metadata.scope_id).alias("scope_id"),
        pl.lit(config.metadata.entity_type).alias("entity_type"),
    ).sort("priority_score", descending=True)
    selected = review_items.select(
        [
            "pair_key",
            "entity_a_id",
            "entity_b_id",
            "final_score",
            "cluster_risk_score",
            "priority_score",
            "team_id",
            "scope_id",
            "entity_type",
        ]
    )
    return MaterializeReviewQueueResult(review_queue_items=selected)


def _empty_review_queue_frame() -> pl.DataFrame:
    """Return canonical empty review-queue artifact."""
    return pl.DataFrame(schema=REVIEW_QUEUE_SCHEMA)


def _pair_cluster_risk_lookup(
    *,
    clusters: pl.DataFrame,
    cluster_pairs: pl.DataFrame,
) -> dict[str, float]:
    """Build pair-key to cluster-risk lookup for queue ranking."""
    if clusters.is_empty() or cluster_pairs.is_empty():
        return {}
    cluster_risk = {
        str(row["cluster_id"]): float(row.get("cluster_risk_score") or 0.0)
        for row in clusters.select(["cluster_id", "cluster_risk_score"]).to_dicts()
    }
    return {
        str(row["pair_key"]): cluster_risk.get(str(row["cluster_id"]), 0.0)
        for row in cluster_pairs.select(["pair_key", "cluster_id"]).to_dicts()
    }


def _review_eligibility_expression(
    *,
    config: EntityResolutionRunConfig,
    available_columns: set[str],
) -> pl.Expr:
    """Return a shared review-eligibility expression with graceful schema fallback."""
    if "review_eligible" in available_columns:
        return pl.col("review_eligible").cast(pl.Boolean).fill_null(False)
    if "pair_outcome" in available_columns:
        return (
            pl.col("pair_outcome")
            .map_elements(is_review_eligible_outcome, return_dtype=pl.Boolean)
            .fill_null(False)
        )
    if "predicted_duplicate" in available_columns:
        return pl.col("predicted_duplicate").cast(pl.Boolean).fill_null(False)
    return (
        pl.col("final_score")
        .map_elements(
            lambda value: is_review_eligible_score(
                final_score=float(value),
                duplicate_threshold=config.scoring.duplicate_threshold,
                maybe_threshold=config.scoring.maybe_threshold,
            ),
            return_dtype=pl.Boolean,
        )
        .fill_null(False)
    )
