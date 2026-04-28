"""Partitioning and merging helpers for sharded service candidate scoring."""

from __future__ import annotations

import polars as pl

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.score_candidates import log_score_diagnostics
from hsds_entity_resolution.types.contracts import ScoreCandidatesResult
from hsds_entity_resolution.types.frames import PAIR_REASONS_SCHEMA, SCORED_PAIRS_SCHEMA


def partition_candidate_pairs_for_sharding(
    candidate_pairs: pl.DataFrame,
    *,
    num_shards: int,
) -> list[pl.DataFrame]:
    """Partition all candidate pairs into ``num_shards`` disjoint shards.

    Every row is assigned to a shard using a stable Polars ``UInt64`` hash
    of ``pair_key`` modulo ``num_shards``.  No entity-type pinning is
    applied — all rows are distributed uniformly.  Use this for
    **organization** jobs where every pair has ``entity_type == "organization"``
    and uniform distribution is desired.

    When ``num_shards <= 1`` the input frame is returned in a single-element
    list without copying.  When the input is empty, ``num_shards`` empty
    frames with the original schema are returned.
    """
    if num_shards <= 1:
        return [candidate_pairs]

    schema = candidate_pairs.schema
    if candidate_pairs.is_empty():
        return [pl.DataFrame(schema=schema) for _ in range(num_shards)]

    with_shard = candidate_pairs.with_columns(
        (pl.col("pair_key").hash() % num_shards).alias("_shard_id")
    )
    return [
        with_shard.filter(pl.col("_shard_id") == i).drop("_shard_id")
        for i in range(num_shards)
    ]


def partition_candidate_pairs_for_service_sharding(
    candidate_pairs: pl.DataFrame,
    *,
    num_shards: int,
) -> list[pl.DataFrame]:
    """Partition candidate pairs into ``num_shards`` disjoint shards.

    Organization rows always land in shard 0.  Service rows are assigned
    using a stable Polars ``UInt64`` hash of ``pair_key`` modulo
    ``num_shards``, so the same pair always lands in the same shard for a
    given ``N`` (deterministic across processes and runs).

    Use this for **service** jobs where the candidate frame may contain a
    mix of ``entity_type == "service"`` and ``entity_type == "organization"``
    pairs.  For pure-organization jobs use
    :func:`partition_candidate_pairs_for_sharding` instead.

    When ``num_shards <= 1`` the input frame is returned in a single-element
    list without copying.  When the input is empty, ``num_shards`` empty
    frames with the original schema are returned.
    """
    if num_shards <= 1:
        return [candidate_pairs]

    schema = candidate_pairs.schema
    if candidate_pairs.is_empty():
        return [pl.DataFrame(schema=schema) for _ in range(num_shards)]

    # Non-service rows (organization) are pinned to shard 0.
    # Service rows are split by stable hash modulo N.
    with_shard = candidate_pairs.with_columns(
        pl.when(pl.col("entity_type") != "service")
        .then(pl.lit(0, dtype=pl.UInt64))
        .otherwise(pl.col("pair_key").hash() % num_shards)
        .alias("_shard_id")
    )

    return [
        with_shard.filter(pl.col("_shard_id") == i).drop("_shard_id")
        for i in range(num_shards)
    ]


def partition_candidate_pairs_by_entity_type(
    candidate_pairs: pl.DataFrame,
    *,
    num_shards: int,
    entity_type: str,
) -> list[pl.DataFrame]:
    """Route to the correct partition function based on ``entity_type``.

    - ``"service"`` → :func:`partition_candidate_pairs_for_service_sharding`
      (org rows pinned to shard 0, service rows distributed by hash).
    - Any other entity type → :func:`partition_candidate_pairs_for_sharding`
      (all rows distributed uniformly by hash).
    """
    if entity_type == "service":
        return partition_candidate_pairs_for_service_sharding(
            candidate_pairs, num_shards=num_shards
        )
    return partition_candidate_pairs_for_sharding(candidate_pairs, num_shards=num_shards)


def merge_score_candidates_results(
    results: list[ScoreCandidatesResult],
    *,
    config: EntityResolutionRunConfig | None = None,
) -> ScoreCandidatesResult:
    """Merge per-shard ``ScoreCandidatesResult`` objects into one combined result.

    ``scored_pairs`` and ``pair_reasons`` are concatenated across all shards.
    ``score_delta_summary`` is recomputed from the merged ``scored_pairs``
    so counts are globally accurate and not summed from stale per-shard
    snapshots.

    Empty shard results are filtered out before concatenation; if every shard
    was empty the first result is returned unchanged (canonical empty shape).

    When ``config`` is provided the merged diagnostics are emitted via
    :func:`~hsds_entity_resolution.core.score_candidates.log_score_diagnostics`
    so op logs remain meaningful even though individual shards do not emit
    their own diagnostics.
    """
    if not results:
        return _empty_merge_result()

    non_empty = [r for r in results if not r.scored_pairs.is_empty()]
    if not non_empty:
        return results[0]

    scored_pairs = pl.concat([r.scored_pairs for r in results], how="diagonal_relaxed")
    pair_reasons = pl.concat([r.pair_reasons for r in results], how="diagonal_relaxed")
    summary = _recompute_score_delta_summary(scored_pairs)

    if config is not None:
        log_score_diagnostics(scored_pairs=scored_pairs, pair_reasons=pair_reasons, config=config)

    return ScoreCandidatesResult(
        scored_pairs=scored_pairs,
        pair_reasons=pair_reasons,
        score_delta_summary=summary,
    )


def _recompute_score_delta_summary(scored_pairs: pl.DataFrame) -> pl.DataFrame:
    """Rebuild score_delta_summary from a merged scored_pairs frame."""
    return pl.DataFrame(
        {
            "candidates_scored": [scored_pairs.height],
            "ml_scored_count": [
                scored_pairs.filter(pl.col("ml_section_score").is_not_null()).height
            ],
            "duplicate_count": [scored_pairs.filter(pl.col("pair_outcome") == "duplicate").height],
            "maybe_count": [scored_pairs.filter(pl.col("pair_outcome") == "maybe").height],
            "retained_count": [scored_pairs.filter(pl.col("review_eligible")).height],
        }
    )


def _empty_merge_result() -> ScoreCandidatesResult:
    """Return canonical empty result for an empty results list."""
    return ScoreCandidatesResult(
        scored_pairs=pl.DataFrame(schema=SCORED_PAIRS_SCHEMA),
        pair_reasons=pl.DataFrame(schema=PAIR_REASONS_SCHEMA),
        score_delta_summary=pl.DataFrame(
            {
                "candidates_scored": [0],
                "ml_scored_count": [0],
                "duplicate_count": [0],
                "maybe_count": [0],
                "retained_count": [0],
            }
        ),
    )
