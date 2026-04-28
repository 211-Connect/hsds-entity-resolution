"""Sharding utilities for the generate_candidates stage.

Partitions the *anchor entity-ID set* into disjoint subsets so that multiple
Dagster ops can run ``generate_candidates`` in parallel.  Each shard receives
the full entity matrix (needed for cosine-similarity look-ups) but iterates
only over its assigned subset of anchor IDs.

Correctness guarantee
---------------------
Candidate ``pair_key`` values are canonical — ``entity_a_id < entity_b_id``
lexicographically.  When anchor A finds pair (A, B) and anchor B
independently finds pair (B, A), both produce the same ``pair_key`` and
identical field values (dot-product is symmetric, canonical ordering is
deterministic).  The merge step deduplicates by ``pair_key`` and discards
the redundant copy, so the merged result equals a monolithic run.
"""

from __future__ import annotations

from dagster import get_dagster_logger
import polars as pl

from hsds_entity_resolution.types.contracts import (
    CleanEntitiesResult,
    GenerateCandidatesResult,
)
from hsds_entity_resolution.types.frames import CANDIDATE_PAIR_SCHEMA


def partition_entity_ids_for_sharding(
    entity_ids: frozenset[str] | set[str],
    *,
    num_shards: int,
) -> list[frozenset[str]]:
    """Partition entity IDs into ``num_shards`` disjoint groups by stable hash.

    Uses the Polars FNV hash — the same algorithm used by
    ``partition_candidate_pairs_for_sharding`` — to keep hashing consistent
    across the pipeline.  Assignment is deterministic across processes (no
    ``PYTHONHASHSEED`` dependency).

    Parameters
    ----------
    entity_ids:
        Full set of anchor entity IDs to distribute across shards.
    num_shards:
        Number of partitions to produce.  ``1`` returns a single-element
        list containing the original set unchanged.
    """
    if num_shards <= 1 or not entity_ids:
        return [frozenset(entity_ids)]
    ids_list = sorted(entity_ids)
    hash_values = pl.Series("id", ids_list).hash().to_list()
    shards: list[set[str]] = [set() for _ in range(num_shards)]
    for eid, h in zip(ids_list, hash_values):
        shards[h % num_shards].add(eid)
    return [frozenset(s) for s in shards]


def compute_anchor_ids(
    *,
    cleaned: CleanEntitiesResult,
    full_scope_rescore: bool,
) -> frozenset[str]:
    """Return the complete set of anchor entity IDs that ``generate_candidates`` will iterate over.

    Mirrors the ``changed_ids`` computation inside ``_generate_for_entity_type``
    (before entity-type filtering) so that the same anchors are distributed
    across shards as would be processed in a monolithic run.

    Parameters
    ----------
    cleaned:
        The :class:`~hsds_entity_resolution.types.contracts.CleanEntitiesResult`
        produced by ``clean_entities``.
    full_scope_rescore:
        ``True`` when ``explicit_backfill or force_rescore``.  When set,
        every entity in the denormalized frames is treated as an anchor,
        rather than only entities with added/changed delta classes.
    """
    if full_scope_rescore:
        org_ids = frozenset(
            cleaned.denormalized_organization.get_column("entity_id").to_list()
        )
        svc_ids = frozenset(
            cleaned.denormalized_service.get_column("entity_id").to_list()
        )
        return org_ids | svc_ids
    delta = cleaned.changed_entities.filter(
        pl.col("delta_class").is_in(["added", "changed"])
    )
    if delta.is_empty():
        return frozenset()
    return frozenset(delta.get_column("entity_id").to_list())


def merge_generate_candidates_results(
    results: list[GenerateCandidatesResult],
) -> GenerateCandidatesResult:
    """Merge per-shard ``GenerateCandidatesResult`` objects into one.

    Concatenates all ``candidate_pairs`` frames and deduplicates by
    ``pair_key`` (keeping the first occurrence).  The same pair may appear in
    multiple shards when both endpoint entities were changed anchors that
    landed in different shards; deduplication is lossless because both
    records carry identical values.

    Parameters
    ----------
    results:
        One result per generate shard.  Empty results are skipped.
    """
    _log = get_dagster_logger()
    non_empty = [r for r in results if not r.candidate_pairs.is_empty()]
    if not non_empty:
        _log.info(
            "merge_generate_candidates: all shards empty — returning empty result"
        )
        return _empty_generate_result()
    combined = pl.concat(
        [r.candidate_pairs for r in non_empty],
        how="diagonal_relaxed",
    )
    before = combined.height
    deduped = combined.unique(subset=["pair_key"], keep="first").sort(
        ["entity_a_id", "entity_b_id"]
    )
    after = deduped.height
    _log.info(
        "merge_generate_candidates: shards=%d raw_pairs=%d deduped_pairs=%d removed_duplicates=%d",
        len(results),
        before,
        after,
        before - after,
    )
    summary = pl.DataFrame(
        {
            "candidate_count": [after],
            "raw_candidate_count": [before],
        }
    )
    return GenerateCandidatesResult(candidate_pairs=deduped, candidate_summary=summary)


def _empty_generate_result() -> GenerateCandidatesResult:
    return GenerateCandidatesResult(
        candidate_pairs=pl.DataFrame(schema=CANDIDATE_PAIR_SCHEMA),
        candidate_summary=pl.DataFrame(
            {"candidate_count": [0], "raw_candidate_count": [0]}
        ),
    )
