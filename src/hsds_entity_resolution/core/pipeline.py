"""Incremental orchestration pipeline for entity-resolution execution."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from dagster import get_dagster_logger

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.apply_mitigation import apply_mitigation
from hsds_entity_resolution.core.clean_entities import clean_entities
from hsds_entity_resolution.core.cluster_pairs import cluster_pairs
from hsds_entity_resolution.core.dataframe_utils import to_dataframe
from hsds_entity_resolution.core.generate_candidates import generate_candidates
from hsds_entity_resolution.core.materialize_review_queue import materialize_review_queue
from hsds_entity_resolution.core.prepare_persistence_artifacts import prepare_persistence_artifacts
from hsds_entity_resolution.core.score_candidates import score_candidates
from hsds_entity_resolution.core.score_candidates_sharded import (
    merge_score_candidates_results,
    partition_candidate_pairs_by_entity_type,
)
from hsds_entity_resolution.observability import FrameTracer, IncrementalProgressLogger
from hsds_entity_resolution.types.contracts import (
    ApplyMitigationResult,
    CleanEntitiesResult,
    ClusterPairsResult,
    GenerateCandidatesResult,
    IncrementalRunResult,
    ScoreCandidatesResult,
)


@dataclass
class CleanEntitiesCheckpointResult:
    """Pipeline state after ``clean_entities`` only.

    Serialized to disk by the ``clean_entities_op`` in the generate-sharded
    Dagster graph so that each ``generate_shard_op`` can load the full
    entity frames and run ``generate_candidates`` over its own anchor subset
    without re-executing the clean stage.
    """

    cleaned: CleanEntitiesResult
    previous_pair_state_index: pl.DataFrame
    config: EntityResolutionRunConfig
    force_rescore: bool
    explicit_backfill: bool
    scope_removed: bool


@dataclass
class CandidatesCheckpointResult:
    """Pipeline state after clean_entities + generate_candidates.

    Serialized to disk by the candidates_checkpoint_op in the sharded Dagster
    graph, then loaded by each score_shard_op and merge_and_finish_op so the
    expensive clean/generate work is never duplicated.
    """

    cleaned: CleanEntitiesResult
    candidates: GenerateCandidatesResult
    previous_pair_state_index: pl.DataFrame
    config: EntityResolutionRunConfig
    force_rescore: bool
    explicit_backfill: bool
    scope_removed: bool


def run_incremental(
    *,
    organization_entities: pl.DataFrame | pl.LazyFrame,
    service_entities: pl.DataFrame | pl.LazyFrame,
    previous_entity_index: pl.DataFrame | pl.LazyFrame,
    previous_pair_state_index: pl.DataFrame | pl.LazyFrame,
    config: EntityResolutionRunConfig,
    taxonomy_embeddings: dict[str, list[float]] | None = None,
    explicit_backfill: bool = False,
    force_rescore: bool = False,
    scope_removed: bool = False,
    progress_logger: IncrementalProgressLogger | None = None,
    score_shards: int = 1,
) -> IncrementalRunResult:
    """Run all incremental stages and return typed artifacts for downstream consumers."""
    _log = get_dagster_logger()
    logger = progress_logger or IncrementalProgressLogger(
        emit_info=_log.info,
        emit_debug=_log.debug,
        context={
            "scope_id": config.metadata.scope_id,
            "entity_type": config.metadata.entity_type,
        },
    )
    logger.stage_started(stage="incremental_pipeline", total=7)

    tracer = _build_tracer(organization_entities)
    if tracer is not None:
        tracer.announce()

    logger.stage_started(stage="clean_entities")
    cleaned = clean_entities(
        organization_entities=organization_entities,
        service_entities=service_entities,
        previous_entity_index=previous_entity_index,
        config=config,
        progress_logger=logger,
        tracer=tracer,
    )
    logger.stage_completed(
        stage="clean_entities",
        detail={
            "denormalized_org_rows": cleaned.denormalized_organization.height,
            "denormalized_service_rows": cleaned.denormalized_service.height,
            "changed_entities": cleaned.changed_entities.height,
        },
    )
    logger.stage_advanced(stage="incremental_pipeline", processed=1, total=7)

    logger.stage_started(stage="generate_candidates")
    candidates = generate_candidates(
        denormalized_organization=cleaned.denormalized_organization,
        denormalized_service=cleaned.denormalized_service,
        changed_entities=cleaned.changed_entities,
        config=config,
        explicit_backfill=explicit_backfill,
        force_rescore=force_rescore,
        progress_logger=logger,
    )
    logger.stage_completed(
        stage="generate_candidates",
        detail={"candidate_pairs": candidates.candidate_pairs.height},
    )
    if tracer is not None:
        tracer.log_frame(candidates.candidate_pairs, "generate_candidates.candidate_pairs")
    logger.stage_advanced(stage="incremental_pipeline", processed=2, total=7)

    logger.stage_started(stage="score_candidates")
    if score_shards > 1:
        shards = partition_candidate_pairs_by_entity_type(
            candidates.candidate_pairs,
            num_shards=score_shards,
            entity_type=config.metadata.entity_type,
        )
        scored_parts = [
            score_candidates(
                candidate_pairs=shard,
                denormalized_organization=cleaned.denormalized_organization,
                denormalized_service=cleaned.denormalized_service,
                config=config,
                taxonomy_embeddings=taxonomy_embeddings,
            )
            for shard in shards
        ]
        scored = merge_score_candidates_results(scored_parts, config=config)
    else:
        scored = score_candidates(
            candidate_pairs=candidates.candidate_pairs,
            denormalized_organization=cleaned.denormalized_organization,
            denormalized_service=cleaned.denormalized_service,
            config=config,
            taxonomy_embeddings=taxonomy_embeddings,
            progress_logger=logger,
        )
    logger.stage_completed(
        stage="score_candidates",
        detail={"scored_pairs": scored.scored_pairs.height},
    )
    if tracer is not None:
        tracer.log_frame(scored.scored_pairs, "score_candidates.scored_pairs")
        tracer.log_frame(scored.pair_reasons, "score_candidates.pair_reasons")
    logger.stage_advanced(stage="incremental_pipeline", processed=3, total=7)

    logger.stage_started(stage="apply_mitigation")
    changed_entity_ids = set(
        cleaned.changed_entities.filter(pl.col("delta_class").is_in(["added", "changed"]))
        .get_column("entity_id")
        .to_list()
    )
    mitigated = apply_mitigation(
        scored_pairs=scored.scored_pairs,
        pair_reasons=scored.pair_reasons,
        removed_entity_ids=cleaned.removed_entity_ids,
        previous_pair_state_index=previous_pair_state_index,
        config=config,
        changed_entity_ids=changed_entity_ids,
        no_change=cleaned.no_change and not explicit_backfill and not force_rescore,
        scope_removed=scope_removed,
    )
    logger.stage_completed(
        stage="apply_mitigation",
        detail={
            "finalized_scored_pairs": mitigated.finalized_scored_pairs.height,
            "removed_pair_ids": mitigated.removed_pair_ids.height,
        },
    )
    if tracer is not None:
        tracer.log_frame(mitigated.finalized_scored_pairs, "apply_mitigation.finalized_pairs")
        tracer.log_frame(mitigated.removed_pair_ids, "apply_mitigation.removed_pair_ids")
        tracer.log_frame(mitigated.mitigation_events, "apply_mitigation.mitigation_events")
    logger.stage_advanced(stage="incremental_pipeline", processed=4, total=7)

    logger.stage_started(stage="cluster_pairs")
    clustered = cluster_pairs(
        finalized_scored_pairs=mitigated.finalized_scored_pairs,
        removed_pair_ids=mitigated.removed_pair_ids,
        config=config,
    )
    logger.stage_completed(
        stage="cluster_pairs",
        detail={
            "clusters": clustered.clusters.height,
            "cluster_pairs": clustered.cluster_pairs.height,
        },
    )
    if tracer is not None:
        tracer.log_frame(clustered.clusters, "cluster_pairs.clusters")
        tracer.log_frame(clustered.cluster_pairs, "cluster_pairs.cluster_pairs")
    logger.stage_advanced(stage="incremental_pipeline", processed=5, total=7)

    logger.stage_started(stage="materialize_review_queue")
    review_queue = materialize_review_queue(
        finalized_scored_pairs=mitigated.finalized_scored_pairs,
        removed_pair_ids=mitigated.removed_pair_ids,
        clusters=clustered.clusters,
        cluster_pairs=clustered.cluster_pairs,
        config=config,
    )
    logger.stage_completed(
        stage="materialize_review_queue",
        detail={"review_queue_items": review_queue.review_queue_items.height},
    )
    if tracer is not None:
        tracer.log_frame(review_queue.review_queue_items, "materialize_review_queue.items")
    logger.stage_advanced(stage="incremental_pipeline", processed=6, total=7)

    logger.stage_started(stage="prepare_persistence_artifacts")
    persistence = prepare_persistence_artifacts(
        denormalized_organization=cleaned.denormalized_organization,
        denormalized_service=cleaned.denormalized_service,
        candidate_pairs=candidates.candidate_pairs,
        finalized_scored_pairs=mitigated.finalized_scored_pairs,
        pair_reasons=scored.pair_reasons,
        mitigation_events=mitigated.mitigation_events,
        clusters=clustered.clusters,
        cluster_pairs=clustered.cluster_pairs,
        pair_state_index=mitigated.pair_state_index,
        removed_entity_ids=cleaned.removed_entity_ids,
        removed_pair_ids=mitigated.removed_pair_ids,
        pair_id_remap=mitigated.pair_id_remap,
        config=config,
    )
    logger.stage_completed(
        stage="prepare_persistence_artifacts",
        detail={
            "run_summary_rows": persistence.run_summary.height,
            "artifact_count": len(persistence.persistence_artifact_bundle),
        },
    )
    logger.stage_advanced(stage="incremental_pipeline", processed=7, total=7)
    run_summary = _augment_summary(
        base_summary=persistence.run_summary,
        cleaned=cleaned,
        candidates=candidates,
        scored=scored,
        mitigated=mitigated,
        clustered=clustered,
    )
    logger.stage_completed(
        stage="incremental_pipeline",
        detail={
            "candidate_pairs": candidates.candidate_pairs.height,
            "final_scored_pairs": mitigated.finalized_scored_pairs.height,
            "clusters": clustered.clusters.height,
        },
    )
    return IncrementalRunResult(
        denormalized_organization=cleaned.denormalized_organization,
        denormalized_service=cleaned.denormalized_service,
        entity_delta_summary=cleaned.entity_delta_summary,
        removed_entity_ids=cleaned.removed_entity_ids,
        candidate_pairs=candidates.candidate_pairs,
        scored_pairs=mitigated.finalized_scored_pairs,
        pair_reasons=scored.pair_reasons,
        mitigation_events=mitigated.mitigation_events,
        removed_pair_ids=mitigated.removed_pair_ids,
        pair_id_remap=mitigated.pair_id_remap,
        clusters=clustered.clusters,
        cluster_pairs=clustered.cluster_pairs,
        pair_state_index=mitigated.pair_state_index,
        review_queue_items=review_queue.review_queue_items,
        run_summary=run_summary,
        persistence_artifact_bundle=persistence.persistence_artifact_bundle,
    )


def run_incremental_until_candidates(
    *,
    organization_entities: pl.DataFrame | pl.LazyFrame,
    service_entities: pl.DataFrame | pl.LazyFrame,
    previous_entity_index: pl.DataFrame | pl.LazyFrame,
    previous_pair_state_index: pl.DataFrame | pl.LazyFrame,
    config: EntityResolutionRunConfig,
    explicit_backfill: bool = False,
    force_rescore: bool = False,
    scope_removed: bool = False,
    progress_logger: IncrementalProgressLogger | None = None,
) -> CandidatesCheckpointResult:
    """Run pipeline through candidate generation only.

    Returns a :class:`CandidatesCheckpointResult` that can be serialized to
    disk and used as the shared input for per-shard scoring ops, avoiding
    redundant clean/generate work across shards.
    """
    _log = get_dagster_logger()
    logger = progress_logger or IncrementalProgressLogger(
        emit_info=_log.info,
        emit_debug=_log.debug,
        context={
            "scope_id": config.metadata.scope_id,
            "entity_type": config.metadata.entity_type,
        },
    )

    tracer = _build_tracer(organization_entities)
    if tracer is not None:
        tracer.announce()

    logger.stage_started(stage="clean_entities")
    cleaned = clean_entities(
        organization_entities=organization_entities,
        service_entities=service_entities,
        previous_entity_index=previous_entity_index,
        config=config,
        progress_logger=logger,
        tracer=tracer,
    )
    logger.stage_completed(
        stage="clean_entities",
        detail={
            "denormalized_org_rows": cleaned.denormalized_organization.height,
            "denormalized_service_rows": cleaned.denormalized_service.height,
            "changed_entities": cleaned.changed_entities.height,
        },
    )

    logger.stage_started(stage="generate_candidates")
    candidates = generate_candidates(
        denormalized_organization=cleaned.denormalized_organization,
        denormalized_service=cleaned.denormalized_service,
        changed_entities=cleaned.changed_entities,
        config=config,
        explicit_backfill=explicit_backfill,
        force_rescore=force_rescore,
        progress_logger=logger,
    )
    logger.stage_completed(
        stage="generate_candidates",
        detail={"candidate_pairs": candidates.candidate_pairs.height},
    )

    return CandidatesCheckpointResult(
        cleaned=cleaned,
        candidates=candidates,
        previous_pair_state_index=to_dataframe(previous_pair_state_index),
        config=config,
        force_rescore=force_rescore,
        explicit_backfill=explicit_backfill,
        scope_removed=scope_removed,
    )


def run_incremental_until_clean_entities(
    *,
    organization_entities: pl.DataFrame | pl.LazyFrame,
    service_entities: pl.DataFrame | pl.LazyFrame,
    previous_entity_index: pl.DataFrame | pl.LazyFrame,
    previous_pair_state_index: pl.DataFrame | pl.LazyFrame,
    config: EntityResolutionRunConfig,
    explicit_backfill: bool = False,
    force_rescore: bool = False,
    scope_removed: bool = False,
    progress_logger: IncrementalProgressLogger | None = None,
) -> CleanEntitiesCheckpointResult:
    """Run the pipeline through ``clean_entities`` only.

    Returns a :class:`CleanEntitiesCheckpointResult` that can be serialized to
    disk and shared by all generate-shard ops.  Each shard op calls
    ``generate_candidates`` with a disjoint anchor subset, avoiding redundant
    entity-cleaning work across shards.
    """
    _log = get_dagster_logger()
    logger = progress_logger or IncrementalProgressLogger(
        emit_info=_log.info,
        emit_debug=_log.debug,
        context={
            "scope_id": config.metadata.scope_id,
            "entity_type": config.metadata.entity_type,
        },
    )

    tracer = _build_tracer(organization_entities)
    if tracer is not None:
        tracer.announce()

    logger.stage_started(stage="clean_entities")
    cleaned = clean_entities(
        organization_entities=organization_entities,
        service_entities=service_entities,
        previous_entity_index=previous_entity_index,
        config=config,
        progress_logger=logger,
        tracer=tracer,
    )
    logger.stage_completed(
        stage="clean_entities",
        detail={
            "denormalized_org_rows": cleaned.denormalized_organization.height,
            "denormalized_service_rows": cleaned.denormalized_service.height,
            "changed_entities": cleaned.changed_entities.height,
        },
    )

    return CleanEntitiesCheckpointResult(
        cleaned=cleaned,
        previous_pair_state_index=to_dataframe(previous_pair_state_index),
        config=config,
        force_rescore=force_rescore,
        explicit_backfill=explicit_backfill,
        scope_removed=scope_removed,
    )


def run_incremental_after_scored(
    *,
    cleaned: CleanEntitiesResult,
    candidates: GenerateCandidatesResult,
    scored: ScoreCandidatesResult,
    previous_pair_state_index: pl.DataFrame | pl.LazyFrame,
    config: EntityResolutionRunConfig,
    explicit_backfill: bool = False,
    force_rescore: bool = False,
    scope_removed: bool = False,
    progress_logger: IncrementalProgressLogger | None = None,
) -> IncrementalRunResult:
    """Complete the incremental pipeline starting from apply_mitigation.

    Accepts a pre-merged :class:`~hsds_entity_resolution.types.contracts.ScoreCandidatesResult`
    (assembled by
    :func:`~hsds_entity_resolution.core.score_candidates_sharded.merge_score_candidates_results`)
    and runs all downstream stages through ``prepare_persistence_artifacts``.
    Returns the same :class:`~hsds_entity_resolution.types.contracts.IncrementalRunResult`
    as :func:`run_incremental` so the consumer mapper and carry-forward logic
    can be applied unchanged.
    """
    _log = get_dagster_logger()
    logger = progress_logger or IncrementalProgressLogger(
        emit_info=_log.info,
        emit_debug=_log.debug,
        context={
            "scope_id": config.metadata.scope_id,
            "entity_type": config.metadata.entity_type,
        },
    )

    tracer = _build_tracer(cleaned.denormalized_organization)

    logger.stage_started(stage="apply_mitigation")
    changed_entity_ids = set(
        cleaned.changed_entities.filter(pl.col("delta_class").is_in(["added", "changed"]))
        .get_column("entity_id")
        .to_list()
    )
    mitigated = apply_mitigation(
        scored_pairs=scored.scored_pairs,
        pair_reasons=scored.pair_reasons,
        removed_entity_ids=cleaned.removed_entity_ids,
        previous_pair_state_index=previous_pair_state_index,
        config=config,
        changed_entity_ids=changed_entity_ids,
        no_change=cleaned.no_change and not explicit_backfill and not force_rescore,
        scope_removed=scope_removed,
    )
    logger.stage_completed(
        stage="apply_mitigation",
        detail={
            "finalized_scored_pairs": mitigated.finalized_scored_pairs.height,
            "removed_pair_ids": mitigated.removed_pair_ids.height,
        },
    )
    if tracer is not None:
        tracer.log_frame(mitigated.finalized_scored_pairs, "apply_mitigation.finalized_pairs")
        tracer.log_frame(mitigated.removed_pair_ids, "apply_mitigation.removed_pair_ids")
        tracer.log_frame(mitigated.mitigation_events, "apply_mitigation.mitigation_events")

    logger.stage_started(stage="cluster_pairs")
    clustered = cluster_pairs(
        finalized_scored_pairs=mitigated.finalized_scored_pairs,
        removed_pair_ids=mitigated.removed_pair_ids,
        config=config,
    )
    logger.stage_completed(
        stage="cluster_pairs",
        detail={
            "clusters": clustered.clusters.height,
            "cluster_pairs": clustered.cluster_pairs.height,
        },
    )
    if tracer is not None:
        tracer.log_frame(clustered.clusters, "cluster_pairs.clusters")
        tracer.log_frame(clustered.cluster_pairs, "cluster_pairs.cluster_pairs")

    logger.stage_started(stage="materialize_review_queue")
    review_queue = materialize_review_queue(
        finalized_scored_pairs=mitigated.finalized_scored_pairs,
        removed_pair_ids=mitigated.removed_pair_ids,
        clusters=clustered.clusters,
        cluster_pairs=clustered.cluster_pairs,
        config=config,
    )
    logger.stage_completed(
        stage="materialize_review_queue",
        detail={"review_queue_items": review_queue.review_queue_items.height},
    )
    if tracer is not None:
        tracer.log_frame(review_queue.review_queue_items, "materialize_review_queue.items")

    logger.stage_started(stage="prepare_persistence_artifacts")
    persistence = prepare_persistence_artifacts(
        denormalized_organization=cleaned.denormalized_organization,
        denormalized_service=cleaned.denormalized_service,
        candidate_pairs=candidates.candidate_pairs,
        finalized_scored_pairs=mitigated.finalized_scored_pairs,
        pair_reasons=scored.pair_reasons,
        mitigation_events=mitigated.mitigation_events,
        clusters=clustered.clusters,
        cluster_pairs=clustered.cluster_pairs,
        pair_state_index=mitigated.pair_state_index,
        removed_entity_ids=cleaned.removed_entity_ids,
        removed_pair_ids=mitigated.removed_pair_ids,
        pair_id_remap=mitigated.pair_id_remap,
        config=config,
    )
    logger.stage_completed(
        stage="prepare_persistence_artifacts",
        detail={
            "run_summary_rows": persistence.run_summary.height,
            "artifact_count": len(persistence.persistence_artifact_bundle),
        },
    )

    run_summary = _augment_summary(
        base_summary=persistence.run_summary,
        cleaned=cleaned,
        candidates=candidates,
        scored=scored,
        mitigated=mitigated,
        clustered=clustered,
    )

    return IncrementalRunResult(
        denormalized_organization=cleaned.denormalized_organization,
        denormalized_service=cleaned.denormalized_service,
        entity_delta_summary=cleaned.entity_delta_summary,
        removed_entity_ids=cleaned.removed_entity_ids,
        candidate_pairs=candidates.candidate_pairs,
        scored_pairs=mitigated.finalized_scored_pairs,
        pair_reasons=scored.pair_reasons,
        mitigation_events=mitigated.mitigation_events,
        removed_pair_ids=mitigated.removed_pair_ids,
        pair_id_remap=mitigated.pair_id_remap,
        clusters=clustered.clusters,
        cluster_pairs=clustered.cluster_pairs,
        pair_state_index=mitigated.pair_state_index,
        review_queue_items=review_queue.review_queue_items,
        run_summary=run_summary,
        persistence_artifact_bundle=persistence.persistence_artifact_bundle,
    )


def _build_tracer(
    organization_entities: pl.DataFrame | pl.LazyFrame,
) -> FrameTracer | None:
    """Pick the first entity_id from the organization frame to use as tracer.

    Returns ``None`` when the frame is empty or has no entity_id column so
    that callers can skip all tracing logic with a single ``if tracer`` check.
    """
    try:
        if isinstance(organization_entities, pl.LazyFrame):
            sample = organization_entities.limit(1).collect()
        else:
            sample = organization_entities
        if not sample.is_empty() and "entity_id" in sample.columns:
            entity_id = sample.get_column("entity_id")[0]
            if entity_id is not None:
                return FrameTracer(entity_id=str(entity_id))
    except Exception:  # noqa: BLE001
        pass
    return None


def _augment_summary(
    *,
    base_summary: pl.DataFrame,
    cleaned: CleanEntitiesResult,
    candidates: GenerateCandidatesResult,
    scored: ScoreCandidatesResult,
    mitigated: ApplyMitigationResult,
    clustered: ClusterPairsResult,
) -> pl.DataFrame:
    """Attach cross-stage metrics and no-change signal to run summary."""
    summary_row = base_summary.row(0, named=True)
    delta_row = cleaned.entity_delta_summary.row(0, named=True)
    candidate_row = candidates.candidate_summary.row(0, named=True)
    score_row = scored.score_delta_summary.row(0, named=True)
    summary_row["added_count"] = int(delta_row["added_count"])
    summary_row["changed_count"] = int(delta_row["changed_count"])
    summary_row["removed_count"] = int(delta_row["removed_count"])
    summary_row["candidate_count"] = int(candidate_row["candidate_count"])
    summary_row["duplicate_count"] = int(
        score_row.get("duplicate_count", score_row["retained_count"])
    )
    summary_row["maybe_count"] = int(score_row.get("maybe_count", 0))
    summary_row["strict_duplicate_count"] = int(score_row.get("strict_duplicate_count", 0))
    summary_row["predicted_duplicate_count"] = int(
        score_row.get("predicted_duplicate_count", score_row.get("duplicate_count", 0))
    )
    summary_row["retained_count"] = int(score_row["retained_count"])
    summary_row["mitigated_count"] = int(mitigated.mitigation_events.height)
    summary_row["cluster_count"] = int(clustered.clusters.height)
    summary_row["no_change"] = bool(cleaned.no_change)
    return pl.DataFrame([summary_row])
