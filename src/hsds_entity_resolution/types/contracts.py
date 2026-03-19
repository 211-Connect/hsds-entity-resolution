"""Typed stage contracts for entity-resolution pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

FrameLike = pl.DataFrame | pl.LazyFrame
ArtifactBundle = dict[str, Any]


@dataclass(frozen=True)
class CleanEntitiesResult:
    """Output contract for entity-cleaning stage."""

    denormalized_organization: pl.DataFrame
    denormalized_service: pl.DataFrame
    entity_index: pl.DataFrame
    entity_delta_summary: pl.DataFrame
    removed_entity_ids: pl.DataFrame
    changed_entities: pl.DataFrame
    no_change: bool


@dataclass(frozen=True)
class GenerateCandidatesResult:
    """Output contract for candidate generation stage."""

    candidate_pairs: pl.DataFrame
    candidate_summary: pl.DataFrame


@dataclass(frozen=True)
class ScoreCandidatesResult:
    """Output contract for scoring stage."""

    scored_pairs: pl.DataFrame
    pair_reasons: pl.DataFrame
    score_delta_summary: pl.DataFrame


@dataclass(frozen=True)
class ApplyMitigationResult:
    """Output contract for mitigation/finalization stage."""

    finalized_scored_pairs: pl.DataFrame
    mitigation_events: pl.DataFrame
    removed_pair_ids: pl.DataFrame
    pair_id_remap: pl.DataFrame
    pair_state_index: pl.DataFrame


@dataclass(frozen=True)
class PreparePersistenceArtifactsResult:
    """Output contract for persistence handoff preparation stage."""

    persistence_artifact_bundle: ArtifactBundle
    run_summary: pl.DataFrame


@dataclass(frozen=True)
class MaterializeReviewQueueResult:
    """Output contract for review queue stage."""

    review_queue_items: pl.DataFrame


@dataclass(frozen=True)
class ClusterPairsResult:
    """Output contract for correlation clustering stage."""

    clusters: pl.DataFrame
    cluster_pairs: pl.DataFrame


@dataclass(frozen=True)
class IncrementalRunResult:
    """Top-level output contract for one incremental component execution."""

    denormalized_organization: pl.DataFrame
    denormalized_service: pl.DataFrame
    entity_delta_summary: pl.DataFrame
    removed_entity_ids: pl.DataFrame
    candidate_pairs: pl.DataFrame
    scored_pairs: pl.DataFrame
    pair_reasons: pl.DataFrame
    mitigation_events: pl.DataFrame
    removed_pair_ids: pl.DataFrame
    pair_id_remap: pl.DataFrame
    clusters: pl.DataFrame
    cluster_pairs: pl.DataFrame
    pair_state_index: pl.DataFrame
    review_queue_items: pl.DataFrame
    run_summary: pl.DataFrame
    persistence_artifact_bundle: ArtifactBundle
