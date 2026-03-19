"""Persistence bundle assembly stage for host-owned storage adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import polars as pl

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.types.contracts import PreparePersistenceArtifactsResult


def prepare_persistence_artifacts(
    *,
    denormalized_organization: pl.DataFrame,
    denormalized_service: pl.DataFrame,
    candidate_pairs: pl.DataFrame,
    finalized_scored_pairs: pl.DataFrame,
    pair_reasons: pl.DataFrame,
    mitigation_events: pl.DataFrame,
    clusters: pl.DataFrame,
    cluster_pairs: pl.DataFrame,
    pair_state_index: pl.DataFrame,
    removed_entity_ids: pl.DataFrame,
    removed_pair_ids: pl.DataFrame,
    pair_id_remap: pl.DataFrame,
    config: EntityResolutionRunConfig,
) -> PreparePersistenceArtifactsResult:
    """Package host-ready typed artifacts and deterministic run summary."""
    artifact_version = "hsds-er-v1"
    bundle = _build_bundle(
        artifact_version=artifact_version,
        denormalized_organization=denormalized_organization,
        denormalized_service=denormalized_service,
        candidate_pairs=candidate_pairs,
        finalized_scored_pairs=finalized_scored_pairs,
        pair_reasons=pair_reasons,
        mitigation_events=mitigation_events,
        clusters=clusters,
        cluster_pairs=cluster_pairs,
        pair_state_index=pair_state_index,
        removed_entity_ids=removed_entity_ids,
        removed_pair_ids=removed_pair_ids,
        pair_id_remap=pair_id_remap,
        config=config,
    )
    run_summary = _build_run_summary(bundle=bundle)
    return PreparePersistenceArtifactsResult(
        persistence_artifact_bundle=bundle,
        run_summary=run_summary,
    )


def _build_bundle(
    *,
    artifact_version: str,
    denormalized_organization: pl.DataFrame,
    denormalized_service: pl.DataFrame,
    candidate_pairs: pl.DataFrame,
    finalized_scored_pairs: pl.DataFrame,
    pair_reasons: pl.DataFrame,
    mitigation_events: pl.DataFrame,
    clusters: pl.DataFrame,
    cluster_pairs: pl.DataFrame,
    pair_state_index: pl.DataFrame,
    removed_entity_ids: pl.DataFrame,
    removed_pair_ids: pl.DataFrame,
    pair_id_remap: pl.DataFrame,
    config: EntityResolutionRunConfig,
) -> dict[str, Any]:
    """Assemble strongly keyed artifact bundle dictionary."""
    return {
        "artifact_version": artifact_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": config.metadata.model_dump(),
        "constants": config.model_dump(),
        "denormalized.organization": denormalized_organization,
        "denormalized.service": denormalized_service,
        "candidate_pairs": candidate_pairs,
        "scored_pairs": finalized_scored_pairs,
        "pair_reasons": pair_reasons,
        "mitigation_events": mitigation_events,
        "clusters": clusters,
        "cluster_pairs": cluster_pairs,
        "pair_state_index": pair_state_index,
        "reconciliation.removed_entity_ids": removed_entity_ids,
        "reconciliation.removed_pair_ids": removed_pair_ids,
        "reconciliation.pair_id_remap": pair_id_remap,
    }


def _build_run_summary(*, bundle: dict[str, Any]) -> pl.DataFrame:
    """Build deterministic one-row summary with per-artifact row counts."""
    keys = [key for key in bundle if isinstance(bundle[key], pl.DataFrame)]
    rows_by_artifact = {f"rows__{key.replace('.', '_')}": bundle[key].height for key in keys}
    return pl.DataFrame(
        {"artifact_count": [len(keys)], **{key: [value] for key, value in rows_by_artifact.items()}}
    )
