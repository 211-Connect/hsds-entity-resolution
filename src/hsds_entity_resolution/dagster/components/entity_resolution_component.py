"""Dagster component wiring incremental entity-resolution stages into assets."""

from __future__ import annotations

from typing import Any, Literal

import dagster as dg
import polars as pl
from pydantic import Field

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core import run_incremental
from hsds_entity_resolution.observability import IncrementalProgressLogger


class EntityResolutionComponent(dg.Component, dg.Model, dg.Resolvable):
    """Configurable Dagster component for storage-agnostic entity-resolution compute."""

    team_id: str = "hsds"
    scope_id: str = "default"
    entity_type: Literal["organization", "service"] = "organization"
    policy_version: str = "hsds-er-v1"
    model_version: str = "embedding-only-v1"
    explicit_backfill: bool = False
    organization_entities_asset_key: str = "organization_entities"
    service_entities_asset_key: str = "service_entities"
    previous_entity_index_asset_key: str = "previous_entity_index"
    previous_pair_state_index_asset_key: str = "previous_pair_state_index"
    output_asset_prefix: list[str] = Field(default_factory=list)
    constants_overrides: dict[str, Any] = Field(default_factory=dict)

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:
        """Build Dagster definitions for this component instance."""
        del context
        run_config = self._resolved_run_config()

        @dg.multi_asset(
            name="entity_resolution_incremental",
            ins={
                "organization_entities": dg.AssetIn(
                    key=dg.AssetKey.from_coercible(self.organization_entities_asset_key)
                ),
                "service_entities": dg.AssetIn(
                    key=dg.AssetKey.from_coercible(self.service_entities_asset_key)
                ),
                "previous_entity_index": dg.AssetIn(
                    key=dg.AssetKey.from_coercible(self.previous_entity_index_asset_key)
                ),
                "previous_pair_state_index": dg.AssetIn(
                    key=dg.AssetKey.from_coercible(self.previous_pair_state_index_asset_key)
                ),
            },
            outs={
                "normalized_organization": dg.AssetOut(
                    key=self._output_key("normalized_organization")
                ),
                "normalized_service": dg.AssetOut(key=self._output_key("normalized_service")),
                "entity_delta_summary": dg.AssetOut(key=self._output_key("entity_delta_summary")),
                "removed_entity_ids": dg.AssetOut(key=self._output_key("removed_entity_ids")),
                "candidate_pairs": dg.AssetOut(key=self._output_key("candidate_pairs")),
                "scored_pairs": dg.AssetOut(key=self._output_key("scored_pairs")),
                "pair_reasons": dg.AssetOut(key=self._output_key("pair_reasons")),
                "mitigation_events": dg.AssetOut(key=self._output_key("mitigation_events")),
                "removed_pair_ids": dg.AssetOut(key=self._output_key("removed_pair_ids")),
                "pair_id_remap": dg.AssetOut(key=self._output_key("pair_id_remap")),
                "clusters": dg.AssetOut(key=self._output_key("clusters")),
                "cluster_pairs": dg.AssetOut(key=self._output_key("cluster_pairs")),
                "pair_state_index": dg.AssetOut(key=self._output_key("pair_state_index")),
                "review_queue_items": dg.AssetOut(key=self._output_key("review_queue_items")),
                "run_summary": dg.AssetOut(key=self._output_key("run_summary")),
                "persistence_artifact_bundle": dg.AssetOut(
                    key=self._output_key("persistence_artifact_bundle")
                ),
            },
        )
        def entity_resolution_incremental(
            organization_entities: Any,
            service_entities: Any,
            previous_entity_index: Any,
            previous_pair_state_index: Any,
        ):
            run_logger = dg.get_dagster_logger()
            progress_logger = IncrementalProgressLogger(
                emit_info=run_logger.info,
                emit_debug=run_logger.debug,
                context={
                    "source": "entity_resolution_component",
                    "team_id": self.team_id,
                    "scope_id": self.scope_id,
                    "entity_type": self.entity_type,
                },
            )
            org_frame = _ensure_frame(organization_entities)
            service_frame = _ensure_frame(service_entities)
            previous_entity_frame = _ensure_frame(previous_entity_index)
            previous_pair_state_frame = _ensure_frame(previous_pair_state_index)
            run_logger.info(
                "Entity resolution component started: team_id=%s "
                "scope_id=%s entity_type=%s org_rows=%d service_rows=%d "
                "prev_entity_rows=%d prev_pair_state_rows=%d explicit_backfill=%s",
                self.team_id,
                self.scope_id,
                self.entity_type,
                _frame_height(org_frame),
                _frame_height(service_frame),
                _frame_height(previous_entity_frame),
                _frame_height(previous_pair_state_frame),
                self.explicit_backfill,
            )
            run_logger.info("Entity resolution pipeline stage started")
            result = run_incremental(
                organization_entities=org_frame,
                service_entities=service_frame,
                previous_entity_index=previous_entity_frame,
                previous_pair_state_index=previous_pair_state_frame,
                config=run_config,
                explicit_backfill=self.explicit_backfill,
                progress_logger=progress_logger,
            )
            run_logger.info(
                "Entity resolution pipeline stage completed: denormalized_org=%d "
                "denormalized_service=%d candidates=%d scored_pairs=%d "
                "removed_pairs=%d clusters=%d review_queue=%d",
                result.denormalized_organization.height,
                result.denormalized_service.height,
                result.candidate_pairs.height,
                result.scored_pairs.height,
                result.removed_pair_ids.height,
                result.clusters.height,
                result.review_queue_items.height,
            )
            run_logger.info("Entity resolution component completed")
            return (
                result.denormalized_organization,
                result.denormalized_service,
                result.entity_delta_summary,
                result.removed_entity_ids,
                result.candidate_pairs,
                result.scored_pairs,
                result.pair_reasons,
                result.mitigation_events,
                result.removed_pair_ids,
                result.pair_id_remap,
                result.clusters,
                result.cluster_pairs,
                result.pair_state_index,
                result.review_queue_items,
                result.run_summary,
                result.persistence_artifact_bundle,
            )

        return dg.Definitions(assets=[entity_resolution_incremental])

    def _resolved_run_config(self) -> EntityResolutionRunConfig:
        """Resolve centralized constants with optional deployment overrides."""
        defaults = EntityResolutionRunConfig.defaults_for_entity_type(
            team_id=self.team_id,
            scope_id=self.scope_id,
            entity_type=self.entity_type,
            policy_version=self.policy_version,
            model_version=self.model_version,
        )
        if not self.constants_overrides:
            return defaults
        merged = _deep_merge(defaults.model_dump(), self.constants_overrides)
        return EntityResolutionRunConfig.model_validate(merged)

    def _output_key(self, name: str) -> dg.AssetKey:
        """Build output asset key from optional component prefix."""
        return dg.AssetKey([*self.output_asset_prefix, name])


def _ensure_frame(value: Any) -> pl.DataFrame | pl.LazyFrame:
    """Coerce supported runtime payloads to Polars frame types.

    The ``list`` branch deliberately uses schema inference: this function is
    called for four different asset inputs (org entities, service entities,
    previous entity index, previous pair state index), each with a different
    schema, so no single explicit schema can be supplied here.  In practice,
    Dagster always delivers these as ``pl.DataFrame`` objects; the ``list``
    branch is a generic escape hatch for ad-hoc testing.

    The ``None`` branch returns a zero-column empty frame; downstream stages
    call ``ensure_columns`` to add the columns they require before accessing
    any data.
    """
    if isinstance(value, (pl.DataFrame, pl.LazyFrame)):
        return value
    if isinstance(value, list):
        return pl.DataFrame(value)
    if value is None:
        return pl.DataFrame()
    message = f"Unsupported frame input type: {type(value).__name__}"
    raise TypeError(message)


def _frame_height(frame: pl.DataFrame | pl.LazyFrame) -> int:
    """Return eager frame row count, collecting lazy frames for diagnostics."""
    if isinstance(frame, pl.DataFrame):
        return frame.height
    return frame.collect().height


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge config dictionaries for hierarchical constant overrides."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
