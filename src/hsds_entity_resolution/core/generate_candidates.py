"""Candidate generation stage with canonical pair identity guarantees."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from dagster import get_dagster_logger

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.dataframe_utils import (
    clean_string_list,
    clean_text_scalar,
    frame_with_schema,
)
from hsds_entity_resolution.core.domain_utils import extract_contact_domains
from hsds_entity_resolution.core.taxonomy_utils import (
    extract_entity_taxonomy_codes,
    taxonomy_codes_match_or_parent_child,
)
from hsds_entity_resolution.observability import IncrementalProgressLogger
from hsds_entity_resolution.types.contracts import GenerateCandidatesResult
from hsds_entity_resolution.types.frames import CANDIDATE_PAIR_SCHEMA

OverlapEvaluator = Callable[[dict[str, Any], dict[str, Any]], bool]

_OVERLAP_CHANNEL_REASON_CODES: dict[str, str] = {
    "email": "shared_email",
    "phone": "shared_phone",
    "website": "shared_domain",
    "taxonomy": "shared_taxonomy",
    "location": "shared_location",
}


@dataclass(frozen=True)
class BlockingOverview:
    """Summary diagnostics for one entity type's blocking pass."""

    entity_type: str
    changed_anchors: int
    active_anchors: int
    anchors_with_above_threshold: int
    anchors_with_retained_candidates: int
    anchors_at_candidate_cap: int
    above_threshold_examined: int
    overlap_blocked: int
    pairs_kept: int
    truncated_above_threshold: int
    last_retained_similarity_sum: float
    last_retained_similarity_count: int
    taxonomy_failed: int
    non_taxonomy_failed: int
    both_failed: int


@dataclass
class BlockingDiagnosticsState:
    """Mutable diagnostics collected while expanding candidates."""

    channel_hits: dict[str, int]
    above_threshold: int = 0
    overlap_blocked: int = 0
    taxonomy_failed: int = 0
    non_taxonomy_failed: int = 0
    both_failed: int = 0
    first_anchor_id: str | None = None
    first_anchor_schema: str | None = None
    first_anchor_samples: list[dict[str, Any]] = field(default_factory=list)
    first_anchor_done: bool = False


@dataclass(frozen=True)
class AnchorProcessingResult:
    """Outcome of expanding one changed anchor."""

    saw_above_threshold: bool
    selected: int
    last_retained_similarity: float | None
    truncated_above_threshold: int


@dataclass(frozen=True)
class AggregateBlockingOverviewMetrics:
    """Aggregated metrics used by the overview logger."""

    active_anchors: int
    anchors_with_retained: int
    no_above_threshold: int
    above_threshold_but_no_retained: int
    anchors_at_cap: int
    above_threshold_examined: int
    overlap_blocked: int
    truncated_above_threshold: int
    avg_last_retained_similarity: float
    last_retained_similarity_count: int
    avg_truncated_per_capped: float
    taxonomy_failed: int
    non_taxonomy_failed: int
    both_failed: int


def generate_candidates(
    *,
    denormalized_organization: pl.DataFrame,
    denormalized_service: pl.DataFrame,
    changed_entities: pl.DataFrame,
    config: EntityResolutionRunConfig,
    explicit_backfill: bool,
    force_rescore: bool = False,
    progress_logger: IncrementalProgressLogger | None = None,
) -> GenerateCandidatesResult:
    """Generate candidate pairs using embedding similarity and overlap prefilter."""
    full_scope_rescore = explicit_backfill or force_rescore
    delta_entities = changed_entities.filter(pl.col("delta_class").is_in(["added", "changed"]))
    if delta_entities.is_empty() and not full_scope_rescore:
        return _empty_result()
    org_pairs, org_overview = _generate_for_entity_type(
        frame=denormalized_organization,
        changed_entities=delta_entities,
        entity_type="organization",
        config=config,
        full_scope_rescore=full_scope_rescore,
        progress_logger=progress_logger,
    )
    service_pairs, service_overview = _generate_for_entity_type(
        frame=denormalized_service,
        changed_entities=delta_entities,
        entity_type="service",
        config=config,
        full_scope_rescore=full_scope_rescore,
        progress_logger=progress_logger,
    )
    candidate_pairs = pl.concat([org_pairs, service_pairs], how="diagonal_relaxed")
    _log_generate_candidates_overview(
        overviews=[org_overview, service_overview],
        candidate_pair_count=candidate_pairs.height,
        config=config,
    )
    summary = pl.DataFrame(
        {
            "candidate_count": [candidate_pairs.height],
            "raw_candidate_count": [candidate_pairs.height],
        }
    )
    return GenerateCandidatesResult(candidate_pairs=candidate_pairs, candidate_summary=summary)


def _generate_for_entity_type(
    *,
    frame: pl.DataFrame,
    changed_entities: pl.DataFrame,
    entity_type: str,
    config: EntityResolutionRunConfig,
    full_scope_rescore: bool,
    progress_logger: IncrementalProgressLogger | None = None,
) -> tuple[pl.DataFrame, BlockingOverview]:
    """Generate candidates for one entity type and cap by anchor top-k."""
    type_frame = frame.filter(pl.col("entity_type") == entity_type)
    if type_frame.is_empty():
        return _empty_candidate_frame(), _empty_blocking_overview(entity_type=entity_type)
    changed_ids = set(
        changed_entities.filter(pl.col("entity_type") == entity_type)
        .get_column("entity_id")
        .to_list()
    )
    if full_scope_rescore:
        changed_ids = set(type_frame.get_column("entity_id").to_list())
    if not changed_ids:
        return _empty_candidate_frame(), _empty_blocking_overview(entity_type=entity_type)
    entity_rows = type_frame.to_dicts()
    _log_entity_sample(entity_rows=entity_rows, entity_type=entity_type)
    matrix = np.array([row["embedding_vector"] for row in entity_rows], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / np.clip(norms, a_min=1e-8, a_max=None)
    id_to_idx = {row["entity_id"]: idx for idx, row in enumerate(entity_rows)}
    pair_records, overview = _collect_candidate_records(
        entity_rows=entity_rows,
        entity_type=entity_type,
        normalized_matrix=normalized_matrix,
        id_to_idx=id_to_idx,
        changed_ids=changed_ids,
        config=config,
        progress_logger=progress_logger,
    )
    if not pair_records:
        return _empty_candidate_frame(), overview
    return (
        frame_with_schema(pair_records, CANDIDATE_PAIR_SCHEMA).sort(["entity_a_id", "entity_b_id"]),
        overview,
    )


def _collect_candidate_records(
    *,
    entity_rows: list[dict[str, Any]],
    entity_type: str,
    normalized_matrix: np.ndarray,
    id_to_idx: dict[str, int],
    changed_ids: set[str],
    config: EntityResolutionRunConfig,
    progress_logger: IncrementalProgressLogger | None = None,
) -> tuple[list[dict[str, Any]], BlockingOverview]:
    """Collect canonical candidate pairs and deduplicate by pair key."""
    pairs_by_key: dict[str, dict[str, Any]] = {}
    threshold = config.blocking.similarity_threshold
    max_per_entity = config.blocking.max_candidates_per_entity
    overlap_channels = config.blocking.overlap_prefilter_channels
    sorted_changed_ids = sorted(changed_ids)
    stage_name = f"generate_candidates.{entity_type}.anchors"
    diagnostics = BlockingDiagnosticsState(channel_hits={ch: 0 for ch in overlap_channels})

    anchors_with_above_threshold = 0
    anchors_with_retained_candidates = 0
    anchors_at_candidate_cap = 0
    active_anchors = 0
    truncated_above_threshold = 0
    last_retained_similarity_sum = 0.0
    last_retained_similarity_count = 0

    if progress_logger is not None:
        progress_logger.stage_started(stage=stage_name, total=len(sorted_changed_ids))
    for index, entity_id in enumerate(sorted_changed_ids, start=1):
        if entity_id not in id_to_idx:
            if progress_logger is not None:
                progress_logger.stage_advanced(
                    stage=stage_name,
                    processed=index,
                    total=len(sorted_changed_ids),
                )
            continue
        active_anchors += 1
        anchor_idx = id_to_idx[entity_id]
        similarities = normalized_matrix @ normalized_matrix[anchor_idx]
        top_indices = np.argsort(similarities)[::-1].tolist()
        anchor = entity_rows[anchor_idx]
        capture_this_anchor = _start_first_anchor_capture(
            diagnostics=diagnostics,
            entity_id=entity_id,
            anchor=anchor,
        )
        anchor_result = _collect_anchor_candidates(
            anchor=anchor,
            anchor_idx=anchor_idx,
            entity_rows=entity_rows,
            similarities=similarities,
            top_indices=top_indices,
            threshold=threshold,
            max_per_entity=max_per_entity,
            overlap_channels=overlap_channels,
            pairs_by_key=pairs_by_key,
            diagnostics=diagnostics,
            capture_this_anchor=capture_this_anchor,
        )
        if anchor_result.saw_above_threshold:
            anchors_with_above_threshold += 1
        if anchor_result.selected > 0:
            anchors_with_retained_candidates += 1
            if anchor_result.last_retained_similarity is not None:
                last_retained_similarity_sum += anchor_result.last_retained_similarity
                last_retained_similarity_count += 1
        if anchor_result.selected >= max_per_entity:
            anchors_at_candidate_cap += 1
        truncated_above_threshold += anchor_result.truncated_above_threshold
        if progress_logger is not None:
            progress_logger.stage_advanced(
                stage=stage_name,
                processed=index,
                total=len(sorted_changed_ids),
            )
    if progress_logger is not None:
        progress_logger.stage_completed(
            stage=stage_name,
            detail={"candidate_pairs": len(pairs_by_key)},
        )
    _log_blocking_summary(
        entity_type=entity_type,
        threshold=threshold,
        above_threshold=diagnostics.above_threshold,
        overlap_blocked=diagnostics.overlap_blocked,
        pairs_kept=len(pairs_by_key),
        overlap_channels=overlap_channels,
        channel_hits=diagnostics.channel_hits,
        first_anchor_id=diagnostics.first_anchor_id,
        first_anchor_schema=diagnostics.first_anchor_schema,
        first_anchor_samples=diagnostics.first_anchor_samples,
    )
    return list(pairs_by_key.values()), BlockingOverview(
        entity_type=entity_type,
        changed_anchors=len(sorted_changed_ids),
        active_anchors=active_anchors,
        anchors_with_above_threshold=anchors_with_above_threshold,
        anchors_with_retained_candidates=anchors_with_retained_candidates,
        anchors_at_candidate_cap=anchors_at_candidate_cap,
        above_threshold_examined=diagnostics.above_threshold,
        overlap_blocked=diagnostics.overlap_blocked,
        pairs_kept=len(pairs_by_key),
        truncated_above_threshold=truncated_above_threshold,
        last_retained_similarity_sum=last_retained_similarity_sum,
        last_retained_similarity_count=last_retained_similarity_count,
        taxonomy_failed=diagnostics.taxonomy_failed,
        non_taxonomy_failed=diagnostics.non_taxonomy_failed,
        both_failed=diagnostics.both_failed,
    )


def _empty_blocking_overview(*, entity_type: str) -> BlockingOverview:
    """Return zeroed blocking diagnostics for one entity type."""
    return BlockingOverview(
        entity_type=entity_type,
        changed_anchors=0,
        active_anchors=0,
        anchors_with_above_threshold=0,
        anchors_with_retained_candidates=0,
        anchors_at_candidate_cap=0,
        above_threshold_examined=0,
        overlap_blocked=0,
        pairs_kept=0,
        truncated_above_threshold=0,
        last_retained_similarity_sum=0.0,
        last_retained_similarity_count=0,
        taxonomy_failed=0,
        non_taxonomy_failed=0,
        both_failed=0,
    )


def _start_first_anchor_capture(
    *,
    diagnostics: BlockingDiagnosticsState,
    entity_id: str,
    anchor: dict[str, Any],
) -> bool:
    """Initialize detailed sampling for the first expanded anchor only."""
    if diagnostics.first_anchor_done:
        return False
    diagnostics.first_anchor_id = entity_id
    diagnostics.first_anchor_schema = str(anchor.get("source_schema") or "?")
    diagnostics.first_anchor_done = True
    return True


def _collect_anchor_candidates(
    *,
    anchor: dict[str, Any],
    anchor_idx: int,
    entity_rows: list[dict[str, Any]],
    similarities: np.ndarray,
    top_indices: list[int],
    threshold: float,
    max_per_entity: int,
    overlap_channels: list[str],
    pairs_by_key: dict[str, dict[str, Any]],
    diagnostics: BlockingDiagnosticsState,
    capture_this_anchor: bool,
) -> AnchorProcessingResult:
    """Expand one anchor and update shared diagnostics/candidate state."""
    selected = 0
    saw_above_threshold = False
    last_retained_similarity: float | None = None
    truncated_above_threshold = 0
    for position, candidate_idx in enumerate(top_indices):
        if selected >= max_per_entity:
            break
        if candidate_idx == anchor_idx:
            continue
        similarity = float(similarities[candidate_idx])
        if similarity < threshold:
            break
        saw_above_threshold = True
        diagnostics.above_threshold += 1
        candidate = entity_rows[candidate_idx]
        taxonomy_reasons, non_taxonomy_reasons = _collect_blocking_reasons(
            anchor=anchor,
            candidate=candidate,
            overlap_channels=overlap_channels,
        )
        overlap_reasons = [*taxonomy_reasons, *non_taxonomy_reasons]
        _record_overlap_channel_hits(
            diagnostics=diagnostics,
            overlap_channels=overlap_channels,
            overlap_reasons=overlap_reasons,
        )
        _capture_first_anchor_sample(
            diagnostics=diagnostics,
            anchor=anchor,
            candidate=candidate,
            similarity=similarity,
            overlap_reasons=overlap_reasons,
            capture_this_anchor=capture_this_anchor,
        )
        taxonomy_pass = bool(taxonomy_reasons)
        non_taxonomy_pass = bool(non_taxonomy_reasons)
        if not taxonomy_pass or not non_taxonomy_pass:
            diagnostics.overlap_blocked += 1
            if not taxonomy_pass and not non_taxonomy_pass:
                diagnostics.both_failed += 1
            elif not taxonomy_pass:
                diagnostics.taxonomy_failed += 1
            else:
                diagnostics.non_taxonomy_failed += 1
            continue
        record = _to_candidate_record(
            anchor=anchor,
            candidate=candidate,
            similarity=similarity,
            overlap_reasons=overlap_reasons,
        )
        pairs_by_key[record["pair_key"]] = record
        selected += 1
        last_retained_similarity = similarity
        if selected >= max_per_entity:
            truncated_above_threshold = _count_truncated_above_threshold(
                similarities=similarities,
                top_indices=top_indices,
                anchor_idx=anchor_idx,
                start_position=position + 1,
                threshold=threshold,
            )
            break
    return AnchorProcessingResult(
        saw_above_threshold=saw_above_threshold,
        selected=selected,
        last_retained_similarity=last_retained_similarity,
        truncated_above_threshold=truncated_above_threshold,
    )


def _count_truncated_above_threshold(
    *,
    similarities: np.ndarray,
    top_indices: list[int],
    anchor_idx: int,
    start_position: int,
    threshold: float,
) -> int:
    """Count above-threshold candidates skipped after hitting the per-anchor cap."""
    truncated = 0
    for candidate_idx in top_indices[start_position:]:
        if candidate_idx == anchor_idx:
            continue
        similarity = float(similarities[candidate_idx])
        if similarity < threshold:
            break
        truncated += 1
    return truncated


def _record_overlap_channel_hits(
    *,
    diagnostics: BlockingDiagnosticsState,
    overlap_channels: list[str],
    overlap_reasons: list[str],
) -> None:
    """Count configured overlap-channel matches across above-threshold comparisons."""
    for channel in overlap_channels:
        if _OVERLAP_CHANNEL_REASON_CODES.get(channel, "") in overlap_reasons:
            diagnostics.channel_hits[channel] += 1


def _capture_first_anchor_sample(
    *,
    diagnostics: BlockingDiagnosticsState,
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    similarity: float,
    overlap_reasons: list[str],
    capture_this_anchor: bool,
) -> None:
    """Capture a compact diagnostic snapshot for the first anchor's top candidates."""
    if not capture_this_anchor or len(diagnostics.first_anchor_samples) >= 5:
        return
    anchor_tax_codes = sorted(
        extract_entity_taxonomy_codes(entity=anchor, include_parent_codes=True)
    )
    cand_tax_codes = sorted(
        extract_entity_taxonomy_codes(entity=candidate, include_parent_codes=True)
    )
    anchor_locs = _extract_location_keys(entity=anchor)
    cand_locs = _extract_location_keys(entity=candidate)
    diagnostics.first_anchor_samples.append(
        {
            "sim": round(similarity, 4),
            "cand_schema": str(candidate.get("source_schema") or "?"),
            "anchor_tax": len(anchor.get("taxonomies") or []),
            "cand_tax": len(candidate.get("taxonomies") or []),
            "anchor_tax_codes": anchor_tax_codes[:4],
            "cand_tax_codes": cand_tax_codes[:4],
            "tax_intersect": sorted(set(anchor_tax_codes) & set(cand_tax_codes))[:3],
            "anchor_loc": len(anchor.get("locations") or []),
            "cand_loc": len(candidate.get("locations") or []),
            "anchor_loc_keys": sorted(
                {
                    key
                    for location in (anchor.get("locations") or [])
                    if isinstance(location, dict)
                    for key in location
                }
            ),
            "loc_intersect": sorted(anchor_locs & cand_locs)[:3],
            "anchor_phones": len(anchor.get("phones") or []),
            "cand_phones": len(candidate.get("phones") or []),
            "overlap": overlap_reasons,
        }
    )


def _log_entity_sample(*, entity_rows: list[dict[str, Any]], entity_type: str) -> None:
    """Emit a DEBUG snapshot of the first 3 entity rows to verify denormalized field population."""
    _log = get_dagster_logger()
    sample = entity_rows[:3]
    schemas = sorted({str(r.get("source_schema") or "?") for r in entity_rows})
    lines = "\n".join(
        f"  [{i}] id={str(r.get('entity_id') or '?')[:20]}"
        f" schema={r.get('source_schema') or '?'}"
        f" tax={len(r.get('taxonomies') or [])}"
        f" loc={len(r.get('locations') or [])}"
        f" phones={len(r.get('phones') or [])}"
        f" websites={len(r.get('websites') or [])}"
        f" emails={len(r.get('emails') or [])}"
        f" vec_len={len(r.get('embedding_vector') or [])}"
        for i, r in enumerate(sample)
    )
    # Embedding uniqueness: fingerprint each vector by its first 8 dims rounded to 3dp.
    # Identical fingerprint = shared embedding = same name+description text in source.
    fingerprints = [
        tuple(round(float(v), 3) for v in (r.get("embedding_vector") or [])[:8])
        for r in entity_rows
        if len(r.get("embedding_vector") or []) >= 8
    ]
    unique_fp = len(set(fingerprints))
    total_fp = len(fingerprints)
    dupe_pct = round(100.0 * (1 - unique_fp / total_fp), 1) if total_fp else 0.0
    _log.debug(
        "🗂 entity_sample entity_type=%s total=%d schemas=%s"
        " unique_embeddings=%d/%d (%.1f%% share a vector)\n%s",
        entity_type,
        len(entity_rows),
        schemas,
        unique_fp,
        total_fp,
        dupe_pct,
        lines,
    )


def _log_blocking_summary(
    *,
    entity_type: str,
    threshold: float,
    above_threshold: int,
    overlap_blocked: int,
    pairs_kept: int,
    overlap_channels: list[str],
    channel_hits: dict[str, int],
    first_anchor_id: str | None,
    first_anchor_schema: str | None,
    first_anchor_samples: list[dict[str, Any]],
) -> None:
    """Emit a single DEBUG summary of the full blocking pass — never called inside a loop."""
    _log = get_dagster_logger()
    channel_hits_str = " ".join(f"{ch}={channel_hits.get(ch, 0)}" for ch in overlap_channels)
    sample_lines = "\n".join(
        f"  [{i + 1}] sim={s['sim']} cand_schema={s['cand_schema']}"
        f" overlap={s['overlap']}\n"
        f"        tax={s['anchor_tax']}/{s['cand_tax']}"
        f" anchor_codes={s.get('anchor_tax_codes', [])}"
        f" cand_codes={s.get('cand_tax_codes', [])}"
        f" intersect={s.get('tax_intersect', [])}\n"
        f"        loc={s['anchor_loc']}/{s['cand_loc']}"
        f" loc_keys={s.get('anchor_loc_keys', [])}"
        f" loc_intersect={s.get('loc_intersect', [])}"
        f" phones={s['anchor_phones']}/{s['cand_phones']}"
        for i, s in enumerate(first_anchor_samples)
    )
    _log.debug(
        "🧮 blocking_summary entity_type=%s threshold=%s"
        " above_threshold=%d overlap_blocked=%d pairs_kept=%d"
        " channel_hits=[%s]\n"
        "🎯 first_anchor=%s schema=%s top_%d_candidates:\n%s",
        entity_type,
        threshold,
        above_threshold,
        overlap_blocked,
        pairs_kept,
        channel_hits_str,
        first_anchor_id or "none",
        first_anchor_schema or "?",
        len(first_anchor_samples),
        sample_lines if sample_lines else "  (no above-threshold candidates for first anchor)",
    )


def _log_generate_candidates_overview(
    *,
    overviews: list[BlockingOverview],
    candidate_pair_count: int,
    config: EntityResolutionRunConfig,
) -> None:
    """Emit one INFO-level overview for coarse blocking-tuning evaluation."""
    _log = get_dagster_logger()
    threshold = config.blocking.similarity_threshold
    max_per_entity = config.blocking.max_candidates_per_entity
    totals = _aggregate_blocking_overview_metrics(overviews=overviews)
    overview_chunks = ", ".join(_format_blocking_overview_chunk(overview) for overview in overviews)
    heuristic_signals = _blocking_heuristic_signals(
        active_anchors=totals.active_anchors,
        no_above_threshold=totals.no_above_threshold,
        anchors_at_cap=totals.anchors_at_cap,
        above_threshold_but_no_retained=totals.above_threshold_but_no_retained,
    )
    _log.info(
        "ℹ️ generate_candidates_overview threshold=%.3f max_candidates_per_entity=%d"
        " candidate_pairs=%d active_anchors=%d anchors_with_retained=%d (%.1f%%)"
        " anchors_with_no_above_threshold=%d (%.1f%%)"
        " anchors_above_threshold_but_no_retained=%d (%.1f%%)"
        " anchors_at_cap=%d (%.1f%%)"
        " avg_last_retained_similarity=%.4f (n=%d)"
        " above_threshold_truncated=%d (avg_per_capped=%.1f)"
        " overlap_blocked=%d/%d (%.1f%% of examined above-threshold)"
        " blocking_failures=[taxonomy_only=%d non_taxonomy_only=%d both=%d]"
        " heuristic=%s per_type=[%s]",
        threshold,
        max_per_entity,
        candidate_pair_count,
        totals.active_anchors,
        totals.anchors_with_retained,
        _percent(totals.anchors_with_retained, totals.active_anchors),
        totals.no_above_threshold,
        _percent(totals.no_above_threshold, totals.active_anchors),
        totals.above_threshold_but_no_retained,
        _percent(totals.above_threshold_but_no_retained, totals.active_anchors),
        totals.anchors_at_cap,
        _percent(totals.anchors_at_cap, totals.active_anchors),
        totals.avg_last_retained_similarity,
        totals.last_retained_similarity_count,
        totals.truncated_above_threshold,
        totals.avg_truncated_per_capped,
        totals.overlap_blocked,
        totals.above_threshold_examined,
        _percent(totals.overlap_blocked, totals.above_threshold_examined),
        totals.taxonomy_failed,
        totals.non_taxonomy_failed,
        totals.both_failed,
        heuristic_signals,
        overview_chunks,
    )


def _aggregate_blocking_overview_metrics(
    *,
    overviews: list[BlockingOverview],
) -> AggregateBlockingOverviewMetrics:
    """Aggregate cross-entity-type blocking metrics for the overview log."""
    active_anchors = sum(overview.active_anchors for overview in overviews)
    anchors_with_above_threshold = sum(
        overview.anchors_with_above_threshold for overview in overviews
    )
    anchors_with_retained = sum(overview.anchors_with_retained_candidates for overview in overviews)
    anchors_at_cap = sum(overview.anchors_at_candidate_cap for overview in overviews)
    above_threshold_examined = sum(overview.above_threshold_examined for overview in overviews)
    overlap_blocked = sum(overview.overlap_blocked for overview in overviews)
    truncated_above_threshold = sum(overview.truncated_above_threshold for overview in overviews)
    last_retained_similarity_sum = sum(
        overview.last_retained_similarity_sum for overview in overviews
    )
    last_retained_similarity_count = sum(
        overview.last_retained_similarity_count for overview in overviews
    )
    return AggregateBlockingOverviewMetrics(
        active_anchors=active_anchors,
        anchors_with_retained=anchors_with_retained,
        no_above_threshold=max(0, active_anchors - anchors_with_above_threshold),
        above_threshold_but_no_retained=max(
            0,
            anchors_with_above_threshold - anchors_with_retained,
        ),
        anchors_at_cap=anchors_at_cap,
        above_threshold_examined=above_threshold_examined,
        overlap_blocked=overlap_blocked,
        truncated_above_threshold=truncated_above_threshold,
        avg_last_retained_similarity=(
            last_retained_similarity_sum / last_retained_similarity_count
            if last_retained_similarity_count
            else 0.0
        ),
        last_retained_similarity_count=last_retained_similarity_count,
        avg_truncated_per_capped=(
            truncated_above_threshold / anchors_at_cap if anchors_at_cap else 0.0
        ),
        taxonomy_failed=sum(overview.taxonomy_failed for overview in overviews),
        non_taxonomy_failed=sum(overview.non_taxonomy_failed for overview in overviews),
        both_failed=sum(overview.both_failed for overview in overviews),
    )


def _format_blocking_overview_chunk(overview: BlockingOverview) -> str:
    """Render one compact per-entity-type overview chunk."""
    no_above_threshold = max(0, overview.active_anchors - overview.anchors_with_above_threshold)
    above_threshold_but_no_retained = max(
        0,
        overview.anchors_with_above_threshold - overview.anchors_with_retained_candidates,
    )
    return (
        f"{overview.entity_type}: active={overview.active_anchors}"
        f" retained={overview.anchors_with_retained_candidates}"
        f" no_above_threshold={no_above_threshold}"
        f" above_threshold_but_no_retained={above_threshold_but_no_retained}"
        f" at_cap={overview.anchors_at_candidate_cap}"
    )


def _blocking_heuristic_signals(
    *,
    active_anchors: int,
    no_above_threshold: int,
    anchors_at_cap: int,
    above_threshold_but_no_retained: int,
) -> str:
    """Return a coarse interpretation string for tuning blocking settings."""
    if active_anchors <= 0:
        return "no_active_anchors"
    signals: list[str] = []
    if _percent(no_above_threshold, active_anchors) >= 20.0:
        signals.append("threshold_may_be_high")
    if _percent(above_threshold_but_no_retained, active_anchors) >= 20.0:
        signals.append("overlap_prefilter_may_be_strict")
    if _percent(anchors_at_cap, active_anchors) >= 20.0:
        signals.append("max_candidates_may_be_low")
    if not signals:
        return "no_obvious_blocking_pressure"
    return ",".join(signals)


def _percent(numerator: int, denominator: int) -> float:
    """Return a rounded percentage without division-by-zero."""
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 1)


def _collect_blocking_reasons(
    *,
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    overlap_channels: list[str],
) -> tuple[list[str], list[str]]:
    """Return taxonomy and non-taxonomy reasons used by the blocking gate."""
    taxonomy_reasons: list[str] = []
    non_taxonomy_reasons: list[str] = []
    evaluators = _overlap_channel_evaluators()
    for channel in overlap_channels:
        evaluator = evaluators.get(channel)
        if evaluator is None:
            continue
        if evaluator(anchor, candidate):
            reason_code = _OVERLAP_CHANNEL_REASON_CODES[channel]
            if channel == "taxonomy":
                taxonomy_reasons.append(reason_code)
            else:
                non_taxonomy_reasons.append(reason_code)
    return sorted(set(taxonomy_reasons)), sorted(set(non_taxonomy_reasons))


def _to_candidate_record(
    *,
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    similarity: float,
    overlap_reasons: list[str],
) -> dict[str, Any]:
    """Build one canonical candidate record with provenance fields."""
    entity_a_id, entity_b_id = _canonical_pair(anchor["entity_id"], candidate["entity_id"])
    reason_codes = sorted(set(["embedding_threshold", *overlap_reasons]))
    pair_key = f"{entity_a_id}__{entity_b_id}"
    source_schema_a = (
        anchor["source_schema"]
        if anchor["entity_id"] == entity_a_id
        else candidate["source_schema"]
    )
    source_schema_b = (
        candidate["source_schema"]
        if candidate["entity_id"] == entity_b_id
        else anchor["source_schema"]
    )
    return {
        "pair_key": pair_key,
        "entity_a_id": entity_a_id,
        "entity_b_id": entity_b_id,
        "entity_type": anchor["entity_type"],
        "embedding_similarity": similarity,
        "candidate_reason_codes": reason_codes,
        "source_schema_a": source_schema_a,
        "source_schema_b": source_schema_b,
    }


def _overlap_channel_evaluators() -> dict[str, OverlapEvaluator]:
    """Define prefilter overlap evaluators keyed by channel."""
    return {
        "email": lambda anchor, candidate: _has_contact_overlap(
            anchor=anchor, candidate=candidate, field_name="emails"
        ),
        "phone": lambda anchor, candidate: _has_contact_overlap(
            anchor=anchor, candidate=candidate, field_name="phones"
        ),
        "website": _has_domain_overlap,
        "taxonomy": _has_taxonomy_overlap,
        "location": _has_location_overlap,
    }


def _has_contact_overlap(
    *, anchor: dict[str, Any], candidate: dict[str, Any], field_name: str
) -> bool:
    """Return true when normalized list values intersect for one contact field."""
    anchor_values = set(clean_string_list(anchor.get(field_name)))
    candidate_values = set(clean_string_list(candidate.get(field_name)))
    return bool(anchor_values.intersection(candidate_values))


def _has_domain_overlap(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Return true when email/website domain sets overlap between entities."""
    anchor_domains = extract_contact_domains(
        emails_value=anchor.get("emails"),
        websites_value=anchor.get("websites"),
    )
    candidate_domains = extract_contact_domains(
        emails_value=candidate.get("emails"),
        websites_value=candidate.get("websites"),
    )
    return bool(anchor_domains.intersection(candidate_domains))


def _has_taxonomy_overlap(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Return true for exact or direct parent-child HSIS taxonomy relationships."""
    anchor_codes = extract_entity_taxonomy_codes(entity=anchor, include_parent_codes=False)
    candidate_codes = extract_entity_taxonomy_codes(entity=candidate, include_parent_codes=False)
    return any(
        taxonomy_codes_match_or_parent_child(left_code=anchor_code, right_code=candidate_code)
        for anchor_code in anchor_codes
        for candidate_code in candidate_codes
    )


def _has_location_overlap(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Return true when (city, state) pairs overlap between entities."""
    anchor_locations = _extract_location_keys(entity=anchor)
    candidate_locations = _extract_location_keys(entity=candidate)
    return bool(anchor_locations.intersection(candidate_locations))


def _extract_location_keys(*, entity: dict[str, Any]) -> set[tuple[str, str]]:
    """Extract normalized (city, state) location keys for overlap matching."""
    locations = entity.get("locations")
    if not isinstance(locations, list):
        return set()
    keys: set[tuple[str, str]] = set()
    for location in locations:
        if not isinstance(location, dict):
            continue
        city_value = location.get("city")
        state_value = location.get("state")
        city = clean_text_scalar(city_value if isinstance(city_value, str) else "")
        state = clean_text_scalar(state_value if isinstance(state_value, str) else "")
        if city and state:
            keys.add((city, state))
    return keys


def _canonical_pair(entity_a_id: str, entity_b_id: str) -> tuple[str, str]:
    """Return lexicographically ordered pair IDs."""
    if entity_a_id < entity_b_id:
        return entity_a_id, entity_b_id
    return entity_b_id, entity_a_id


def _empty_result() -> GenerateCandidatesResult:
    """Return empty candidate stage outputs."""
    return GenerateCandidatesResult(
        candidate_pairs=_empty_candidate_frame(),
        candidate_summary=pl.DataFrame({"candidate_count": [0], "raw_candidate_count": [0]}),
    )


def _empty_candidate_frame() -> pl.DataFrame:
    """Return canonical empty candidate-pairs frame."""
    return pl.DataFrame(schema=CANDIDATE_PAIR_SCHEMA)
