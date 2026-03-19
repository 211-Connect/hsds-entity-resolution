"""Candidate generation stage with canonical pair identity guarantees."""

from __future__ import annotations

from collections.abc import Callable
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
from hsds_entity_resolution.core.taxonomy_utils import extract_entity_taxonomy_codes
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
    org_pairs = _generate_for_entity_type(
        frame=denormalized_organization,
        changed_entities=delta_entities,
        entity_type="organization",
        config=config,
        full_scope_rescore=full_scope_rescore,
        progress_logger=progress_logger,
    )
    service_pairs = _generate_for_entity_type(
        frame=denormalized_service,
        changed_entities=delta_entities,
        entity_type="service",
        config=config,
        full_scope_rescore=full_scope_rescore,
        progress_logger=progress_logger,
    )
    candidate_pairs = pl.concat([org_pairs, service_pairs], how="diagonal_relaxed")
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
) -> pl.DataFrame:
    """Generate candidates for one entity type and cap by anchor top-k."""
    type_frame = frame.filter(pl.col("entity_type") == entity_type)
    if type_frame.is_empty():
        return _empty_candidate_frame()
    changed_ids = set(
        changed_entities.filter(pl.col("entity_type") == entity_type)
        .get_column("entity_id")
        .to_list()
    )
    if full_scope_rescore:
        changed_ids = set(type_frame.get_column("entity_id").to_list())
    if not changed_ids:
        return _empty_candidate_frame()
    entity_rows = type_frame.to_dicts()
    _log_entity_sample(entity_rows=entity_rows, entity_type=entity_type)
    matrix = np.array([row["embedding_vector"] for row in entity_rows], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / np.clip(norms, a_min=1e-8, a_max=None)
    id_to_idx = {row["entity_id"]: idx for idx, row in enumerate(entity_rows)}
    pair_records = _collect_candidate_records(
        entity_rows=entity_rows,
        entity_type=entity_type,
        normalized_matrix=normalized_matrix,
        id_to_idx=id_to_idx,
        changed_ids=changed_ids,
        config=config,
        progress_logger=progress_logger,
    )
    if not pair_records:
        return _empty_candidate_frame()
    return frame_with_schema(pair_records, CANDIDATE_PAIR_SCHEMA).sort(
        ["entity_a_id", "entity_b_id"]
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
) -> list[dict[str, Any]]:
    """Collect canonical candidate pairs and deduplicate by pair key."""
    pairs_by_key: dict[str, dict[str, Any]] = {}
    threshold = config.blocking.similarity_threshold
    max_per_entity = config.blocking.max_candidates_per_entity
    overlap_channels = config.blocking.overlap_prefilter_channels
    sorted_changed_ids = sorted(changed_ids)
    stage_name = f"generate_candidates.{entity_type}.anchors"

    # Diagnostic accumulators — collected across the full loop, emitted once after.
    _diag_above_threshold: int = 0
    _diag_overlap_blocked: int = 0
    _diag_channel_hits: dict[str, int] = {ch: 0 for ch in overlap_channels}
    _diag_first_anchor_id: str | None = None
    _diag_first_anchor_schema: str | None = None
    _diag_first_anchor_samples: list[dict[str, Any]] = []
    _diag_first_anchor_done = False

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
        anchor_idx = id_to_idx[entity_id]
        similarities = normalized_matrix @ normalized_matrix[anchor_idx]
        top_indices = np.argsort(similarities)[::-1].tolist()
        selected = 0

        capture_this_anchor = not _diag_first_anchor_done
        if capture_this_anchor:
            _diag_first_anchor_id = entity_id
            _diag_first_anchor_schema = str(entity_rows[anchor_idx].get("source_schema") or "?")
            _diag_first_anchor_done = True

        for candidate_idx in top_indices:
            if selected >= max_per_entity:
                break
            if candidate_idx == anchor_idx:
                continue
            similarity = float(similarities[candidate_idx])
            if similarity < threshold:
                # Cosine similarities are processed in descending order, so once a value
                # drops below threshold all remaining candidates will also be below it.
                break
            _diag_above_threshold += 1
            anchor = entity_rows[anchor_idx]
            candidate = entity_rows[candidate_idx]
            overlap_reasons = _collect_overlap_reason_codes(
                anchor=anchor,
                candidate=candidate,
                overlap_channels=overlap_channels,
            )
            # Count per-channel fires across ALL above-threshold pairs.
            for ch in overlap_channels:
                if _OVERLAP_CHANNEL_REASON_CODES.get(ch, "") in overlap_reasons:
                    _diag_channel_hits[ch] += 1
            # Capture detailed snapshot for the first anchor's top-5 candidates only.
            if capture_this_anchor and len(_diag_first_anchor_samples) < 5:
                anchor_tax_codes = sorted(
                    extract_entity_taxonomy_codes(entity=anchor, include_parent_codes=True)
                )
                cand_tax_codes = sorted(
                    extract_entity_taxonomy_codes(entity=candidate, include_parent_codes=True)
                )
                anchor_locs = _extract_location_keys(entity=anchor)
                cand_locs = _extract_location_keys(entity=candidate)
                _diag_first_anchor_samples.append(
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
                                k
                                for loc in (anchor.get("locations") or [])
                                if isinstance(loc, dict)
                                for k in loc
                            }
                        ),
                        "loc_intersect": sorted(anchor_locs & cand_locs)[:3],
                        "anchor_phones": len(anchor.get("phones") or []),
                        "cand_phones": len(candidate.get("phones") or []),
                        "overlap": overlap_reasons,
                    }
                )
            if not overlap_reasons:
                _diag_overlap_blocked += 1
                continue
            record = _to_candidate_record(
                anchor=anchor,
                candidate=candidate,
                similarity=similarity,
                overlap_reasons=overlap_reasons,
            )
            # Multiple changed anchors can surface the same pair; keyed replacement keeps
            # one canonical record and prevents duplicate downstream scoring work.
            pairs_by_key[record["pair_key"]] = record
            selected += 1
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
        above_threshold=_diag_above_threshold,
        overlap_blocked=_diag_overlap_blocked,
        pairs_kept=len(pairs_by_key),
        overlap_channels=overlap_channels,
        channel_hits=_diag_channel_hits,
        first_anchor_id=_diag_first_anchor_id,
        first_anchor_schema=_diag_first_anchor_schema,
        first_anchor_samples=_diag_first_anchor_samples,
    )
    return list(pairs_by_key.values())


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


def _collect_overlap_reason_codes(
    *,
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    overlap_channels: list[str],
) -> list[str]:
    """Return configured overlap reason codes shared by both entities."""
    reason_codes: list[str] = []
    evaluators = _overlap_channel_evaluators()
    for channel in overlap_channels:
        evaluator = evaluators.get(channel)
        if evaluator is None:
            continue
        if evaluator(anchor, candidate):
            reason_codes.append(_OVERLAP_CHANNEL_REASON_CODES[channel])
    return sorted(set(reason_codes))


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
    pair_key = f"{entity_a_id}::{entity_b_id}"
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
    """Return true when taxonomy codes overlap, including parent hierarchy codes."""
    anchor_codes = extract_entity_taxonomy_codes(entity=anchor, include_parent_codes=True)
    candidate_codes = extract_entity_taxonomy_codes(entity=candidate, include_parent_codes=True)
    return bool(anchor_codes.intersection(candidate_codes))


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
