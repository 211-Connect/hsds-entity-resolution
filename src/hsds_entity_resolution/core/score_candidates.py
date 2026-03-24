"""Candidate scoring stage for HSDS entity resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import polars as pl
from dagster import get_dagster_logger

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.dataframe_utils import (
    clean_string_list,
    clean_text_scalar,
    frame_with_schema,
)
from hsds_entity_resolution.core.domain_utils import domain_overlap_score
from hsds_entity_resolution.core.evidence_policy import is_contributing_evidence
from hsds_entity_resolution.core.ml_inference import score_pairs_with_model, to_legacy_entity
from hsds_entity_resolution.core.nlp import compute_nlp_score
from hsds_entity_resolution.core.pair_tiering import (
    classify_pair_outcome,
    is_review_eligible_outcome,
)
from hsds_entity_resolution.observability import IncrementalProgressLogger
from hsds_entity_resolution.types.contracts import ScoreCandidatesResult
from hsds_entity_resolution.types.frames import PAIR_REASONS_SCHEMA, SCORED_PAIRS_SCHEMA


@dataclass(frozen=True)
class ScoredPairRecord:
    """Internal normalized score object before frame conversion."""

    row: dict[str, Any]
    reasons: list[dict[str, Any]]


@dataclass(frozen=True)
class PreMlPairRecord:
    """Intermediate pair score payload before optional ML inference."""

    candidate: dict[str, Any]
    det_score: float
    nlp_score: float
    pre_ml_score: float
    det_reasons: list[dict[str, Any]]
    nlp_reasons: list[dict[str, Any]]


def score_candidates(
    *,
    candidate_pairs: pl.DataFrame,
    denormalized_organization: pl.DataFrame,
    denormalized_service: pl.DataFrame,
    config: EntityResolutionRunConfig,
    progress_logger: IncrementalProgressLogger | None = None,
) -> ScoreCandidatesResult:
    """Score candidate pairs and emit explainability reasons."""
    if candidate_pairs.is_empty():
        return _empty_result()
    entity_lookup = _build_entity_lookup(
        denormalized_organization=denormalized_organization,
        denormalized_service=denormalized_service,
    )
    candidate_rows = candidate_pairs.to_dicts()
    if progress_logger is not None:
        progress_logger.stage_started(
            stage="score_candidates.pre_score_pairs", total=len(candidate_rows)
        )
    pre_ml_records: list[PreMlPairRecord] = []
    for index, row in enumerate(candidate_rows, start=1):
        pre_ml_records.append(
            _pre_score_pair(candidate=row, entity_lookup=entity_lookup, config=config)
        )
        if progress_logger is not None:
            progress_logger.stage_advanced(
                stage="score_candidates.pre_score_pairs",
                processed=index,
                total=len(candidate_rows),
            )
    if progress_logger is not None:
        progress_logger.stage_completed(
            stage="score_candidates.pre_score_pairs",
            detail={"pairs": len(pre_ml_records)},
        )
    model_scores = _score_ml_subset(
        pre_ml_records=pre_ml_records,
        entity_lookup=entity_lookup,
        config=config,
    )
    if progress_logger is not None:
        progress_logger.stage_started(
            stage="score_candidates.finalize_pairs",
            total=len(pre_ml_records),
        )
    scored_records: list[ScoredPairRecord] = []
    for index, record in enumerate(pre_ml_records, start=1):
        scored_records.append(
            _finalize_pair(record=record, config=config, model_scores=model_scores)
        )
        if progress_logger is not None:
            progress_logger.stage_advanced(
                stage="score_candidates.finalize_pairs",
                processed=index,
                total=len(pre_ml_records),
            )
    if progress_logger is not None:
        progress_logger.stage_completed(
            stage="score_candidates.finalize_pairs",
            detail={"pairs": len(scored_records)},
        )
    scored_pairs = frame_with_schema([record.row for record in scored_records], SCORED_PAIRS_SCHEMA)
    reason_rows = [reason for record in scored_records for reason in record.reasons]
    pair_reasons = frame_with_schema(reason_rows, PAIR_REASONS_SCHEMA)
    _log_signal_band_diagnostics(
        scored_pairs=scored_pairs, pair_reasons=pair_reasons, config=config
    )
    summary = pl.DataFrame(
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
    return ScoreCandidatesResult(
        scored_pairs=scored_pairs,
        pair_reasons=pair_reasons,
        score_delta_summary=summary,
    )


def _build_entity_lookup(
    *,
    denormalized_organization: pl.DataFrame,
    denormalized_service: pl.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build key-based lookup for fast pair feature retrieval."""
    combined = pl.concat([denormalized_organization, denormalized_service], how="diagonal_relaxed")
    return {
        (row["entity_type"], row["entity_id"]): row
        for row in combined.select(
            [
                "entity_id",
                "entity_type",
                "name",
                "description",
                "emails",
                "phones",
                "websites",
                "locations",
                "taxonomies",
                "identifiers",
                "services_rollup",
                "organization_name",
                "organization_id",
                "embedding_vector",
                "source_schema",
            ]
        ).to_dicts()
    }


def _pre_score_pair(
    *,
    candidate: dict[str, Any],
    entity_lookup: dict[tuple[str, str], dict[str, Any]],
    config: EntityResolutionRunConfig,
) -> PreMlPairRecord:
    """Compute deterministic/NLP sections and ML gate prerequisites."""
    entity_type = candidate["entity_type"]
    left = entity_lookup[(entity_type, candidate["entity_a_id"])]
    right = entity_lookup[(entity_type, candidate["entity_b_id"])]
    det_score, det_reasons = _deterministic_score(left=left, right=right, config=config)
    nlp_score, nlp_reasons = _nlp_score(
        left=left,
        right=right,
        config=config,
        deterministic_score=det_score,
    )
    pre_ml_score = _compose_pre_ml_score(det_score=det_score, nlp_score=nlp_score, config=config)
    return PreMlPairRecord(
        candidate=candidate,
        det_score=det_score,
        nlp_score=nlp_score,
        pre_ml_score=pre_ml_score,
        det_reasons=det_reasons,
        nlp_reasons=nlp_reasons,
    )


def _score_ml_subset(
    *,
    pre_ml_records: list[PreMlPairRecord],
    entity_lookup: dict[tuple[str, str], dict[str, Any]],
    config: EntityResolutionRunConfig,
) -> dict[str, float]:
    """Run model inference for gated pairs, grouped by entity type."""
    if not config.scoring.ml.ml_enabled:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in pre_ml_records:
        if record.pre_ml_score < config.scoring.ml.ml_gate_threshold:
            continue
        candidate = record.candidate
        entity_type = candidate["entity_type"]
        left = entity_lookup[(entity_type, candidate["entity_a_id"])]
        right = entity_lookup[(entity_type, candidate["entity_b_id"])]
        payload = {
            "pair_key": candidate["pair_key"],
            "embedding_similarity": candidate["embedding_similarity"],
            "entity_a": to_legacy_entity(row=left),
            "entity_b": to_legacy_entity(row=right),
        }
        grouped.setdefault(entity_type, []).append(payload)
    scores: dict[str, float] = {}
    for entity_type, pairs in grouped.items():
        scores.update(score_pairs_with_model(pairs=pairs, entity_type=entity_type))
    return scores


def _finalize_pair(
    *,
    record: PreMlPairRecord,
    config: EntityResolutionRunConfig,
    model_scores: dict[str, float],
) -> ScoredPairRecord:
    """Finalize candidate score and build explainability reasons."""
    candidate = record.candidate
    entity_type = candidate["entity_type"]
    ml_score = _ml_score(
        candidate=candidate,
        pre_ml_score=record.pre_ml_score,
        config=config,
        model_scores=model_scores,
    )
    final_score = _final_score(
        det_score=record.det_score,
        nlp_score=record.nlp_score,
        ml_score=ml_score,
        config=config,
    )
    pair_outcome = classify_pair_outcome(
        final_score=final_score,
        duplicate_threshold=config.scoring.duplicate_threshold,
        maybe_threshold=config.scoring.maybe_threshold,
    )
    predicted_duplicate = pair_outcome == "duplicate"
    review_eligible = is_review_eligible_outcome(pair_outcome)
    reasons = [*record.det_reasons, *record.nlp_reasons]
    if ml_score is not None:
        ml_reason = _ml_reason(ml_score=ml_score, config=config)
        if ml_reason is not None:
            reasons.append(ml_reason)
    reasons = [{**reason, "pair_key": candidate["pair_key"]} for reason in reasons]
    row = {
        "pair_key": candidate["pair_key"],
        "entity_a_id": candidate["entity_a_id"],
        "entity_b_id": candidate["entity_b_id"],
        "source_schema_a": candidate["source_schema_a"],
        "source_schema_b": candidate["source_schema_b"],
        "entity_type": entity_type,
        "policy_version": config.metadata.policy_version,
        "model_version": config.metadata.model_version,
        "deterministic_section_score": record.det_score,
        "nlp_section_score": record.nlp_score,
        "ml_section_score": ml_score,
        "final_score": final_score,
        "predicted_duplicate": predicted_duplicate,
        "pair_outcome": pair_outcome,
        "review_eligible": review_eligible,
        "embedding_similarity": float(candidate["embedding_similarity"]),
    }
    return ScoredPairRecord(row=row, reasons=reasons)


def _deterministic_score(
    *,
    left: dict[str, Any],
    right: dict[str, Any],
    config: EntityResolutionRunConfig,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute deterministic overlap score using legacy-compatible normalization."""
    domain_raw = domain_overlap_score(
        left_emails=left.get("emails"),
        left_websites=left.get("websites"),
        right_emails=right.get("emails"),
        right_websites=right.get("websites"),
    )
    shared_address_left = _canonical_address_values(left.get("locations"))
    shared_address_right = _canonical_address_values(right.get("locations"))
    shared_identifier_left = _canonical_identifier_values(left.get("identifiers"))
    shared_identifier_right = _canonical_identifier_values(right.get("identifiers"))
    channels = {
        "shared_email": (
            left.get("emails"),
            right.get("emails"),
            config.scoring.deterministic.shared_email,
        ),
        "shared_phone": (
            left.get("phones"),
            right.get("phones"),
            config.scoring.deterministic.shared_phone,
        ),
        "shared_domain": (
            [],
            [],
            config.scoring.deterministic.shared_domain,
        ),
        "shared_address": (
            shared_address_left,
            shared_address_right,
            config.scoring.deterministic.shared_address,
        ),
        "shared_identifier": (
            shared_identifier_left,
            shared_identifier_right,
            config.scoring.deterministic.shared_identifier,
        ),
    }
    contributions: list[dict[str, Any]] = []
    weighted_total = 0.0
    enabled_weight_total = 0.0
    for match_type, (left_values, right_values, signal) in channels.items():
        raw = (
            domain_raw
            if match_type == "shared_domain"
            else _overlap_ratio(left_values=left_values, right_values=right_values)
        )
        weighted = _signal_weighted_contribution(
            raw=raw,
            signal_enabled=signal.enabled,
            weight=signal.weight,
        )
        weighted_total += weighted
        enabled_weight_total += _enabled_signal_weight(
            signal_enabled=signal.enabled,
            weight=signal.weight,
        )
        if is_contributing_evidence(raw_contribution=raw, weighted_contribution=weighted):
            contributions.append(
                _reason_row(
                    match_type=match_type,
                    raw=raw,
                    weighted=weighted,
                    signal_weight=signal.weight,
                )
            )
    return _normalize_section_score(
        weighted_sum=weighted_total,
        enabled_weight_sum=enabled_weight_total,
    ), contributions


def _signal_weighted_contribution(*, raw: float, signal_enabled: bool, weight: float) -> float:
    """Apply per-signal enable and weight controls to one raw contribution."""
    if not signal_enabled:
        return 0.0
    return raw * weight


def _enabled_signal_weight(*, signal_enabled: bool, weight: float) -> float:
    """Return effective signal weight used for deterministic section normalization."""
    if not signal_enabled or weight <= 0.0:
        return 0.0
    return weight


def _normalize_section_score(*, weighted_sum: float, enabled_weight_sum: float) -> float:
    """Normalize section weighted sum to a bounded [0, 1] score."""
    if enabled_weight_sum <= 0.0:
        return 0.0
    return min(weighted_sum / enabled_weight_sum, 1.0)


def _overlap_ratio(*, left_values: Any, right_values: Any) -> float:
    """Compute overlap ratio between normalized list-like values."""
    left_set = set(clean_string_list(left_values))
    right_set = set(clean_string_list(right_values))
    if not left_set or not right_set:
        return 0.0
    overlap_count = len(left_set.intersection(right_set))
    denominator = max(len(left_set), len(right_set))
    return overlap_count / denominator


def _canonical_address_values(locations_value: Any) -> list[str]:
    """Build canonical address tokens from normalized location payloads."""
    if not isinstance(locations_value, list):
        return []
    output: list[str] = []
    for location in locations_value:
        if not isinstance(location, dict):
            continue
        street = _normalize_address_component(
            _first_present(location, ("address_1", "address1", "line1", "street", "address"))
        )
        city = _normalize_address_component(_first_present(location, ("city",)))
        state = _normalize_address_component(_first_present(location, ("state",)))
        postal = _normalize_address_component(
            _first_present(location, ("postal_code", "postal", "zip", "zipcode"))
        )
        parts = [part for part in (street, city, state, postal) if part]
        if parts:
            output.append("|".join(parts))
    return clean_string_list(output)


def _normalize_address_component(value: object) -> str:
    """Normalize one address component for deterministic exact equality."""
    normalized = clean_text_scalar(value)
    if not normalized:
        return ""
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = f" {normalized} "
    token_replacements = {
        " n ": " north ",
        " s ": " south ",
        " e ": " east ",
        " w ": " west ",
        " st ": " street ",
        " ave ": " avenue ",
        " blvd ": " boulevard ",
        " rd ": " road ",
        " dr ": " drive ",
        " ln ": " lane ",
        " ct ": " court ",
    }
    for short, full in token_replacements.items():
        normalized = normalized.replace(short, full)
    return " ".join(normalized.split())


def _canonical_identifier_values(identifiers_value: Any) -> list[str]:
    """Build canonical identifier tokens as exact `(system, value)` tuples."""
    if not isinstance(identifiers_value, list):
        return []
    output: list[str] = []
    for identifier in identifiers_value:
        if not isinstance(identifier, dict):
            continue
        system = clean_text_scalar(
            _first_present(identifier, ("system", "identifier_system", "identifier_type", "type"))
        )
        value = clean_text_scalar(_first_present(identifier, ("value", "identifier", "id")))
        if system and value:
            output.append(f"{system}|{value}")
    return clean_string_list(output)


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> object:
    """Return first present key value from a dict-like payload."""
    lowered_lookup = {clean_text_scalar(key): value for key, value in payload.items()}
    for key in keys:
        normalized_key = clean_text_scalar(key)
        if normalized_key in lowered_lookup:
            return lowered_lookup[normalized_key]
    return None


def _nlp_score(
    *,
    left: dict[str, Any],
    right: dict[str, Any],
    config: EntityResolutionRunConfig,
    deterministic_score: float,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute name similarity with safeguards and penalty controls."""
    weighted, similarity = compute_nlp_score(
        left=left,
        right=right,
        config=config,
        deterministic_score=deterministic_score,
    )
    reason = _reason_row(
        match_type="name_similarity",
        raw=similarity,
        weighted=weighted,
        signal_weight=1.0,
    )
    reasons = (
        [reason]
        if is_contributing_evidence(
            raw_contribution=similarity,
            weighted_contribution=weighted,
        )
        else []
    )
    return weighted, reasons


def _compose_pre_ml_score(
    *, det_score: float, nlp_score: float, config: EntityResolutionRunConfig
) -> float:
    """Compose deterministic and NLP sections into the pre-ML gate score."""
    return (
        det_score * config.scoring.deterministic_section_weight
        + nlp_score * config.scoring.nlp_section_weight
    )


def _ml_score(
    *,
    candidate: dict[str, Any],
    pre_ml_score: float,
    config: EntityResolutionRunConfig,
    model_scores: dict[str, float],
) -> float | None:
    """Resolve optional ML score from model output, with embedding fallback."""
    if not config.scoring.ml.ml_enabled:
        return None
    if pre_ml_score < config.scoring.ml.ml_gate_threshold:
        return None
    model_score = model_scores.get(candidate["pair_key"])
    if model_score is not None:
        return float(model_score)
    return float(candidate["embedding_similarity"])


def _final_score(
    *,
    det_score: float,
    nlp_score: float,
    ml_score: float | None,
    config: EntityResolutionRunConfig,
) -> float:
    """Compute final ensemble score for prediction thresholding."""
    final_score = _compose_pre_ml_score(det_score=det_score, nlp_score=nlp_score, config=config)
    if ml_score is None:
        return final_score
    return final_score + (ml_score * config.scoring.ml_section_weight)


def _reason_row(
    *, match_type: str, raw: float, weighted: float, signal_weight: float
) -> dict[str, Any]:
    """Create standardized reason row payload."""
    return {
        "match_type": match_type,
        "raw_contribution": raw,
        "weighted_contribution": weighted,
        "signal_weight": signal_weight,
    }


def _ml_reason(
    *,
    ml_score: float,
    config: EntityResolutionRunConfig,
) -> dict[str, Any] | None:
    """Create reason row for ML contribution."""
    weighted = ml_score * config.scoring.ml_section_weight
    if not is_contributing_evidence(raw_contribution=ml_score, weighted_contribution=weighted):
        return None
    return _reason_row(
        match_type="ml_similarity",
        raw=ml_score,
        weighted=weighted,
        signal_weight=config.scoring.ml_section_weight,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Signal × band diagnostics
# ─────────────────────────────────────────────────────────────────────────────

# Ordered so the log tables are always printed in a consistent sequence.
_STRONG_SIGNALS: tuple[str, ...] = (
    "shared_email",
    "shared_phone",
    "shared_domain",
    "shared_address",
    "name_similarity",
    "shared_identifier",
)

_SIGNAL_SHORT: dict[str, str] = {
    "shared_email": "email",
    "shared_phone": "phone",
    "shared_domain": "domain",
    "shared_address": "address",
    "name_similarity": "name",
    "shared_identifier": "identifier",
}


def _log_signal_band_diagnostics(
    *,
    scored_pairs: pl.DataFrame,
    pair_reasons: pl.DataFrame,
    config: EntityResolutionRunConfig,
) -> None:
    """Emit signal × band diagnostics to guide ML gate and threshold calibration.

    Logs two tables:
      1. Signal-count × band: how pairs group by number of contributing signals
         and which score band they land in, plus ML gate pass rate per group.
      2. Individual signal × band: per-signal breakdown showing which band pairs
         with each evidence type typically end up in.

    A WARNING is emitted when pairs with 2+ strong signals are not passing the
    ML gate — those pairs have the most evidence and should always reach the model.
    """
    if scored_pairs.is_empty():
        return
    records = _build_signal_analysis(
        scored_pairs=scored_pairs, pair_reasons=pair_reasons, config=config
    )
    _log_signal_count_table(records=records, config=config)
    _log_individual_signal_table(records=records)


def _build_signal_analysis(
    *,
    scored_pairs: pl.DataFrame,
    pair_reasons: pl.DataFrame,
    config: EntityResolutionRunConfig,
) -> list[dict[str, object]]:
    """Annotate each scored pair with contributing-signal flags and ML gate status."""
    det_w = config.scoring.deterministic_section_weight
    nlp_w = config.scoring.nlp_section_weight
    gate = config.scoring.ml.ml_gate_threshold

    # Build one set-per-signal of pair_keys that have that contributing signal.
    signal_index: dict[str, set[str]] = {s: set() for s in _STRONG_SIGNALS}
    if not pair_reasons.is_empty() and "match_type" in pair_reasons.columns:
        for row in (
            pair_reasons.filter(pl.col("match_type").is_in(list(_STRONG_SIGNALS)))
            .select(["pair_key", "match_type"])
            .to_dicts()
        ):
            signal_index[str(row["match_type"])].add(str(row["pair_key"]))

    records: list[dict[str, object]] = []
    for row in scored_pairs.select(
        ["pair_key", "pair_outcome", "deterministic_section_score", "nlp_section_score"]
    ).to_dicts():
        pk = str(row["pair_key"])
        pre_ml = (
            float(row["deterministic_section_score"]) * det_w
            + float(row["nlp_section_score"]) * nlp_w
        )
        present = {s for s in _STRONG_SIGNALS if pk in signal_index[s]}
        records.append(
            {
                "pair_key": pk,
                "pair_outcome": str(row["pair_outcome"]),
                "pre_ml_score": pre_ml,
                "ml_gate_pass": pre_ml >= gate,
                "signal_count": len(present),
                **{f"has_{s}": s in present for s in _STRONG_SIGNALS},
            }
        )
    return records


def _log_signal_count_table(
    *,
    records: list[dict[str, object]],
    config: EntityResolutionRunConfig,
) -> None:
    """Log the signal-count × band table with ML gate pass rates per tier."""
    logger = get_dagster_logger()
    gate = config.scoring.ml.ml_gate_threshold

    # Aggregate by signal count.
    band_counts: dict[int, dict[str, int]] = {}
    gate_pass: dict[int, int] = {}
    for rec in records:
        sc = int(rec["signal_count"])  # type: ignore[arg-type]
        band = str(rec["pair_outcome"])
        band_counts.setdefault(sc, {"duplicate": 0, "maybe": 0, "below_maybe": 0, "total": 0})
        band_counts[sc][band] = band_counts[sc].get(band, 0) + 1
        band_counts[sc]["total"] += 1
        gate_pass.setdefault(sc, 0)
        if rec["ml_gate_pass"]:
            gate_pass[sc] += 1

    logger.info(
        "score_candidates signal×band | entity_type=%s dup_t=%.2f maybe_t=%.2f ml_gate=%.2f",
        config.metadata.entity_type,
        config.scoring.duplicate_threshold,
        config.scoring.maybe_threshold,
        gate,
    )
    logger.info(
        "  %-8s  %-8s  %-8s  %-8s  %-8s  %-10s",
        "signals",
        "total",
        "dup",
        "maybe",
        "below",
        "ml_gated%",
    )
    for sc in sorted(band_counts):
        row = band_counts[sc]
        total = row["total"]
        pct = 100.0 * gate_pass[sc] / max(total, 1)
        logger.info(
            "  %-8d  %-8d  %-8d  %-8d  %-8d  %-9.1f%%",
            sc,
            total,
            row.get("duplicate", 0),
            row.get("maybe", 0),
            row.get("below_maybe", 0),
            pct,
        )

    # Warn when 2+ signal pairs are missing the ML gate — they have the most
    # evidence and should always reach the model.
    strong_miss = sum(
        1
        for rec in records
        if int(rec["signal_count"]) >= 2 and not bool(rec["ml_gate_pass"])  # type: ignore[arg-type]
    )
    if strong_miss > 0:
        logger.warning(
            "ML gate miss: %d pair(s) with 2+ contributing signals have "
            "pre_ml_score < %.2f and will NOT be sent to the model. "
            "Consider lowering ml_gate_threshold in the run config.",
            strong_miss,
            gate,
        )


def _log_individual_signal_table(*, records: list[dict[str, object]]) -> None:
    """Log per-signal breakdown: how many pairs with each signal land in each band."""
    logger = get_dagster_logger()
    logger.info(
        "  %-14s  %-8s  %-8s  %-8s  %-8s",
        "signal",
        "total",
        "dup",
        "maybe",
        "below",
    )
    for signal in _STRONG_SIGNALS:
        col = f"has_{signal}"
        subset = [rec for rec in records if rec.get(col)]
        if not subset:
            continue
        dup = sum(1 for r in subset if r["pair_outcome"] == "duplicate")
        maybe = sum(1 for r in subset if r["pair_outcome"] == "maybe")
        below = sum(1 for r in subset if r["pair_outcome"] == "below_maybe")
        logger.info(
            "  %-14s  %-8d  %-8d  %-8d  %-8d",
            _SIGNAL_SHORT.get(signal, signal),
            len(subset),
            dup,
            maybe,
            below,
        )


def _empty_result() -> ScoreCandidatesResult:
    """Return empty scored artifacts."""
    return ScoreCandidatesResult(
        scored_pairs=_empty_scored_pairs_frame(),
        pair_reasons=_empty_reasons_frame(),
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


def _empty_scored_pairs_frame() -> pl.DataFrame:
    """Return canonical empty scored-pairs frame."""
    return pl.DataFrame(schema=SCORED_PAIRS_SCHEMA)


def _empty_reasons_frame() -> pl.DataFrame:
    """Return canonical empty pair-reasons frame."""
    return pl.DataFrame(schema=PAIR_REASONS_SCHEMA)
