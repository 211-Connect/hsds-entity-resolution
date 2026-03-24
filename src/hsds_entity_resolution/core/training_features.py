"""Shared ML feature contract and payload assembly helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hsds_entity_resolution.types.domain import EntityType

ORGANIZATION_FEATURES: tuple[str, ...] = (
    "both_have_email",
    "both_have_cities",
    "both_have_zip",
    "token_overlap_count",
    "total_token_count",
    "name_length_diff_in_tokens",
    "same_state",
    "same_zipcode",
    "name_levenshtein",
    "min_services",
    "max_services",
    "total_services",
    "both_have_services",
    "shared_service_names",
    "max_service_name_jaccard",
    "shared_service_taxonomies",
    "total_unique_service_taxonomies",
    "shared_address",
    "embedding_similarity",
    "name_complexity_diff",
    "name_complexity_ratio",
    "num_services_min",
    "tfidf_weighted_similarity",
    "shared_identifier",
    "shared_identifier_similarity",
)

SERVICE_FEATURES: tuple[str, ...] = (
    "both_have_email",
    "both_have_phone",
    "both_have_website",
    "both_have_zip",
    "same_city",
    "name_token_sort",
    "shared_taxonomy_count",
    "both_have_taxonomies",
    "total_taxonomy_count",
    "name_complexity_max",
    "description_complexity_ratio",
    "num_services_max",
    "num_services_diff",
    "percent_shared_taxonomies",
    "taxonomy_hierarchy_match_score",
    "taxonomy_pairs_sim_gt_085",
    "taxonomy_pairs_sim_gt_085_ratio",
    "is_virtual_service_diff",
    "both_are_virtual_services",
    "same_org_name_fuzzy",
    "fuzzy_name",
    "shared_address",
    "shared_phone",
    "embedding_similarity",
    "bigram_overlap",
    "description_length_ratio",
    "containment_asymmetry",
)

FEATURE_SCHEMA_VERSION = "ml-features-v1"


def feature_names_for_entity_type(entity_type: EntityType | str) -> tuple[str, ...]:
    """Return the stable model-input feature names for one entity type."""
    if entity_type == "organization":
        return ORGANIZATION_FEATURES
    return SERVICE_FEATURES


def build_signal_overrides_from_reason_sets(
    *,
    det_reasons: Sequence[Mapping[str, Any]],
    nlp_reasons: Sequence[Mapping[str, Any]],
    nlp_score: float,
) -> dict[str, float]:
    """Build override features from deterministic and NLP scoring reasons."""
    overrides: dict[str, float] = {}
    for reason in det_reasons:
        match_type = _safe_str(reason.get("match_type") or reason.get("MATCH_TYPE"))
        if match_type not in {"shared_address", "shared_phone"}:
            continue
        overrides[match_type] = _safe_float(
            reason.get("raw_contribution") or reason.get("RAW_CONTRIBUTION"),
            default=0.0,
        )
    for reason in nlp_reasons:
        match_type = _safe_str(reason.get("match_type") or reason.get("MATCH_TYPE"))
        if match_type != "name_similarity":
            continue
        overrides["fuzzy_name"] = _safe_float(
            reason.get("raw_contribution") or reason.get("RAW_CONTRIBUTION"),
            default=0.0,
        )
        break
    if "fuzzy_name" not in overrides:
        overrides["fuzzy_name"] = _safe_float(nlp_score, default=0.0)
    return overrides


def build_signal_overrides_from_pipeline_signals(
    *,
    pipeline_signals: Any,
    nlp_score: float,
) -> dict[str, float]:
    """Build override features from persisted pipeline signal payloads."""
    if not isinstance(pipeline_signals, Sequence) or isinstance(pipeline_signals, (str, bytes)):
        return {"fuzzy_name": _safe_float(nlp_score, default=0.0)}
    return build_signal_overrides_from_reason_sets(
        det_reasons=[signal for signal in pipeline_signals if isinstance(signal, Mapping)],
        nlp_reasons=[signal for signal in pipeline_signals if isinstance(signal, Mapping)],
        nlp_score=nlp_score,
    )


def build_api_feature_payload(
    *,
    pair: Mapping[str, Any],
    extractor: Any,
    entity_type: EntityType | str,
) -> dict[str, float]:
    """Build the exact ML API feature payload for one pair."""
    raw_features = extractor.extract_features(pair["entity_a"], pair["entity_b"])
    raw_features["embedding_similarity"] = _safe_float(
        pair.get("embedding_similarity"),
        default=0.0,
    )
    for key, value in (pair.get("signal_overrides") or {}).items():
        raw_features[key] = _safe_float(value, default=0.0)
    return {
        name: _safe_float(raw_features.get(name), default=0.0)
        for name in feature_names_for_entity_type(entity_type)
    }


def _safe_float(value: Any, *, default: float) -> float:
    """Convert unknown numeric-ish values to bounded float."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return parsed


def _safe_str(value: Any) -> str:
    """Normalize unknown value to stripped string."""
    return str(value).strip() if value is not None else ""
