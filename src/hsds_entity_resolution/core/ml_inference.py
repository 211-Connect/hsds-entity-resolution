"""AI-Utils ML inference adapter using legacy feature extraction semantics."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib import error, request

from hsds_entity_resolution.core.taxonomy_utils import to_legacy_services, to_legacy_taxonomies

_log = logging.getLogger(__name__)

ORGANIZATION_FEATURES = [
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
]

SERVICE_FEATURES = [
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
]


def score_pairs_with_model(
    *,
    pairs: list[dict[str, Any]],
    entity_type: str,
    taxonomy_embeddings: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    """Score candidate pairs via AI-Utils endpoint and return `pair_key -> score`."""
    endpoint = os.getenv("AI_UTILS_ENDPOINT", "").strip().rstrip("/")
    if not endpoint or not pairs:
        return {}
    timeout = _int_env("AI_UTILS_TIMEOUT_SECONDS", 30)
    batch_size = max(_int_env("AI_UTILS_BATCH_SIZE", 100), 1)
    api_key = os.getenv("AI_UTILS_API_KEY")
    try:
        extractor = _build_feature_extractor(
            entity_type=entity_type,
            taxonomy_embeddings=taxonomy_embeddings,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        _log.warning("ML scoring skipped: feature extractor unavailable (%s)", exc)
        return {}
    features = [
        _extract_pair_features(pair=pair, extractor=extractor, entity_type=entity_type)
        for pair in pairs
    ]
    scores: dict[str, float] = {}
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]
        payload = {
            "model_type": entity_type,
            "features": [item["api_features"] for item in batch],
        }
        response = _post_json(
            url=f"{endpoint}/api/v1/dedupe_ml",
            payload=payload,
            api_key=api_key,
            timeout=timeout,
        )
        if response is None:
            continue
        results = response.get("results")
        if not isinstance(results, list):
            continue
        for index, result in enumerate(results):
            if index >= len(batch) or not isinstance(result, dict):
                continue
            confidence = _safe_float(result.get("isDupeConfidence"), default=0.0)
            is_dupe = bool(result.get("isDupe", False))
            score = confidence if is_dupe else (1.0 - confidence)
            scores[batch[index]["pair_key"]] = max(0.0, min(1.0, score))
    return scores


def _build_feature_extractor(
    *,
    entity_type: str,
    taxonomy_embeddings: dict[str, list[float]] | None = None,
) -> Any:
    """Create legacy feature extractor with matching TF-IDF model path semantics."""
    from hsds_entity_resolution.core.feature_extractor import FeatureExtractor

    vectorizer_path = _resolve_tfidf_vectorizer_path(entity_type=entity_type)
    return FeatureExtractor(
        rollup_services=True,
        model_type=entity_type,
        tfidf_vectorizer_path=str(vectorizer_path) if vectorizer_path else None,
        taxonomy_embeddings=taxonomy_embeddings or {},
    )


def _resolve_tfidf_vectorizer_path(*, entity_type: str) -> Path | None:
    """Resolve TF-IDF vectorizer path, preferring package-local models."""
    filename = f"tfidf_vectorizer_{entity_type}.joblib"
    override_dir = os.getenv("AI_UTILS_TFIDF_MODEL_DIR", "").strip()
    package_root = Path(__file__).resolve().parents[1]
    candidates: list[Path] = []
    if override_dir:
        candidates.append(Path(override_dir) / filename)
    candidates.append(package_root / "tf_idf_models" / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _extract_pair_features(
    *,
    pair: dict[str, Any],
    extractor: Any,
    entity_type: str,
) -> dict[str, Any]:
    """Extract legacy ML feature payload for one pair and keep pair identity."""
    entity_a = pair["entity_a"]
    entity_b = pair["entity_b"]
    raw_features = extractor.extract_features(entity_a, entity_b)
    raw_features["embedding_similarity"] = _safe_float(
        pair.get("embedding_similarity"), default=0.0
    )
    # Merge pre-computed deterministic/NLP signal values (fuzzy_name, shared_address,
    # shared_phone) that the model expects but FeatureExtractor never produces on its own.
    for key, value in (pair.get("signal_overrides") or {}).items():
        raw_features[key] = _safe_float(value, default=0.0)
    feature_names = ORGANIZATION_FEATURES if entity_type == "organization" else SERVICE_FEATURES
    api_features = {
        name: _safe_float(raw_features.get(name), default=0.0) for name in feature_names
    }
    return {"pair_key": pair["pair_key"], "api_features": api_features}


def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: int,
) -> dict[str, Any] | None:
    """POST JSON payload and parse JSON response; return `None` on transport errors."""
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return parsed if isinstance(parsed, dict) else None
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _int_env(name: str, default: int) -> int:
    """Read integer environment variable with safe fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _safe_float(value: Any, *, default: float) -> float:
    """Convert unknown numeric-ish values to bounded float."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return parsed


def to_legacy_entity(*, row: dict[str, Any]) -> dict[str, Any]:
    """Pass the denormalized row through to the feature extractor with normalized fields."""
    return {
        "entity_id": _safe_str(row.get("entity_id")),
        "name": _safe_str(row.get("name")),
        "description": _safe_str(row.get("description")),
        "emails": _safe_list(row.get("emails")),
        "phones": _safe_list(row.get("phones")),
        "websites": _safe_list(row.get("websites")),
        "locations": _safe_object_list(row.get("locations")),
        "taxonomies": to_legacy_taxonomies(row.get("taxonomies")),
        "identifiers": _safe_object_list(row.get("identifiers")),
        "services_rollup": to_legacy_services(row.get("services_rollup")),
        "organization_name": _safe_str(row.get("organization_name")),
        "organization_id": _safe_str(row.get("organization_id")),
        "embedding_vector": row.get("embedding_vector") or [],
    }


def _safe_object_list(value: Any) -> list[dict[str, Any]]:
    """Normalize unknown value to list of object dictionaries."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _safe_list(value: Any) -> list[str]:
    """Normalize unknown value to list of strings."""
    if not isinstance(value, list):
        return []
    return [_safe_str(item) for item in value if item is not None]


def _safe_str(value: Any) -> str:
    """Normalize unknown value to stripped string."""
    return str(value).strip() if value is not None else ""
