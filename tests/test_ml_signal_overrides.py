"""Tests for ML signal override forwarding.

Covers the fix for the constant-ML-score bug: fuzzy_name, shared_address, and
shared_phone are pre-computed deterministic/NLP signals that must be merged into
the FeatureExtractor output before the ML feature payload is assembled.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from hsds_entity_resolution.core.ml_inference import (
    _build_feature_extractor,
    _extract_pair_features,
    score_pairs_with_model,
)
from hsds_entity_resolution.core.score_candidates import (
    PreMlPairRecord,
    _extract_signal_overrides,
)

# ---------------------------------------------------------------------------
# _extract_signal_overrides
# ---------------------------------------------------------------------------


def _make_record(
    *,
    det_reasons: list[dict[str, Any]],
    nlp_reasons: list[dict[str, Any]],
    nlp_score: float = 0.9,
) -> PreMlPairRecord:
    return PreMlPairRecord(
        candidate={
            "pair_key": "a::b",
            "entity_type": "service",
            "entity_a_id": "a",
            "entity_b_id": "b",
            "embedding_similarity": 0.85,
        },
        det_score=0.4,
        nlp_score=nlp_score,
        pre_ml_score=0.6,
        det_reasons=det_reasons,
        nlp_reasons=nlp_reasons,
    )


def test_extract_signal_overrides_all_present() -> None:
    """All three override features extracted when all reasons are present."""
    record = _make_record(
        det_reasons=[
            {"match_type": "shared_phone", "raw_contribution": 1.0, "weighted_contribution": 0.18},
            {
                "match_type": "shared_address",
                "raw_contribution": 0.5,
                "weighted_contribution": 0.06,
            },
            {"match_type": "shared_email", "raw_contribution": 1.0, "weighted_contribution": 0.22},
        ],
        nlp_reasons=[
            {
                "match_type": "name_similarity",
                "raw_contribution": 0.92,
                "weighted_contribution": 0.92,
            },
        ],
    )

    overrides = _extract_signal_overrides(record=record)

    assert overrides["shared_phone"] == pytest.approx(1.0)
    assert overrides["shared_address"] == pytest.approx(0.5)
    assert overrides["fuzzy_name"] == pytest.approx(0.92)
    # shared_email is not an ML feature override; must not leak through
    assert "shared_email" not in overrides


def test_extract_signal_overrides_no_det_reasons() -> None:
    """shared_address and shared_phone absent when det_reasons is empty."""
    record = _make_record(
        det_reasons=[],
        nlp_reasons=[
            {
                "match_type": "name_similarity",
                "raw_contribution": 0.87,
                "weighted_contribution": 0.87,
            },
        ],
    )

    overrides = _extract_signal_overrides(record=record)

    assert "shared_phone" not in overrides
    assert "shared_address" not in overrides
    assert overrides["fuzzy_name"] == pytest.approx(0.87)


def test_extract_signal_overrides_fuzzy_name_fallback_to_nlp_score() -> None:
    """fuzzy_name falls back to nlp_score when name_similarity reason is absent."""
    record = _make_record(
        det_reasons=[],
        nlp_reasons=[],  # no contributing name_similarity reason
        nlp_score=0.78,
    )

    overrides = _extract_signal_overrides(record=record)

    assert overrides["fuzzy_name"] == pytest.approx(0.78)


def test_extract_signal_overrides_nlp_reason_takes_priority_over_fallback() -> None:
    """When a name_similarity reason is present its raw_contribution wins over nlp_score."""
    record = _make_record(
        det_reasons=[],
        nlp_reasons=[
            {
                "match_type": "name_similarity",
                "raw_contribution": 0.95,
                "weighted_contribution": 0.95,
            },
        ],
        nlp_score=0.60,  # different value — reason should win
    )

    overrides = _extract_signal_overrides(record=record)

    assert overrides["fuzzy_name"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# _extract_pair_features — signal_overrides merged into raw_features
# ---------------------------------------------------------------------------


def _make_mock_extractor(raw_features: dict[str, float]) -> MagicMock:
    extractor = MagicMock()
    extractor.extract_features.return_value = dict(raw_features)
    return extractor


def test_extract_pair_features_merges_signal_overrides() -> None:
    """signal_overrides values appear in the assembled api_features payload."""
    extractor = _make_mock_extractor(
        {"name_token_sort": 0.88, "embedding_similarity": 0.0, "bigram_overlap": 0.5}
    )
    pair: dict[str, Any] = {
        "pair_key": "a::b",
        "embedding_similarity": 0.91,
        "entity_a": {},
        "entity_b": {},
        "signal_overrides": {
            "fuzzy_name": 0.93,
            "shared_address": 0.0,
            "shared_phone": 1.0,
        },
    }

    result = _extract_pair_features(pair=pair, extractor=extractor, entity_type="service")

    features = result["api_features"]
    assert features["fuzzy_name"] == pytest.approx(0.93)
    assert features["shared_phone"] == pytest.approx(1.0)
    assert features["shared_address"] == pytest.approx(0.0)
    assert features["embedding_similarity"] == pytest.approx(0.91)
    assert features["name_token_sort"] == pytest.approx(0.88)


def test_extract_pair_features_no_signal_overrides_defaults_to_zero() -> None:
    """When signal_overrides is absent the three features default to 0.0."""
    extractor = _make_mock_extractor({"name_token_sort": 0.7})
    pair: dict[str, Any] = {
        "pair_key": "a::b",
        "embedding_similarity": 0.82,
        "entity_a": {},
        "entity_b": {},
        # no signal_overrides key
    }

    result = _extract_pair_features(pair=pair, extractor=extractor, entity_type="service")

    features = result["api_features"]
    assert features["fuzzy_name"] == pytest.approx(0.0)
    assert features["shared_phone"] == pytest.approx(0.0)
    assert features["shared_address"] == pytest.approx(0.0)


def test_extract_pair_features_signal_overrides_do_not_overwrite_existing_keys_for_org() -> None:
    """Organization features list does not include fuzzy_name; override is silently unused."""
    extractor = _make_mock_extractor({"name_levenshtein": 0.9})
    pair: dict[str, Any] = {
        "pair_key": "a::b",
        "embedding_similarity": 0.8,
        "entity_a": {},
        "entity_b": {},
        "signal_overrides": {"fuzzy_name": 0.9, "shared_address": 1.0},
    }

    result = _extract_pair_features(pair=pair, extractor=extractor, entity_type="organization")

    features = result["api_features"]
    # fuzzy_name is not in ORGANIZATION_FEATURES so it will not appear
    assert "fuzzy_name" not in features
    # shared_address IS in ORGANIZATION_FEATURES
    assert features["shared_address"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _build_feature_extractor — taxonomy_embeddings forwarded
# ---------------------------------------------------------------------------


def test_build_feature_extractor_passes_taxonomy_embeddings(monkeypatch) -> None:
    """taxonomy_embeddings kwarg is forwarded to FeatureExtractor constructor."""
    captured: dict[str, Any] = {}

    class FakeExtractor:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "hsds_entity_resolution.core.feature_extractor.FeatureExtractor",
        FakeExtractor,
    )
    monkeypatch.setattr(
        "hsds_entity_resolution.core.ml_inference._resolve_tfidf_vectorizer_path",
        lambda entity_type: None,
    )

    embeddings = {"BD-1800": [0.1, 0.2, 0.3]}
    _build_feature_extractor(entity_type="service", taxonomy_embeddings=embeddings)

    assert captured["taxonomy_embeddings"] == embeddings


def test_build_feature_extractor_empty_dict_when_none(monkeypatch) -> None:
    """Passing None for taxonomy_embeddings results in an empty dict, not None."""
    captured: dict[str, Any] = {}

    class FakeExtractor:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "hsds_entity_resolution.core.feature_extractor.FeatureExtractor",
        FakeExtractor,
    )
    monkeypatch.setattr(
        "hsds_entity_resolution.core.ml_inference._resolve_tfidf_vectorizer_path",
        lambda entity_type: None,
    )

    _build_feature_extractor(entity_type="service", taxonomy_embeddings=None)

    assert captured["taxonomy_embeddings"] == {}


# ---------------------------------------------------------------------------
# score_pairs_with_model — taxonomy_embeddings plumbed through
# ---------------------------------------------------------------------------


def test_score_pairs_with_model_passes_taxonomy_embeddings_to_extractor(
    monkeypatch,
) -> None:
    """taxonomy_embeddings reaches FeatureExtractor when passed to score_pairs_with_model."""
    captured: dict[str, Any] = {}

    class FakeExtractor:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def extract_features(self, a: Any, b: Any) -> dict[str, float]:
            return {}

    monkeypatch.setenv("AI_UTILS_ENDPOINT", "http://fake-endpoint")
    monkeypatch.setattr(
        "hsds_entity_resolution.core.feature_extractor.FeatureExtractor",
        FakeExtractor,
    )
    monkeypatch.setattr(
        "hsds_entity_resolution.core.ml_inference._resolve_tfidf_vectorizer_path",
        lambda entity_type: None,
    )
    monkeypatch.setattr(
        "hsds_entity_resolution.core.ml_inference._post_json",
        lambda **_kwargs: {"results": [{"isDupe": True, "isDupeConfidence": 0.9}]},
    )

    embeddings = {"BD-1800": [0.1, 0.2]}
    pairs = [
        {
            "pair_key": "a::b",
            "embedding_similarity": 0.88,
            "entity_a": {"name": "Foo"},
            "entity_b": {"name": "Foo"},
            "signal_overrides": {"fuzzy_name": 0.9},
        }
    ]
    score_pairs_with_model(
        pairs=pairs,
        entity_type="service",
        taxonomy_embeddings=embeddings,
    )

    assert captured.get("taxonomy_embeddings") == embeddings
