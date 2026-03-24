"""Tests for dataset cutting filter and stratification logic."""

from __future__ import annotations

from datetime import datetime

from scripts.cut_training_dataset import ResolvedPair, _apply_filters


def _pair(
    *,
    pair_id: str,
    label: str,
    band: str,
    source_run_id: str = "run-latest",
) -> ResolvedPair:
    """Create one resolved pair fixture with typed features present."""
    return ResolvedPair(
        training_pair_id=pair_id,
        entity_type="service",
        pair_canonical_key=f"{pair_id}|peer",
        baseline_confidence_score=0.7,
        baseline_predicted_duplicate=band == "duplicate",
        resolved_label=label,
        label_source="SINGLE_REVIEW",
        review_id=f"review-{pair_id}",
        review_count=1,
        distinct_decisions=1,
        min_confidence="HIGH",
        latest_reviewed_at=datetime(2026, 3, 24, 12, 0, 0),
        team_id="IL211",
        source_run_id=source_run_id,
        policy_version="hsds-er-v1",
        feature_schema_version="ml-features-v1",
        baseline_score_band=band,
        pair_created_at=datetime(2026, 3, 24, 12, 0, 0),
    )


def test_apply_filters_balances_negatives_by_band() -> None:
    """Default filter path should keep all positives and balanced reviewed negatives."""
    pairs = [
        _pair(pair_id="pos-1", label="TRUE_DUPLICATE", band="duplicate"),
        _pair(pair_id="pos-2", label="TRUE_DUPLICATE", band="maybe"),
        _pair(pair_id="neg-dup-1", label="FALSE_POSITIVE", band="duplicate"),
        _pair(pair_id="neg-dup-2", label="FALSE_POSITIVE", band="duplicate"),
        _pair(pair_id="neg-maybe-1", label="FALSE_POSITIVE", band="maybe"),
        _pair(pair_id="neg-maybe-2", label="FALSE_POSITIVE", band="maybe"),
        _pair(pair_id="neg-low-1", label="FALSE_POSITIVE", band="below_maybe"),
    ]

    plan = _apply_filters(
        pairs,
        team_id="IL211",
        entity_type="service",
        policy_version="hsds-er-v1",
        run_selection="all",
        source_run_id=None,
        feature_schema_version="ml-features-v1",
        min_confidence="HIGH",
        include_unsure=False,
        stratify_negatives=True,
        duplicate_negative_ratio=0.4,
        maybe_negative_ratio=0.4,
        below_maybe_negative_ratio=0.2,
    )

    included_ids = {pair.training_pair_id for pair in plan.included}
    assert {"pos-1", "pos-2"}.issubset(included_ids)
    assert {"neg-dup-1", "neg-maybe-1", "neg-low-1"}.issubset(included_ids)
    assert len([pair for pair in plan.included if pair.resolved_label == "FALSE_POSITIVE"]) == 5


def test_apply_filters_limits_to_latest_run_by_default() -> None:
    """Latest-run mode should exclude reviewed pairs from older source runs."""
    older = _pair(
        pair_id="old-1",
        label="TRUE_DUPLICATE",
        band="duplicate",
        source_run_id="run-old",
    )
    older = ResolvedPair(**{**older.__dict__, "pair_created_at": datetime(2026, 3, 20, 12, 0, 0)})
    latest = _pair(
        pair_id="new-1",
        label="TRUE_DUPLICATE",
        band="duplicate",
        source_run_id="run-new",
    )
    latest = ResolvedPair(**{**latest.__dict__, "pair_created_at": datetime(2026, 3, 24, 12, 0, 0)})

    plan = _apply_filters(
        [older, latest],
        team_id="IL211",
        entity_type="service",
        policy_version="hsds-er-v1",
        run_selection="latest",
        source_run_id=None,
        feature_schema_version="ml-features-v1",
        min_confidence="HIGH",
        include_unsure=False,
        stratify_negatives=False,
        duplicate_negative_ratio=0.4,
        maybe_negative_ratio=0.4,
        below_maybe_negative_ratio=0.2,
    )

    assert [pair.training_pair_id for pair in plan.included] == ["new-1"]
    assert [pair.training_pair_id for pair in plan.excluded_source_run] == ["old-1"]
