"""Tests for the NLP scoring stage.

Tests specify correct HSDS entity resolution behavior for name-based fuzzy
matching, safeguard policies, and scoring gates.  Tier C tests lock in
intentional design decisions that must not silently change.
"""

from __future__ import annotations

import pytest

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.nlp.algorithms import (
    jaro_winkler_similarity,
    resolve_fuzzy_similarity,
    token_sort_ratio_similarity,
)
from hsds_entity_resolution.core.nlp.safeguards import (
    apply_nlp_safeguards,
    number_mismatch_safeguard,
)
from hsds_entity_resolution.core.nlp.scoring import compute_nlp_score
from hsds_entity_resolution.core.nlp.types import NlpSafeguardContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(
    *,
    fuzzy_threshold: float = 0.88,
    standalone_fuzzy_threshold: float = 0.94,
    number_mismatch_veto_enabled: bool = True,
    fuzzy_algorithm: str = "sequence_matcher",
) -> EntityResolutionRunConfig:
    """Build a config with NLP overrides for targeted testing."""
    base = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="test-team",
        scope_id="test-scope",
        entity_type="organization",
    )
    data = base.model_dump()
    data["scoring"]["nlp"]["fuzzy_threshold"] = fuzzy_threshold
    data["scoring"]["nlp"]["standalone_fuzzy_threshold"] = standalone_fuzzy_threshold
    data["scoring"]["nlp"]["number_mismatch_veto_enabled"] = number_mismatch_veto_enabled
    data["scoring"]["nlp"]["fuzzy_algorithm"] = fuzzy_algorithm
    return EntityResolutionRunConfig.model_validate(data)


def _ctx(
    left: str, right: str, cfg: EntityResolutionRunConfig | None = None
) -> NlpSafeguardContext:
    """Build an NlpSafeguardContext for safeguard unit tests."""
    return NlpSafeguardContext(
        left_name=left,
        right_name=right,
        config=cfg or _config(),
    )


def _nlp_score(
    left: str,
    right: str,
    deterministic_score: float = 0.5,
    cfg: EntityResolutionRunConfig | None = None,
) -> tuple[float, float]:
    """Compute (nlp_contribution, raw_fuzzy_similarity) for a name pair."""
    config = cfg or _config()
    return compute_nlp_score(
        left={"name": left},
        right={"name": right},
        config=config,
        deterministic_score=deterministic_score,
    )


# ---------------------------------------------------------------------------
# Number mismatch safeguard — unit tests on the policy function
# ---------------------------------------------------------------------------


def test_number_mismatch_veto_fires_when_both_names_have_different_digits() -> None:
    """Two org names with different numeric tokens must be vetoed.

    This prevents "Camp 1 Food Bank" from scoring as a duplicate of
    "Camp 2 Food Bank", which would represent a different program instance.
    """
    ctx = _ctx("Camp 1 Food Bank", "Camp 2 Food Bank")
    outcome = number_mismatch_safeguard(ctx)
    assert outcome.veto is True


def test_number_mismatch_does_not_fire_when_only_one_side_has_digits() -> None:
    """Veto must not fire if only one side contains digit runs.

    An org name with a number is not a mismatch against one without any
    number — the number may simply be absent from the other record.
    """
    ctx = _ctx("First Street Shelter", "Main Street Shelter No. 5")
    outcome = number_mismatch_safeguard(ctx)
    assert outcome.veto is False


def test_number_mismatch_does_not_fire_when_both_sides_have_same_digits() -> None:
    """Veto must not fire when both names contain the same digit runs.

    "24-Hour Services" and "24-Hour Crisis Line" both contain "24".  They
    share the same number, so no mismatch.
    """
    ctx = _ctx("24-Hour Services", "24-Hour Crisis Line")
    outcome = number_mismatch_safeguard(ctx)
    assert outcome.veto is False


def test_number_mismatch_does_not_fire_when_neither_name_has_digits() -> None:
    """Veto must not fire when neither name contains any digit run."""
    ctx = _ctx("North Shelter Services", "North Shelter Programs")
    outcome = number_mismatch_safeguard(ctx)
    assert outcome.veto is False


def test_number_mismatch_veto_disabled_via_config() -> None:
    """Veto must not fire when number_mismatch_veto_enabled=False in config."""
    ctx = _ctx("Camp 1 Food Bank", "Camp 2 Food Bank", _config(number_mismatch_veto_enabled=False))
    outcome = number_mismatch_safeguard(ctx)
    assert outcome.veto is False


# ---------------------------------------------------------------------------
# Number mismatch — design-contract (Tier C known limitation)
# ---------------------------------------------------------------------------


def test_hyphenated_token_digit_extraction_known_limitation() -> None:
    """Tier C: hyphenated tokens are split into separate digit runs.

    re.findall(r'\\d+', '1-800 Services') yields ['1', '800'], so
    '1-800 Services' and '800 Services' trigger a veto because their digit
    sets differ ({'1','800'} != {'800'}).

    This is a *known limitation*, not desired behavior.  The test documents
    the current behavior so that any future change to the extraction logic
    (e.g., treating '1-800' as a single token) will be caught explicitly.
    See todo.md for potential improvements.
    """
    ctx = _ctx("1-800 Services", "800 Services")
    outcome = number_mismatch_safeguard(ctx)
    # Current behavior: veto fires because '1-800' → {'1','800'} ≠ {'800'}
    assert outcome.veto is True, (
        "Known limitation: '1-800 Services' vs '800 Services' incorrectly triggers veto. "
        "See todo.md for proposed fix."
    )


# ---------------------------------------------------------------------------
# apply_nlp_safeguards — integration of safeguard chain
# ---------------------------------------------------------------------------


def test_veto_zeroes_out_similarity() -> None:
    """A veto safeguard must collapse any similarity value to 0.0."""
    adjusted = apply_nlp_safeguards(
        similarity=0.95,
        context=_ctx("Camp 1", "Camp 2"),
    )
    assert adjusted == 0.0


def test_no_safeguard_firing_returns_original_similarity() -> None:
    """When no safeguard fires, the input similarity must pass through unchanged."""
    adjusted = apply_nlp_safeguards(
        similarity=0.92,
        context=_ctx("North Shelter Services", "North Shelter Services"),
    )
    assert adjusted == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# compute_nlp_score — fuzzy_threshold gate
# ---------------------------------------------------------------------------


def test_nlp_score_zero_when_raw_similarity_below_fuzzy_threshold() -> None:
    """Names with raw similarity below fuzzy_threshold contribute nothing to NLP.

    Low-similarity names should not accumulate partial NLP signal — if the
    names are too different, the NLP section should contribute 0 so the pair
    must meet the duplicate threshold through other signals alone.
    """
    contribution, raw = _nlp_score("North Shelter Services", "Youth Services Alliance")
    assert raw < 0.88, f"Expected low raw similarity, got {raw}"
    assert contribution == 0.0


def test_nlp_score_positive_when_names_are_identical() -> None:
    """Identical names must produce a positive NLP contribution equal to 1.0."""
    contribution, raw = _nlp_score("North Shelter Services", "North Shelter Services")
    assert raw == pytest.approx(1.0)
    assert contribution == pytest.approx(1.0)


def test_nlp_score_positive_when_names_are_very_similar() -> None:
    """Near-identical names above the fuzzy threshold must contribute positively."""
    contribution, raw = _nlp_score("North Shelter Services", "North Shelter Service")
    assert raw > 0.88, f"Expected high raw similarity, got {raw}"
    assert contribution > 0.0


# ---------------------------------------------------------------------------
# compute_nlp_score — standalone_fuzzy_threshold gate
# ---------------------------------------------------------------------------


def test_nlp_score_zero_when_deterministic_zero_and_similarity_below_standalone() -> None:
    """With no deterministic signal, names below standalone_fuzzy_threshold score 0.

    When there is no overlapping contact information (emails, phones, domains),
    a very high name similarity bar is required to claim a match.  Names that
    are merely 'similar' are insufficient evidence on their own.
    """
    contribution, raw = _nlp_score(
        "North Shelter Services",
        "North Shelter Programs",
        deterministic_score=0.0,
    )
    # raw similarity for these names is well below 0.94
    assert raw < 0.94, f"Expected similarity below standalone threshold, got {raw}"
    assert contribution == 0.0


def test_nlp_score_positive_when_deterministic_zero_and_names_identical() -> None:
    """Identical names must still contribute positively even with no deterministic signal."""
    contribution, _ = _nlp_score(
        "North Shelter Services",
        "North Shelter Services",
        deterministic_score=0.0,
    )
    assert contribution > 0.0


def test_nlp_score_positive_with_nonzero_deterministic_and_similar_names() -> None:
    """With deterministic signal present, the standalone threshold does not apply.

    Once there is some contact-info overlap, even moderately similar names
    should contribute NLP signal (above the regular fuzzy_threshold).
    """
    cfg = _config(fuzzy_threshold=0.80, standalone_fuzzy_threshold=0.94)
    contribution, raw = _nlp_score(
        "North Shelter Services",
        "North Shelter Services Inc",
        deterministic_score=0.4,
        cfg=cfg,
    )
    assert raw >= 0.80, f"Expected raw similarity >= fuzzy_threshold, got {raw}"
    assert contribution > 0.0


# ---------------------------------------------------------------------------
# Algorithm routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["sequence_matcher", "token_sort_ratio", "jaro_winkler"])
def test_resolve_fuzzy_similarity_returns_value_in_unit_interval(algorithm: str) -> None:
    """All supported fuzzy algorithms must return a value in [0.0, 1.0]."""
    result = resolve_fuzzy_similarity(
        left_name="North Shelter Services",
        right_name="North Shelter Service",
        algorithm=algorithm,
        strict_validation_mode=True,
    )
    assert 0.0 <= result <= 1.0


def test_unknown_algorithm_raises_in_strict_mode() -> None:
    """An unsupported algorithm name must raise ValueError in strict mode."""
    with pytest.raises(ValueError, match="Unsupported fuzzy_algorithm"):
        resolve_fuzzy_similarity(
            left_name="a",
            right_name="b",
            algorithm="totally_made_up",
            strict_validation_mode=True,
        )


def test_unknown_algorithm_falls_back_to_sequence_matcher_in_non_strict_mode() -> None:
    """An unsupported algorithm name must fall back to sequence_matcher in non-strict mode."""
    result = resolve_fuzzy_similarity(
        left_name="North Shelter",
        right_name="North Shelter",
        algorithm="nonexistent",
        strict_validation_mode=False,
    )
    assert result == pytest.approx(1.0)


def test_token_sort_ratio_is_order_invariant() -> None:
    """token_sort_ratio must produce the same result regardless of word order."""
    similarity_1 = token_sort_ratio_similarity(
        left_name="Community Health Center North",
        right_name="North Community Health Center",
    )
    similarity_2 = token_sort_ratio_similarity(
        left_name="North Community Health Center",
        right_name="Community Health Center North",
    )
    assert similarity_1 == pytest.approx(similarity_2)


def test_token_sort_ratio_identical_names_score_one() -> None:
    """Identical names must score 1.0 with token_sort_ratio."""
    result = token_sort_ratio_similarity(
        left_name="North Shelter Services",
        right_name="North Shelter Services",
    )
    assert result == pytest.approx(1.0)


def test_jaro_winkler_identical_names_score_one() -> None:
    """Identical names must score 1.0 with jaro_winkler."""
    result = jaro_winkler_similarity(
        left_name="North Shelter Services",
        right_name="North Shelter Services",
    )
    assert result == pytest.approx(1.0)


def test_jaro_winkler_empty_names_score_zero() -> None:
    """An empty name on either side must score 0.0 with jaro_winkler."""
    assert jaro_winkler_similarity(left_name="", right_name="North Shelter") == 0.0
    assert jaro_winkler_similarity(left_name="North Shelter", right_name="") == 0.0


def test_jaro_winkler_completely_different_names_score_low() -> None:
    """Completely different names must produce a low jaro_winkler similarity."""
    result = jaro_winkler_similarity(
        left_name="aaa",
        right_name="zzz",
    )
    assert result < 0.5
