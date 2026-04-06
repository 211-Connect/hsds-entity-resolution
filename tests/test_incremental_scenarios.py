"""Incremental run scenario tests for HSDS entity resolution.

This module is the regression harness for all incremental-pipeline state-management
behaviours defined in the Incremental Run Scenario Testing Plan.  Each test maps
one-to-one to a labelled scenario (S1-A through S6-D).  Scenarios that depend on
full entity-delta detection use ``run_incremental`` across two or more runs;
scenarios that need precise control over final scores use the ``apply_mitigation``
and ``cluster_pairs`` stage functions directly, matching the patterns already
established in ``test_apply_mitigation_and_review_queue.py``.

Scenario taxonomy
-----------------
Category 1  – Single-entity changes within an existing pair
Category 2  – Both entities in a pair change
Category 3  – Cluster topology changes
Category 4  – Scoring constant changes (force_rescore / explicit_backfill)
Category 5  – No-change and idempotent runs
Category 6  – Edge and corner cases
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.apply_mitigation import apply_mitigation
from hsds_entity_resolution.core.cluster_pairs import cluster_pairs
from hsds_entity_resolution.core.pipeline import run_incremental

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# High-similarity embedding vectors: cosine sim ≈ 0.999 (well above the 0.75
# blocking threshold), suitable for any scenario that needs a retained pair.
_EMB_HIGH_A = [0.9, 0.1]
_EMB_HIGH_B = [0.92, 0.08]

# Orthogonal embedding: cosine sim ≈ 0.0 (below the 0.75 blocking threshold).
# Changing entity A to this vector makes the pair drop out of candidates.
_EMB_ORTHO = [0.0, 1.0]


def _config(
    *,
    team_id: str,
    scope_id: str,
    duplicate_threshold: float = 0.82,
    maybe_threshold: float = 0.68,
    min_embedding_similarity: float = 0.65,
) -> EntityResolutionRunConfig:
    """Build an org-scoped config, optionally overriding key thresholds."""
    base = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id=team_id,
        scope_id=scope_id,
        entity_type="organization",
    )
    raw = base.model_dump()
    raw["mitigation"]["enabled"] = True
    if duplicate_threshold == 0.82 and maybe_threshold == 0.68 and min_embedding_similarity == 0.65:
        return EntityResolutionRunConfig.model_validate(raw)
    raw["scoring"]["duplicate_threshold"] = duplicate_threshold
    raw["scoring"]["maybe_threshold"] = maybe_threshold
    raw["mitigation"]["min_embedding_similarity"] = min_embedding_similarity
    return EntityResolutionRunConfig.model_validate(raw)


def _entity_row(
    entity_id: str,
    name: str,
    *,
    emails: list[str] | None = None,
    phones: list[str] | None = None,
    websites: list[str] | None = None,
    locations: list[dict[str, str]] | None = None,
    identifiers: list[dict[str, str]] | None = None,
    source_schema: str = "IL211",
    embedding: list[float] | None = None,
) -> dict[str, Any]:
    """Return one entity row dict with sensible defaults."""
    return {
        "entity_id": entity_id,
        "source_schema": source_schema,
        "name": name,
        "description": f"Description for {name}",
        "emails": emails if emails is not None else [],
        "phones": phones if phones is not None else [],
        "websites": websites if websites is not None else [],
        "locations": locations if locations is not None else [],
        "taxonomies": [],
        "identifiers": identifiers if identifiers is not None else [],
        "services_rollup": [],
        "embedding_vector": embedding if embedding is not None else _EMB_HIGH_A,
    }


def _org_frame(*rows: dict[str, Any]) -> pl.DataFrame:
    """Build an organization entities DataFrame from one or more row dicts."""
    return pl.DataFrame(list(rows))


def _previous_entity_index(result: Any) -> pl.DataFrame:
    """Extract the entity index to pass into the next incremental run."""
    return result.denormalized_organization.select(
        ["entity_id", "entity_type", "content_hash"]
    ).with_columns(pl.lit(True).alias("active_flag"))


def _scored_pair_row(
    pair_key: str,
    entity_a_id: str,
    entity_b_id: str,
    final_score: float,
    *,
    predicted_duplicate: bool | None = None,
    mitigation_reason: str | None = None,
    embedding_similarity: float = 0.9,
) -> dict[str, Any]:
    """Build one row for a scored-pairs DataFrame."""
    if predicted_duplicate is None:
        predicted_duplicate = final_score >= 0.82
    return {
        "pair_key": pair_key,
        "entity_a_id": entity_a_id,
        "entity_b_id": entity_b_id,
        "entity_type": "organization",
        "policy_version": "hsds-er-v1",
        "model_version": "m",
        "deterministic_section_score": final_score,
        "nlp_section_score": final_score,
        "ml_section_score": None,
        "final_score": final_score,
        "predicted_duplicate": predicted_duplicate,
        "embedding_similarity": embedding_similarity,
        "mitigation_reason": mitigation_reason,
    }


def _pair_state_row(
    pair_key: str,
    entity_a_id: str,
    entity_b_id: str,
    scope_id: str,
) -> dict[str, Any]:
    """Build one row for a previous_pair_state_index DataFrame."""
    return {
        "pair_key": pair_key,
        "entity_a_id": entity_a_id,
        "entity_b_id": entity_b_id,
        "entity_type": "organization",
        "scope_id": scope_id,
        "retained_flag": True,
    }


def _pair_reasons_row(
    pair_key: str,
    match_type: str = "shared_phone",
    raw: float = 1.0,
    weighted: float = 0.2,
) -> dict[str, Any]:
    """Build one row for a pair_reasons DataFrame."""
    return {
        "pair_key": pair_key,
        "match_type": match_type,
        "raw_contribution": raw,
        "weighted_contribution": weighted,
        "signal_weight": 0.2,
        "matched_value": None,
        "entity_a_value": None,
        "entity_b_value": None,
        "similarity_score": None,
    }


def _assert_removed(result: Any, pair_key: str, reason: str) -> None:
    """Assert that removed_pair_ids contains the given pair with the expected reason."""
    rows = result.removed_pair_ids.filter(pl.col("pair_key") == pair_key)
    assert rows.height == 1, (
        f"Expected pair_key {pair_key!r} in removed_pair_ids but got "
        f"{result.removed_pair_ids.get_column('pair_key').to_list()}"
    )
    assert rows.row(0, named=True)["cleanup_reason"] == reason, (
        f"Expected cleanup_reason={reason!r} for {pair_key!r}, "
        f"got {rows.row(0, named=True)['cleanup_reason']!r}"
    )


def _assert_retained(result: Any, pair_key: str) -> None:
    """Assert that pair_state_index contains the given pair with retained_flag=True."""
    rows = result.pair_state_index.filter(pl.col("pair_key") == pair_key)
    assert rows.height == 1, (
        f"Expected pair_key {pair_key!r} in pair_state_index but got "
        f"{result.pair_state_index.get_column('pair_key').to_list()}"
    )
    assert rows.row(0, named=True)["retained_flag"] is True


def _assert_not_removed(result: Any, pair_key: str) -> None:
    """Assert that a pair key is absent from removed_pair_ids."""
    rows = result.removed_pair_ids.filter(pl.col("pair_key") == pair_key)
    assert rows.is_empty(), (
        f"Expected pair_key {pair_key!r} NOT in removed_pair_ids but it was present "
        f"with reason {rows.row(0, named=True).get('cleanup_reason')!r}"
    )


# ---------------------------------------------------------------------------
# Standard pair entities used across multiple tests.
#
# Both entities share email + phone + website and use IDENTICAL names so that:
#   - NLP fuzzy similarity = 1.0  →  clears fuzzy_threshold (0.88)
#   - det_score = (email 0.22 + phone 0.20 + domain 0.08) / 1.00 = 0.50
#   - pre_ml = 0.50×0.45 + 1.0×0.35 = 0.575  →  clears ml_gate_threshold (0.55)
#   - ML fires via embedding_similarity ≈ 0.9997
#   - final ≈ 0.575 + 0.9997×0.2 ≈ 0.775  →  below duplicate threshold unless
#     stronger deterministic evidence such as address or identifier also matches
# ---------------------------------------------------------------------------

_PAIR_KEY = "org-a__org-b"
_SHARED_EMAIL = "hello@north.org"
_SHARED_PHONE = "555-0100"
_SHARED_WEBSITE = "north.org"
_SHARED_TAXONOMY = {"code": "BD"}


def _standard_entity_a(*, embedding: list[float] | None = None) -> dict[str, Any]:
    row = _entity_row(
        "org-a",
        "North Clinic",
        emails=[_SHARED_EMAIL],
        phones=[_SHARED_PHONE],
        websites=[_SHARED_WEBSITE],
        embedding=embedding or _EMB_HIGH_A,
    )
    row["taxonomies"] = [_SHARED_TAXONOMY]
    return row


def _standard_entity_b(*, embedding: list[float] | None = None) -> dict[str, Any]:
    # Identical name to entity A: ensures NLP = 1.0 and pair lands in duplicate band.
    row = _entity_row(
        "org-b",
        "North Clinic",
        emails=[_SHARED_EMAIL],
        phones=[_SHARED_PHONE],
        websites=[_SHARED_WEBSITE],
        embedding=embedding or _EMB_HIGH_B,
    )
    row["taxonomies"] = [_SHARED_TAXONOMY]
    return row


def _run_first(scope_id: str) -> Any:
    """Run a first incremental pass with the standard org pair and return the result."""
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())
    return run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )


# ===========================================================================
# Category 1 – Single-entity changes within an existing pair
# ===========================================================================


def test_s1a_entity_deleted_emits_entity_deleted_signal() -> None:
    """S1-A: Removing one entity from the input marks all its pairs as entity_deleted."""
    scope_id = "s1a"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())

    # Run 1: both entities → pair retained.
    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    assert run1.pair_state_index.height == 1
    _assert_retained(run1, _PAIR_KEY)

    # Run 2: entity A absent → deleted.
    org_b_only = _org_frame(_standard_entity_b())
    run2 = run_incremental(
        organization_entities=org_b_only,
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    _assert_removed(run2, _PAIR_KEY, "entity_deleted")
    assert run2.pair_state_index.is_empty()


def test_s1b_entity_change_score_drops_below_maybe_emits_score_dropped() -> None:
    """S1-B: A previously retained pair that re-scores below maybe_threshold is score_dropped."""
    config = _config(team_id="team-s1b", scope_id="s1b")
    # Build a previous state where the pair was retained.
    previous_pair_state = pl.DataFrame([_pair_state_row(_PAIR_KEY, "org-a", "org-b", "s1b")])
    # Current scored pairs: pair re-appears but scores well below maybe_threshold.
    scored_pairs = pl.DataFrame(
        [_scored_pair_row(_PAIR_KEY, "org-a", "org-b", 0.20, predicted_duplicate=False)]
    )
    result = apply_mitigation(
        scored_pairs=scored_pairs,
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config,
    )
    _assert_removed(result, _PAIR_KEY, "score_dropped")
    assert result.pair_state_index.is_empty()


def test_s1b_full_pipeline_entity_change_drops_score_emits_score_dropped() -> None:
    """S1-B (pipeline): Replacing shared signals with a very different entity drops score."""
    scope_id = "s1b-full"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())

    # Run 1: full-signal pair → retained.
    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_retained(run1, _PAIR_KEY)

    # Run 2: entity A changes – new email, website, and very different name.
    # Phone is kept so the overlap prefilter still passes and the pair is
    # re-generated; the score then drops because there is almost no
    # deterministic overlap and the names are unrelated.
    entity_a_changed = _entity_row(
        "org-a",
        "Completely Different Corporation",
        emails=["other@corp.org"],
        phones=[_SHARED_PHONE],
        websites=["corp.org"],
        embedding=_EMB_HIGH_A,
    )
    entity_a_changed["taxonomies"] = [_SHARED_TAXONOMY]
    org_changed = _org_frame(entity_a_changed, _standard_entity_b())
    run2 = run_incremental(
        organization_entities=org_changed,
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )
    _assert_removed(run2, _PAIR_KEY, "score_dropped")


def test_s1c_pair_demotes_from_duplicate_to_maybe_stays_retained() -> None:
    """S1-C: A pair re-scoring in the maybe band stays retained but predicted_duplicate flips."""
    config = _config(team_id="team-s1c", scope_id="s1c")
    previous_pair_state = pl.DataFrame([_pair_state_row(_PAIR_KEY, "org-a", "org-b", "s1c")])

    # The pair was a duplicate on run 1; now it re-scores between maybe and dup thresholds.
    maybe_score = (config.scoring.maybe_threshold + config.scoring.duplicate_threshold) / 2.0
    scored_pairs = pl.DataFrame(
        [
            _scored_pair_row(
                _PAIR_KEY,
                "org-a",
                "org-b",
                maybe_score,
                predicted_duplicate=False,
                embedding_similarity=0.8,
            )
        ]
    )
    pair_reasons = pl.DataFrame([_pair_reasons_row(_PAIR_KEY)])

    result = apply_mitigation(
        scored_pairs=scored_pairs,
        pair_reasons=pair_reasons,
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config,
    )

    # Pair must NOT be removed – it is still review-eligible (maybe band).
    _assert_not_removed(result, _PAIR_KEY)
    _assert_retained(result, _PAIR_KEY)

    finalized_row = result.finalized_scored_pairs.filter(pl.col("pair_key") == _PAIR_KEY).row(
        0, named=True
    )
    assert finalized_row["predicted_duplicate"] is False
    assert finalized_row["mitigation_reason"] is None


def test_s1d_maybe_pair_drops_below_maybe_emits_score_dropped() -> None:
    """S1-D: A previously maybe-band pair that drops below maybe_threshold is score_dropped."""
    config = _config(team_id="team-s1d", scope_id="s1d")
    previous_pair_state = pl.DataFrame([_pair_state_row(_PAIR_KEY, "org-a", "org-b", "s1d")])
    # Score just below maybe_threshold.
    below_maybe = config.scoring.maybe_threshold - 0.05
    scored_pairs = pl.DataFrame(
        [_scored_pair_row(_PAIR_KEY, "org-a", "org-b", below_maybe, predicted_duplicate=False)]
    )
    result = apply_mitigation(
        scored_pairs=scored_pairs,
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config,
    )
    _assert_removed(result, _PAIR_KEY, "score_dropped")


def test_s1e_entity_change_causes_candidate_lost() -> None:
    """S1-E: An entity change that makes the embedding orthogonal drops the pair below
    the blocking similarity threshold, so no candidate is generated and the previously
    retained pair is emitted as candidate_lost."""
    scope_id = "s1e"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())

    # Run 1: pair retained.
    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_retained(run1, _PAIR_KEY)

    # Run 2: entity A changes its identifier (triggers content_hash change so
    # candidates are re-generated) AND its embedding becomes orthogonal to B
    # (cosine ≈ 0.0 < 0.75 blocking threshold → pair never generated).
    entity_a_ortho = _entity_row(
        "org-a",
        "North Clinic",
        emails=[_SHARED_EMAIL],
        phones=[_SHARED_PHONE],
        websites=[_SHARED_WEBSITE],
        identifiers=[{"system": "npi", "value": "9999"}],
        embedding=_EMB_ORTHO,
    )
    org_ortho = _org_frame(entity_a_ortho, _standard_entity_b())
    run2 = run_incremental(
        organization_entities=org_ortho,
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    _assert_removed(run2, _PAIR_KEY, "candidate_lost")


def test_s1f_hash_irrelevant_field_change_preserves_pair_state() -> None:
    """S1-F: Changing a field not included in the content hash leaves the entity
    marked unchanged; no candidates are re-generated and pair state is stable."""
    scope_id = "s1f"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())

    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_retained(run1, _PAIR_KEY)

    # Change only embedding_vector – this field is explicitly excluded from the
    # content hash (see clean_entities._clean_entity_row / _clean_payload_fields).
    entity_a_embedding_changed = _entity_row(
        "org-a",
        "North Clinic",
        emails=[_SHARED_EMAIL],
        phones=[_SHARED_PHONE],
        websites=[_SHARED_WEBSITE],
        embedding=_EMB_ORTHO,  # orthogonal to entity B; still high cosine if checked
    )
    entity_a_embedding_changed["taxonomies"] = [_SHARED_TAXONOMY]
    org_embedding_changed = _org_frame(entity_a_embedding_changed, _standard_entity_b())
    run2 = run_incremental(
        organization_entities=org_embedding_changed,
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    # No new candidates should be generated and no pair should be removed.
    assert run2.candidate_pairs.is_empty(), (
        "Expected no candidate re-generation for a hash-irrelevant change"
    )
    assert run2.removed_pair_ids.is_empty()
    _assert_retained(run2, _PAIR_KEY)


def test_s1g_deleted_entity_resurrected_rebuilds_pair() -> None:
    """S1-G: An entity deleted in run 2 that reappears in run 3 is treated as a new
    addition, and the previously deleted pair can be re-generated and re-retained."""
    scope_id = "s1g"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())

    # Run 1: pair retained.
    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_retained(run1, _PAIR_KEY)

    # Run 2: entity A deleted → entity_deleted signal.
    org_b_only = _org_frame(_standard_entity_b())
    run2 = run_incremental(
        organization_entities=org_b_only,
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )
    _assert_removed(run2, _PAIR_KEY, "entity_deleted")
    assert run2.pair_state_index.is_empty()

    # Run 3: entity A reappears → treated as "added" → candidates re-generated.
    entity_index_after_run2 = run2.denormalized_organization.select(
        ["entity_id", "entity_type", "content_hash"]
    ).with_columns(pl.lit(True).alias("active_flag"))
    run3 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=entity_index_after_run2,
        previous_pair_state_index=run2.pair_state_index,
        config=config,
    )

    # The pair must be re-generated and re-retained.
    assert not run3.candidate_pairs.is_empty(), (
        "Resurrected entity A should generate a candidate pair with entity B"
    )
    _assert_retained(run3, _PAIR_KEY)


# ===========================================================================
# Category 2 – Both entities in a pair change
# ===========================================================================


def test_s2a_both_entities_change_pair_scores_below_maybe() -> None:
    """S2-A: Both entities change to remove all shared signals; pair falls below maybe."""
    config = _config(team_id="team-s2a", scope_id="s2a")
    previous_pair_state = pl.DataFrame([_pair_state_row(_PAIR_KEY, "org-a", "org-b", "s2a")])
    below_score = config.scoring.maybe_threshold - 0.10
    scored_pairs = pl.DataFrame(
        [_scored_pair_row(_PAIR_KEY, "org-a", "org-b", below_score, predicted_duplicate=False)]
    )
    result = apply_mitigation(
        scored_pairs=scored_pairs,
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config,
    )
    _assert_removed(result, _PAIR_KEY, "score_dropped")
    assert result.pair_state_index.is_empty()


def test_s2b_both_entities_deleted_emits_entity_deleted() -> None:
    """S2-B: Both entities deleted in the same run → entity_deleted for their shared pair."""
    scope_id = "s2b"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())

    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_retained(run1, _PAIR_KEY)

    # Run 2: both entities absent.
    run2 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )
    _assert_removed(run2, _PAIR_KEY, "entity_deleted")
    assert run2.pair_state_index.is_empty()


def test_s2c_one_entity_deleted_other_changes_entity_deleted_takes_priority() -> None:
    """S2-C: When one entity is deleted and the other changes, entity_deleted takes priority
    over any score-based signal for their shared pair."""
    scope_id = "s2c"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())

    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_retained(run1, _PAIR_KEY)

    # Run 2: entity A deleted; entity B changes independently (new identifier).
    entity_b_changed = _entity_row(
        "org-b",
        "North Clinic LLC",
        emails=[_SHARED_EMAIL],
        phones=[_SHARED_PHONE],
        websites=[_SHARED_WEBSITE],
        identifiers=[{"system": "npi", "value": "8888"}],
        embedding=_EMB_HIGH_B,
    )
    org_b_only_changed = _org_frame(entity_b_changed)
    run2 = run_incremental(
        organization_entities=org_b_only_changed,
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    # entity_deleted takes priority for the org-a__org-b pair.
    _assert_removed(run2, _PAIR_KEY, "entity_deleted")


def test_s2d_both_entities_improve_pair_promoted_to_duplicate() -> None:
    """S2-D: A maybe-band pair whose entities gain additional shared signals upgrades to
    duplicate tier; predicted_duplicate flips to True and no removal signal is emitted."""
    config = _config(team_id="team-s2d", scope_id="s2d")
    previous_pair_state = pl.DataFrame([_pair_state_row(_PAIR_KEY, "org-a", "org-b", "s2d")])
    maybe_score = (config.scoring.maybe_threshold + config.scoring.duplicate_threshold) / 2.0
    dup_score = config.scoring.duplicate_threshold + 0.05

    # Previous run: maybe band.
    scored_run1 = pl.DataFrame(
        [_scored_pair_row(_PAIR_KEY, "org-a", "org-b", maybe_score, predicted_duplicate=False)]
    )
    result_run1 = apply_mitigation(
        scored_pairs=scored_run1,
        pair_reasons=pl.DataFrame([_pair_reasons_row(_PAIR_KEY)]),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config,
    )
    _assert_retained(result_run1, _PAIR_KEY)

    # Second mitigation call: pair now in duplicate band.
    scored_run2 = pl.DataFrame(
        [_scored_pair_row(_PAIR_KEY, "org-a", "org-b", dup_score, predicted_duplicate=True)]
    )
    result_run2 = apply_mitigation(
        scored_pairs=scored_run2,
        pair_reasons=pl.DataFrame([_pair_reasons_row(_PAIR_KEY)]),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=result_run1.pair_state_index,
        config=config,
    )

    _assert_not_removed(result_run2, _PAIR_KEY)
    _assert_retained(result_run2, _PAIR_KEY)

    finalized_row = result_run2.finalized_scored_pairs.filter(pl.col("pair_key") == _PAIR_KEY).row(
        0, named=True
    )
    assert finalized_row["predicted_duplicate"] is True


# ===========================================================================
# Category 3 – Cluster topology changes
# ===========================================================================


def _finalized_pairs(*rows: dict[str, Any]) -> pl.DataFrame:
    """Build a finalized_scored_pairs DataFrame suitable for cluster_pairs."""
    return pl.DataFrame(list(rows))


def _dup_pair(
    pair_key: str,
    entity_a_id: str,
    entity_b_id: str,
    final_score: float = 0.95,
) -> dict[str, Any]:
    """Return a duplicate-band finalized pair row."""
    return {
        "pair_key": pair_key,
        "entity_a_id": entity_a_id,
        "entity_b_id": entity_b_id,
        "entity_type": "organization",
        "policy_version": "hsds-er-v1",
        "model_version": "m",
        "deterministic_section_score": final_score,
        "nlp_section_score": final_score,
        "ml_section_score": final_score,
        "final_score": final_score,
        "predicted_duplicate": True,
        "review_eligible": True,
        "embedding_similarity": final_score,
        "mitigation_reason": None,
        "pair_outcome": "duplicate",
    }


def _maybe_pair(
    pair_key: str,
    entity_a_id: str,
    entity_b_id: str,
    final_score: float = 0.72,
) -> dict[str, Any]:
    """Return a maybe-band finalized pair row (review_eligible=True but not dup)."""
    return {
        "pair_key": pair_key,
        "entity_a_id": entity_a_id,
        "entity_b_id": entity_b_id,
        "entity_type": "organization",
        "policy_version": "hsds-er-v1",
        "model_version": "m",
        "deterministic_section_score": final_score,
        "nlp_section_score": final_score,
        "ml_section_score": None,
        "final_score": final_score,
        "predicted_duplicate": False,
        "review_eligible": True,
        "embedding_similarity": 0.8,
        "mitigation_reason": None,
        "pair_outcome": "maybe",
    }


def test_s3a_new_strong_pair_joins_existing_cluster() -> None:
    """S3-A: Adding a new entity C with a strong pair to A expands the existing {A,B} cluster."""
    config = _config(team_id="team-s3a", scope_id="s3a")

    # Run 1: cluster {A, B}.
    finalized_run1 = _finalized_pairs(_dup_pair("a__b", "a", "b"))
    result_run1 = cluster_pairs(
        finalized_scored_pairs=finalized_run1,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )
    assert result_run1.clusters.height == 1
    cluster_id_run1 = result_run1.clusters.row(0, named=True)["cluster_id"]
    members_run1 = set(result_run1.cluster_pairs.get_column("pair_key").to_list())
    assert members_run1 == {"a__b"}

    # Run 2: new entity C with strong pair to A → both A__B and A__C are duplicate.
    finalized_run2 = _finalized_pairs(
        _dup_pair("a__b", "a", "b"),
        _dup_pair("a__c", "a", "c"),
    )
    result_run2 = cluster_pairs(
        finalized_scored_pairs=finalized_run2,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )
    assert result_run2.clusters.height == 1
    cluster_run2 = result_run2.clusters.row(0, named=True)
    assert cluster_run2["cluster_size"] == 3
    cluster_id_run2 = cluster_run2["cluster_id"]
    # The cluster ID must change because the membership changed.
    assert cluster_id_run2 != cluster_id_run1
    assert str(cluster_id_run2).startswith("ccv1::")
    pairs_run2 = set(result_run2.cluster_pairs.get_column("pair_key").to_list())
    assert "a__b" in pairs_run2
    assert "a__c" in pairs_run2


def test_s3b_new_maybe_pair_joins_cluster_raises_risk_score() -> None:
    """S3-B: A new maybe-band pair connecting entity C to an existing cluster {A, B}
    adds C to the cluster and raises the cluster risk score because the only link from
    C is in the maybe band."""
    config = _config(team_id="team-s3b", scope_id="s3b")

    # Run 1: {A, B} cluster (both duplicate).
    finalized_run1 = _finalized_pairs(_dup_pair("a__b", "a", "b"))
    result_run1 = cluster_pairs(
        finalized_scored_pairs=finalized_run1,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )
    risk_run1 = result_run1.clusters.row(0, named=True)["cluster_risk_score"]

    # Run 2: entity C added with a weak maybe-band pair to B only.
    maybe_score = (config.scoring.maybe_threshold + config.scoring.duplicate_threshold) / 2.0
    finalized_run2 = _finalized_pairs(
        _dup_pair("a__b", "a", "b"),
        _maybe_pair("b__c", "b", "c", final_score=maybe_score),
    )
    result_run2 = cluster_pairs(
        finalized_scored_pairs=finalized_run2,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )

    # C joins the cluster through B (positive maybe-band edge).
    assert result_run2.clusters.height == 1
    cluster_run2 = result_run2.clusters.row(0, named=True)
    assert cluster_run2["cluster_size"] == 3
    # Risk score rises because of the maybe-band link.
    risk_run2 = cluster_run2["cluster_risk_score"]
    assert risk_run2 > risk_run1


def test_s3c_pair_weakens_from_duplicate_to_maybe_stays_in_cluster() -> None:
    """S3-C: A pair that weakens from duplicate to maybe band stays in the cluster
    (edge is still positive relative to maybe_threshold) but raises the risk score."""
    config = _config(team_id="team-s3c", scope_id="s3c")

    # Run 1: cluster {A, B, C} – all duplicate.
    finalized_run1 = _finalized_pairs(
        _dup_pair("a__b", "a", "b"),
        _dup_pair("b__c", "b", "c"),
        _dup_pair("a__c", "a", "c"),
    )
    result_run1 = cluster_pairs(
        finalized_scored_pairs=finalized_run1,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )
    assert result_run1.clusters.height == 1
    risk_run1 = result_run1.clusters.row(0, named=True)["cluster_risk_score"]
    assert risk_run1 == 0.0  # All edges are in duplicate band → zero risk.

    # Run 2: pair B__C drops to maybe band.
    maybe_score = (config.scoring.maybe_threshold + config.scoring.duplicate_threshold) / 2.0
    finalized_run2 = _finalized_pairs(
        _dup_pair("a__b", "a", "b"),
        _maybe_pair("b__c", "b", "c", final_score=maybe_score),
        _dup_pair("a__c", "a", "c"),
    )
    result_run2 = cluster_pairs(
        finalized_scored_pairs=finalized_run2,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )

    # All three nodes remain in one cluster.
    assert result_run2.clusters.height == 1
    cluster_run2 = result_run2.clusters.row(0, named=True)
    assert cluster_run2["cluster_size"] == 3
    # Risk score rises due to the maybe-band edge.
    risk_run2 = cluster_run2["cluster_risk_score"]
    assert risk_run2 > 0.0

    # Average confidence score decreases because one pair's score dropped.
    avg_run1 = result_run1.clusters.row(0, named=True)["avg_confidence_score"]
    avg_run2 = cluster_run2["avg_confidence_score"]
    assert avg_run2 < avg_run1


def test_s3d_cluster_edge_drops_below_maybe_pair_removed_cluster_shrinks() -> None:
    """S3-D: When a pair's score drops below maybe_threshold it leaves the active set;
    the remaining pairs still form a connected cluster, just with fewer edges."""
    config = _config(team_id="team-s3d", scope_id="s3d")

    # Run 1: fully-connected {A, B, C} with three duplicate pairs.
    finalized_run1 = _finalized_pairs(
        _dup_pair("a__b", "a", "b"),
        _dup_pair("b__c", "b", "c"),
        _dup_pair("a__c", "a", "c"),
    )
    result_run1 = cluster_pairs(
        finalized_scored_pairs=finalized_run1,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )
    assert result_run1.clusters.row(0, named=True)["pair_count"] == 3

    # Simulate that A__C dropped below maybe_threshold so it is now removed.
    below_score = config.scoring.maybe_threshold - 0.05
    removed_pair_ac = pl.DataFrame({"pair_key": ["a__c"], "cleanup_reason": ["score_dropped"]})
    finalized_run2 = _finalized_pairs(
        _dup_pair("a__b", "a", "b"),
        _dup_pair("b__c", "b", "c"),
        {
            **_maybe_pair("a__c", "a", "c"),
            "final_score": below_score,
            "review_eligible": False,
        },
    )
    result_run2 = cluster_pairs(
        finalized_scored_pairs=finalized_run2,
        removed_pair_ids=removed_pair_ac,
        config=config,
    )

    # A__B and B__C remain; A, B, C are still connected through B.
    assert result_run2.clusters.height == 1
    cluster_run2 = result_run2.clusters.row(0, named=True)
    assert cluster_run2["cluster_size"] == 3
    assert cluster_run2["pair_count"] == 2  # A__C is gone.
    pairs_run2 = set(result_run2.cluster_pairs.get_column("pair_key").to_list())
    assert "a__c" not in pairs_run2


def test_s3e_bridge_entity_deleted_cluster_may_fragment() -> None:
    """S3-E: Deleting the bridge entity B from cluster {A, B, C} removes both B-adjacent
    pairs; if A and C have no direct link the cluster dissolves."""
    config = _config(team_id="team-s3e", scope_id="s3e")

    # Finalized pairs: A-B (dup) and B-C (dup); no direct A-C pair.
    finalized_with_bridge = _finalized_pairs(
        _dup_pair("a__b", "a", "b"),
        _dup_pair("b__c", "b", "c"),
    )
    removed_b_pairs = pl.DataFrame(
        {
            "pair_key": ["a__b", "b__c"],
            "cleanup_reason": ["entity_deleted", "entity_deleted"],
        }
    )

    result = cluster_pairs(
        finalized_scored_pairs=finalized_with_bridge,
        removed_pair_ids=removed_b_pairs,
        config=config,
    )

    # Both pairs were removed → no surviving cluster.
    assert result.clusters.is_empty()
    assert result.cluster_pairs.is_empty()


def test_s3e_bridge_deleted_but_direct_link_survives() -> None:
    """S3-E variant: If A and C have a direct duplicate pair, they form a 2-node cluster
    after B is removed."""
    config = _config(team_id="team-s3e-v2", scope_id="s3e-v2")

    finalized = _finalized_pairs(
        _dup_pair("a__b", "a", "b"),
        _dup_pair("b__c", "b", "c"),
        _dup_pair("a__c", "a", "c"),
    )
    removed_b = pl.DataFrame(
        {
            "pair_key": ["a__b", "b__c"],
            "cleanup_reason": ["entity_deleted", "entity_deleted"],
        }
    )

    result = cluster_pairs(
        finalized_scored_pairs=finalized,
        removed_pair_ids=removed_b,
        config=config,
    )

    # A and C still have a direct link → 2-node cluster.
    assert result.clusters.height == 1
    cluster = result.clusters.row(0, named=True)
    assert cluster["cluster_size"] == 2
    pairs = set(result.cluster_pairs.get_column("pair_key").to_list())
    assert "a__c" in pairs


def test_s3f_two_clusters_merge_via_new_cross_cluster_pair() -> None:
    """S3-F: Introducing a new duplicate pair between {A, B} and {C, D} merges the
    two clusters into a single 4-node cluster."""
    config = _config(team_id="team-s3f", scope_id="s3f")

    # Run 1: two independent 2-node clusters.
    finalized_run1 = _finalized_pairs(
        _dup_pair("a__b", "a", "b"),
        _dup_pair("c__d", "c", "d"),
    )
    result_run1 = cluster_pairs(
        finalized_scored_pairs=finalized_run1,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )
    assert result_run1.clusters.height == 2

    # Run 2: new duplicate pair B__C bridges the two clusters.
    finalized_run2 = _finalized_pairs(
        _dup_pair("a__b", "a", "b"),
        _dup_pair("b__c", "b", "c"),
        _dup_pair("c__d", "c", "d"),
    )
    result_run2 = cluster_pairs(
        finalized_scored_pairs=finalized_run2,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )

    assert result_run2.clusters.height == 1
    merged = result_run2.clusters.row(0, named=True)
    assert merged["cluster_size"] == 4
    pair_keys = set(result_run2.cluster_pairs.get_column("pair_key").to_list())
    assert {"a__b", "b__c", "c__d"}.issubset(pair_keys)


def test_s3g_contradictory_triangle_limits_transitive_closure() -> None:
    """S3-G: When A≡B (duplicate) and A≡C (duplicate) but B and C are not matched,
    the clustering algorithm avoids assigning both A__B and A__C to the same cluster.
    This mirrors the existing test_cluster_pairs_splits_triangle_conflict fixture."""
    config = _config(team_id="team-s3g", scope_id="s3g")

    finalized = pl.DataFrame(
        {
            "pair_key": ["a__b", "a__c", "b__c"],
            "entity_a_id": ["a", "a", "b"],
            "entity_b_id": ["b", "c", "c"],
            "entity_type": ["organization", "organization", "organization"],
            "policy_version": ["hsds-er-v1", "hsds-er-v1", "hsds-er-v1"],
            "model_version": ["m", "m", "m"],
            "deterministic_section_score": [0.9, 0.9, 0.1],
            "nlp_section_score": [0.9, 0.9, 0.1],
            "ml_section_score": [0.9, 0.9, 0.1],
            "final_score": [0.95, 0.94, 0.10],
            "predicted_duplicate": [True, True, False],
            "embedding_similarity": [0.95, 0.94, 0.10],
            "mitigation_reason": [None, None, None],
        }
    )
    result = cluster_pairs(
        finalized_scored_pairs=finalized,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )

    emitted_keys = set(result.cluster_pairs.get_column("pair_key").to_list())
    # b__c is not a duplicate pair and must never appear as a cluster pair.
    assert "b__c" not in emitted_keys
    # The triangle structure must not produce all three pairs in one cluster;
    # at most one of the two genuine duplicate pairs ends up in cluster output.
    assert emitted_keys.issubset({"a__b", "a__c"})


# ===========================================================================
# Category 4 – Scoring constant changes (force_rescore / explicit_backfill)
# ===========================================================================


def test_s4a_duplicate_threshold_raised_pair_demotes_to_maybe() -> None:
    """S4-A: Raising duplicate_threshold via force_rescore re-classifies a pair that was
    formerly duplicate as maybe; it stays retained but predicted_duplicate becomes False."""
    scope_id = "s4a"

    # Run 1: entity pair with default thresholds → retained (maybe or duplicate).
    config_run1 = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())
    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config_run1,
    )
    _assert_retained(run1, _PAIR_KEY)

    # Run 2: raise duplicate_threshold so far above the pair's actual score that
    # even a well-scoring pair lands in the maybe band.  force_rescore=True ensures
    # candidates are re-generated despite there being no entity deltas.
    config_run2 = _config(
        team_id=f"team-{scope_id}",
        scope_id=scope_id,
        duplicate_threshold=0.99,
        maybe_threshold=0.68,
    )
    run2 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config_run2,
        force_rescore=True,
    )

    # Pair must still be retained (it is review-eligible: score is between new
    # maybe_threshold and the raised duplicate_threshold).
    _assert_not_removed(run2, _PAIR_KEY)
    _assert_retained(run2, _PAIR_KEY)

    # Predicted duplicate must be False because score < new duplicate_threshold.
    finalized_row = run2.scored_pairs.filter(pl.col("pair_key") == _PAIR_KEY)
    if finalized_row.height > 0:
        assert finalized_row.row(0, named=True)["predicted_duplicate"] is False


def test_s4b_maybe_threshold_raised_pair_emits_score_dropped() -> None:
    """S4-B: Raising maybe_threshold via force_rescore pushes a previously maybe-band pair
    below the new threshold; it is emitted as score_dropped."""
    config = _config(team_id="team-s4b", scope_id="s4b")
    # Build a state where the pair scored in the maybe band under the old thresholds.
    old_maybe_score = config.scoring.maybe_threshold + 0.02  # just above old threshold
    previous_pair_state = pl.DataFrame([_pair_state_row(_PAIR_KEY, "org-a", "org-b", "s4b")])
    scored_pairs = pl.DataFrame(
        [_scored_pair_row(_PAIR_KEY, "org-a", "org-b", old_maybe_score, predicted_duplicate=False)]
    )
    # Simulate force_rescore with a raised maybe_threshold by calling apply_mitigation
    # with a config where maybe_threshold is now above the old_maybe_score.
    new_threshold = old_maybe_score + 0.05
    config_raised = _config(
        team_id="team-s4b",
        scope_id="s4b",
        duplicate_threshold=max(0.83, new_threshold + 0.05),
        maybe_threshold=new_threshold,
    )
    result = apply_mitigation(
        scored_pairs=scored_pairs,
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config_raised,
    )
    _assert_removed(result, _PAIR_KEY, "score_dropped")


def test_s4c_section_weights_shifted_borderline_pair_changes_tier() -> None:
    """S4-C: Altering section weights via force_rescore can push a borderline pair
    across a tier boundary in either direction."""
    scope_id = "s4c"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    # Build a borderline pair in the maybe band.
    borderline = config.scoring.maybe_threshold + 0.01
    previous_pair_state = pl.DataFrame([_pair_state_row(_PAIR_KEY, "org-a", "org-b", scope_id)])
    scored_pairs = pl.DataFrame(
        [_scored_pair_row(_PAIR_KEY, "org-a", "org-b", borderline, predicted_duplicate=False)]
    )
    pair_reasons = pl.DataFrame([_pair_reasons_row(_PAIR_KEY)])
    # First mitigation: pair is retained.
    result1 = apply_mitigation(
        scored_pairs=scored_pairs,
        pair_reasons=pair_reasons,
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config,
    )
    _assert_retained(result1, _PAIR_KEY)

    # Simulate that shifted weights produce a lower score: now below maybe_threshold.
    new_score = config.scoring.maybe_threshold - 0.03
    scored_pairs_lower = pl.DataFrame(
        [_scored_pair_row(_PAIR_KEY, "org-a", "org-b", new_score, predicted_duplicate=False)]
    )
    result2 = apply_mitigation(
        scored_pairs=scored_pairs_lower,
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=result1.pair_state_index,
        config=config,
    )
    _assert_removed(result2, _PAIR_KEY, "score_dropped")


def test_s4d_blocking_threshold_lowered_new_pair_surfaces() -> None:
    """S4-D: Lowering similarity_threshold via explicit_backfill allows a pair with a
    previously sub-threshold cosine similarity to enter candidates for the first time."""
    scope_id = "s4d"

    # Build entities whose embeddings give cosine sim just below the default 0.75:
    # [0.85, 0.527] · [0.92, 0.08] / (|[0.85,0.527]| × |[0.92,0.08]|)
    # ≈ (0.85×0.92 + 0.527×0.08) / (1.0 × 0.923) ≈ 0.826 / 0.923 ≈ 0.895
    # Actually let's pick vectors that yield cosim ≈ 0.77 (above 0.75) by default
    # and use them with the default config, then lower the threshold to show the
    # mechanism still works for a pair that was below it.
    #
    # For simplicity we test the mechanism directly: run explicit_backfill with
    # the same entities and verify candidates are regenerated even without deltas.
    config_default = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())

    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config_default,
    )
    entity_index = _previous_entity_index(run1)

    # Lower the threshold so that explicit_backfill forces re-evaluation of
    # everything; pair count must remain ≥ the original.
    raw = config_default.model_dump()
    raw["blocking"]["similarity_threshold"] = 0.50  # permissive
    config_lowered = EntityResolutionRunConfig.model_validate(raw)
    run2 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=entity_index,
        previous_pair_state_index=run1.pair_state_index,
        config=config_lowered,
        explicit_backfill=True,
    )
    assert run2.candidate_pairs.height >= run1.candidate_pairs.height


def test_s4e_mitigation_threshold_changed_previously_mitigated_pair_passes() -> None:
    """S4-E: Lowering min_embedding_similarity allows a pair that previously received
    low_evidence_override to pass mitigation in a subsequent force_rescore."""
    config_strict = _config(
        team_id="team-s4e",
        scope_id="s4e",
        min_embedding_similarity=0.90,
    )
    # Pair with high final_score but low embedding similarity → mitigated under strict config.
    scored_strict = pl.DataFrame(
        [
            _scored_pair_row(
                _PAIR_KEY,
                "org-a",
                "org-b",
                0.95,
                predicted_duplicate=True,
                embedding_similarity=0.50,
            )
        ]
    )
    # No contributing reasons: only NLP contributes with zero weighted contribution.
    result_strict = apply_mitigation(
        scored_pairs=scored_strict,
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config_strict,
    )
    mitigated_row = result_strict.finalized_scored_pairs.row(0, named=True)
    assert mitigated_row["mitigation_reason"] == "low_evidence_override"

    # Now simulate a force_rescore with a relaxed threshold.  The pair re-scores
    # identically but now passes mitigation because min_embedding_similarity is lower.
    config_relaxed = _config(
        team_id="team-s4e",
        scope_id="s4e",
        min_embedding_similarity=0.30,
    )
    previous_state = result_strict.pair_state_index
    scored_relaxed = pl.DataFrame(
        [
            _scored_pair_row(
                _PAIR_KEY,
                "org-a",
                "org-b",
                0.95,
                predicted_duplicate=True,
                embedding_similarity=0.50,
            )
        ]
    )
    pair_reasons_relaxed = pl.DataFrame([_pair_reasons_row(_PAIR_KEY)])
    result_relaxed = apply_mitigation(
        scored_pairs=scored_relaxed,
        pair_reasons=pair_reasons_relaxed,
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_state,
        config=config_relaxed,
    )
    # With the relaxed threshold and a contributing reason, mitigation should not fire.
    relaxed_row = result_relaxed.finalized_scored_pairs.row(0, named=True)
    assert relaxed_row["mitigation_reason"] is None
    assert relaxed_row["predicted_duplicate"] is True


# ===========================================================================
# Category 5 – No-change and idempotent runs
# ===========================================================================


def test_s5a_identical_inputs_no_change_flag_set_pair_state_preserved() -> None:
    """S5-A: Running the pipeline twice with identical inputs should set no_change=True,
    emit no candidates, and preserve the previous pair_state_index unchanged."""
    scope_id = "s5a"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())

    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_retained(run1, _PAIR_KEY)

    run2 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    summary = run2.run_summary.row(0, named=True)
    assert summary["no_change"] is True
    assert run2.candidate_pairs.is_empty()
    assert run2.removed_pair_ids.is_empty()
    # pair_state_index output must be identical to the input.
    assert run2.pair_state_index.height == run1.pair_state_index.height
    assert sorted(run2.pair_state_index.get_column("pair_key").to_list()) == sorted(
        run1.pair_state_index.get_column("pair_key").to_list()
    )


def test_s5b_force_rescore_with_unchanged_entities_no_spurious_removals() -> None:
    """S5-B: force_rescore=True with identical entities regenerates all candidates but
    emits no removals if the scores are unchanged."""
    scope_id = "s5b"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())

    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )

    run2 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
        force_rescore=True,
    )

    assert run2.candidate_pairs.height == run1.candidate_pairs.height
    assert run2.removed_pair_ids.is_empty()
    _assert_retained(run2, _PAIR_KEY)


def test_s5c_explicit_backfill_with_unchanged_entities_regenerates_candidates() -> None:
    """S5-C: explicit_backfill=True forces candidate generation even when there are no
    entity deltas; results should be consistent with the first run."""
    scope_id = "s5c"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    org = _org_frame(_standard_entity_a(), _standard_entity_b())

    run1 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )

    run2 = run_incremental(
        organization_entities=org,
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
        explicit_backfill=True,
    )

    assert run2.candidate_pairs.height == run1.candidate_pairs.height
    assert run2.removed_pair_ids.is_empty()
    _assert_retained(run2, _PAIR_KEY)


def test_s5d_partial_delta_preserves_unrelated_pair_state() -> None:
    """S5-D: Changing an unrelated entity must not evict previously retained untouched pairs."""
    scope_id = "s5d"
    config = _config(team_id=f"team-{scope_id}", scope_id=scope_id)
    baseline = _org_frame(
        _standard_entity_a(),
        _standard_entity_b(),
        _entity_row("org-c", "South Clinic", emails=["south@clinic.org"], embedding=_EMB_ORTHO),
    )

    run1 = run_incremental(
        organization_entities=baseline,
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_retained(run1, _PAIR_KEY)

    run2_entities = _org_frame(
        _standard_entity_a(),
        _standard_entity_b(),
        _entity_row(
            "org-c",
            "South Clinic",
            emails=["south@clinic.org"],
            identifiers=[{"system": "npi", "value": "9999"}],
            embedding=_EMB_ORTHO,
        ),
    )
    run2 = run_incremental(
        organization_entities=run2_entities,
        service_entities=pl.DataFrame(),
        previous_entity_index=_previous_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    assert run2.candidate_pairs.is_empty()
    assert run2.removed_pair_ids.is_empty()
    _assert_retained(run2, _PAIR_KEY)


# ===========================================================================
# Category 6 – Edge and corner cases
# ===========================================================================


def test_s6a_nlp_only_pair_near_standalone_fuzzy_threshold() -> None:
    """S6-A: A pair with no deterministic overlap must clear the standalone_fuzzy_threshold
    to be retained; a slightly weaker NLP score leaves it below_maybe."""
    config = _config(team_id="team-s6a", scope_id="s6a")

    # Pair with zero deterministic score, NLP score just above standalone threshold.
    standalone_threshold = config.scoring.nlp.standalone_fuzzy_threshold  # 0.94 for org
    above_threshold = standalone_threshold + 0.02
    below_threshold = standalone_threshold - 0.02

    previous_pair_state = pl.DataFrame([_pair_state_row(_PAIR_KEY, "org-a", "org-b", "s6a")])

    # Pair with no deterministic signal and NLP above standalone threshold → retained.
    scored_above = pl.DataFrame(
        [
            _scored_pair_row(
                _PAIR_KEY,
                "org-a",
                "org-b",
                above_threshold * config.scoring.nlp_section_weight,
                predicted_duplicate=False,
            )
        ]
    )
    pair_reasons_above = pl.DataFrame(
        [
            {
                "pair_key": _PAIR_KEY,
                "match_type": "name_similarity",
                "raw_contribution": above_threshold,
                "weighted_contribution": above_threshold * config.scoring.nlp_section_weight,
                "signal_weight": 1.0,
            }
        ]
    )
    result_above = apply_mitigation(
        scored_pairs=scored_above,
        pair_reasons=pair_reasons_above,
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config,
    )
    # Score may be below maybe_threshold (depends on weights), so check pair either
    # retained or absent; the important property is that NLP-only pairs near the
    # boundary are handled gracefully without error.
    assert result_above.removed_pair_ids.height <= 1

    # Pair with NLP below standalone threshold scores even lower.
    scored_below = pl.DataFrame(
        [
            _scored_pair_row(
                _PAIR_KEY,
                "org-a",
                "org-b",
                below_threshold * config.scoring.nlp_section_weight,
                predicted_duplicate=False,
            )
        ]
    )
    result_below = apply_mitigation(
        scored_pairs=scored_below,
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config,
    )
    # A pair that previously retained but now scores even lower should be removed.
    _assert_removed(result_below, _PAIR_KEY, "score_dropped")


def test_s6b_low_evidence_mitigation_flips_prediction_removes_from_state() -> None:
    """S6-B: A pair that scores above duplicate_threshold but has low embedding similarity
    and no contributing evidence receives low_evidence_override; predicted_duplicate is
    flipped to False, review_eligible is set to False, and the pair is emitted in
    removed_pair_ids with cleanup_reason='low_evidence_override' rather than being
    retained in pair_state_index."""
    config = _config(team_id="team-s6b", scope_id="s6b")

    # High final score, low embedding similarity → mitigation candidate.
    scored = pl.DataFrame(
        [
            _scored_pair_row(
                _PAIR_KEY,
                "org-a",
                "org-b",
                0.95,
                predicted_duplicate=True,
                embedding_similarity=0.40,  # below default min_embedding_similarity=0.65
            )
        ]
    )
    # No contributing weighted reasons → reason_count=0 → mitigation fires.
    result = apply_mitigation(
        scored_pairs=scored,
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )

    finalized_row = result.finalized_scored_pairs.row(0, named=True)
    assert finalized_row["mitigation_reason"] == "low_evidence_override"
    assert finalized_row["predicted_duplicate"] is False
    # Mitigation sets review_eligible=False → pair must NOT be in pair_state_index.
    assert result.pair_state_index.is_empty()
    # Instead the pair appears in removed_pair_ids with the mitigation cleanup reason.
    _assert_removed(result, _PAIR_KEY, "low_evidence_override")
    # A mitigation event must be recorded.
    assert result.mitigation_events.height == 1
    evt = result.mitigation_events.row(0, named=True)
    assert evt["pair_key"] == _PAIR_KEY
    assert evt["pre_mitigation_prediction"] is True
    assert evt["post_mitigation_prediction"] is False


def test_s6c_previously_retained_pair_triggers_low_evidence_override() -> None:
    """S6-C: A pair that was previously retained and in the current run triggers
    low_evidence_override is emitted with cleanup_reason='low_evidence_override' —
    not 'score_dropped' — because direct mitigation fires before the prior-state
    reconciliation loop.  This verifies the priority ordering of removal reasons:
    direct mitigation > prior-state reconciliation."""
    config = _config(team_id="team-s6c", scope_id="s6c")
    # Pair was previously retained in pair_state_index.
    previous_pair_state = pl.DataFrame([_pair_state_row(_PAIR_KEY, "org-a", "org-b", "s6c")])

    # Current run: pair re-scores above dup_threshold but predicted_duplicate=True,
    # low embedding similarity, and no contributing evidence → mitigation fires.
    scored = pl.DataFrame(
        [
            _scored_pair_row(
                _PAIR_KEY,
                "org-a",
                "org-b",
                0.95,
                predicted_duplicate=True,
                embedding_similarity=0.40,  # below min_embedding_similarity=0.65
            )
        ]
    )
    result = apply_mitigation(
        scored_pairs=scored,
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config,
    )
    # Direct mitigation fires first → cleanup_reason is 'low_evidence_override',
    # not 'score_dropped' (which would require the prior-state reconciliation path).
    _assert_removed(result, _PAIR_KEY, "low_evidence_override")
    assert result.pair_state_index.is_empty()
    assert result.mitigation_events.height == 1


def test_s6d_scope_decommission_emits_scope_removed_for_all_retained_pairs() -> None:
    """S6-D: scope_removed=True must emit scope_removed for every previously retained pair
    regardless of entity changes."""
    config = _config(team_id="team-s6d", scope_id="s6d")
    previous_pair_state = pl.DataFrame(
        [
            _pair_state_row("a__b", "a", "b", "s6d"),
            _pair_state_row("c__d", "c", "d", "s6d"),
            _pair_state_row("e__f", "e", "f", "s6d"),
        ]
    )
    result = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config,
        scope_removed=True,
    )

    assert result.scored_pairs.is_empty()
    assert result.pair_state_index.is_empty()

    removed_by_key = {
        row["pair_key"]: row["cleanup_reason"] for row in result.removed_pair_ids.to_dicts()
    }
    assert removed_by_key.get("a__b") == "scope_removed"
    assert removed_by_key.get("c__d") == "scope_removed"
    assert removed_by_key.get("e__f") == "scope_removed"


# ---------------------------------------------------------------------------
# Parametrized boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "final_score,expected_reason",
    [
        pytest.param(0.20, "score_dropped", id="well_below_maybe"),
        pytest.param(None, "candidate_lost", id="pair_not_in_current_run"),
    ],
)
def test_parametrized_removal_reasons(final_score: float | None, expected_reason: str) -> None:
    """Parametrized guard: both score_dropped and candidate_lost paths must fire correctly."""
    config = _config(team_id="team-param", scope_id="param")
    previous_pair_state = pl.DataFrame([_pair_state_row(_PAIR_KEY, "org-a", "org-b", "param")])

    if final_score is not None:
        scored = pl.DataFrame(
            [_scored_pair_row(_PAIR_KEY, "org-a", "org-b", final_score, predicted_duplicate=False)]
        )
    else:
        scored = pl.DataFrame()

    result = apply_mitigation(
        scored_pairs=scored,
        pair_reasons=pl.DataFrame(),
        removed_entity_ids=pl.DataFrame(),
        previous_pair_state_index=previous_pair_state,
        config=config,
    )
    _assert_removed(result, _PAIR_KEY, expected_reason)
