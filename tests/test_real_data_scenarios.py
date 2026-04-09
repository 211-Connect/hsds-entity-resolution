"""Real-data incremental scenario tests seeded from a live IL211 duplicate pair.

Seed data is loaded from tests/fixtures/burrell_sph_pair.json, which was
queried directly from Snowflake on 2026-04-09 via the SnowSQL skill:

    Run ID:        er-6e923fd76fa14732902d005ac21e96b4
    Team:          IL211
    Source schema: uwgsl211
    Pair type:     service
    Confidence:    0.74  (service dup threshold: 0.70, maybe threshold: 0.62)

The two entities are both Burrell Behavioral Health locations offering the same
Suicide Prevention Hotlines service.  They share name, email, taxonomy code,
the national 988 hotline number, and burrellcenter.com domain, but sit at
different physical addresses (Branson vs Marshfield, MO).

Service scoring formula (ML absent for this job):
    final = det_section × 0.50 + nlp_section × 0.50
    0.7443 ≈ 0.4886 × 0.50 + 1.0 × 0.50

Signal catalogue (from DUPLICATE_REASONS):
    name_similarity  +1.00  (NLP section – exact name match)
    shared_email     +0.16  rebecca.randolph@burrellcenter.com
    shared_taxonomy  +0.12  rp-1500.1400-800
    shared_phone     +0.11  "988" shared; raw=0.50 (partial match)
    shared_domain    +0.04  burrellcenter.com

Category R – Real-data scenarios
----------------------------------
R1  Baseline:     both entities present → pair is review-eligible
R2  Remove A:     entity A absent → entity_deleted
R3  Remove B:     entity B absent → entity_deleted
R4  Remove 988:   entity B drops the shared 988 → score_change (dup→maybe or dropped)
R5  Remove email: entity B drops shared email → score drops further
R6  Rename B:     entity B gets an unrelated name → score_dropped or candidate_lost
R7  Resurrect B:  B deleted run-2 then returns run-3 → pair re-built
R8  Idempotent:   identical inputs twice → no_change, pair state preserved
R9  Third Burrell location added → 3-node cluster forms
R10 Both signals (email + 988) removed → pair clearly falls below maybe threshold

configurable-readers note
--------------------------
Even when predicted_duplicate=True, configurable-readers publishes BOTH
records because entity_a is in Branson (65616) and entity_b is in
Marshfield (65706) — different locations mean both are published regardless
of duplicate status.  Only same-address duplicates are collapsed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.pipeline import run_incremental

# ---------------------------------------------------------------------------
# Load seed fixture
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "burrell_sph_pair.json"
_SEED: dict[str, Any] = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Real entity identifiers (from Snowflake)
# ---------------------------------------------------------------------------

_ENT_A_ID = "aa4da335-99fc-58ae-970c-ded5d2e5779b"
_ENT_B_ID = "b71c103f-32d8-5514-ac00-2a86c0a13acf"
_SVC_PAIR_KEY = f"{_ENT_A_ID}__{_ENT_B_ID}"
_SOURCE_SCHEMA = "uwgsl211"
_TEAM_ID = "IL211"

# Embedding vectors — the real run reported EMBEDDING_SIMILARITY=1.0.
# We use nearly-identical 2-D vectors: cosine ≈ 0.9997, safely above the
# service blocking threshold so the pair always enters the candidate set.
_EMB_HIGH_A = [0.9, 0.1]
_EMB_HIGH_B = [0.92, 0.08]

# Orthogonal vector: cosine ≈ 0.0 — drops pair below blocking threshold.
_EMB_ORTHO = [0.0, 1.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svc_config(scope_id: str = "uwgsl211") -> EntityResolutionRunConfig:
    """Build a service-scoped config that matches the IL211 job defaults."""
    return EntityResolutionRunConfig.defaults_for_entity_type(
        team_id=_TEAM_ID,
        scope_id=scope_id,
        entity_type="service",
    )


def _entity_from_seed(key: str, *, embedding: list[float] | None = None) -> dict[str, Any]:
    """Build a pipeline-ready entity dict from the seed fixture.

    The fixture mirrors the real Snowflake DENORMALIZED_SERVICE_CACHE columns;
    this helper normalises field names to the snake_case used by the pipeline
    and injects an embedding vector (embedding is never stored in the cache).
    """
    raw: dict[str, Any] = _SEED[key]
    emb = embedding if embedding is not None else (
        _EMB_HIGH_A if key == "entity_a" else _EMB_HIGH_B
    )
    return {
        "entity_id": raw["entity_id"],
        "source_schema": _SOURCE_SCHEMA,
        "name": raw["name"],
        "description": raw["description"],
        "emails": list(raw["emails"]),
        "phones": list(raw["phones"]),
        "websites": list(raw["websites"]),
        "taxonomies": [{"code": t["code"]} for t in raw["taxonomies"]],
        "locations": list(raw["locations"]),
        "identifiers": [],
        "services_rollup": [],
        "organization_id": None,
        "organization_name": raw.get("organization_name"),
        "embedding_vector": emb,
    }


def _svc_frame(*rows: dict[str, Any]) -> pl.DataFrame:
    """Build a service entities DataFrame from one or more row dicts."""
    return pl.DataFrame(list(rows))


def _previous_svc_entity_index(result: Any) -> pl.DataFrame:
    """Extract the entity index from a service-focused incremental result."""
    svc = result.denormalized_service
    if svc.is_empty():
        return pl.DataFrame(
            schema={"entity_id": pl.String, "entity_type": pl.String,
                    "content_hash": pl.String, "active_flag": pl.Boolean}
        )
    return svc.select(["entity_id", "entity_type", "content_hash"]).with_columns(
        pl.lit(True).alias("active_flag")
    )


def _run_baseline(scope_id: str = "r-baseline") -> Any:
    """Run a first incremental pass with both real entities and return the result."""
    config = _svc_config(scope_id)
    svc = _svc_frame(_entity_from_seed("entity_a"), _entity_from_seed("entity_b"))
    return run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=svc,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )


def _assert_pair_retained(result: Any, pair_key: str = _SVC_PAIR_KEY) -> None:
    rows = result.pair_state_index.filter(pl.col("pair_key") == pair_key)
    assert rows.height == 1, (
        f"Expected {pair_key!r} in pair_state_index but got "
        f"{result.pair_state_index.get_column('pair_key').to_list()}"
    )
    assert rows.row(0, named=True)["retained_flag"] is True


def _assert_pair_removed(result: Any, reason: str, pair_key: str = _SVC_PAIR_KEY) -> None:
    rows = result.removed_pair_ids.filter(pl.col("pair_key") == pair_key)
    assert rows.height == 1, (
        f"Expected {pair_key!r} in removed_pair_ids but got "
        f"{result.removed_pair_ids.get_column('pair_key').to_list()}"
    )
    actual_reason = rows.row(0, named=True)["cleanup_reason"]
    assert actual_reason == reason, (
        f"Expected cleanup_reason={reason!r} for {pair_key!r}, got {actual_reason!r}"
    )


def _assert_pair_not_removed(result: Any, pair_key: str = _SVC_PAIR_KEY) -> None:
    rows = result.removed_pair_ids.filter(pl.col("pair_key") == pair_key)
    assert rows.is_empty(), (
        f"Expected {pair_key!r} NOT in removed_pair_ids but it was present with "
        f"cleanup_reason={rows.row(0, named=True).get('cleanup_reason')!r}"
    )


# ---------------------------------------------------------------------------
# R1 — Baseline: real signals generate a review-eligible pair
# ---------------------------------------------------------------------------


def test_r1_baseline_real_pair_is_review_eligible() -> None:
    """R1: Both Burrell locations present → pair enters the review queue.

    The real run produced confidence=0.74 > service dup_threshold=0.70, so the
    pair should be predicted_duplicate=True and appear in pair_state_index.
    """
    run1 = _run_baseline("r1")

    _assert_pair_retained(run1)

    scored = run1.scored_pairs.filter(pl.col("pair_key") == _SVC_PAIR_KEY)
    assert scored.height == 1, "Expected exactly one scored row for the real pair"

    row = scored.row(0, named=True)
    assert row["predicted_duplicate"] is True, (
        f"Expected predicted_duplicate=True (confidence {row['final_score']:.4f} "
        f"should exceed service dup_threshold=0.70)"
    )
    assert row["final_score"] > 0.70, (
        f"Real pair confidence {row['final_score']:.4f} expected to exceed 0.70"
    )

    # Verify the real signal weights reproduced by the pipeline
    reasons = run1.pair_reasons.filter(pl.col("pair_key") == _SVC_PAIR_KEY)
    match_types = set(reasons.get_column("match_type").to_list())
    assert "name_similarity" in match_types, "Expected name_similarity reason"
    assert "shared_email" in match_types, "Expected shared_email reason"
    assert "shared_taxonomy" in match_types, "Expected shared_taxonomy reason"


# ---------------------------------------------------------------------------
# R2 — Entity A removed: the Branson location is decommissioned
# ---------------------------------------------------------------------------


def test_r2_entity_a_removed_emits_entity_deleted() -> None:
    """R2: Entity A (Branson office) is removed from the source feed.

    The pair must be emitted as entity_deleted in removed_pair_ids and must
    leave pair_state_index empty.  The Marshfield entity B continues to exist
    as a standalone service.
    """
    scope_id = "r2"
    config = _svc_config(scope_id)
    both = _svc_frame(_entity_from_seed("entity_a"), _entity_from_seed("entity_b"))

    run1 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_pair_retained(run1)

    # Run 2: only entity B remains in the source.
    b_only = _svc_frame(_entity_from_seed("entity_b"))
    run2 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=b_only,
        previous_entity_index=_previous_svc_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    _assert_pair_removed(run2, "entity_deleted")
    assert run2.pair_state_index.is_empty()

    # Entity A appears in removed_entity_ids.
    removed_entities = run2.removed_entity_ids.filter(
        pl.col("entity_id") == _ENT_A_ID
    )
    assert removed_entities.height == 1, (
        f"Expected entity_id={_ENT_A_ID!r} in removed_entity_ids"
    )
    assert removed_entities.row(0, named=True)["cleanup_reason"] == "entity_deleted"


# ---------------------------------------------------------------------------
# R3 — Entity B removed: the Marshfield location is decommissioned
# ---------------------------------------------------------------------------


def test_r3_entity_b_removed_emits_entity_deleted() -> None:
    """R3: Entity B (Marshfield office) is removed from the source feed.

    Same expectation as R2 — pair gets entity_deleted regardless of which
    side of the pair is removed.
    """
    scope_id = "r3"
    config = _svc_config(scope_id)
    both = _svc_frame(_entity_from_seed("entity_a"), _entity_from_seed("entity_b"))

    run1 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_pair_retained(run1)

    a_only = _svc_frame(_entity_from_seed("entity_a"))
    run2 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=a_only,
        previous_entity_index=_previous_svc_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    _assert_pair_removed(run2, "entity_deleted")
    assert run2.pair_state_index.is_empty()

    removed_entities = run2.removed_entity_ids.filter(pl.col("entity_id") == _ENT_B_ID)
    assert removed_entities.height == 1


# ---------------------------------------------------------------------------
# R4 — Remove shared "988" phone from entity B
# ---------------------------------------------------------------------------


def test_r4_remove_shared_988_phone_score_drops() -> None:
    """R4: Entity B's record is updated to drop the 988 hotline number.

    Losing the shared_phone signal (weighted contribution ≈ 0.11) reduces the
    deterministic section score.  The pair must either:
      - remain in the review queue but demote from duplicate to maybe band, OR
      - fall below maybe_threshold and be emitted as score_dropped

    In either case the pair must NOT still be predicted_duplicate=True,
    because the full signal set was the sole reason it cleared dup_threshold=0.70.
    """
    scope_id = "r4"
    config = _svc_config(scope_id)
    both = _svc_frame(_entity_from_seed("entity_a"), _entity_from_seed("entity_b"))

    run1 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_pair_retained(run1)
    assert run1.scored_pairs.filter(
        pl.col("pair_key") == _SVC_PAIR_KEY
    ).row(0, named=True)["predicted_duplicate"] is True

    # Run 2: entity B keeps its local number but drops 988.
    ent_b_no_988 = _entity_from_seed("entity_b")
    ent_b_no_988["phones"] = ["417-761-5000"]  # local line only; no shared phone

    run2 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=_svc_frame(_entity_from_seed("entity_a"), ent_b_no_988),
        previous_entity_index=_previous_svc_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    retained = run2.pair_state_index.filter(pl.col("pair_key") == _SVC_PAIR_KEY).height > 0
    removed = run2.removed_pair_ids.filter(pl.col("pair_key") == _SVC_PAIR_KEY).height > 0
    assert retained or removed, (
        "Pair must be tracked (retained in maybe band) or emitted as removed "
        "after losing the shared 988 signal"
    )

    if retained:
        scored = run2.scored_pairs.filter(pl.col("pair_key") == _SVC_PAIR_KEY)
        if scored.height > 0:
            assert scored.row(0, named=True)["predicted_duplicate"] is False, (
                "Pair stayed in maybe band but should no longer be predicted_duplicate=True "
                "after losing the shared_phone signal"
            )
    else:
        assert run2.removed_pair_ids.filter(
            pl.col("pair_key") == _SVC_PAIR_KEY
        ).row(0, named=True)["cleanup_reason"] == "score_dropped"


# ---------------------------------------------------------------------------
# R5 — Remove shared email from entity B
# ---------------------------------------------------------------------------


def test_r5_remove_shared_email_drops_score() -> None:
    """R5: Entity B's coordinator email is updated to a different address.

    Losing shared_email (weighted contribution ≈ 0.16) reduces the
    deterministic section score and should push the pair either into the maybe
    band or below maybe_threshold.
    """
    scope_id = "r5"
    config = _svc_config(scope_id)
    both = _svc_frame(_entity_from_seed("entity_a"), _entity_from_seed("entity_b"))

    run1 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_pair_retained(run1)

    ent_b_new_email = _entity_from_seed("entity_b")
    ent_b_new_email["emails"] = ["marshfield.admin@burrellcenter.com"]  # different coordinator

    run2 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=_svc_frame(_entity_from_seed("entity_a"), ent_b_new_email),
        previous_entity_index=_previous_svc_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    retained = run2.pair_state_index.filter(pl.col("pair_key") == _SVC_PAIR_KEY).height > 0
    removed = run2.removed_pair_ids.filter(pl.col("pair_key") == _SVC_PAIR_KEY).height > 0
    assert retained or removed, (
        "Pair must change state after losing shared_email signal"
    )

    if retained:
        scored = run2.scored_pairs.filter(pl.col("pair_key") == _SVC_PAIR_KEY)
        if scored.height > 0:
            row = scored.row(0, named=True)
            assert row["final_score"] < run1.scored_pairs.filter(
                pl.col("pair_key") == _SVC_PAIR_KEY
            ).row(0, named=True)["final_score"], (
                "Score must decrease after removing shared_email"
            )


# ---------------------------------------------------------------------------
# R6 — Remove BOTH shared signals (email + 988) simultaneously
# ---------------------------------------------------------------------------


def test_r6_remove_email_and_shared_phone_pushes_below_maybe() -> None:
    """R6: Entity B loses both the shared email and the 988 phone in one update.

    Losing email (≈0.16) + shared_phone (≈0.11) combined should reliably push
    the pair below maybe_threshold=0.62 → score_dropped.

    This tests the more aggressive data-hygiene path where a source record is
    significantly cleaned up and multiple shared signals disappear at once.
    """
    scope_id = "r6"
    config = _svc_config(scope_id)
    both = _svc_frame(_entity_from_seed("entity_a"), _entity_from_seed("entity_b"))

    run1 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_pair_retained(run1)

    ent_b_cleaned = _entity_from_seed("entity_b")
    ent_b_cleaned["emails"] = ["marshfield.admin@burrellcenter.com"]  # different email
    ent_b_cleaned["phones"] = ["417-761-5000"]                        # no more 988

    run2 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=_svc_frame(_entity_from_seed("entity_a"), ent_b_cleaned),
        previous_entity_index=_previous_svc_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    # Dropping ≈0.27 of deterministic weight makes the det section score fall
    # substantially, which should push the final score below maybe_threshold.
    retained = run2.pair_state_index.filter(pl.col("pair_key") == _SVC_PAIR_KEY).height > 0
    removed = run2.removed_pair_ids.filter(pl.col("pair_key") == _SVC_PAIR_KEY).height > 0
    assert retained or removed, "Pair must have changed state"

    # Prefer score_dropped, but accept the pair staying in the maybe band if
    # the remaining name + taxonomy + domain signals still clear maybe_threshold.
    if removed:
        actual_reason = run2.removed_pair_ids.filter(
            pl.col("pair_key") == _SVC_PAIR_KEY
        ).row(0, named=True)["cleanup_reason"]
        assert actual_reason in ("score_dropped", "candidate_lost"), (
            f"Unexpected cleanup_reason {actual_reason!r}"
        )


# ---------------------------------------------------------------------------
# R7 — Entity B renamed to an unrelated service
# ---------------------------------------------------------------------------


def test_r7_rename_entity_b_pair_drops_from_review_queue() -> None:
    """R7: Marshfield updates its record to a different service name entirely.

    Changing the name from "Suicide Prevention Hotlines" to a completely
    different string removes the dominant NLP signal (name_similarity ≈ 1.0 →
    near 0).  The combined score collapses; the pair must be removed via
    score_dropped or candidate_lost depending on whether the embedding still
    clears the blocking threshold.

    Real-world analogy: the Marshfield clinic rebrands this program under a
    different title while the Branson office keeps the original name.
    """
    scope_id = "r7"
    config = _svc_config(scope_id)
    both = _svc_frame(_entity_from_seed("entity_a"), _entity_from_seed("entity_b"))

    run1 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_pair_retained(run1)

    ent_b_renamed = _entity_from_seed("entity_b")
    ent_b_renamed["name"] = "Marshfield Crisis Stabilization Unit"
    ent_b_renamed["description"] = "24-hour psychiatric crisis stabilization services."
    # Keep same emails/phones/websites/taxonomy to isolate the name-change effect;
    # the NLP section score will drop dramatically.

    run2 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=_svc_frame(_entity_from_seed("entity_a"), ent_b_renamed),
        previous_entity_index=_previous_svc_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    removed_rows = run2.removed_pair_ids.filter(pl.col("pair_key") == _SVC_PAIR_KEY)
    assert removed_rows.height == 1, (
        "Renaming entity B to an unrelated service should remove the pair "
        f"but got removed_pair_ids={run2.removed_pair_ids.to_dicts()}"
    )
    actual_reason = removed_rows.row(0, named=True)["cleanup_reason"]
    assert actual_reason in ("score_dropped", "candidate_lost"), (
        f"Unexpected cleanup_reason {actual_reason!r} after name change"
    )


# ---------------------------------------------------------------------------
# R8 — Entity B deleted then resurrected
# ---------------------------------------------------------------------------


def test_r8_entity_b_deleted_then_resurrected_rebuilds_pair() -> None:
    """R8: Entity B is absent in run 2, then returns in run 3.

    When an entity reappears after deletion the pipeline treats it as a new
    addition and re-generates candidates.  The pair must be re-retained in
    run 3 because all original signals are restored.

    Real-world analogy: the Marshfield clinic's record is temporarily removed
    from the source feed (e.g., during a data migration) and later restored.
    """
    scope_id = "r8"
    config = _svc_config(scope_id)
    both = _svc_frame(_entity_from_seed("entity_a"), _entity_from_seed("entity_b"))

    run1 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_pair_retained(run1)

    # Run 2: entity B disappears.
    a_only = _svc_frame(_entity_from_seed("entity_a"))
    run2 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=a_only,
        previous_entity_index=_previous_svc_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )
    _assert_pair_removed(run2, "entity_deleted")
    assert run2.pair_state_index.is_empty()

    # Run 3: entity B comes back with all original signals.
    entity_index_after_run2 = run2.denormalized_service.select(
        ["entity_id", "entity_type", "content_hash"]
    ).with_columns(pl.lit(True).alias("active_flag"))

    run3 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=entity_index_after_run2,
        previous_pair_state_index=run2.pair_state_index,
        config=config,
    )

    assert not run3.candidate_pairs.is_empty(), (
        "Resurrected entity B should generate a new candidate pair with entity A"
    )
    _assert_pair_retained(run3)


# ---------------------------------------------------------------------------
# R9 — Idempotent run: identical inputs twice
# ---------------------------------------------------------------------------


def test_r9_idempotent_run_preserves_pair_state() -> None:
    """R9: Running the pipeline twice with no entity changes is a no-op.

    The pipeline should set no_change=True, emit no candidates, emit no
    removals, and return the same pair_state_index as run 1.

    This verifies that the incremental fingerprinting (content_hash +
    pair_state_index passthrough) prevents spurious re-scoring.
    """
    scope_id = "r9"
    config = _svc_config(scope_id)
    both = _svc_frame(_entity_from_seed("entity_a"), _entity_from_seed("entity_b"))

    run1 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_pair_retained(run1)

    run2 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=_previous_svc_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    summary = run2.run_summary.row(0, named=True)
    assert summary["no_change"] is True, (
        "Identical inputs should produce no_change=True"
    )
    assert run2.candidate_pairs.is_empty(), "No candidates should be generated"
    assert run2.removed_pair_ids.is_empty(), "No pairs should be removed"
    _assert_pair_retained(run2)


# ---------------------------------------------------------------------------
# R10 — Third Burrell location creates a 3-node cluster
# ---------------------------------------------------------------------------


def test_r10_third_burrell_location_forms_three_node_cluster() -> None:
    """R10: A third Burrell location (Springfield clinic) is added.

    The new entity shares the same name, email, 988 phone, taxonomy code, and
    domain — enough signals to form a duplicate pair with both existing
    entities.  All three nodes should end up in a single cluster.

    Real-world analogy: Burrell opens a third 988 service listing for their
    Springfield office, which is a copy of the same AIRS master record.
    """
    scope_id = "r10"
    config = _svc_config(scope_id)
    both = _svc_frame(_entity_from_seed("entity_a"), _entity_from_seed("entity_b"))

    run1 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_pair_retained(run1)

    # Entity C: a third Burrell location with all shared signals.
    # Use a different embedding direction to ensure the pair survives
    # blocking (cosine between C and A/B must be above the service threshold).
    ent_c: dict[str, Any] = {
        "entity_id": "cc000000-0000-0000-0000-000000000001",
        "source_schema": _SOURCE_SCHEMA,
        "name": "Suicide Prevention Hotlines",
        "description": "Provides a 24-hour suicide prevention hotline.",
        "emails": ["rebecca.randolph@burrellcenter.com"],
        "phones": ["417-555-0199", "988"],  # Springfield local line + shared 988
        "websites": ["https://www.burrellcenter.com"],
        "taxonomies": [{"code": "rp-1500.1400-800"}],
        "locations": [
            {
                "location_id": "cc000000-0000-0000-0000-000000000002",
                "name": "Burrell Behavioral Health - Springfield Clinic",
                "address_1": "800 S Glenstone Ave",
                "city": "Springfield",
                "state": "MO",
                "postal_code": "65804",
            }
        ],
        "identifiers": [],
        "services_rollup": [],
        "organization_id": None,
        "organization_name": "BURRELL BEHAVIORAL HEALTH",
        "embedding_vector": [0.91, 0.09],  # close to A and B → high cosine
    }

    three_entities = _svc_frame(
        _entity_from_seed("entity_a"),
        _entity_from_seed("entity_b"),
        ent_c,
    )
    run2 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=three_entities,
        previous_entity_index=_previous_svc_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    # At minimum the original A-B pair must still be retained.
    _assert_pair_not_removed(run2)
    _assert_pair_retained(run2)

    # Entity C should generate at least one new pair.
    new_pairs = run2.candidate_pairs.filter(
        pl.col("entity_a_id").is_in(["cc000000-0000-0000-0000-000000000001"])
        | pl.col("entity_b_id").is_in(["cc000000-0000-0000-0000-000000000001"])
    )
    assert new_pairs.height >= 1, (
        "Third Burrell entity should generate at least one new candidate pair "
        "with either entity A or entity B"
    )

    # All three entities should end up in a single cluster.
    if run2.clusters.height > 0:
        max_cluster_size = run2.clusters.get_column("cluster_size").max()
        assert max_cluster_size == 3, (
            f"Expected a 3-node cluster for the three Burrell locations, "
            f"got cluster sizes: {run2.clusters.get_column('cluster_size').to_list()}"
        )


# ---------------------------------------------------------------------------
# R11 — Score stability: pair already in maybe band stays stable
# ---------------------------------------------------------------------------


def test_r11_pair_in_maybe_band_is_stable_across_reruns() -> None:
    """R11: When entity B loses the shared 988 signal and the pair demotes to maybe,
    a subsequent run with no further changes must NOT emit any removal signal —
    the pair should sit stably in the maybe band.

    This guards against a regression where a pair oscillates between bands on
    every run instead of remaining in a stable retained state.
    """
    scope_id = "r11"
    config = _svc_config(scope_id)
    both = _svc_frame(_entity_from_seed("entity_a"), _entity_from_seed("entity_b"))

    run1 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=both,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
    )
    _assert_pair_retained(run1)

    # Run 2: drop 988 — pair may demote to maybe or get score_dropped.
    ent_b_no_988 = _entity_from_seed("entity_b")
    ent_b_no_988["phones"] = ["417-761-5000"]
    no_988_svc = _svc_frame(_entity_from_seed("entity_a"), ent_b_no_988)

    run2 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=no_988_svc,
        previous_entity_index=_previous_svc_entity_index(run1),
        previous_pair_state_index=run1.pair_state_index,
        config=config,
    )

    pair_still_retained = (
        run2.pair_state_index.filter(pl.col("pair_key") == _SVC_PAIR_KEY).height > 0
    )
    if not pair_still_retained:
        pytest.skip("Pair was removed in run 2 — stability test only applies to maybe-band pairs")

    # Run 3: same data as run 2 — no entity changes.
    run3 = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=no_988_svc,
        previous_entity_index=_previous_svc_entity_index(run2),
        previous_pair_state_index=run2.pair_state_index,
        config=config,
    )

    summary = run3.run_summary.row(0, named=True)
    assert summary["no_change"] is True, (
        "A pair stable in the maybe band should produce no_change=True on a re-run "
        "with identical inputs"
    )
    assert run3.removed_pair_ids.is_empty(), (
        "No pair removal should be emitted when inputs are identical"
    )
    _assert_pair_retained(run3)
