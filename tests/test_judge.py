"""Tests for the Pair Judge contract: state builder, answers schema, stage and pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.pipeline import run_incremental
from hsds_entity_resolution.judge import (
    ORGANIZATION_RELATIONS,
    PairAnswers,
    PairJudge,
    PairState,
    ReferenceJudge,
    SiteState,
    answers_from_frame,
    answers_to_frame,
    build_pair_state,
    compare_sites,
    judge_scored_pairs,
)
from hsds_entity_resolution.types.frames import JUDGE_ANSWERS_SCHEMA

_UNIFORM = dict.fromkeys(ORGANIZATION_RELATIONS, 0.2)


def _service_row(entity_id: str, **overrides: Any) -> dict[str, Any]:
    """Return one clean-entity-shaped service row."""
    row: dict[str, Any] = {
        "entity_id": entity_id,
        "entity_type": "service",
        "source_schema": "source_a",
        "name": "food pantry",
        "display_name": "Food Pantry",
        "description": "weekly groceries",
        "display_description": "Weekly groceries for residents.",
        "alternate_name": "Community Pantry",
        "short_description": "Groceries",
        "eligibility_description": "Residents of the county",
        "fees_description": "Free",
        "application_process": "Walk in",
        "emails": ["pantry@example.org"],
        "phones": ["‪(555) 010-0199‬", "5550100199 ext. 4"],
        "websites": ["WWW.Example.org/Pantry/"],
        "locations": [
            {
                "address_1": "100 Example Way.",
                "city": "Westfield",
                "state": "ZZ",
                "postal_code": "00101-1234",
            }
        ],
        "taxonomies": [{"code": "BD-1800.2000", "name": "Food Pantries"}],
        "identifiers": [],
        "services_rollup": [],
        "organization_name": "Example Org",
        "organization_id": "org-1",
        "embedding_vector": [0.9, 0.1],
    }
    row.update(overrides)
    return row


def _site(address: str, city: str, state: str, postal_code: str) -> SiteState:
    """An unnamed, untyped site with the given address components."""
    return SiteState(
        name="", location_type="", address=address, city=city, state=state, postal_code=postal_code
    )


def test_sites_carry_their_name_and_location_type() -> None:
    """A site's name and HSDS location type reach the judge; matching is unchanged."""
    state = build_pair_state(
        pair_key="a__b",
        entity_type="service",
        entity_a=_service_row(
            "a",
            locations=[
                {
                    "name": " Main Office ",
                    "location_type": " Physical ",
                    "address_1": "100 Example Way.",
                    "city": "Westfield",
                    "postal_code": "00101",
                }
            ],
        ),
        entity_b=_service_row("b"),
        pair_outcome="maybe",
    )

    (site,) = state.record_a.sites
    assert site.name == "Main Office"
    assert site.location_type == "physical"
    assert state.site_comparison == "same address"


def test_a_state_without_a_prior_score_sends_no_trace_of_it() -> None:
    """``pair_outcome=None`` leaves the previous matcher's verdict out of the state."""
    state = build_pair_state(
        pair_key="a__b",
        entity_type="service",
        entity_a=_service_row("a"),
        entity_b=_service_row("b"),
        pair_outcome=None,
    )

    assert state.prior_score is None
    assert "prior_score" not in state.to_dict()
    assert "previous matcher" not in str(state.to_dict())


def test_state_fills_hsds_fields_and_leaves_profile_slots_empty() -> None:
    """The engine fills HSDS text, computes site and prior words, and leaves profiles empty."""
    state = build_pair_state(
        pair_key="a__b",
        entity_type="service",
        entity_a=_service_row("a"),
        entity_b=_service_row(
            "b",
            locations=[
                {"address_1": "100  example way", "city": "WESTFIELD", "postal_code": "00101"}
            ],
        ),
        pair_outcome="maybe",
    )
    record = state.record_a
    assert record.name == "Food Pantry"
    assert record.description == "Weekly groceries for residents."
    assert record.alternate_name == "Community Pantry"
    assert record.eligibility == "Residents of the county"
    assert record.fees == "Free"
    assert record.application_process == "Walk in"
    assert record.taxonomies == ("Food Pantries",)
    assert record.phones == ("5550100199", "5550100199 x 4")
    assert record.websites == ("https://www.example.org/Pantry",)
    assert record.sites == (
        SiteState(
            name="",
            location_type="",
            address="100 example way",
            city="westfield",
            state="zz",
            postal_code="00101",
        ),
    )
    assert state.site_comparison == "same address"
    assert state.prior_score == "the previous matcher would send this pair to human review"
    assert state.source_profile_a == ""
    assert state.source_profile_b == ""


def test_state_never_carries_taxonomy_codes() -> None:
    """Codes are for code; only taxonomy names reach the judge."""
    state = build_pair_state(
        pair_key="a__b",
        entity_type="service",
        entity_a=_service_row("a"),
        entity_b=_service_row("b"),
        pair_outcome="duplicate",
    )
    assert "BD-1800.2000" not in str(state.to_dict())


@pytest.mark.parametrize(
    "profile",
    [
        "",
        "Service names in this source are taxonomy terms.\n  Phones live on the location.",
        "  leading and trailing whitespace kept  ",
        "Unicode ‪ kept é verbatim",
    ],
)
def test_source_profile_slots_pass_through_verbatim(profile: str) -> None:
    """Caller-supplied Source Profile text reaches the state byte for byte, per side."""
    scored = pl.DataFrame(
        {
            "pair_key": ["a__b"],
            "entity_a_id": ["a"],
            "entity_b_id": ["b"],
            "entity_type": ["service"],
            "pair_outcome": ["maybe"],
        }
    )
    services = pl.DataFrame(
        [_service_row("a", source_schema="source_a"), _service_row("b", source_schema="source_b")]
    )
    from hsds_entity_resolution.judge import build_pair_states

    (state,) = build_pair_states(
        scored_pairs=scored,
        denormalized_organization=pl.DataFrame(),
        denormalized_service=services,
        source_profiles={"source_a": profile, "source_b": "other"},
    )
    assert state.source_profile_a == profile
    assert state.source_profile_b == "other"


@pytest.mark.parametrize(
    ("sites_a", "sites_b", "expected"),
    [
        ([], [_site("1 a st", "x", "zz", "00001")], "unknown"),
        (
            [_site("1 a st", "x", "zz", "00001")],
            [_site("1 a st", "x", "zz", "00001")],
            "same address",
        ),
        (
            [_site("1 a st", "x", "zz", "00001")],
            [_site("2 b st", "x", "zz", "00001")],
            "same city",
        ),
        (
            [_site("1 a st", "x", "zz", "00001")],
            [_site("1 a st", "y", "zz", "00002")],
            "different",
        ),
    ],
)
def test_compare_sites_words(
    sites_a: list[SiteState], sites_b: list[SiteState], expected: str
) -> None:
    """Site comparison is computed in code and sent as a word."""
    assert compare_sites(sites_a, sites_b) == expected


def _answer(entity_type: str = "service", **overrides: Any) -> PairAnswers:
    """Return a valid answer, optionally overridden."""
    values: dict[str, Any] = {
        "pair_key": "a__b",
        "entity_type": entity_type,
        "model_id": "m",
        "question_set_version": "q1",
        "same_site": 0.9,
        "same_offering": None if entity_type == "organization" else 0.8,
        "physically_delivered": None if entity_type == "organization" else 0.7,
        "organization_relation": {
            "same": 0.1,
            "parent_and_chapter": 0.6,
            "affiliated": 0.1,
            "unrelated": 0.1,
            "cannot_tell": 0.1,
        },
    }
    values.update(overrides)
    return PairAnswers(**values)


def test_answers_round_trip_through_the_artifact_schema(tmp_path: Path) -> None:
    """Answers survive frame conversion and a Parquet write/read without reshaping."""
    answers = [_answer("service"), _answer("organization", pair_key="c__d")]
    frame = answers_to_frame(answers)
    assert frame.schema == pl.Schema(JUDGE_ANSWERS_SCHEMA)
    assert frame.get_column("organization_relation").to_list() == [
        "parent_and_chapter",
        "parent_and_chapter",
    ]
    path = tmp_path / "judge_answers.parquet"
    frame.write_parquet(path)
    restored = pl.read_parquet(path)
    assert restored.schema == frame.schema
    assert answers_from_frame(restored) == answers


@pytest.mark.parametrize(
    ("entity_type", "overrides", "message"),
    [
        ("service", {"same_site": 1.5}, "same_site must be a probability"),
        ("service", {"organization_relation": {"same": 1.0}}, "exactly"),
        (
            "service",
            {"organization_relation": dict.fromkeys(ORGANIZATION_RELATIONS, 0.5)},
            "sum to 1",
        ),
        ("organization", {"same_offering": 0.5}, "organization pairs carry no"),
        ("service", {"physically_delivered": None}, "service pairs need"),
    ],
)
def test_answers_reject_invalid_values(
    entity_type: str, overrides: dict[str, Any], message: str
) -> None:
    """Answer validation names the broken rule."""
    with pytest.raises(ValueError, match=message):
        _answer(entity_type, **overrides)


def test_reference_judge_is_uninformative_and_satisfies_the_protocol() -> None:
    """The reference judge answers 0.5 everywhere and a uniform relation."""
    judge = ReferenceJudge()
    assert isinstance(judge, PairJudge)
    state = build_pair_state(
        pair_key="a__b",
        entity_type="organization",
        entity_a=_service_row("a", entity_type="organization"),
        entity_b=_service_row("b", entity_type="organization"),
        pair_outcome="below_maybe",
    )
    (answer,) = judge.judge([state])
    assert answer.same_site == 0.5
    assert answer.same_offering is None
    assert answer.physically_delivered is None
    assert answer.organization_relation == _UNIFORM


def _seed_frames() -> tuple[pl.DataFrame, EntityResolutionRunConfig]:
    """Return two near-identical services that always score as a candidate pair."""
    services = pl.DataFrame(
        [_service_row("svc-a"), _service_row("svc-b", embedding_vector=[0.92, 0.08])]
    )
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team", scope_id="scope", entity_type="service"
    )
    return services, config


def test_run_incremental_emits_answers_beside_scored_pairs() -> None:
    """With a judge, every scored pair gets one answer and the bundle carries them."""
    services, config = _seed_frames()
    kwargs: dict[str, Any] = {
        "organization_entities": pl.DataFrame(),
        "service_entities": services,
        "previous_entity_index": pl.DataFrame(),
        "previous_pair_state_index": pl.DataFrame(),
        "config": config,
    }
    without = run_incremental(**kwargs, judge=None)
    judged = run_incremental(
        **kwargs, judge=ReferenceJudge(), source_profiles={"source_a": "profile text"}
    )

    assert judged.scored_pairs.height >= 1
    assert sorted(judged.judge_answers.get_column("pair_key").to_list()) == sorted(
        judged.scored_pairs.get_column("pair_key").to_list()
    )
    assert judged.judge_answers.schema == pl.Schema(JUDGE_ANSWERS_SCHEMA)
    assert judged.persistence_artifact_bundle["judge_answers"].equals(judged.judge_answers)
    assert judged.scored_pairs.equals(without.scored_pairs)
    assert judged.clusters.equals(without.clusters)
    assert without.judge_answers.is_empty()
    assert "judge_answers" not in without.persistence_artifact_bundle


class _OverBudgetJudge(ReferenceJudge):
    """A judge whose budget no real state fits."""

    state_token_budget = 10


class _DroppingJudge(ReferenceJudge):
    """A judge that loses an answer."""

    def judge(self, states: Sequence[PairState]) -> list[PairAnswers]:
        """Return one answer fewer than asked."""
        return super().judge(states)[:-1]


@pytest.mark.parametrize(
    ("judge", "message"),
    [
        (_OverBudgetJudge(), "exceed the judge's 10-token budget"),
        (_DroppingJudge(), "returned 0 answers"),
    ],
)
def test_judge_stage_rejects_oversized_states_and_lost_answers(
    judge: PairJudge, message: str
) -> None:
    """The stage refuses states over budget and answers that do not match one for one."""
    services, config = _seed_frames()
    result = run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=services,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=config,
        judge=None,
    )
    with pytest.raises(ValueError, match=message):
        judge_scored_pairs(
            scored_pairs=result.scored_pairs.head(1),
            denormalized_organization=result.denormalized_organization,
            denormalized_service=result.denormalized_service,
            judge=judge,
            source_profiles=None,
        )
