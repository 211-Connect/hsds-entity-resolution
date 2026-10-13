"""Tests for the judge state's record evidence (2.4.0).

Assurance date and assurer, emails, organization identifiers, service area as level
and place, schedules with a scope tag, and the organization link.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args

import polars as pl
import pytest

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.pipeline import run_incremental
from hsds_entity_resolution.judge import (
    MAX_PLACES_PER_LEVEL,
    SMALLEST_MODEL_STATE_BUDGET_TOKENS,
    AssuranceRecency,
    IdentifierState,
    PairAnswers,
    PairState,
    ReferenceJudge,
    ScheduleState,
    ServiceAreaState,
    build_pair_state,
    build_record_state,
    compare_assurance,
    is_low_information_schedule,
)

_NOT_MATCH_EVIDENCE = "assurance dates are not evidence of whether the records match"
_FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_hotline_pair.json"


def _row(entity_id: str, **overrides: Any) -> dict[str, Any]:
    """Return a minimal clean-entity-shaped service row."""
    row: dict[str, Any] = {
        "entity_id": entity_id,
        "entity_type": "service",
        "source_schema": "source_a",
        "name": "food pantry",
        "description": "weekly groceries",
        "emails": [],
        "phones": [],
        "websites": [],
        "locations": [],
        "taxonomies": [],
        "identifiers": [],
        "organization_name": "Example Org",
        "embedding_vector": [0.9, 0.1],
    }
    row.update(overrides)
    return row


def _pair(entity_type: str = "service", **sides: dict[str, Any]) -> PairState:
    """Build a pair state from ``a`` and ``b`` row overrides."""
    return build_pair_state(
        pair_key="a__b",
        entity_type=entity_type,
        entity_a=_row("a", entity_type=entity_type, **sides.get("a", {})),
        entity_b=_row("b", entity_type=entity_type, **sides.get("b", {})),
        pair_outcome=None,
    )


# --- Assurance -----------------------------------------------------------------------


def test_assurance_date_and_assurer_render_per_record() -> None:
    """The assurance date renders as a date, the assurer as a normalised email."""
    record = build_record_state(
        _row("a", assured_date="2025-03-04T10:15:00", assurer_email=" Staff@Example.ORG ")
    )
    assert record.assured_date == "2025-03-04"
    assert record.assurer_email == "staff@example.org"
    assert build_record_state(_row("b", assured_date="spring 2025")).assured_date == "spring 2025"


@pytest.mark.parametrize(
    ("assured_a", "assured_b", "newer", "outdated"),
    [
        ("2025-06-01", "2023-01-15", "record_a", "record_b"),
        ("2023-01-15", "2025-06-01 08:00:00", "record_b", "record_a"),
        ("2025-06-01", "", "record_a", "record_b"),
        ("", "2025-06-01", "record_b", "record_a"),
    ],
)
def test_assurance_recency_names_the_outdated_record_never_a_match(
    assured_a: str, assured_b: str, newer: str, outdated: str
) -> None:
    """Recency says which record may be outdated, and says it is not match evidence."""
    state = _pair(a={"assured_date": assured_a}, b={"assured_date": assured_b})
    recency = state.to_dict()["assurance_recency"]
    assert recency.startswith(newer) or recency.startswith(f"only {newer}")
    assert f"so {outdated} may be the outdated one" in recency
    assert recency.endswith(_NOT_MATCH_EVIDENCE)


def test_every_recency_sentence_disclaims_match_evidence() -> None:
    """No recency wording, now or later, may read as evidence for or against a match."""
    sentences: tuple[str, ...] = get_args(AssuranceRecency)
    assert len(sentences) == 5
    for sentence in sentences:
        assert sentence.endswith(_NOT_MATCH_EVIDENCE), sentence
        for loaded in ("duplicate", "same thing", "likely", "different"):
            assert loaded not in sentence, sentence
    assert compare_assurance("2025-01-01", "2025-01-01") == (
        f"both records were assured on the same day; {_NOT_MATCH_EVIDENCE}"
    )


def test_no_assurance_dates_send_no_recency() -> None:
    """With no parseable date on either side, the state carries no recency key."""
    state = _pair(a={"assured_date": "unknown"})
    assert state.assurance_recency is None
    assert "assurance_recency" not in state.to_dict()


# --- Emails and identifiers ----------------------------------------------------------


def test_service_emails_reach_the_state() -> None:
    """Emails are normalised and de-duplicated, in first-seen order."""
    record = build_record_state(
        _row("a", emails=["mailto:Intake@Example.org", "intake@example.org", " ", "b@x.org"])
    )
    assert record.emails == ("intake@example.org", "b@x.org")


def test_organization_identifiers_reach_the_state() -> None:
    """Identifiers keep their type and value; blank values are dropped."""
    record = build_record_state(
        _row(
            "a",
            entity_type="organization",
            identifiers=[
                {"identifier_type": "EIN", "identifier": " 12-3456789 "},
                {"identifier_scheme": "US-EIN", "identifier": "98-7654321"},
                {"identifier_type": "EIN", "identifier": ""},
                {"identifier_type": "EIN", "identifier": "12-3456789"},
            ],
        )
    )
    assert record.identifiers == (
        IdentifierState(type="EIN", identifier="12-3456789"),
        IdentifierState(type="US-EIN", identifier="98-7654321"),
    )


# --- Service area --------------------------------------------------------------------


def test_service_area_reads_level_and_place_not_the_free_text_name() -> None:
    """Areas come from structured level and place, coarsest first; names are ignored."""
    record = build_record_state(
        _row(
            "a",
            service_areas=[
                {"level": "ZIP", "place": "00102"},
                {"level": "county", "place": "Example County, ZZ"},
                {"name": "Serves all of Example County and beyond"},
                {"level": "city", "name": "Serves all of Example County and beyond"},
                {"level": "neighborhood", "place": "Old Town"},
                {"level": "state", "place": " ZZ "},
                {"level": "zip_code", "place": "00101"},
            ],
        )
    )
    assert record.service_areas == (
        ServiceAreaState(level="state", places=("ZZ",), omitted=0),
        ServiceAreaState(level="county", places=("Example County, ZZ",), omitted=0),
        ServiceAreaState(level="postal_code", places=("00101", "00102"), omitted=0),
    )
    assert "Serves all of" not in json.dumps(record.service_areas, default=str)


def test_usa_placeholder_on_a_county_level_record_reads_as_unknown() -> None:
    """A "USA" row beside narrower areas is a source default and is dropped."""
    county = build_record_state(
        _row(
            "a",
            service_areas=[
                {"level": "national", "place": "USA"},
                {"level": "county", "place": "Example County, ZZ"},
            ],
        )
    )
    assert county.service_areas == (
        ServiceAreaState(level="county", places=("Example County, ZZ",), omitted=0),
    )
    only_name = build_record_state(_row("b", service_areas=[{"name": "USA"}]))
    assert only_name.service_areas == ()
    national = build_record_state(
        _row("c", service_areas=[{"level": "national", "place": "U.S.A."}])
    )
    assert national.service_areas == (
        ServiceAreaState(level="national", places=("United States",), omitted=0),
    )


def test_service_area_caps_places_per_level() -> None:
    """Hundreds of postal codes render as the first ones and a count."""
    zips = [{"level": "postal_code", "place": f"{code:05d}"} for code in range(300, 0, -1)]
    (area,) = build_record_state(_row("a", service_areas=zips)).service_areas
    assert len(area.places) == MAX_PLACES_PER_LEVEL
    assert area.places[0] == "00001"
    assert area.omitted == 300 - MAX_PLACES_PER_LEVEL


# --- Schedules -----------------------------------------------------------------------


def test_schedules_carry_text_and_scope() -> None:
    """Each schedule keeps its English text and is tagged service or location."""
    record = build_record_state(
        _row(
            "a",
            schedules=[
                {"description": " Mon-Fri  9am-5pm ", "scope": "service"},
                {"description": "Tue 1pm-4pm", "scope": "LOCATION"},
                {"description": "Sat 10am-noon", "location_id": "loc-1"},
                {"text": "Sun closed"},
                {"description": "", "scope": "service"},
            ],
        )
    )
    assert record.schedules == (
        ScheduleState(scope="service", text="Mon-Fri 9am-5pm", low_information=False),
        ScheduleState(scope="location", text="Tue 1pm-4pm", low_information=False),
        ScheduleState(scope="location", text="Sat 10am-noon", low_information=False),
        ScheduleState(scope="unknown", text="Sun closed", low_information=False),
    )


@pytest.mark.parametrize(
    "text",
    [
        "Same as site hours",
        "Same as agency hours.",
        "Varies",
        "Hours vary by location",
        "Call for hours",
        "See website",
        "N/A",
    ],
)
def test_placeholder_schedules_are_low_information(text: str) -> None:
    """Text that names no hours is marked low-information, and still shown."""
    (schedule,) = build_record_state(
        _row("a", schedules=[{"description": text, "scope": "service"}])
    ).schedules
    assert schedule.text == text
    assert schedule.low_information is True


@pytest.mark.parametrize(
    "text",
    ["Mon-Fri 9am-5pm", "24 hours, 7 days a week", "By appointment only", "Varies: Mon 9-5"],
)
def test_real_hours_are_not_low_information(text: str) -> None:
    """Schedules that name hours keep full weight."""
    assert is_low_information_schedule(text) is False


# --- Organization link ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("entity_type", "side_a", "side_b", "expected"),
    [
        (
            "service",
            {"organization_id": "ORG-1"},
            {"organization_id": "org-1"},
            "both records belong to the same organization record",
        ),
        ("service", {"organization_id": "org-1"}, {"organization_id": "org-2"}, None),
        (
            "organization",
            {},
            {"parent_organization_id": "A"},
            "record_a is record_b's parent organization",
        ),
        (
            "organization",
            {"parent_organization_id": "b"},
            {},
            "record_b is record_a's parent organization",
        ),
        (
            "organization",
            {"parent_organization_id": "p"},
            {"parent_organization_id": "p"},
            "both organizations have the same parent organization",
        ),
        ("organization", {}, {}, None),
    ],
)
def test_organization_link_flags_a_shared_or_parent_organization(
    entity_type: str, side_a: dict[str, Any], side_b: dict[str, Any], expected: str | None
) -> None:
    """Ids alone link organizations; with no link the key is left out."""
    state = _pair(entity_type, a=side_a, b=side_b)
    assert state.organization_link == expected
    assert ("organization_link" in state.to_dict()) is (expected is not None)


# --- Compatibility, pipeline and size ------------------------------------------------


def test_callers_without_the_new_columns_get_empty_evidence() -> None:
    """A row without any 2.4.0 column renders empty evidence and no pair sentences."""
    state = _pair()
    record = state.to_dict()["record_a"]
    for key in ("emails", "identifiers", "service_areas", "schedules"):
        assert record[key] == ()
    assert record["assured_date"] == ""
    assert record["assurer_email"] == ""
    assert {"assurance_recency", "organization_link"}.isdisjoint(state.to_dict())


class _CapturingJudge(ReferenceJudge):
    """A reference judge that keeps the states it was shown."""

    def __init__(self) -> None:
        self.states: list[PairState] = []

    def judge(self, states: Sequence[PairState]) -> list[PairAnswers]:
        """Record the states, then answer uninformatively."""
        self.states.extend(states)
        return super().judge(states)


def test_new_columns_survive_the_clean_stage_into_the_state() -> None:
    """Schedules, service areas and parent ids pass through clean_entities to the judge."""
    extra = {
        "assured_date": "2025-06-01",
        "assurer_email": "staff@example.org",
        "emails": ["intake@example.org"],
        "schedules": [{"description": "Same as site hours", "scope": "location"}],
        "service_areas": [{"level": "county", "place": "Example County, ZZ"}],
        "organization_id": "org-1",
    }
    services = pl.DataFrame(
        [
            _row("svc-a", **extra),
            _row("svc-b", **{**extra, "assured_date": "2024-01-01"}, embedding_vector=[0.92, 0.08]),
        ]
    )
    judge = _CapturingJudge()
    run_incremental(
        organization_entities=pl.DataFrame(),
        service_entities=services,
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
        config=EntityResolutionRunConfig.defaults_for_entity_type(
            team_id="team", scope_id="scope", entity_type="service"
        ),
        judge=judge,
    )
    (state,) = judge.states
    for record in (state.record_a, state.record_b):
        assert record.emails == ("intake@example.org",)
        assert record.assurer_email == "staff@example.org"
        assert record.schedules == (
            ScheduleState(scope="location", text="Same as site hours", low_information=True),
        )
        assert record.service_areas == (
            ServiceAreaState(level="county", places=("Example County, ZZ",), omitted=0),
        )
    assert state.organization_link == "both records belong to the same organization record"
    assert state.assurance_recency is not None
    assert state.assurance_recency.endswith(_NOT_MATCH_EVIDENCE)


def large_realistic_service_row(entity_id: str) -> dict[str, Any]:
    """A service row at the heavy end of what 211 sources hold, every field filled.

    The fixture pair is small; this row is built so the size check covers a record
    with long text, many sites and taxonomies, and a postal-code service area.
    """
    fixture = json.loads(_FIXTURE.read_text())["entity_a"]
    paragraph = fixture["description"] + " " + "Crisis counselors answer calls and texts. " * 10
    return _row(
        entity_id,
        display_name=fixture["name"],
        display_description=(paragraph + "\n") * 8,
        alternate_name="Crisis Line",
        short_description=paragraph,
        eligibility_description="Anyone in crisis. " * 30,
        fees_description="Free. " * 10,
        application_process="Call, text or chat. " * 20,
        emails=[f"team{n}@riverbend.example.org" for n in range(5)],
        phones=[f"555010{n:04d}" for n in range(12)],
        websites=[f"https://www.riverbend.example.org/page/{n}" for n in range(5)],
        locations=[
            {
                "name": f"Riverbend Behavioral Health - Clinic {n}",
                "location_type": "physical",
                "address_1": f"{100 + n} Example Way",
                "city": "Westfield",
                "state": "ZZ",
                "postal_code": f"{101 + n:05d}",
            }
            for n in range(20)
        ],
        taxonomies=[{"code": f"rp-{n}", "name": f"Taxonomy term {n}"} for n in range(25)],
        identifiers=[{"identifier_type": "EIN", "identifier": "12-3456789"}],
        organization_name=fixture["organization_name"],
        organization_description="Behavioral health provider. " * 40,
        assured_date="2025-06-01",
        assurer_email="staff@example.org",
        service_areas=[
            {"level": "national", "place": "USA"},
            *({"level": "county", "place": f"County {n}, ZZ"} for n in range(12)),
            *({"level": "postal_code", "place": f"{n:05d}"} for n in range(400)),
        ],
        schedules=[
            *(
                {"description": f"Clinic {n}: Mon-Fri 8am-6pm", "scope": "location"}
                for n in range(20)
            ),
            {"description": "24 hours, 7 days a week", "scope": "service"},
            {"description": "Same as site hours", "scope": "service"},
        ],
    )


def test_a_large_realistic_pair_fits_the_smallest_budget() -> None:
    """The heaviest realistic pair stays well inside the smallest model's 32k budget."""
    state = build_pair_state(
        pair_key="a__b",
        entity_type="service",
        entity_a=large_realistic_service_row("a"),
        entity_b=large_realistic_service_row("b"),
        pair_outcome=None,
        source_profile_a="Profile text. " * 50,
        source_profile_b="Profile text. " * 50,
    )
    assert state.estimated_tokens() < SMALLEST_MODEL_STATE_BUDGET_TOKENS // 2
