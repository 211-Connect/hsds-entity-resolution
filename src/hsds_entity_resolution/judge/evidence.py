"""Record evidence beyond the HSDS text: identifiers, assurance, service area, schedules.

Each builder reads one optional clean-entity column and returns compact, judge-ready
values. Every column is optional: a record without it renders an empty value, so
callers that do not supply it keep working. Two pair-level facts are computed in code
and sent as sentences, like the site comparison:

* :func:`compare_assurance` names which record was assured more recently. Recency says
  which record may be outdated; the sentence says outright that it is not evidence of
  whether the two records match.
* :func:`link_organizations` names a shared or parent organization when the ids say so.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, cast

ServiceAreaLevel = Literal["national", "state", "county", "city", "postal_code"]
ScheduleScope = Literal["service", "location", "unknown"]
AssuranceRecency = Literal[
    "record_a was assured more recently than record_b, so record_b may be the outdated one; "
    "assurance dates are not evidence of whether the records match",
    "record_b was assured more recently than record_a, so record_a may be the outdated one; "
    "assurance dates are not evidence of whether the records match",
    "both records were assured on the same day; "
    "assurance dates are not evidence of whether the records match",
    "only record_a has an assurance date, so record_b may be the outdated one; "
    "assurance dates are not evidence of whether the records match",
    "only record_b has an assurance date, so record_a may be the outdated one; "
    "assurance dates are not evidence of whether the records match",
]
OrganizationLink = Literal[
    "both records belong to the same organization record",
    "both organizations have the same parent organization",
    "record_a is record_b's parent organization",
    "record_b is record_a's parent organization",
]

# Coarsest first: the order levels are listed in a record's state.
_LEVELS: tuple[ServiceAreaLevel, ...] = ("national", "state", "county", "city", "postal_code")
_LEVEL_ALIASES: dict[str, ServiceAreaLevel] = {
    "national": "national",
    "nation": "national",
    "country": "national",
    "state": "state",
    "county": "county",
    "city": "city",
    "postal_code": "postal_code",
    "postal": "postal_code",
    "zip": "postal_code",
    "zip_code": "postal_code",
    "zipcode": "postal_code",
}
# A national row naming no more than the country: on a record that also lists narrower
# areas it is a source default, not a claim to serve the whole country.
_NATIONAL_PLACEHOLDERS = frozenset(
    {"", "us", "usa", "u s", "u s a", "united states", "united states of america", "nationwide"}
)
# Beyond this many places at one level the state lists the first ones (sorted) and a
# count, so a record covering hundreds of postal codes stays small.
MAX_PLACES_PER_LEVEL = 25

_SCHEDULE_SCOPES: frozenset[str] = frozenset({"service", "location"})
# Schedule text that names no hours: it points elsewhere or says nothing fixed.
_LOW_INFORMATION_SCHEDULE = re.compile(
    r"^(?:"
    r"same as .*hours?"
    r"|(?:hours )?(?:vary|varies)(?: by (?:site|location|program|season|day|week))?"
    r"|call(?: (?:for|to confirm) (?:hours|times|schedule|information|details))?"
    r"|contact (?:agency|us|the agency|office)(?: for (?:hours|times|schedule))?"
    r"|see (?:website|web site|description|agency|site|location)"
    r"|n/?a|none|tbd|unknown|not available|no set hours"
    r")$"
)

_RECENCY_A_NEWER: AssuranceRecency = (
    "record_a was assured more recently than record_b, so record_b may be the outdated one; "
    "assurance dates are not evidence of whether the records match"
)
_RECENCY_B_NEWER: AssuranceRecency = (
    "record_b was assured more recently than record_a, so record_a may be the outdated one; "
    "assurance dates are not evidence of whether the records match"
)
_RECENCY_SAME_DAY: AssuranceRecency = (
    "both records were assured on the same day; "
    "assurance dates are not evidence of whether the records match"
)
_RECENCY_ONLY_A: AssuranceRecency = (
    "only record_a has an assurance date, so record_b may be the outdated one; "
    "assurance dates are not evidence of whether the records match"
)
_RECENCY_ONLY_B: AssuranceRecency = (
    "only record_b has an assurance date, so record_a may be the outdated one; "
    "assurance dates are not evidence of whether the records match"
)


@dataclass(frozen=True)
class IdentifierState:
    """One organization identifier (HSDS ``organization_identifier``), e.g. an EIN."""

    type: str
    identifier: str


@dataclass(frozen=True)
class ServiceAreaState:
    """The places a record serves at one level, coarsest level first in a record."""

    level: ServiceAreaLevel
    places: tuple[str, ...]
    omitted: int


@dataclass(frozen=True)
class ScheduleState:
    """One schedule as English text, tagged with what it is attached to.

    ``low_information`` marks text that names no hours ("Same as site hours",
    "Varies"): it shows the source had a schedule slot, not when the service runs.
    """

    scope: ScheduleScope
    text: str
    low_information: bool


def identifiers(value: object) -> tuple[IdentifierState, ...]:
    """Build identifiers from ``identifiers`` objects, dropping blank values.

    Args:
        value: List of objects with ``identifier_type`` (or ``identifier_scheme`` /
            ``type``) and ``identifier`` (or ``value``).

    Returns:
        Unique identifiers in first-seen order.
    """
    found: list[IdentifierState] = []
    for item in _objects(value):
        identifier = _first_text(item, ("identifier", "value"))
        if identifier:
            kind = _first_text(item, ("identifier_type", "identifier_scheme", "type"))
            found.append(IdentifierState(type=kind, identifier=identifier))
    return tuple(dict.fromkeys(found))


def assured_on(value: object) -> str:
    """Render an assurance date as ``YYYY-MM-DD`` when it parses, else as given.

    Args:
        value: The HSDS ``assured_date``, a date or a date-time string.

    Returns:
        The date, the trimmed text when it does not parse, or ``""``.
    """
    text = value.strip() if isinstance(value, str) else ""
    parsed = _parse_date(text)
    return parsed.isoformat() if parsed is not None else text


def compare_assurance(assured_a: str, assured_b: str) -> AssuranceRecency | None:
    """Name which record was assured more recently, never whether they match.

    Args:
        assured_a: Side A's rendered assurance date.
        assured_b: Side B's rendered assurance date.

    Returns:
        A sentence naming the more recently assured side, or ``None`` when neither
        side has a date that parses.
    """
    date_a = _parse_date(assured_a)
    date_b = _parse_date(assured_b)
    if date_a is None and date_b is None:
        return None
    if date_b is None:
        return _RECENCY_ONLY_A
    if date_a is None:
        return _RECENCY_ONLY_B
    if date_a == date_b:
        return _RECENCY_SAME_DAY
    return _RECENCY_A_NEWER if date_a > date_b else _RECENCY_B_NEWER


def service_areas(value: object) -> tuple[ServiceAreaState, ...]:
    """Build a record's service area as level and place, from structured extent only.

    Rows are read for ``level`` and ``place``; an area's free-text name is never read,
    and a row whose level is not national, state, county, city or postal code is
    dropped. A national row naming only the country is dropped when the record also
    lists a narrower area: a county-level record's "USA" row is a source default. An
    empty result means the service area is unknown.

    Args:
        value: List of ``{"level": ..., "place": ...}`` objects.

    Returns:
        One entry per level present, coarsest first, places sorted and capped at
        :data:`MAX_PLACES_PER_LEVEL`.
    """
    by_level: dict[ServiceAreaLevel, set[str]] = {}
    national_placeholder = False
    for item in _objects(value):
        level = _LEVEL_ALIASES.get(_folded(item.get("level")).replace(" ", "_"))
        place = " ".join(_first_text(item, ("place",)).split())
        if level == "national" and _is_national_placeholder(place):
            national_placeholder = True
        elif level is not None and place:
            by_level.setdefault(level, set()).add(place)
    if national_placeholder and not by_level:
        by_level["national"] = {"United States"}
    return tuple(_area(level, by_level[level]) for level in _LEVELS if level in by_level)


def schedules(value: object) -> tuple[ScheduleState, ...]:
    """Build schedules from canonical English text with a service or location tag.

    Args:
        value: List of objects with ``description`` (or ``text``) and ``scope``
            (``service`` or ``location``). Without a scope, a row carrying a
            ``location_id`` or ``service_at_location_id`` is a location schedule.

    Returns:
        Unique schedules in first-seen order; blank text is dropped.
    """
    found: list[ScheduleState] = []
    for item in _objects(value):
        text = " ".join(_first_text(item, ("description", "text")).split())
        if text:
            found.append(
                ScheduleState(
                    scope=_schedule_scope(item),
                    text=text,
                    low_information=is_low_information_schedule(text),
                )
            )
    return tuple(dict.fromkeys(found))


def is_low_information_schedule(text: str) -> bool:
    """Whether schedule text names no hours ("Same as site hours", "Varies", "Call").

    Args:
        text: Schedule text.

    Returns:
        ``True`` when the text points elsewhere or says nothing fixed.
    """
    folded = " ".join(text.lower().replace(".", " ").replace("!", " ").split()).strip(" ,;:-")
    return bool(_LOW_INFORMATION_SCHEDULE.match(folded))


def link_organizations(
    entity_type: str, entity_a: Mapping[str, Any], entity_b: Mapping[str, Any]
) -> OrganizationLink | None:
    """Name a shared or parent organization when the records' ids say so.

    Services are linked when both carry the same ``organization_id``. Organizations are
    linked through ``parent_organization_id``: one names the other, or both name the
    same parent.

    Args:
        entity_type: ``organization`` or ``service``.
        entity_a: Clean entity row for side A.
        entity_b: Clean entity row for side B.

    Returns:
        The link in words, or ``None`` when the ids show none.
    """
    if entity_type == "service":
        org_a = _folded(entity_a.get("organization_id"))
        same = org_a and org_a == _folded(entity_b.get("organization_id"))
        return "both records belong to the same organization record" if same else None
    id_a = _folded(entity_a.get("entity_id"))
    id_b = _folded(entity_b.get("entity_id"))
    parent_a = _folded(entity_a.get("parent_organization_id"))
    parent_b = _folded(entity_b.get("parent_organization_id"))
    if parent_b and parent_b == id_a:
        return "record_a is record_b's parent organization"
    if parent_a and parent_a == id_b:
        return "record_b is record_a's parent organization"
    if parent_a and parent_a == parent_b:
        return "both organizations have the same parent organization"
    return None


def _area(level: ServiceAreaLevel, places: set[str]) -> ServiceAreaState:
    """One level's entry: sorted places up to the cap, and how many were left out."""
    ordered = sorted(places, key=str.lower)
    kept = tuple(ordered[:MAX_PLACES_PER_LEVEL])
    return ServiceAreaState(level=level, places=kept, omitted=len(ordered) - len(kept))


def _is_national_placeholder(place: str) -> bool:
    """Whether a national row's place names only the country."""
    letters = re.sub(r"[^a-z ]", " ", place.lower())
    return " ".join(letters.split()) in _NATIONAL_PLACEHOLDERS


def _schedule_scope(item: Mapping[str, Any]) -> ScheduleScope:
    """A schedule's scope: the given one, else location when it names a location."""
    scope = _folded(item.get("scope"))
    if scope in _SCHEDULE_SCOPES:
        return "service" if scope == "service" else "location"
    if _first_text(item, ("location_id", "service_at_location_id")):
        return "location"
    return "unknown"


def _parse_date(text: str) -> date | None:
    """Parse the leading ``YYYY-MM-DD`` of a date or date-time string."""
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _objects(value: object) -> list[Mapping[str, Any]]:
    """Return the mapping items of a list value."""
    if not isinstance(value, list | tuple):
        return []
    items = cast("Sequence[object]", value)
    return [cast("Mapping[str, Any]", item) for item in items if isinstance(item, Mapping)]


def _first_text(item: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-blank string among alternative keys."""
    for key in keys:
        text = item.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _folded(value: object) -> str:
    """Trim, lowercase and collapse whitespace; ``""`` for null (ids compare this way)."""
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())
