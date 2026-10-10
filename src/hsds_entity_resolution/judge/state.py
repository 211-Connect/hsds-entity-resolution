"""The open state schema a Pair Judge reads for one candidate pair.

The state carries both records as structured HSDS text, the facts code already
computed (how the two records' sites compare, optionally the prior score as a named
bucket), and one free-text Source Profile slot per side. It deliberately carries nothing numeric
that code could have compared: phones are digit strings for reading, not for the
model to match, and the site comparison arrives as a word.

The engine fills every HSDS field and leaves the Source Profile slots empty. What a
profile says about a source system is the caller's knowledge, supplied per source
schema.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from hsds_entity_resolution.normalize import (
    normalize_address_component,
    normalize_phone,
    normalize_postal_code,
    normalize_url,
)

SiteComparison = Literal["same address", "same city", "different", "unknown"]
PriorScoreBucket = Literal[
    "the previous matcher would merge this pair",
    "the previous matcher would send this pair to human review",
    "the previous matcher would not flag this pair",
]

# The smallest state budget among the System One models a judge may run on. A state
# plus its longest question must fit in this many tokens.
SMALLEST_MODEL_STATE_BUDGET_TOKENS = 32_768

_PRIOR_BUCKET_BY_OUTCOME: dict[str, PriorScoreBucket] = {
    "duplicate": "the previous matcher would merge this pair",
    "maybe": "the previous matcher would send this pair to human review",
    "below_maybe": "the previous matcher would not flag this pair",
}
_STREET_KEYS = ("address_1", "address1", "line1", "street", "address")
_POSTAL_KEYS = ("postal_code", "postal", "zip", "zipcode")


@dataclass(frozen=True)
class SiteState:
    """One site of a record: its name, its HSDS location type and its address components."""

    name: str
    location_type: str
    address: str
    city: str
    state: str
    postal_code: str


@dataclass(frozen=True)
class RecordState:
    """One side of a pair: the HSDS text a person would read to compare it."""

    name: str
    alternate_name: str
    description: str
    short_description: str
    eligibility: str
    fees: str
    application_process: str
    taxonomies: tuple[str, ...]
    sites: tuple[SiteState, ...]
    phones: tuple[str, ...]
    websites: tuple[str, ...]
    organization_name: str
    organization_description: str


@dataclass(frozen=True)
class PairState:
    """Everything a Pair Judge is shown about one candidate pair."""

    pair_key: str
    entity_type: str
    record_a: RecordState
    record_b: RecordState
    site_comparison: SiteComparison
    prior_score: PriorScoreBucket | None
    source_profile_a: str
    source_profile_b: str

    def to_dict(self) -> dict[str, Any]:
        """Return the state as JSON-ready nested dicts and lists.

        ``prior_score`` is left out when ``None``, so a caller that does not want the
        judge anchored on the previous matcher sends no trace of it.
        """
        payload = asdict(self)
        if self.prior_score is None:
            del payload["prior_score"]
        return payload

    def estimated_tokens(self) -> int:
        """Estimate the state's size in model tokens (four characters per token).

        Returns:
            A conservative token estimate for budget checks against
            :data:`SMALLEST_MODEL_STATE_BUDGET_TOKENS`.
        """
        return len(json.dumps(self.to_dict(), ensure_ascii=False)) // 4 + 1


def prior_score_bucket(pair_outcome: str) -> PriorScoreBucket:
    """Name the prior matcher's verdict in words.

    Args:
        pair_outcome: ``duplicate``, ``maybe`` or ``below_maybe`` from scoring.

    Returns:
        A sentence fragment naming what the previous matcher would have done.
    """
    if pair_outcome not in _PRIOR_BUCKET_BY_OUTCOME:
        expected = sorted(_PRIOR_BUCKET_BY_OUTCOME)
        message = f"Unknown pair_outcome {pair_outcome!r}; expected one of {expected}"
        raise ValueError(message)
    return _PRIOR_BUCKET_BY_OUTCOME[pair_outcome]


def build_pair_state(
    *,
    pair_key: str,
    entity_type: str,
    entity_a: Mapping[str, Any],
    entity_b: Mapping[str, Any],
    pair_outcome: str | None,
    source_profile_a: str = "",
    source_profile_b: str = "",
) -> PairState:
    """Build the judge state for one pair from two clean entity rows.

    Args:
        pair_key: The pair's stable key.
        entity_type: ``organization`` or ``service``.
        entity_a: Clean entity row for side A (a ``CLEAN_ENTITY_SCHEMA`` row).
        entity_b: Clean entity row for side B.
        pair_outcome: The scoring stage's outcome for the pair, named as a bucket, or
            ``None`` to leave the prior score out of the state.
        source_profile_a: Caller-supplied Source Profile text for side A's source;
            passed through verbatim.
        source_profile_b: Caller-supplied Source Profile text for side B's source.

    Returns:
        The pair's state.
    """
    record_a = build_record_state(entity_a)
    record_b = build_record_state(entity_b)
    return PairState(
        pair_key=pair_key,
        entity_type=entity_type,
        record_a=record_a,
        record_b=record_b,
        site_comparison=compare_sites(record_a.sites, record_b.sites),
        prior_score=None if pair_outcome is None else prior_score_bucket(pair_outcome),
        source_profile_a=source_profile_a,
        source_profile_b=source_profile_b,
    )


def build_record_state(entity: Mapping[str, Any]) -> RecordState:
    """Build one side's state from a clean entity row.

    Display-cased name and description are preferred over the lowercased matching
    copies when present.

    Args:
        entity: Clean entity row.

    Returns:
        The record's state.
    """
    return RecordState(
        name=_text(entity.get("display_name")) or _text(entity.get("name")),
        alternate_name=_text(entity.get("alternate_name")),
        description=_text(entity.get("display_description")) or _text(entity.get("description")),
        short_description=_text(entity.get("short_description")),
        eligibility=_text(entity.get("eligibility_description")),
        fees=_text(entity.get("fees_description")),
        application_process=_text(entity.get("application_process")),
        taxonomies=_taxonomy_names(entity.get("taxonomies")),
        sites=_sites(entity.get("locations")),
        phones=_phones(entity.get("phones")),
        websites=_unique(normalize_url(value) for value in _strings(entity.get("websites"))),
        organization_name=_text(entity.get("organization_name")),
        organization_description=_text(entity.get("organization_description")),
    )


def compare_sites(sites_a: Iterable[SiteState], sites_b: Iterable[SiteState]) -> SiteComparison:
    """Compare two records' sites and name the closest relation in words.

    ``same address`` needs an equal street address, city and postal code on some
    pair of sites; ``same city`` needs an equal city and state; ``unknown`` means
    either side has no site.

    Args:
        sites_a: Side A's sites.
        sites_b: Side B's sites.

    Returns:
        The closest relation found across all site pairs.
    """
    left = list(sites_a)
    right = list(sites_b)
    if not left or not right:
        return "unknown"
    best: SiteComparison = "different"
    for site_a in left:
        for site_b in right:
            if (
                site_a.address
                and site_a.address == site_b.address
                and site_a.city == site_b.city
                and site_a.postal_code == site_b.postal_code
            ):
                return "same address"
            if site_a.city and site_a.city == site_b.city and site_a.state == site_b.state:
                best = "same city"
    return best


def _text(value: object) -> str:
    """Return a trimmed string, or ``""`` for null and non-string values."""
    return value.strip() if isinstance(value, str) else ""


def _strings(value: object) -> list[str]:
    """Return the non-blank strings of a list value."""
    if not isinstance(value, list | tuple):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    """Drop blanks and repeats, keeping first-seen order."""
    return tuple(dict.fromkeys(value for value in values if value))


def _phones(value: object) -> tuple[str, ...]:
    """Normalise phones to digits, keeping an extension as ``digits x ext``."""
    rendered: list[str] = []
    for raw in _strings(value):
        digits, extension = normalize_phone(raw)
        if digits:
            rendered.append(f"{digits} x {extension}" if extension else digits)
    return _unique(rendered)


def _taxonomy_names(value: object) -> tuple[str, ...]:
    """Return taxonomy term names; codes are never sent to the judge."""
    if not isinstance(value, list | tuple):
        return ()
    names = [_text(item.get("name")) for item in value if isinstance(item, Mapping)]
    return _unique(names)


def _first_present(location: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-blank string among alternative location keys."""
    for key in keys:
        text = _text(location.get(key))
        if text:
            return text
    return ""


def _sites(value: object) -> tuple[SiteState, ...]:
    """Build normalised sites from location payloads, dropping empty ones."""
    if not isinstance(value, list | tuple):
        return ()
    sites: list[SiteState] = []
    for location in value:
        if not isinstance(location, Mapping):
            continue
        site = SiteState(
            name=_text(location.get("name")),
            location_type=_text(location.get("location_type")).lower(),
            address=normalize_address_component(_first_present(location, _STREET_KEYS)),
            city=normalize_address_component(_first_present(location, ("city",))),
            state=normalize_address_component(_first_present(location, ("state",))),
            postal_code=normalize_postal_code(_first_present(location, _POSTAL_KEYS)),
        )
        if site.address or site.city or site.postal_code:
            sites.append(site)
    return tuple(dict.fromkeys(sites))
