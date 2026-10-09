"""The judge stage: build states for scored pairs, ask a judge, return answers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import polars as pl

from hsds_entity_resolution.judge.answers import answers_to_frame
from hsds_entity_resolution.judge.protocol import PairJudge
from hsds_entity_resolution.judge.state import PairState, build_pair_state
from hsds_entity_resolution.types.frames import JUDGE_ANSWERS_SCHEMA

_LOGGER = logging.getLogger(__name__)


def build_pair_states(
    *,
    scored_pairs: pl.DataFrame,
    denormalized_organization: pl.DataFrame,
    denormalized_service: pl.DataFrame,
    source_profiles: Mapping[str, str] | None,
) -> list[PairState]:
    """Build one judge state per scored pair.

    Args:
        scored_pairs: Scored pairs (``SCORED_PAIRS_SCHEMA``).
        denormalized_organization: Clean organization entities.
        denormalized_service: Clean service entities.
        source_profiles: Source Profile text keyed by source schema; a schema with
            no entry gets an empty slot. The text is passed through verbatim.

    Returns:
        States in ``scored_pairs`` order.
    """
    if scored_pairs.is_empty():
        return []
    profiles = source_profiles or {}
    lookup = _entity_lookup(denormalized_organization, denormalized_service)
    states: list[PairState] = []
    for row in scored_pairs.iter_rows(named=True):
        entity_type = str(row["entity_type"])
        entity_a = lookup[(entity_type, str(row["entity_a_id"]))]
        entity_b = lookup[(entity_type, str(row["entity_b_id"]))]
        states.append(
            build_pair_state(
                pair_key=str(row["pair_key"]),
                entity_type=entity_type,
                entity_a=entity_a,
                entity_b=entity_b,
                pair_outcome=str(row["pair_outcome"]),
                source_profile_a=profiles.get(str(entity_a.get("source_schema") or ""), ""),
                source_profile_b=profiles.get(str(entity_b.get("source_schema") or ""), ""),
            )
        )
    return states


def judge_scored_pairs(
    *,
    scored_pairs: pl.DataFrame,
    denormalized_organization: pl.DataFrame,
    denormalized_service: pl.DataFrame,
    judge: PairJudge,
    source_profiles: Mapping[str, str] | None,
) -> pl.DataFrame:
    """Ask a judge about every scored pair and return its answers as a frame.

    Args:
        scored_pairs: Scored pairs (``SCORED_PAIRS_SCHEMA``).
        denormalized_organization: Clean organization entities.
        denormalized_service: Clean service entities.
        judge: The Pair Judge to ask.
        source_profiles: Source Profile text keyed by source schema.

    Returns:
        One row per scored pair, ``JUDGE_ANSWERS_SCHEMA``.

    Raises:
        ValueError: When a state exceeds the judge's token budget, or the judge
            returns answers that do not match the states one for one.
    """
    states = build_pair_states(
        scored_pairs=scored_pairs,
        denormalized_organization=denormalized_organization,
        denormalized_service=denormalized_service,
        source_profiles=source_profiles,
    )
    if not states:
        return pl.DataFrame(schema=JUDGE_ANSWERS_SCHEMA)
    oversized = [
        state.pair_key for state in states if state.estimated_tokens() > judge.state_token_budget
    ]
    if oversized:
        message = (
            f"{len(oversized)} pair state(s) exceed the judge's {judge.state_token_budget}-token "
            f"budget, e.g. {oversized[:3]}; trim the source records or Source Profiles"
        )
        raise ValueError(message)
    answers = judge.judge(states)
    expected = [state.pair_key for state in states]
    returned = [answer.pair_key for answer in answers]
    if returned != expected:
        message = (
            f"Judge {judge.model_id!r} returned {len(returned)} answers for "
            f"{len(expected)} states, or out of order"
        )
        raise ValueError(message)
    _LOGGER.info(
        "judge_scored_pairs model_id=%s question_set_version=%s pairs=%d",
        judge.model_id,
        judge.question_set_version,
        len(answers),
    )
    return answers_to_frame(answers)


def _entity_lookup(
    denormalized_organization: pl.DataFrame, denormalized_service: pl.DataFrame
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index clean entity rows by ``(entity_type, entity_id)``."""
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for frame in (denormalized_organization, denormalized_service):
        if frame.is_empty():
            continue
        for row in frame.iter_rows(named=True):
            lookup[(str(row["entity_type"]), str(row["entity_id"]))] = row
    return lookup
