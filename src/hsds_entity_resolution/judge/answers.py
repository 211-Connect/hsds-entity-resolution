"""The open answer schema a Pair Judge returns for one pair.

The answers are the fixed questions of ADR-0027 as probabilities: Same Site, Same
Offering, physical delivery, and the Organization Relation as a distribution over
five labels. The judge never answers "duplicate"; the caller composes that in code
from these answers and its own policy.

Any answer may be ``None``: the question was not asked. A judge that asks the
questions one at a time can stop once the answers so far decide the caller's policy
(2.3.0); what it did not ask stays unknown rather than being guessed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from hsds_entity_resolution.types.frames import JUDGE_ANSWERS_SCHEMA

OrganizationRelation = Literal[
    "same", "parent_and_chapter", "affiliated", "unrelated", "cannot_tell"
]
ORGANIZATION_RELATIONS: tuple[OrganizationRelation, ...] = (
    "same",
    "parent_and_chapter",
    "affiliated",
    "unrelated",
    "cannot_tell",
)
_PROBABILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PairAnswers:
    """One judge's answers for one pair.

    ``None`` means the question was not asked. Organization pairs are never asked
    ``same_offering`` or ``physically_delivered``; at least one question is answered.
    """

    pair_key: str
    entity_type: str
    model_id: str
    question_set_version: str
    same_site: float | None
    same_offering: float | None
    physically_delivered: float | None
    organization_relation: Mapping[OrganizationRelation, float] | None

    def __post_init__(self) -> None:
        """Validate probability ranges, relation labels and per-entity-type fields."""
        probabilities = (self.same_site, self.same_offering, self.physically_delivered)
        for name, value in zip(
            ("same_site", "same_offering", "physically_delivered"), probabilities, strict=True
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                message = f"{name} must be a probability in [0, 1], got {value!r}"
                raise ValueError(message)
        if self.organization_relation is not None:
            _validate_relation(self.organization_relation)
        has_service_answers = (
            self.same_offering is not None or self.physically_delivered is not None
        )
        if self.entity_type == "organization" and has_service_answers:
            message = "organization pairs carry no same_offering or physically_delivered answer"
            raise ValueError(message)
        if self.organization_relation is None and all(value is None for value in probabilities):
            message = "a judgment must answer at least one question"
            raise ValueError(message)

    @property
    def organization_relation_choice(self) -> OrganizationRelation | None:
        """Return the most probable relation (ties in label order); ``None`` if not asked."""
        relation = self.organization_relation
        if relation is None:
            return None
        return max(ORGANIZATION_RELATIONS, key=lambda label: relation[label])


def _validate_relation(relation: Mapping[OrganizationRelation, float]) -> None:
    """Require a probability for exactly the five relation labels, summing to one."""
    if set(relation) != set(ORGANIZATION_RELATIONS):
        message = (
            "organization_relation must give a probability for exactly "
            f"{list(ORGANIZATION_RELATIONS)}, got {sorted(relation)}"
        )
        raise ValueError(message)
    total = sum(relation.values())
    if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
        message = f"organization_relation probabilities must sum to 1, got {total!r}"
        raise ValueError(message)


def answers_to_frame(answers: Iterable[PairAnswers]) -> pl.DataFrame:
    """Convert answers to a frame with :data:`JUDGE_ANSWERS_SCHEMA`.

    Args:
        answers: Judge answers, one per pair.

    Returns:
        One row per pair; relation probabilities as a struct column.
    """
    rows = [
        {
            "pair_key": answer.pair_key,
            "entity_type": answer.entity_type,
            "model_id": answer.model_id,
            "question_set_version": answer.question_set_version,
            "same_site": answer.same_site,
            "same_offering": answer.same_offering,
            "physically_delivered": answer.physically_delivered,
            "organization_relation": answer.organization_relation_choice,
            "organization_relation_probabilities": _relation_struct(answer),
        }
        for answer in answers
    ]
    return pl.DataFrame(rows, schema=JUDGE_ANSWERS_SCHEMA)


def _relation_struct(answer: PairAnswers) -> dict[str, float] | None:
    """Return the relation distribution as a struct value, or ``None`` if not asked."""
    relation = answer.organization_relation
    if relation is None:
        return None
    return {label: relation[label] for label in ORGANIZATION_RELATIONS}


def answers_from_frame(frame: pl.DataFrame) -> list[PairAnswers]:
    """Rebuild answers from a :data:`JUDGE_ANSWERS_SCHEMA` frame.

    Args:
        frame: Frame produced by :func:`answers_to_frame` or read back from storage.

    Returns:
        The answers, in frame order.
    """
    return [_answer_from_row(row) for row in frame.iter_rows(named=True)]


def _answer_from_row(row: Mapping[str, Any]) -> PairAnswers:
    """Rebuild one answer from a frame row."""
    probabilities = row["organization_relation_probabilities"]
    return PairAnswers(
        pair_key=row["pair_key"],
        entity_type=row["entity_type"],
        model_id=row["model_id"],
        question_set_version=row["question_set_version"],
        same_site=row["same_site"],
        same_offering=row["same_offering"],
        physically_delivered=row["physically_delivered"],
        organization_relation=(
            None
            if probabilities is None
            else {label: probabilities[label] for label in ORGANIZATION_RELATIONS}
        ),
    )
