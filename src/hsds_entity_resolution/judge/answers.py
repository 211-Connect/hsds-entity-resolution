"""The open answer schema a Pair Judge returns for one pair.

The answers are the fixed questions of ADR-0027 as probabilities: Same Site, Same
Offering, physical delivery, and the Organization Relation as a distribution over
five labels. The judge never answers "duplicate"; the caller composes that in code
from these answers and its own policy.
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

    ``same_offering`` and ``physically_delivered`` are ``None`` for organization
    pairs, which are not asked them.
    """

    pair_key: str
    entity_type: str
    model_id: str
    question_set_version: str
    same_site: float
    same_offering: float | None
    physically_delivered: float | None
    organization_relation: Mapping[OrganizationRelation, float]

    def __post_init__(self) -> None:
        """Validate probability ranges, relation labels and per-entity-type fields."""
        for name in ("same_site", "same_offering", "physically_delivered"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                message = f"{name} must be a probability in [0, 1], got {value!r}"
                raise ValueError(message)
        if set(self.organization_relation) != set(ORGANIZATION_RELATIONS):
            message = (
                "organization_relation must give a probability for exactly "
                f"{list(ORGANIZATION_RELATIONS)}, got {sorted(self.organization_relation)}"
            )
            raise ValueError(message)
        total = sum(self.organization_relation.values())
        if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
            message = f"organization_relation probabilities must sum to 1, got {total!r}"
            raise ValueError(message)
        is_organization = self.entity_type == "organization"
        has_service_answers = (
            self.same_offering is not None or self.physically_delivered is not None
        )
        if is_organization and has_service_answers:
            message = "organization pairs carry no same_offering or physically_delivered answer"
            raise ValueError(message)
        if not is_organization and (
            self.same_offering is None or self.physically_delivered is None
        ):
            message = "service pairs need same_offering and physically_delivered answers"
            raise ValueError(message)

    @property
    def organization_relation_choice(self) -> OrganizationRelation:
        """Return the most probable relation; ties resolve in label order."""
        return max(ORGANIZATION_RELATIONS, key=lambda label: self.organization_relation[label])


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
            "organization_relation_probabilities": {
                label: answer.organization_relation[label] for label in ORGANIZATION_RELATIONS
            },
        }
        for answer in answers
    ]
    return pl.DataFrame(rows, schema=JUDGE_ANSWERS_SCHEMA)


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
        organization_relation={label: probabilities[label] for label in ORGANIZATION_RELATIONS},
    )
