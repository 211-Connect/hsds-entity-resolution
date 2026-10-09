"""A reference judge that knows nothing, so the pipeline runs end to end without a model."""

from __future__ import annotations

from collections.abc import Sequence

from hsds_entity_resolution.judge.answers import ORGANIZATION_RELATIONS, PairAnswers
from hsds_entity_resolution.judge.state import SMALLEST_MODEL_STATE_BUDGET_TOKENS, PairState

_UNINFORMATIVE = 0.5


class ReferenceJudge:
    """Answers every question with a fixed, uninformative distribution.

    Every probability is 0.5 and the Organization Relation is uniform over its five
    labels. It reads no state, calls nothing, and exists to exercise the judge stage
    and the answer schema; it is not a model.
    """

    model_id = "reference-uninformative"
    question_set_version = "reference-v1"
    state_token_budget = SMALLEST_MODEL_STATE_BUDGET_TOKENS

    def judge(self, states: Sequence[PairState]) -> list[PairAnswers]:
        """Return the uninformative answer for each state.

        Args:
            states: Pair states.

        Returns:
            One uninformative answer per state, in input order.
        """
        uniform = 1.0 / len(ORGANIZATION_RELATIONS)
        relation = dict.fromkeys(ORGANIZATION_RELATIONS, uniform)
        return [
            PairAnswers(
                pair_key=state.pair_key,
                entity_type=state.entity_type,
                model_id=self.model_id,
                question_set_version=self.question_set_version,
                same_site=_UNINFORMATIVE,
                same_offering=None if state.entity_type == "organization" else _UNINFORMATIVE,
                physically_delivered=(
                    None if state.entity_type == "organization" else _UNINFORMATIVE
                ),
                organization_relation=relation,
            )
            for state in states
        ]
