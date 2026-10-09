"""The Pair Judge contract: anything that answers the fixed questions for a batch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from hsds_entity_resolution.judge.answers import PairAnswers
from hsds_entity_resolution.judge.state import PairState


@runtime_checkable
class PairJudge(Protocol):
    """Reads pair states and answers Same Site, Same Offering, physical delivery and
    Organization Relation as probabilities.

    Implementations own their question wording, model choice and batching against
    vendor limits. The engine only fixes the state they read and the answers they
    return.

    Attributes:
        model_id: Identifier of the model behind the judge, recorded on every answer.
        question_set_version: Version of the question wording and criteria, recorded
            on every answer so a pair can be re-judged under a new version.
        state_token_budget: Largest state, in tokens, the judge accepts. Callers size
            states against it; :data:`~hsds_entity_resolution.judge.state.
            SMALLEST_MODEL_STATE_BUDGET_TOKENS` is the floor for any model a caller
            may switch to.
    """

    model_id: str
    question_set_version: str
    state_token_budget: int

    def judge(self, states: Sequence[PairState]) -> list[PairAnswers]:
        """Answer every state, returning one answer per state in the same order.

        Args:
            states: Pair states, each within ``state_token_budget``.

        Returns:
            One :class:`PairAnswers` per state, in input order.
        """
        ...
