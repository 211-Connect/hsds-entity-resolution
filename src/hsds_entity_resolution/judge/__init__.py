"""The Pair Judge contract: open state and answer schemas, protocol and reference judge.

A Pair Judge reads one candidate pair and answers fixed questions — Same Site, Same
Offering, physical delivery and Organization Relation — as probabilities. It never
answers "duplicate"; callers compose that from the answers with their own policy.
The question wording, Source Profile text and thresholds belong to the caller.
"""

from hsds_entity_resolution.judge.answers import (
    ORGANIZATION_RELATIONS,
    OrganizationRelation,
    PairAnswers,
    answers_from_frame,
    answers_to_frame,
)
from hsds_entity_resolution.judge.protocol import PairJudge
from hsds_entity_resolution.judge.reference import ReferenceJudge
from hsds_entity_resolution.judge.stage import build_pair_states, judge_scored_pairs
from hsds_entity_resolution.judge.state import (
    SMALLEST_MODEL_STATE_BUDGET_TOKENS,
    PairState,
    PriorScoreBucket,
    RecordState,
    SiteComparison,
    SiteState,
    build_pair_state,
    build_record_state,
    compare_sites,
    prior_score_bucket,
)

__all__ = [
    "ORGANIZATION_RELATIONS",
    "SMALLEST_MODEL_STATE_BUDGET_TOKENS",
    "OrganizationRelation",
    "PairAnswers",
    "PairJudge",
    "PairState",
    "PriorScoreBucket",
    "RecordState",
    "ReferenceJudge",
    "SiteComparison",
    "SiteState",
    "answers_from_frame",
    "answers_to_frame",
    "build_pair_state",
    "build_pair_states",
    "build_record_state",
    "compare_sites",
    "judge_scored_pairs",
    "prior_score_bucket",
]
