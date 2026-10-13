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
from hsds_entity_resolution.judge.evidence import (
    MAX_PLACES_PER_LEVEL,
    AssuranceRecency,
    IdentifierState,
    OrganizationLink,
    ScheduleScope,
    ScheduleState,
    ServiceAreaLevel,
    ServiceAreaState,
    compare_assurance,
    is_low_information_schedule,
    link_organizations,
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
    "MAX_PLACES_PER_LEVEL",
    "ORGANIZATION_RELATIONS",
    "SMALLEST_MODEL_STATE_BUDGET_TOKENS",
    "AssuranceRecency",
    "IdentifierState",
    "OrganizationLink",
    "OrganizationRelation",
    "PairAnswers",
    "PairJudge",
    "PairState",
    "PriorScoreBucket",
    "RecordState",
    "ReferenceJudge",
    "ScheduleScope",
    "ScheduleState",
    "ServiceAreaLevel",
    "ServiceAreaState",
    "SiteComparison",
    "SiteState",
    "answers_from_frame",
    "answers_to_frame",
    "build_pair_state",
    "build_pair_states",
    "build_record_state",
    "compare_assurance",
    "compare_sites",
    "is_low_information_schedule",
    "judge_scored_pairs",
    "link_organizations",
    "prior_score_bucket",
]
