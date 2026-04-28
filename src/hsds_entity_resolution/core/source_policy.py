"""Generic source-aware policy resolution for candidate admission and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hsds_entity_resolution.config.entity_resolution_run_config import (
    AdmissionRuleConfig,
    EntityResolutionRunConfig,
    FeatureOverrideConfig,
    ScoringConfig,
)

DEFAULT_BLOCKING_RULE_ID = "default_taxonomy_and_non_taxonomy"
DEFAULT_POLICY_RULE_ID = "default"


@dataclass(frozen=True)
class PairPolicyContext:
    """Minimal metadata needed to select a source-aware rule."""

    entity_type: str
    source_schema_a: str | None
    source_schema_b: str | None


@dataclass(frozen=True)
class AdmissionDecision:
    """Candidate-admission result for one above-threshold pair."""

    admitted: bool
    rule_id: str | None


@dataclass(frozen=True)
class EffectiveScoringPolicy:
    """Scoring controls resolved for one pair."""

    rule_id: str
    duplicate_threshold: float
    maybe_threshold: float
    low_maybe_threshold: float
    suppressed_signals: list[str]
    feature_overrides: FeatureOverrideConfig


def collect_source_profiles(
    *,
    config: EntityResolutionRunConfig,
    source_schema: str | None,
) -> set[str]:
    """Return host-assigned profile ids for one source schema."""
    if not source_schema:
        return set()
    normalized = source_schema.strip().upper()
    profiles: set[str] = set()
    for profile_id, profile in config.source_policy.source_profiles.items():
        if normalized in set(profile.source_schemas):
            profiles.add(profile_id)
    return profiles


def source_relation_matches(
    *,
    relation: str,
    context: PairPolicyContext,
    config: EntityResolutionRunConfig,
) -> bool:
    """Evaluate a generic source relation predicate."""
    if relation == "any":
        return True
    left = (context.source_schema_a or "").strip().upper()
    right = (context.source_schema_b or "").strip().upper()
    same_source = bool(left and right and left == right)
    if relation == "same_source":
        return same_source
    if relation == "cross_source":
        return bool(left and right and left != right)
    left_profiles = collect_source_profiles(config=config, source_schema=left)
    right_profiles = collect_source_profiles(config=config, source_schema=right)
    shared_profiles = bool(left_profiles.intersection(right_profiles))
    if relation == "same_profile":
        return shared_profiles
    if relation == "cross_source_same_profile":
        return bool(left and right and left != right and shared_profiles)
    if relation == "cross_profile":
        return bool(left_profiles and right_profiles and not shared_profiles)
    return False


def admission_rule_matches_context(
    *,
    rule: AdmissionRuleConfig,
    context: PairPolicyContext,
    config: EntityResolutionRunConfig,
    similarity: float,
) -> bool:
    """Return true when a candidate admission rule applies to the pair context."""
    if context.entity_type not in rule.entity_types:
        return False
    if rule.min_embedding_similarity is not None and similarity < rule.min_embedding_similarity:
        return False
    if not source_relation_matches(relation=rule.source_relation, context=context, config=config):
        return False
    return _source_profiles_match(
        config=config,
        context=context,
        relation=rule.source_relation,
        source_profiles=rule.source_profiles,
    )


def decide_candidate_admission(
    *,
    config: EntityResolutionRunConfig,
    context: PairPolicyContext,
    similarity: float,
    channel_hits: set[str],
    taxonomy_pass: bool,
    non_taxonomy_pass: bool,
    default_admission_allowed: bool = True,
) -> AdmissionDecision:
    """Apply ordered generic admission rules with backward-compatible fallback."""
    for rule in config.source_policy.admission_rules:
        if not admission_rule_matches_context(
            rule=rule,
            context=context,
            config=config,
            similarity=similarity,
        ):
            continue
        if rule.none_of and channel_hits.intersection(rule.none_of):
            continue
        if rule.all_of and not set(rule.all_of).issubset(channel_hits):
            continue
        if rule.any_of and not channel_hits.intersection(rule.any_of):
            continue
        return AdmissionDecision(admitted=True, rule_id=rule.rule_id)
    return AdmissionDecision(
        admitted=default_admission_allowed and taxonomy_pass and non_taxonomy_pass,
        rule_id=(
            DEFAULT_BLOCKING_RULE_ID
            if default_admission_allowed and taxonomy_pass and non_taxonomy_pass
            else None
        ),
    )


def resolve_scoring_policy(
    *,
    config: EntityResolutionRunConfig,
    context: PairPolicyContext,
    contributed_signals: set[str] | None = None,
) -> EffectiveScoringPolicy:
    """Resolve effective pair-level scoring controls."""
    rule_id = DEFAULT_POLICY_RULE_ID
    overrides = FeatureOverrideConfig()
    for rule in config.source_policy.pair_rules:
        if context.entity_type not in rule.entity_types:
            continue
        if not source_relation_matches(
            relation=rule.source_relation,
            context=context,
            config=config,
        ):
            continue
        if not _source_profiles_match(
            config=config,
            context=context,
            relation=rule.source_relation,
            source_profiles=rule.source_profiles,
        ):
            continue
        rule_id = rule.rule_id
        overrides = rule.feature_overrides
        break
    scoring = config.scoring
    duplicate_threshold = overrides.duplicate_threshold or scoring.duplicate_threshold
    maybe_threshold = overrides.maybe_threshold or scoring.maybe_threshold
    low_maybe_threshold = overrides.low_maybe_threshold or scoring.low_maybe_threshold
    suppressed = _resolve_suppressed_signals(
        overrides=overrides,
        contributed_signals=contributed_signals or set(),
    )
    return EffectiveScoringPolicy(
        rule_id=rule_id,
        duplicate_threshold=duplicate_threshold,
        maybe_threshold=maybe_threshold,
        low_maybe_threshold=low_maybe_threshold,
        suppressed_signals=suppressed,
        feature_overrides=overrides,
    )


def effective_scoring_values(
    *,
    scoring: ScoringConfig,
    overrides: FeatureOverrideConfig,
) -> dict[str, Any]:
    """Return scalar scoring settings after applying pair-level overrides."""
    return {
        "deterministic_section_weight": (
            overrides.deterministic_section_weight
            if overrides.deterministic_section_weight is not None
            else scoring.deterministic_section_weight
        ),
        "nlp_section_weight": (
            overrides.nlp_section_weight
            if overrides.nlp_section_weight is not None
            else scoring.nlp_section_weight
        ),
        "ml_section_weight": (
            overrides.ml_section_weight
            if overrides.ml_section_weight is not None
            else scoring.ml_section_weight
        ),
        "ml_gate_threshold": (
            overrides.ml_gate_threshold
            if overrides.ml_gate_threshold is not None
            else scoring.ml.ml_gate_threshold
        ),
    }


def deterministic_signal_config(
    *,
    config: EntityResolutionRunConfig,
    signal_name: str,
    overrides: FeatureOverrideConfig,
) -> Any:
    """Return deterministic signal config after applying pair-level overrides."""
    override = overrides.deterministic.get(signal_name)
    if override is not None:
        return override
    return getattr(config.scoring.deterministic, signal_name)


def _source_profiles_match(
    *,
    config: EntityResolutionRunConfig,
    context: PairPolicyContext,
    relation: str,
    source_profiles: list[str],
) -> bool:
    """Return whether a rule's optional profile filter matches the pair."""
    if not source_profiles:
        return True
    left_profiles = collect_source_profiles(config=config, source_schema=context.source_schema_a)
    right_profiles = collect_source_profiles(config=config, source_schema=context.source_schema_b)
    requested_profiles = set(source_profiles)
    if relation in {"same_profile", "cross_source_same_profile"}:
        return bool((left_profiles & right_profiles) & requested_profiles)
    return bool((left_profiles | right_profiles) & requested_profiles)


def _resolve_suppressed_signals(
    *,
    overrides: FeatureOverrideConfig,
    contributed_signals: set[str],
) -> list[str]:
    """Resolve suppression rules against signals observed for the pair."""
    suppressed: list[str] = []
    for rule in overrides.suppressions:
        if set(rule.when_all_present).issubset(contributed_signals):
            suppressed.append(rule.signal)
    return sorted(set(suppressed))
