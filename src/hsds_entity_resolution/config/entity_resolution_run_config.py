"""Centralized run configuration for entity-resolution stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hsds_entity_resolution.types.domain import EntityType

_SUPPORTED_BLOCKING_OVERLAP_CHANNELS = {
    "email",
    "phone",
    "website",
    "taxonomy",
    "location",
    "address_exact",
}

_SUPPORTED_SIGNAL_NAMES = {
    "shared_email",
    "shared_phone",
    "shared_domain",
    "shared_taxonomy",
    "shared_address",
    "shared_identifier",
    "name_similarity",
    "ml_score",
}


class BaseStrictModel(BaseModel):
    """Shared strict pydantic model behavior."""

    model_config = ConfigDict(extra="forbid")


class BlockingConfig(BaseStrictModel):
    """Candidate blocking and fanout controls."""

    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_candidates_per_entity: int = Field(default=50, ge=1, le=500)
    blocking_batch_size: int = Field(default=5000, ge=1, le=50000)
    overlap_prefilter_channels: list[str] = Field(
        default_factory=lambda: ["email", "phone", "website", "taxonomy", "location"]
    )

    @field_validator("overlap_prefilter_channels")
    @classmethod
    def validate_overlap_prefilter_channels(cls, values: list[str]) -> list[str]:
        """Validate overlap prefilter channel selections."""
        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str):
                message = "overlap_prefilter_channels entries must be strings"
                raise ValueError(message)
            normalized_value = value.strip().lower()
            if normalized_value:
                normalized.append(normalized_value)
        unique_values = list(dict.fromkeys(normalized))
        if not unique_values:
            message = "overlap_prefilter_channels must contain at least one channel"
            raise ValueError(message)
        unsupported = sorted(set(unique_values).difference(_SUPPORTED_BLOCKING_OVERLAP_CHANNELS))
        if unsupported:
            message = f"Unsupported overlap prefilter channels: {unsupported!r}"
            raise ValueError(message)
        return unique_values


class AdmissionRuleConfig(BaseStrictModel):
    """Generic candidate-admission rule evaluated after embedding threshold."""

    rule_id: str
    entity_types: list[EntityType] = Field(default_factory=lambda: ["organization", "service"])
    source_relation: Literal[
        "any",
        "same_source",
        "cross_source",
        "same_profile",
        "cross_source_same_profile",
        "cross_profile",
    ] = "any"
    source_profiles: list[str] = Field(default_factory=list)
    min_embedding_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    all_of: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list)
    none_of: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_admission_rule(self) -> AdmissionRuleConfig:
        """Validate supported channel names and source-profile references."""
        referenced = [*self.all_of, *self.any_of, *self.none_of]
        unsupported = sorted(set(referenced).difference(_SUPPORTED_BLOCKING_OVERLAP_CHANNELS))
        if unsupported:
            message = f"Unsupported admission rule channels: {unsupported!r}"
            raise ValueError(message)
        if not self.all_of and not self.any_of:
            message = "Admission rule must define at least one all_of or any_of channel"
            raise ValueError(message)
        return self


class DeterministicSignalConfig(BaseStrictModel):
    """Configuration for one deterministic overlap signal."""

    enabled: bool = True
    weight: float = Field(default=0.2, ge=0.0, le=0.6)


class DeterministicConfig(BaseStrictModel):
    """Deterministic scoring controls."""

    shared_email: DeterministicSignalConfig
    shared_phone: DeterministicSignalConfig
    shared_domain: DeterministicSignalConfig
    shared_taxonomy: DeterministicSignalConfig
    shared_address: DeterministicSignalConfig
    shared_identifier: DeterministicSignalConfig


class NlpConfig(BaseStrictModel):
    """Name/description fuzzy matching controls."""

    fuzzy_algorithm: str = "sequence_matcher"
    fuzzy_threshold: float = Field(default=0.88, ge=0.6, le=0.98)
    number_mismatch_veto_enabled: bool = True
    standalone_fuzzy_threshold: float = Field(default=0.94, ge=0.7, le=0.99)


class SignalSuppressionConfig(BaseStrictModel):
    """Suppress one signal when all configured trigger signals contributed."""

    signal: str
    when_all_present: list[str]

    @model_validator(mode="after")
    def validate_signal_names(self) -> SignalSuppressionConfig:
        """Validate signal identifiers used by suppression rules."""
        names = [self.signal, *self.when_all_present]
        unsupported = sorted(set(names).difference(_SUPPORTED_SIGNAL_NAMES))
        if unsupported:
            message = f"Unsupported signal suppression names: {unsupported!r}"
            raise ValueError(message)
        if not self.when_all_present:
            message = "Signal suppression requires at least one trigger signal"
            raise ValueError(message)
        return self


class FeatureOverrideConfig(BaseStrictModel):
    """Pair-rule scoring overrides applied on top of entity-type defaults."""

    deterministic: dict[str, DeterministicSignalConfig] = Field(default_factory=dict)
    nlp_enabled: bool | None = None
    deterministic_section_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    nlp_section_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    ml_section_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    duplicate_threshold: float | None = Field(default=None, ge=0.5, le=0.99)
    maybe_threshold: float | None = Field(default=None, ge=0.3, le=0.95)
    low_maybe_threshold: float | None = Field(default=None, ge=0.2, le=0.9)
    ml_gate_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    suppressions: list[SignalSuppressionConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_feature_overrides(self) -> FeatureOverrideConfig:
        """Validate override keys and threshold ordering when both are present."""
        unsupported = sorted(set(self.deterministic).difference(_SUPPORTED_SIGNAL_NAMES))
        if unsupported:
            message = f"Unsupported deterministic override names: {unsupported!r}"
            raise ValueError(message)
        non_deterministic = sorted(
            set(self.deterministic).intersection({"name_similarity", "ml_score"})
        )
        if non_deterministic:
            message = (
                "Non-deterministic signals cannot use deterministic overrides: "
                f"{non_deterministic!r}"
            )
            raise ValueError(message)
        if (
            self.duplicate_threshold is not None
            and self.maybe_threshold is not None
            and self.duplicate_threshold <= self.maybe_threshold
        ):
            message = "duplicate_threshold must be strictly greater than maybe_threshold"
            raise ValueError(message)
        if (
            self.maybe_threshold is not None
            and self.low_maybe_threshold is not None
            and self.maybe_threshold <= self.low_maybe_threshold
        ):
            message = "maybe_threshold must be strictly greater than low_maybe_threshold"
            raise ValueError(message)
        section_values = [
            self.deterministic_section_weight,
            self.nlp_section_weight,
            self.ml_section_weight,
        ]
        if any(value is not None for value in section_values):
            if not all(value is not None for value in section_values):
                message = "Section weight overrides must set all three section weights together"
                raise ValueError(message)
            total = sum(float(value) for value in section_values if value is not None)
            if abs(total - 1.0) > 0.001:
                message = "Section weights must sum to 1.0 +/- 0.001"
                raise ValueError(message)
        return self


class PairRuleConfig(BaseStrictModel):
    """Generic pair policy selected from source relation/profile metadata."""

    rule_id: str
    entity_types: list[EntityType] = Field(default_factory=lambda: ["organization", "service"])
    source_relation: Literal[
        "any",
        "same_source",
        "cross_source",
        "same_profile",
        "cross_source_same_profile",
        "cross_profile",
    ] = "any"
    source_profiles: list[str] = Field(default_factory=list)
    feature_overrides: FeatureOverrideConfig = Field(default_factory=FeatureOverrideConfig)


class SourceProfileConfig(BaseStrictModel):
    """Host-assigned abstract source profile membership."""

    source_schemas: list[str] = Field(default_factory=list)

    @field_validator("source_schemas")
    @classmethod
    def normalize_source_schemas(cls, values: list[str]) -> list[str]:
        """Normalize source-schema names for case-insensitive matching."""
        normalized = [value.strip().upper() for value in values if value.strip()]
        return list(dict.fromkeys(normalized))


class SourcePolicyConfig(BaseStrictModel):
    """Generic source-aware policy extension supplied by host applications."""

    source_profiles: dict[str, SourceProfileConfig] = Field(default_factory=dict)
    admission_rules: list[AdmissionRuleConfig] = Field(default_factory=list)
    pair_rules: list[PairRuleConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_policy(self) -> SourcePolicyConfig:
        """Validate rule references against configured source profiles."""
        profile_ids = set(self.source_profiles)
        referenced: set[str] = set()
        for rule in [*self.admission_rules, *self.pair_rules]:
            referenced.update(rule.source_profiles)
        unknown = sorted(referenced.difference(profile_ids))
        if unknown:
            message = f"Unknown source profile references: {unknown!r}"
            raise ValueError(message)
        return self


class MlConfig(BaseStrictModel):
    """ML gating controls for optional third scoring section."""

    ml_enabled: bool = False
    ml_gate_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    ml_base_weight: float = Field(default=0.2, ge=0.0, le=0.6)
    ml_dynamic_weighting_enabled: bool = False
    ml_threshold_fallback: float = Field(default=0.5, ge=0.0, le=1.0)


class CalibrationConfig(BaseStrictModel):
    """Shadow confidence calibration controls."""

    enabled: bool = True
    prior_log_odds: float = Field(default=-0.5, ge=-10.0, le=10.0)
    calibration_version: str = "shadow-log-odds-v1"


class ScoringConfig(BaseStrictModel):
    """Top-level scoring constants for one run scope."""

    deterministic_section_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    nlp_section_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    ml_section_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    duplicate_threshold: float = Field(default=0.82, ge=0.5, le=0.99)
    maybe_threshold: float = Field(default=0.68, ge=0.3, le=0.95)
    low_maybe_threshold: float = Field(default=0.58, ge=0.2, le=0.9)
    min_reason_count_for_keep: int = Field(default=1, ge=0, le=5)
    deterministic: DeterministicConfig
    nlp: NlpConfig
    ml: MlConfig
    calibration: CalibrationConfig

    @model_validator(mode="after")
    def validate_weighting_rules(self) -> ScoringConfig:
        """Validate cross-field constraints required by the RFC."""
        total = self.deterministic_section_weight + self.nlp_section_weight + self.ml_section_weight
        if abs(total - 1.0) > 0.001:
            message = "Section weights must sum to 1.0 +/- 0.001"
            raise ValueError(message)
        if self.duplicate_threshold <= self.maybe_threshold:
            message = "duplicate_threshold must be strictly greater than maybe_threshold"
            raise ValueError(message)
        if self.maybe_threshold <= self.low_maybe_threshold:
            message = "maybe_threshold must be strictly greater than low_maybe_threshold"
            raise ValueError(message)
        return self


class MitigationConfig(BaseStrictModel):
    """Mitigation stage controls and thresholds."""

    enabled: bool = False
    min_embedding_similarity: float = Field(default=0.65, ge=0.0, le=1.0)
    require_reason_match: bool = True


class ClusteringConfig(BaseStrictModel):
    """Correlation clustering solver controls."""

    algorithm: str = "correlative_greedy_v1"
    max_iter: int = Field(default=20, ge=1, le=500)
    min_edge_weight: float = Field(default=0.0, ge=-1.0, le=1.0)
    min_cluster_size: int = Field(default=2, ge=2, le=5000)


class ExecutionConfig(BaseStrictModel):
    """Execution behavior controls."""

    strict_validation_mode: bool = True
    emit_removals_only: bool = True


class MetadataConfig(BaseStrictModel):
    """Run metadata and version identity."""

    team_id: str
    scope_id: str
    entity_type: EntityType
    policy_version: str = "hsds-er-v1"
    model_version: str = "embedding-only-v1"


class EntityResolutionRunConfig(BaseStrictModel):
    """Resolved centralized constants used across all stages in one run."""

    blocking: BlockingConfig
    scoring: ScoringConfig
    mitigation: MitigationConfig
    clustering: ClusteringConfig
    execution: ExecutionConfig
    metadata: MetadataConfig
    source_policy: SourcePolicyConfig = Field(default_factory=SourcePolicyConfig)

    @classmethod
    def defaults_for_entity_type(
        cls,
        *,
        team_id: str,
        scope_id: str,
        entity_type: EntityType,
        policy_version: str = "hsds-er-v1",
        model_version: str = "embedding-only-v1",
    ) -> EntityResolutionRunConfig:
        """Build RFC-aligned defaults for organization or service scope."""
        blocking = BlockingConfig(max_candidates_per_entity=125 if entity_type == "service" else 50)
        deterministic_weights = _build_deterministic_defaults(entity_type=entity_type)
        scoring_values = _build_scoring_defaults(entity_type=entity_type)
        return cls(
            blocking=blocking,
            scoring=ScoringConfig(
                deterministic=deterministic_weights,
                nlp=NlpConfig(
                    fuzzy_threshold=scoring_values["fuzzy_threshold"],
                    standalone_fuzzy_threshold=scoring_values["standalone_fuzzy_threshold"],
                ),
                ml=MlConfig(ml_gate_threshold=scoring_values["ml_gate_threshold"]),
                calibration=CalibrationConfig(
                    prior_log_odds=scoring_values["prior_log_odds"],
                    calibration_version="shadow-log-odds-v1",
                ),
                deterministic_section_weight=scoring_values["deterministic_section_weight"],
                nlp_section_weight=scoring_values["nlp_section_weight"],
                ml_section_weight=scoring_values["ml_section_weight"],
                duplicate_threshold=scoring_values["duplicate_threshold"],
                maybe_threshold=scoring_values["maybe_threshold"],
                low_maybe_threshold=scoring_values["low_maybe_threshold"],
                min_reason_count_for_keep=1,
            ),
            mitigation=MitigationConfig(),
            clustering=ClusteringConfig(),
            execution=ExecutionConfig(),
            metadata=MetadataConfig(
                team_id=team_id,
                scope_id=scope_id,
                entity_type=entity_type,
                policy_version=policy_version,
                model_version=model_version,
            ),
        )


def _build_deterministic_defaults(*, entity_type: EntityType) -> DeterministicConfig:
    """Return per-entity-type deterministic signal defaults."""
    if entity_type == "organization":
        return DeterministicConfig(
            shared_email=DeterministicSignalConfig(weight=0.22),
            shared_phone=DeterministicSignalConfig(weight=0.20),
            shared_domain=DeterministicSignalConfig(weight=0.06),
            shared_taxonomy=DeterministicSignalConfig(weight=0.08),
            shared_address=DeterministicSignalConfig(weight=0.25),
            shared_identifier=DeterministicSignalConfig(weight=0.25),
        )
    return DeterministicConfig(
        shared_email=DeterministicSignalConfig(weight=0.16),
        shared_phone=DeterministicSignalConfig(weight=0.22),
        shared_domain=DeterministicSignalConfig(weight=0.04),
        shared_taxonomy=DeterministicSignalConfig(weight=0.12),
        shared_address=DeterministicSignalConfig(weight=0.34),
        shared_identifier=DeterministicSignalConfig(enabled=False, weight=0.0),
    )


def _build_scoring_defaults(*, entity_type: EntityType) -> dict[str, float]:
    """Return scalar scoring defaults aligned with RFC baseline table."""
    if entity_type == "organization":
        return {
            "deterministic_section_weight": 0.45,
            "nlp_section_weight": 0.35,
            "ml_section_weight": 0.20,
            "fuzzy_threshold": 0.88,
            "standalone_fuzzy_threshold": 0.94,
            "ml_gate_threshold": 0.55,
            "duplicate_threshold": 0.82,
            "maybe_threshold": 0.68,
            "low_maybe_threshold": 0.58,
            "prior_log_odds": -0.5,
        }
    # Service-specific calibration notes:
    # - HSDS services from different 211 schemas are copies of the same AIRS master
    #   record. Names and phones are identical text, so embedding cosine similarity
    #   is uniformly ~0.81 for all candidate pairs — the ML section adds zero
    #   discriminative power for this dataset (confirmed by audit: ML_BIN=0.81 for
    #   all 13,667 pairs).
    # - The only discriminative signal is whether the pair also shares an address:
    #     phone + name only:         score ≈ 0.665  → needs human review
    #     phone + name + address:    score ≈ 0.733  → high-confidence, auto-cluster
    # - Section weights are kept at original proportions. Thresholds are calibrated
    #   to bracket the two observed score clusters:
    #     duplicate_threshold = 0.70  → phone+name+address (0.733) auto-clusters
    #     maybe_threshold     = 0.62  → phone+name-only (0.665) enters review queue
    return {
        "deterministic_section_weight": 0.40,
        "nlp_section_weight": 0.40,
        "ml_section_weight": 0.20,
        "fuzzy_threshold": 0.86,
        "standalone_fuzzy_threshold": 0.92,
        "ml_gate_threshold": 0.50,
        "duplicate_threshold": 0.70,
        "maybe_threshold": 0.62,
        "low_maybe_threshold": 0.54,
        "prior_log_odds": -0.25,
    }
