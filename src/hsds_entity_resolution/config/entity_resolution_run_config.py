"""Centralized run configuration for entity-resolution stages."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hsds_entity_resolution.types.domain import EntityType

_SUPPORTED_BLOCKING_OVERLAP_CHANNELS = {
    "email",
    "phone",
    "website",
    "taxonomy",
    "location",
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


class DeterministicSignalConfig(BaseStrictModel):
    """Configuration for one deterministic overlap signal."""

    enabled: bool = True
    weight: float = Field(default=0.2, ge=0.0, le=0.6)


class DeterministicConfig(BaseStrictModel):
    """Deterministic scoring controls."""

    shared_email: DeterministicSignalConfig
    shared_phone: DeterministicSignalConfig
    shared_domain: DeterministicSignalConfig
    shared_address: DeterministicSignalConfig
    shared_identifier: DeterministicSignalConfig


class NlpConfig(BaseStrictModel):
    """Name/description fuzzy matching controls."""

    fuzzy_algorithm: str = "sequence_matcher"
    fuzzy_threshold: float = Field(default=0.88, ge=0.6, le=0.98)
    number_mismatch_veto_enabled: bool = True
    standalone_fuzzy_threshold: float = Field(default=0.94, ge=0.7, le=0.99)


class MlConfig(BaseStrictModel):
    """ML gating controls for optional third scoring section."""

    ml_enabled: bool = False
    ml_gate_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    ml_base_weight: float = Field(default=0.2, ge=0.0, le=0.6)
    ml_dynamic_weighting_enabled: bool = False
    ml_threshold_fallback: float = Field(default=0.5, ge=0.0, le=1.0)


class ScoringConfig(BaseStrictModel):
    """Top-level scoring constants for one run scope."""

    deterministic_section_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    nlp_section_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    ml_section_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    duplicate_threshold: float = Field(default=0.82, ge=0.5, le=0.99)
    maybe_threshold: float = Field(default=0.68, ge=0.3, le=0.95)
    min_reason_count_for_keep: int = Field(default=1, ge=0, le=5)
    deterministic: DeterministicConfig
    nlp: NlpConfig
    ml: MlConfig

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
        deterministic_weights = _build_deterministic_defaults(entity_type=entity_type)
        scoring_values = _build_scoring_defaults(entity_type=entity_type)
        return cls(
            blocking=BlockingConfig(),
            scoring=ScoringConfig(
                deterministic=deterministic_weights,
                nlp=NlpConfig(
                    fuzzy_threshold=scoring_values["fuzzy_threshold"],
                    standalone_fuzzy_threshold=scoring_values["standalone_fuzzy_threshold"],
                ),
                ml=MlConfig(ml_gate_threshold=scoring_values["ml_gate_threshold"]),
                deterministic_section_weight=scoring_values["deterministic_section_weight"],
                nlp_section_weight=scoring_values["nlp_section_weight"],
                ml_section_weight=scoring_values["ml_section_weight"],
                duplicate_threshold=scoring_values["duplicate_threshold"],
                maybe_threshold=scoring_values["maybe_threshold"],
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
            shared_domain=DeterministicSignalConfig(weight=0.08),
            shared_address=DeterministicSignalConfig(weight=0.25),
            shared_identifier=DeterministicSignalConfig(weight=0.25),
        )
    return DeterministicConfig(
        shared_email=DeterministicSignalConfig(weight=0.16),
        shared_phone=DeterministicSignalConfig(weight=0.22),
        shared_domain=DeterministicSignalConfig(weight=0.04),
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
    }
