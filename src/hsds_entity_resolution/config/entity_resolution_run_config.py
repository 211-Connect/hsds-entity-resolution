"""Centralized run configuration for entity-resolution stages."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hsds_entity_resolution.types.domain import EntityType

_ML_SECTION_REMOVED_MESSAGE = (
    "The ML scoring section was removed in hsds-entity-resolution 1.2.0: the score is "
    "deterministic plus NLP only, and the shadow/legacy calibration scores are gone. "
    "Remove {keys} from this config; deterministic_section_weight and nlp_section_weight "
    "must now sum to 1.0."
)


_ADMISSION_REPLACED_MESSAGE = (
    "Per-lane candidate admission was replaced in hsds-entity-resolution 2.1.0 (ISS-2177) "
    "by one rule: a pair is a candidate when it shares an Informative Key value or its "
    "embedding similarity is at or above blocking.similarity_threshold, unless a "
    "Structural Exclusion vetoes it. Remove {keys} from this config and pass "
    "informative_keys / structural_exclusion to the pipeline instead."
)


def _reject_replaced_admission_keys(data: Any, *, removed: tuple[str, ...]) -> Any:
    """Raise a migration error when a config still carries replaced admission keys.

    Args:
        data: Raw mapping handed to a pydantic ``mode="before"`` validator.
        removed: Key names that no longer exist in this model.

    Returns:
        ``data`` unchanged when none of the removed keys are present.
    """
    if not isinstance(data, dict):
        return data
    present = sorted(key for key in removed if key in data)
    if present:
        raise ValueError(_ADMISSION_REPLACED_MESSAGE.format(keys=", ".join(present)))
    return data


def _reject_removed_ml_keys(data: Any, *, removed: tuple[str, ...]) -> Any:
    """Raise a migration error when a config still carries removed ML keys.

    Args:
        data: Raw mapping handed to a pydantic ``mode="before"`` validator.
        removed: Key names that no longer exist in this model.

    Returns:
        ``data`` unchanged when none of the removed keys are present.
    """
    if not isinstance(data, dict):
        return data
    present = sorted(key for key in removed if key in data)
    if present:
        raise ValueError(_ML_SECTION_REMOVED_MESSAGE.format(keys=", ".join(present)))
    return data


_SUPPORTED_SIGNAL_NAMES = {
    "shared_email",
    "shared_phone",
    "shared_domain",
    "shared_taxonomy",
    "shared_address",
    "address_plus_taxonomy",
    "address_plus_taxonomy_plus_contact",
    "shared_identifier",
    "name_similarity",
    "organization_name_similarity",
}


class BaseStrictModel(BaseModel):
    """Shared strict pydantic model behavior."""

    model_config = ConfigDict(extra="forbid")


class BlockingConfig(BaseStrictModel):
    """Candidate admission and fanout controls.

    ``similarity_threshold`` is the single embedding floor: a pair at or above it is a
    candidate on similarity alone. ``max_candidates_per_entity`` caps how many such
    pairs one anchor keeps; pairs admitted by a shared Informative Key are not capped by
    it (see ``ChunkingConfig`` for the optional key caps).
    """

    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_candidates_per_entity: int = Field(default=50, ge=1, le=500)
    blocking_batch_size: int = Field(default=5000, ge=1, le=50000)

    @model_validator(mode="before")
    @classmethod
    def reject_replaced_admission_keys(cls, data: Any) -> Any:
        """Fail with a migration message when the removed overlap prefilter is passed."""
        return _reject_replaced_admission_keys(data, removed=("overlap_prefilter_channels",))


class ChunkingConfig(BaseStrictModel):
    """Memory-bounded chunking controls for monolithic (single-shard) ER stages.

    Defaults keep historical unbounded behavior: unset values mean "no chunking"
    and "no contact-overlap caps". Consumers enable knobs explicitly when large
    explicit-backfill jobs must stay within a fixed pod memory budget.
    """

    generate_anchor_chunk_size: int | None = Field(
        default=None,
        ge=1,
        le=500_000,
        description=(
            "When set, process generate-candidate anchors in batches of this size "
            "instead of one monolithic loop."
        ),
    )
    score_candidate_chunk_size: int | None = Field(
        default=None,
        ge=1,
        le=500_000,
        description=(
            "When set, score candidate pairs in batches of this size to bound peak "
            "memory from full-frame materialization."
        ),
    )
    max_contact_overlap_pairs_per_anchor: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        description=(
            "When set, cap the Informative Key pairs considered per anchor. "
            "Unset leaves Informative Key admission uncapped."
        ),
    )
    max_contact_index_fanout: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        description=(
            "When set, truncate Informative Key index values that map to more than this "
            "many entities (deterministic sorted keep). Unset leaves fanout uncapped."
        ),
    )


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
    address_plus_taxonomy: DeterministicSignalConfig = Field(
        default_factory=lambda: DeterministicSignalConfig(enabled=False, weight=0.0)
    )
    address_plus_taxonomy_plus_contact: DeterministicSignalConfig = Field(
        default_factory=lambda: DeterministicSignalConfig(enabled=False, weight=0.0)
    )
    shared_identifier: DeterministicSignalConfig
    organization_name_similarity: DeterministicSignalConfig = Field(
        default_factory=lambda: DeterministicSignalConfig(enabled=False, weight=0.0)
    )


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
    duplicate_threshold: float | None = Field(default=None, ge=0.5, le=0.99)
    maybe_threshold: float | None = Field(default=None, ge=0.3, le=0.95)
    low_maybe_threshold: float | None = Field(default=None, ge=0.2, le=0.9)
    min_review_embedding_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    min_duplicate_embedding_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    embedding_floor_exempt_signals: list[str] = Field(default_factory=list)
    review_on_signals: list[str] = Field(default_factory=list)
    suppressions: list[SignalSuppressionConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_ml_overrides(cls, data: Any) -> Any:
        """Fail with a migration message when removed ML override keys are passed."""
        return _reject_removed_ml_keys(data, removed=("ml_gate_threshold", "ml_section_weight"))

    @model_validator(mode="after")
    def validate_feature_overrides(self) -> FeatureOverrideConfig:
        """Validate override keys and threshold ordering when both are present."""
        unsupported = sorted(set(self.deterministic).difference(_SUPPORTED_SIGNAL_NAMES))
        if unsupported:
            message = f"Unsupported deterministic override names: {unsupported!r}"
            raise ValueError(message)
        non_deterministic = sorted(set(self.deterministic).intersection({"name_similarity"}))
        if non_deterministic:
            message = (
                "Non-deterministic signals cannot use deterministic overrides: "
                f"{non_deterministic!r}"
            )
            raise ValueError(message)
        unsupported_exemptions = sorted(
            set(self.embedding_floor_exempt_signals).difference(_SUPPORTED_SIGNAL_NAMES)
        )
        if unsupported_exemptions:
            message = (
                f"Unsupported embedding floor exemption signal names: {unsupported_exemptions!r}"
            )
            raise ValueError(message)
        unsupported_review_overrides = sorted(
            set(self.review_on_signals).difference(_SUPPORTED_SIGNAL_NAMES)
        )
        if unsupported_review_overrides:
            message = (
                f"Unsupported review_on_signals signal names: {unsupported_review_overrides!r}"
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
        if (
            self.min_review_embedding_similarity is not None
            and self.min_duplicate_embedding_similarity is not None
            and self.min_duplicate_embedding_similarity < self.min_review_embedding_similarity
        ):
            message = (
                "min_duplicate_embedding_similarity must be greater than or equal to "
                "min_review_embedding_similarity"
            )
            raise ValueError(message)
        section_values = [self.deterministic_section_weight, self.nlp_section_weight]
        if any(value is not None for value in section_values):
            if not all(value is not None for value in section_values):
                message = "Section weight overrides must set both section weights together"
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
    """Source-aware scoring overrides supplied by host applications.

    Only scoring reads it: ``pair_rules`` pick per-pair overrides by entity type and the
    two records' source profiles. Candidate admission is generic (ISS-2177).
    """

    source_profiles: dict[str, SourceProfileConfig] = Field(default_factory=dict)
    pair_rules: list[PairRuleConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_replaced_admission_rules(cls, data: Any) -> Any:
        """Fail with a migration message when per-lane admission rules are passed."""
        return _reject_replaced_admission_keys(data, removed=("admission_rules",))

    @model_validator(mode="after")
    def validate_source_policy(self) -> SourcePolicyConfig:
        """Validate pair-rule references against configured source profiles."""
        profile_ids = set(self.source_profiles)
        referenced: set[str] = set()
        for rule in self.pair_rules:
            referenced.update(rule.source_profiles)
        unknown = sorted(referenced.difference(profile_ids))
        if unknown:
            message = f"Unknown source profile references: {unknown!r}"
            raise ValueError(message)
        return self


class ScoringConfig(BaseStrictModel):
    """Top-level scoring constants for one run scope."""

    deterministic_section_weight: float = Field(default=0.5625, ge=0.0, le=1.0)
    nlp_section_weight: float = Field(default=0.4375, ge=0.0, le=1.0)
    duplicate_threshold: float = Field(default=0.82, ge=0.5, le=0.99)
    maybe_threshold: float = Field(default=0.68, ge=0.3, le=0.95)
    low_maybe_threshold: float = Field(default=0.58, ge=0.2, le=0.9)
    min_reason_count_for_keep: int = Field(default=1, ge=0, le=5)
    deterministic: DeterministicConfig
    nlp: NlpConfig

    @model_validator(mode="before")
    @classmethod
    def reject_removed_ml_scoring(cls, data: Any) -> Any:
        """Fail with a migration message when removed ML or calibration keys are passed."""
        return _reject_removed_ml_keys(data, removed=("calibration", "ml", "ml_section_weight"))

    @model_validator(mode="after")
    def validate_weighting_rules(self) -> ScoringConfig:
        """Validate cross-field constraints required by the RFC."""
        total = self.deterministic_section_weight + self.nlp_section_weight
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
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

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
                deterministic_section_weight=scoring_values["deterministic_section_weight"],
                nlp_section_weight=scoring_values["nlp_section_weight"],
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
            chunking=ChunkingConfig(),
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
            address_plus_taxonomy=DeterministicSignalConfig(enabled=False, weight=0.0),
            address_plus_taxonomy_plus_contact=DeterministicSignalConfig(enabled=False, weight=0.0),
            shared_identifier=DeterministicSignalConfig(weight=0.25),
            organization_name_similarity=DeterministicSignalConfig(enabled=False, weight=0.0),
        )
    return DeterministicConfig(
        shared_email=DeterministicSignalConfig(weight=0.16),
        shared_phone=DeterministicSignalConfig(weight=0.22),
        shared_domain=DeterministicSignalConfig(weight=0.04),
        shared_taxonomy=DeterministicSignalConfig(weight=0.12),
        shared_address=DeterministicSignalConfig(weight=0.34),
        address_plus_taxonomy=DeterministicSignalConfig(enabled=False, weight=0.0),
        address_plus_taxonomy_plus_contact=DeterministicSignalConfig(enabled=False, weight=0.0),
        shared_identifier=DeterministicSignalConfig(enabled=False, weight=0.0),
        organization_name_similarity=DeterministicSignalConfig(enabled=False, weight=0.0),
    )


def _build_scoring_defaults(*, entity_type: EntityType) -> dict[str, float]:
    """Return scalar scoring defaults aligned with RFC baseline table.

    Section weights are the pre-1.2.0 deterministic/NLP weights renormalised over the
    two sections that were ever active (organization 0.45/0.35, service 0.40/0.40), so
    every score computed without the removed ML section is reproduced exactly.
    """
    if entity_type == "organization":
        return {
            "deterministic_section_weight": 0.5625,
            "nlp_section_weight": 0.4375,
            "fuzzy_threshold": 0.88,
            "standalone_fuzzy_threshold": 0.94,
            "duplicate_threshold": 0.82,
            "maybe_threshold": 0.68,
            "low_maybe_threshold": 0.58,
        }
    # Service thresholds bracket the two observed score clusters:
    #   phone + name only:         score ≈ 0.665  → needs human review
    #   phone + name + address:    score ≈ 0.733  → high-confidence, auto-cluster
    #   duplicate_threshold = 0.70, maybe_threshold = 0.62
    return {
        "deterministic_section_weight": 0.5,
        "nlp_section_weight": 0.5,
        "fuzzy_threshold": 0.86,
        "standalone_fuzzy_threshold": 0.92,
        "duplicate_threshold": 0.70,
        "maybe_threshold": 0.62,
        "low_maybe_threshold": 0.54,
    }
