"""Validation tests for centralized run-config rules."""

from __future__ import annotations

import pytest

from hsds_entity_resolution.config.entity_resolution_run_config import (
    EntityResolutionRunConfig,
)


def test_weight_sum_validation_rejects_invalid_configuration() -> None:
    """Section weights must sum to approximately one."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="organization",
    ).model_dump()
    payload["scoring"]["deterministic_section_weight"] = 0.9
    payload["scoring"]["nlp_section_weight"] = 0.9
    payload["scoring"]["ml_section_weight"] = 0.9

    with pytest.raises(ValueError, match="Section weights"):
        _ = EntityResolutionRunConfig.model_validate(payload)


def test_overlap_prefilter_channels_reject_unknown_channel() -> None:
    """Blocking config should reject unsupported overlap prefilter channels."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team",
        scope_id="scope",
        entity_type="organization",
    ).model_dump()
    payload["blocking"]["overlap_prefilter_channels"] = ["email", "zipcode"]

    with pytest.raises(ValueError, match="Unsupported overlap prefilter channels"):
        _ = EntityResolutionRunConfig.model_validate(payload)
