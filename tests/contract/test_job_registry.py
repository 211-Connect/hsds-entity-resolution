"""Contract tests for consumer job registry configuration."""

from __future__ import annotations

from pathlib import Path

from consumer.consumer_adapter.job_registry import (
    get_job,
    get_training_feature_job,
    load_job_registry,
)

JOB_REGISTRY_PATH = Path("consumer/config/entity_resolution_jobs.yaml")


def test_job_registry_contains_il211_tester_job() -> None:
    """Registry should include the IL211 initial tester job."""
    registry = load_job_registry(JOB_REGISTRY_PATH)
    assert "il211_regional" in registry.entity_resolution_jobs
    job = registry.entity_resolution_jobs["il211_regional"]
    assert job.team_id == "IL211"
    assert set(job.entity_types) == {"organization", "service"}
    assert "NE211" in job.target_schemas


def test_get_job_resolves_enabled_job() -> None:
    """Job lookup should return loaded key/config for enabled job."""
    loaded = get_job(JOB_REGISTRY_PATH, "il211_regional")
    assert loaded.job_key == "il211_regional"
    assert loaded.config.enabled is True


def test_training_feature_job_registry_contains_il211_job() -> None:
    """Registry should include the IL211 training feature materialization job."""
    registry = load_job_registry(JOB_REGISTRY_PATH)
    assert "il211_regional_training" in registry.training_feature_jobs
    job = registry.training_feature_jobs["il211_regional_training"]
    assert job.team_id == "IL211"
    assert job.scope_id == "il211_regional"
    assert set(job.entity_types) == {"organization", "service"}
    assert job.review_database == "DEDUPLICATION"
    assert job.runtime_schema == "ER_RUNTIME"
    assert job.feature_schema_version == "ml-features-v1"


def test_get_training_feature_job_resolves_enabled_job() -> None:
    """Training feature job lookup should return loaded key/config."""
    loaded = get_training_feature_job(JOB_REGISTRY_PATH, "il211_regional_training")
    assert loaded.job_key == "il211_regional_training"
    assert loaded.config.enabled is True
