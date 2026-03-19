"""Contract tests for consumer job registry configuration."""

from __future__ import annotations

from pathlib import Path

from consumer.consumer_adapter.job_registry import get_job, load_job_registry

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
