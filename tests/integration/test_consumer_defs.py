"""Tests for consumer Dagster definitions module."""

from __future__ import annotations

from consumer import defs


def test_consumer_defs_exposes_il211_jobs() -> None:
    """Consumer definitions should include generated IL211 jobs from registry."""
    assert defs.jobs is not None
    job_names = sorted(job.name for job in defs.jobs)
    assert "entity_resolution__il211_regional__organization" in job_names
    assert "entity_resolution__il211_regional__service" in job_names
