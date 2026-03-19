"""Tests for correlation clustering stage behavior."""

from __future__ import annotations

import polars as pl

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.cluster_pairs import cluster_pairs


def test_cluster_pairs_deterministic_for_identical_input() -> None:
    """Repeated runs over identical input should produce the same cluster IDs."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-cluster-1",
        scope_id="scope-cluster-1",
        entity_type="organization",
    )
    finalized = _triangle_fixture()
    first = cluster_pairs(
        finalized_scored_pairs=finalized,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )
    second = cluster_pairs(
        finalized_scored_pairs=finalized,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )
    assert first.clusters.height == second.clusters.height
    assert first.cluster_pairs.height == second.cluster_pairs.height
    assert first.clusters.select("cluster_id").to_series().to_list() == (
        second.clusters.select("cluster_id").to_series().to_list()
    )


def test_cluster_pairs_splits_triangle_conflict() -> None:
    """Contradictory triangles should avoid full transitive closure under CC objective."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-cluster-2",
        scope_id="scope-cluster-2",
        entity_type="organization",
    )
    finalized = _triangle_fixture()
    result = cluster_pairs(
        finalized_scored_pairs=finalized,
        removed_pair_ids=pl.DataFrame(),
        config=config,
    )
    emitted_keys = set(result.cluster_pairs.get_column("pair_key").to_list())
    assert emitted_keys.issubset({"a::b", "a::c"})
    assert len(emitted_keys) == 1
    assert "b::c" not in emitted_keys


def test_cluster_pairs_ignores_removed_keys() -> None:
    """Removed pair keys should be excluded from clustering bridge outputs."""
    config = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-cluster-3",
        scope_id="scope-cluster-3",
        entity_type="organization",
    )
    finalized = _triangle_fixture()
    removed = pl.DataFrame({"pair_key": ["a::b"], "cleanup_reason": ["entity_deleted"]})
    result = cluster_pairs(
        finalized_scored_pairs=finalized,
        removed_pair_ids=removed,
        config=config,
    )
    emitted_keys = set(result.cluster_pairs.get_column("pair_key").to_list())
    assert "a::b" not in emitted_keys


def _triangle_fixture() -> pl.DataFrame:
    """Build one triangle with two strong matches and one strong mismatch."""
    return pl.DataFrame(
        {
            "pair_key": ["a::b", "a::c", "b::c"],
            "entity_a_id": ["a", "a", "b"],
            "entity_b_id": ["b", "c", "c"],
            "entity_type": ["organization", "organization", "organization"],
            "policy_version": ["hsds-er-v1", "hsds-er-v1", "hsds-er-v1"],
            "model_version": ["m", "m", "m"],
            "deterministic_section_score": [0.9, 0.9, 0.1],
            "nlp_section_score": [0.9, 0.9, 0.1],
            "ml_section_score": [0.9, 0.9, 0.1],
            "final_score": [0.95, 0.94, 0.10],
            "predicted_duplicate": [True, True, False],
            "embedding_similarity": [0.95, 0.94, 0.10],
            "mitigation_reason": [None, None, None],
        }
    )
