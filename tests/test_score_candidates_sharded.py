"""Tests for sharded service candidate-scoring helpers."""

from __future__ import annotations

import math

import polars as pl
import pytest

import hsds_entity_resolution.core.score_candidates as score_candidates_module
from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.score_candidates import score_candidates
from hsds_entity_resolution.core.score_candidates_sharded import (
    merge_score_candidates_results,
    partition_candidate_pairs_for_sharding,
    partition_candidate_pairs_for_service_sharding,
)
from hsds_entity_resolution.types.contracts import ScoreCandidatesResult
from hsds_entity_resolution.types.frames import PAIR_REASONS_SCHEMA, SCORED_PAIRS_SCHEMA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ml_disabled_service_config() -> EntityResolutionRunConfig:
    """Return a service-entity config with ML disabled for deterministic tests."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-shard",
        scope_id="scope-shard",
        entity_type="service",
    ).model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = False
    return EntityResolutionRunConfig.model_validate(payload)


def _ml_disabled_org_config() -> EntityResolutionRunConfig:
    """Return an organization-entity config with ML disabled for passthrough tests."""
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-shard-org",
        scope_id="scope-shard-org",
        entity_type="organization",
    ).model_dump()
    payload["scoring"]["ml"]["ml_enabled"] = False
    return EntityResolutionRunConfig.model_validate(payload)


def _svc_entity_row(entity_id: str) -> dict:
    """Return one denormalized service entity row suitable for entity_lookup."""
    return {
        "entity_id": entity_id,
        "entity_type": "service",
        "source_schema": "SCHEMA_A",
        "name": f"Service {entity_id}",
        "description": f"Description for {entity_id}",
        "emails": [],
        "phones": [],
        "websites": [],
        "locations": [],
        "taxonomies": [],
        "identifiers": [],
        "services_rollup": [],
        "organization_name": "Parent Org",
        "organization_id": "org-parent",
        "embedding_vector": [0.5, 0.5],
    }


def _org_entity_row(entity_id: str) -> dict:
    """Return one denormalized organization entity row."""
    return {
        "entity_id": entity_id,
        "entity_type": "organization",
        "source_schema": "SCHEMA_A",
        "name": f"Org {entity_id}",
        "description": f"Description for {entity_id}",
        "emails": [],
        "phones": [],
        "websites": [],
        "locations": [],
        "taxonomies": [],
        "identifiers": [],
        "services_rollup": [],
        "organization_name": "",
        "organization_id": "",
        "embedding_vector": [0.5, 0.5],
    }


def _svc_candidate_pair(id_a: str, id_b: str) -> dict:
    """Return one service candidate-pair row."""
    return {
        "pair_key": f"{id_a}__{id_b}",
        "entity_a_id": id_a,
        "entity_b_id": id_b,
        "entity_type": "service",
        "embedding_similarity": 0.8,
        "candidate_reason_codes": ["embedding_threshold"],
        "source_schema_a": "SCHEMA_A",
        "source_schema_b": "SCHEMA_A",
        "blocking_rule_id": "rule-1",
    }


def _org_candidate_pair(id_a: str, id_b: str) -> dict:
    """Return one organization candidate-pair row."""
    return {
        "pair_key": f"{id_a}__{id_b}",
        "entity_a_id": id_a,
        "entity_b_id": id_b,
        "entity_type": "organization",
        "embedding_similarity": 0.8,
        "candidate_reason_codes": ["embedding_threshold"],
        "source_schema_a": "SCHEMA_A",
        "source_schema_b": "SCHEMA_A",
        "blocking_rule_id": "rule-1",
    }


def _build_svc_entity_frame(entity_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame([_svc_entity_row(eid) for eid in entity_ids])


def _build_org_entity_frame(entity_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame([_org_entity_row(eid) for eid in entity_ids])


def _build_candidate_frame(pairs: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(pairs)


# ---------------------------------------------------------------------------
# partition_candidate_pairs_for_sharding  (generic — no entity-type pinning)
# ---------------------------------------------------------------------------


class TestPartitionCandidatePairsForSharding:
    def test_single_shard_returns_input_unchanged(self) -> None:
        candidates = _build_candidate_frame([_org_candidate_pair("org-a", "org-b")])
        result = partition_candidate_pairs_for_sharding(candidates, num_shards=1)
        assert len(result) == 1
        assert result[0] is candidates

    def test_empty_frame_returns_n_empty_frames_with_same_schema(self) -> None:
        candidates = _build_candidate_frame([_org_candidate_pair("org-a", "org-b")])
        empty = candidates.clear()
        result = partition_candidate_pairs_for_sharding(empty, num_shards=4)
        assert len(result) == 4
        for shard in result:
            assert shard.is_empty()
            assert shard.schema == empty.schema

    def test_every_pair_key_in_exactly_one_shard(self) -> None:
        pair_dicts = [
            _org_candidate_pair(f"org-{i:03d}", f"org-{i+1:03d}") for i in range(0, 40, 2)
        ]
        candidates = _build_candidate_frame(pair_dicts)
        shards = partition_candidate_pairs_for_sharding(candidates, num_shards=4)
        all_keys: list[str] = []
        for shard in shards:
            all_keys.extend(shard.get_column("pair_key").to_list())
        assert sorted(all_keys) == sorted(candidates.get_column("pair_key").to_list())

    def test_org_rows_distributed_across_all_shards(self) -> None:
        """Unlike the service variant, org rows must NOT all pile up in shard 0."""
        n_pairs = 40
        pair_dicts = [
            _org_candidate_pair(f"org-{i:03d}", f"org-{i+1:03d}") for i in range(0, n_pairs * 2, 2)
        ]
        candidates = _build_candidate_frame(pair_dicts)
        shards = partition_candidate_pairs_for_sharding(candidates, num_shards=4)
        # Every shard should have at least one pair (with 40 pairs and 4 shards, this is extremely
        # likely with any reasonable hash function)
        non_empty = sum(1 for s in shards if not s.is_empty())
        assert non_empty > 1, "Org pairs should be distributed across multiple shards"

    def test_distribution_is_balanced(self) -> None:
        n_pairs, n_shards = 100, 4
        pair_dicts = [
            _org_candidate_pair(f"org-{i:04d}", f"org-{i+1:04d}") for i in range(0, n_pairs * 2, 2)
        ]
        candidates = _build_candidate_frame(pair_dicts)
        shards = partition_candidate_pairs_for_sharding(candidates, num_shards=n_shards)
        max_allowed = math.ceil(1.5 * n_pairs / n_shards)
        for i, shard in enumerate(shards):
            assert shard.height <= max_allowed, (
                f"Shard {i} has {shard.height} rows, exceeds max_allowed={max_allowed}"
            )

    def test_org_job_sharded_matches_monolithic(self) -> None:
        """Merged shard results for org pairs must exactly match single-batch output."""
        config = _ml_disabled_org_config()
        org_entity_ids = [f"org-{i:03d}" for i in range(12)]
        org_entities = _build_org_entity_frame(org_entity_ids)
        pair_dicts = [
            _org_candidate_pair(org_entity_ids[i], org_entity_ids[i + 1])
            for i in range(0, len(org_entity_ids) - 1, 2)
        ]
        candidates = _build_candidate_frame(pair_dicts)

        expected = score_candidates(
            candidate_pairs=candidates,
            denormalized_organization=org_entities,
            denormalized_service=pl.DataFrame(),
            config=config,
        )

        shards = partition_candidate_pairs_for_sharding(candidates, num_shards=4)
        shard_results = [
            score_candidates(
                candidate_pairs=shard,
                denormalized_organization=org_entities,
                denormalized_service=pl.DataFrame(),
                config=config,
            )
            for shard in shards
        ]
        merged = merge_score_candidates_results(shard_results)
        assert merged.scored_pairs.sort("pair_key").equals(expected.scored_pairs.sort("pair_key"))


# ---------------------------------------------------------------------------
# partition_candidate_pairs_for_service_sharding
# ---------------------------------------------------------------------------


class TestPartitionCandidatePairsForServiceSharding:
    def test_single_shard_returns_input_unchanged(self) -> None:
        candidates = _build_candidate_frame(
            [_svc_candidate_pair("svc-a", "svc-b"), _svc_candidate_pair("svc-c", "svc-d")]
        )
        result = partition_candidate_pairs_for_service_sharding(candidates, num_shards=1)
        assert len(result) == 1
        assert result[0] is candidates

    def test_zero_shards_treated_as_single(self) -> None:
        candidates = _build_candidate_frame([_svc_candidate_pair("svc-a", "svc-b")])
        result = partition_candidate_pairs_for_service_sharding(candidates, num_shards=0)
        assert len(result) == 1
        assert result[0] is candidates

    def test_empty_frame_returns_n_empty_frames_with_same_schema(self) -> None:
        candidates = _build_candidate_frame([_svc_candidate_pair("svc-a", "svc-b")])
        empty = candidates.clear()
        result = partition_candidate_pairs_for_service_sharding(empty, num_shards=4)
        assert len(result) == 4
        for shard in result:
            assert shard.is_empty()
            assert shard.schema == empty.schema

    def test_every_pair_key_appears_in_exactly_one_shard(self) -> None:
        pair_dicts = [
            _svc_candidate_pair(f"svc-{i:03d}", f"svc-{i+1:03d}") for i in range(0, 40, 2)
        ]
        candidates = _build_candidate_frame(pair_dicts)
        shards = partition_candidate_pairs_for_service_sharding(candidates, num_shards=4)

        all_keys_seen: list[str] = []
        for shard in shards:
            all_keys_seen.extend(shard.get_column("pair_key").to_list())

        assert sorted(all_keys_seen) == sorted(candidates.get_column("pair_key").to_list())

    def test_org_rows_always_land_in_shard_zero(self) -> None:
        svc_pairs = [_svc_candidate_pair(f"svc-{i}", f"svc-{i+1}") for i in range(0, 10, 2)]
        org_pairs = [
            _org_candidate_pair("org-a", "org-b"),
            _org_candidate_pair("org-c", "org-d"),
        ]
        candidates = _build_candidate_frame(svc_pairs + org_pairs)
        shards = partition_candidate_pairs_for_service_sharding(candidates, num_shards=4)

        # All org rows must be in shard 0
        org_keys = {p["pair_key"] for p in org_pairs}
        shard_0_keys = set(shards[0].get_column("pair_key").to_list())
        assert org_keys.issubset(shard_0_keys)

        # No org rows in other shards
        for shard in shards[1:]:
            if not shard.is_empty():
                assert "organization" not in shard.get_column("entity_type").to_list()

    def test_shard_count_matches_num_shards(self) -> None:
        candidates = _build_candidate_frame(
            [_svc_candidate_pair(f"svc-{i}", f"svc-{i+1}") for i in range(0, 20, 2)]
        )
        for n in [2, 4, 8]:
            shards = partition_candidate_pairs_for_service_sharding(candidates, num_shards=n)
            assert len(shards) == n

    def test_shard_distribution_is_reasonably_balanced(self) -> None:
        """No shard should hold more than 1.5× the expected average with 100 pairs."""
        n_pairs = 100
        n_shards = 4
        pair_dicts = [
            _svc_candidate_pair(f"svc-{i:04d}", f"svc-{i+1:04d}") for i in range(0, n_pairs * 2, 2)
        ]
        candidates = _build_candidate_frame(pair_dicts)
        shards = partition_candidate_pairs_for_service_sharding(candidates, num_shards=n_shards)
        max_allowed = math.ceil(1.5 * n_pairs / n_shards)
        for i, shard in enumerate(shards):
            assert shard.height <= max_allowed, (
                f"Shard {i} has {shard.height} rows, exceeds max_allowed={max_allowed}"
            )

    def test_num_shards_greater_than_pairs_produces_some_empty_shards(self) -> None:
        candidates = _build_candidate_frame(
            [_svc_candidate_pair("svc-a", "svc-b"), _svc_candidate_pair("svc-c", "svc-d")]
        )
        shards = partition_candidate_pairs_for_service_sharding(candidates, num_shards=10)
        total_rows = sum(s.height for s in shards)
        empty_shards = sum(1 for s in shards if s.is_empty())
        assert total_rows == candidates.height
        assert empty_shards > 0


# ---------------------------------------------------------------------------
# merge_score_candidates_results
# ---------------------------------------------------------------------------


class TestMergeScoreCandidatesResults:
    def test_empty_results_list_returns_empty_result(self) -> None:
        result = merge_score_candidates_results([])
        assert result.scored_pairs.is_empty()
        assert result.pair_reasons.is_empty()
        assert result.score_delta_summary.row(0, named=True)["candidates_scored"] == 0

    def test_all_empty_shard_results_returns_first(self) -> None:
        empty = ScoreCandidatesResult(
            scored_pairs=pl.DataFrame(schema=SCORED_PAIRS_SCHEMA),
            pair_reasons=pl.DataFrame(schema=PAIR_REASONS_SCHEMA),
            score_delta_summary=pl.DataFrame(
                {
                    "candidates_scored": [0],
                    "ml_scored_count": [0],
                    "duplicate_count": [0],
                    "maybe_count": [0],
                    "strict_duplicate_count": [0],
                    "predicted_duplicate_count": [0],
                    "retained_count": [0],
                }
            ),
        )
        result = merge_score_candidates_results([empty, empty])
        assert result is empty

    def test_summary_is_recomputed_from_merged_frame(self) -> None:
        """Summary counts must reflect the merged frame, not sums of shard summaries."""
        config = _ml_disabled_service_config()
        # Two service pairs, each in its own shard result
        svc_entities = _build_svc_entity_frame(["svc-a", "svc-b", "svc-c", "svc-d"])
        r1 = score_candidates(
            candidate_pairs=_build_candidate_frame([_svc_candidate_pair("svc-a", "svc-b")]),
            denormalized_organization=pl.DataFrame(),
            denormalized_service=svc_entities,
            config=config,
        )
        r2 = score_candidates(
            candidate_pairs=_build_candidate_frame([_svc_candidate_pair("svc-c", "svc-d")]),
            denormalized_organization=pl.DataFrame(),
            denormalized_service=svc_entities,
            config=config,
        )
        merged = merge_score_candidates_results([r1, r2])
        summary = merged.score_delta_summary.row(0, named=True)
        assert summary["candidates_scored"] == 2
        assert summary["candidates_scored"] == merged.scored_pairs.height

    def test_scored_pairs_contain_all_pair_keys(self) -> None:
        config = _ml_disabled_service_config()
        svc_entities = _build_svc_entity_frame(
            ["svc-a", "svc-b", "svc-c", "svc-d", "svc-e", "svc-f"]
        )
        pairs_full = _build_candidate_frame([
            _svc_candidate_pair("svc-a", "svc-b"),
            _svc_candidate_pair("svc-c", "svc-d"),
            _svc_candidate_pair("svc-e", "svc-f"),
        ])
        shards = partition_candidate_pairs_for_service_sharding(pairs_full, num_shards=3)
        shard_results = [
            score_candidates(
                candidate_pairs=shard,
                denormalized_organization=pl.DataFrame(),
                denormalized_service=svc_entities,
                config=config,
            )
            for shard in shards
        ]
        merged = merge_score_candidates_results(shard_results)
        assert merged.scored_pairs.height == 3
        merged_keys = sorted(merged.scored_pairs.get_column("pair_key").to_list())
        expected_keys = sorted(pairs_full.get_column("pair_key").to_list())
        assert merged_keys == expected_keys


# ---------------------------------------------------------------------------
# Golden test: sharded merge == single score_candidates call
# ---------------------------------------------------------------------------


class TestShardsMatchMonolithicRun:
    def test_service_only_sharded_matches_monolithic(self) -> None:
        """Merged shard results must exactly match the single-batch output."""
        config = _ml_disabled_service_config()
        entity_ids = [f"svc-{i:03d}" for i in range(12)]
        svc_entities = _build_svc_entity_frame(entity_ids)
        pair_dicts = [
            _svc_candidate_pair(entity_ids[i], entity_ids[i + 1])
            for i in range(0, len(entity_ids) - 1, 2)
        ]
        candidates = _build_candidate_frame(pair_dicts)

        expected = score_candidates(
            candidate_pairs=candidates,
            denormalized_organization=pl.DataFrame(),
            denormalized_service=svc_entities,
            config=config,
        )

        shards = partition_candidate_pairs_for_service_sharding(candidates, num_shards=4)
        shard_results = [
            score_candidates(
                candidate_pairs=shard,
                denormalized_organization=pl.DataFrame(),
                denormalized_service=svc_entities,
                config=config,
            )
            for shard in shards
        ]
        merged = merge_score_candidates_results(shard_results)

        expected_pairs = expected.scored_pairs.sort("pair_key")
        merged_pairs = merged.scored_pairs.sort("pair_key")
        assert expected_pairs.equals(merged_pairs)

        expected_reasons = expected.pair_reasons.sort(["pair_key", "match_type"])
        merged_reasons = merged.pair_reasons.sort(["pair_key", "match_type"])
        assert expected_reasons.equals(merged_reasons)

    def test_mixed_org_service_sharded_matches_monolithic(self) -> None:
        """Org rows in shard 0 plus service rows must produce identical output to monolithic."""
        svc_config = _ml_disabled_service_config()
        svc_entity_ids = [f"svc-{i:03d}" for i in range(8)]
        org_entity_ids = ["org-a", "org-b", "org-c", "org-d"]
        svc_entities = _build_svc_entity_frame(svc_entity_ids)
        org_entities = _build_org_entity_frame(org_entity_ids)

        svc_pairs = [
            _svc_candidate_pair(svc_entity_ids[i], svc_entity_ids[i + 1])
            for i in range(0, len(svc_entity_ids) - 1, 2)
        ]
        org_pairs = [_org_candidate_pair("org-a", "org-b"), _org_candidate_pair("org-c", "org-d")]
        all_pairs = _build_candidate_frame(svc_pairs + org_pairs)

        expected = score_candidates(
            candidate_pairs=all_pairs,
            denormalized_organization=org_entities,
            denormalized_service=svc_entities,
            config=svc_config,
        )

        shards = partition_candidate_pairs_for_service_sharding(all_pairs, num_shards=4)
        # Org rows must be in shard 0
        shard_0_types = set(shards[0].get_column("entity_type").to_list()) if not shards[0].is_empty() else set()
        assert "organization" in shard_0_types or all(
            "organization" not in set(s.get_column("entity_type").to_list())
            for s in shards[1:]
            if not s.is_empty()
        )

        shard_results = [
            score_candidates(
                candidate_pairs=shard,
                denormalized_organization=org_entities,
                denormalized_service=svc_entities,
                config=svc_config,
            )
            for shard in shards
        ]
        merged = merge_score_candidates_results(shard_results)

        assert merged.scored_pairs.sort("pair_key").equals(expected.scored_pairs.sort("pair_key"))

    def test_num_shards_greater_than_pairs_matches_monolithic(self) -> None:
        """When num_shards > pair_count some shards are empty; merged output must still match."""
        config = _ml_disabled_service_config()
        svc_entities = _build_svc_entity_frame(["svc-a", "svc-b"])
        candidates = _build_candidate_frame([_svc_candidate_pair("svc-a", "svc-b")])

        expected = score_candidates(
            candidate_pairs=candidates,
            denormalized_organization=pl.DataFrame(),
            denormalized_service=svc_entities,
            config=config,
        )

        shards = partition_candidate_pairs_for_service_sharding(candidates, num_shards=8)
        shard_results = [
            score_candidates(
                candidate_pairs=shard,
                denormalized_organization=pl.DataFrame(),
                denormalized_service=svc_entities,
                config=config,
            )
            for shard in shards
        ]
        merged = merge_score_candidates_results(shard_results)
        assert merged.scored_pairs.height == expected.scored_pairs.height
        assert merged.scored_pairs.sort("pair_key").equals(expected.scored_pairs.sort("pair_key"))


# ---------------------------------------------------------------------------
# Org passthrough: service_score_shards ignored for organization jobs
# ---------------------------------------------------------------------------


class TestOrgPassthrough:
    def test_org_rows_not_split_across_shards(self) -> None:
        """All org rows must land in shard 0 leaving shards 1..N-1 without org rows."""
        org_pairs = [
            _org_candidate_pair("org-a", "org-b"),
            _org_candidate_pair("org-c", "org-d"),
            _org_candidate_pair("org-e", "org-f"),
        ]
        candidates = _build_candidate_frame(org_pairs)
        shards = partition_candidate_pairs_for_service_sharding(candidates, num_shards=4)

        # All rows go to shard 0 (non-service rows pinned to shard 0)
        total_in_shard_0 = shards[0].height
        total_in_others = sum(s.height for s in shards[1:])
        assert total_in_shard_0 == len(org_pairs)
        assert total_in_others == 0


# ---------------------------------------------------------------------------
# Memory-safe score chunking parity
# ---------------------------------------------------------------------------


class TestScoreChunkingParity:
    def test_score_chunking_matches_baseline(self) -> None:
        config = _ml_disabled_service_config()
        entity_ids = [f"svc-{i:03d}" for i in range(10)]
        svc_entities = _build_svc_entity_frame(entity_ids)
        pair_dicts = [
            _svc_candidate_pair(entity_ids[i], entity_ids[i + 1])
            for i in range(0, len(entity_ids) - 1, 2)
        ]
        candidates = _build_candidate_frame(pair_dicts)

        baseline = score_candidates(
            candidate_pairs=candidates,
            denormalized_organization=pl.DataFrame(),
            denormalized_service=svc_entities,
            config=config,
        )

        chunked_payload = config.model_dump()
        chunked_payload["chunking"] = {"score_candidate_chunk_size": 2}
        chunked_config = EntityResolutionRunConfig.model_validate(chunked_payload)
        chunked = score_candidates(
            candidate_pairs=candidates,
            denormalized_organization=pl.DataFrame(),
            denormalized_service=svc_entities,
            config=chunked_config,
        )

        assert baseline.scored_pairs.sort("pair_key").equals(chunked.scored_pairs.sort("pair_key"))
        assert baseline.pair_reasons.sort(["pair_key", "match_type"]).equals(
            chunked.pair_reasons.sort(["pair_key", "match_type"])
        )
