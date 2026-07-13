"""Tests for sharded generate-candidates helpers."""

from __future__ import annotations

import polars as pl
import pytest

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.generate_candidates import generate_candidates
from hsds_entity_resolution.core.generate_candidates_sharded import (
    compute_anchor_ids,
    merge_generate_candidates_results,
    partition_entity_ids_for_sharding,
)
from hsds_entity_resolution.types.contracts import (
    CleanEntitiesResult,
    GenerateCandidatesResult,
)
from hsds_entity_resolution.types.frames import CANDIDATE_PAIR_SCHEMA


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _org_config() -> EntityResolutionRunConfig:
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-gen-shard",
        scope_id="scope-gen-shard",
        entity_type="organization",
    ).model_dump()
    payload["blocking"]["similarity_threshold"] = 0.01
    payload["blocking"]["max_candidates_per_entity"] = 50
    return EntityResolutionRunConfig.model_validate(payload)


def _svc_config() -> EntityResolutionRunConfig:
    payload = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-gen-shard",
        scope_id="scope-gen-shard",
        entity_type="service",
    ).model_dump()
    payload["blocking"]["similarity_threshold"] = 0.01
    payload["blocking"]["max_candidates_per_entity"] = 50
    return EntityResolutionRunConfig.model_validate(payload)


def _make_entity_row(entity_id: str, entity_type: str, embedding: list[float]) -> dict:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "source_schema": "SCHEMA_A",
        "name": f"Entity {entity_id}",
        "description": "",
        "emails": [],
        "phones": [],
        "websites": [],
        "locations": [],
        "taxonomies": [],
        "identifiers": [],
        "services_rollup": [],
        "organization_name": "",
        "organization_id": "",
        "organization_original_id": "",
        "display_name": "",
        "display_description": "",
        "alternate_name": "",
        "short_description": "",
        "application_process": "",
        "fees_description": "",
        "eligibility_description": "",
        "resource_writer_name": "",
        "assured_date": "",
        "assurer_email": "",
        "original_id": "",
        "content_hash": entity_id,
        "embedding_vector": embedding,
    }


def _changed_entities(entity_ids: list[str], entity_type: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": entity_ids,
            "entity_type": [entity_type] * len(entity_ids),
            "delta_class": ["changed"] * len(entity_ids),
        }
    )


def _empty_changed() -> pl.DataFrame:
    return pl.DataFrame(
        {"entity_id": [], "entity_type": [], "delta_class": []}
    ).cast({"entity_id": pl.String, "entity_type": pl.String, "delta_class": pl.String})


def _simple_candidate_record(pair_key: str, entity_a: str, entity_b: str) -> dict:
    return {
        "pair_key": pair_key,
        "entity_a_id": entity_a,
        "entity_b_id": entity_b,
        "entity_type": "organization",
        "embedding_similarity": 0.9,
        "candidate_reason_codes": ["embedding_threshold"],
        "source_schema_a": "SCHEMA_A",
        "source_schema_b": "SCHEMA_A",
        "blocking_rule_id": "rule-1",
    }


def _empty_pair_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=CANDIDATE_PAIR_SCHEMA)


def _gen_result(pairs: list[dict]) -> GenerateCandidatesResult:
    df = pl.DataFrame(pairs, schema_overrides=CANDIDATE_PAIR_SCHEMA) if pairs else _empty_pair_frame()
    summary = pl.DataFrame({"candidate_count": [len(pairs)], "raw_candidate_count": [len(pairs)]})
    return GenerateCandidatesResult(candidate_pairs=df, candidate_summary=summary)


# ---------------------------------------------------------------------------
# partition_entity_ids_for_sharding
# ---------------------------------------------------------------------------


class TestPartitionEntityIdsForSharding:
    def test_single_shard_returns_full_set(self) -> None:
        ids = frozenset({"a", "b", "c"})
        result = partition_entity_ids_for_sharding(ids, num_shards=1)
        assert len(result) == 1
        assert result[0] == ids

    def test_empty_set_returns_single_empty_partition(self) -> None:
        result = partition_entity_ids_for_sharding(frozenset(), num_shards=4)
        assert len(result) == 1
        assert result[0] == frozenset()

    def test_partitions_are_disjoint(self) -> None:
        ids = frozenset(f"entity-{i:04d}" for i in range(60))
        shards = partition_entity_ids_for_sharding(ids, num_shards=4)
        assert len(shards) == 4
        seen: set[str] = set()
        for shard in shards:
            assert shard.isdisjoint(seen), "Shards overlap"
            seen |= shard

    def test_union_equals_input(self) -> None:
        ids = frozenset(f"entity-{i:04d}" for i in range(60))
        shards = partition_entity_ids_for_sharding(ids, num_shards=5)
        combined = frozenset().union(*shards)
        assert combined == ids

    def test_num_shards_exceeds_ids_produces_some_empty(self) -> None:
        ids = frozenset({"a", "b"})
        shards = partition_entity_ids_for_sharding(ids, num_shards=10)
        assert sum(len(s) for s in shards) == len(ids)
        assert any(len(s) == 0 for s in shards)

    def test_deterministic_across_calls(self) -> None:
        ids = frozenset(f"entity-{i:04d}" for i in range(40))
        shards_a = partition_entity_ids_for_sharding(ids, num_shards=4)
        shards_b = partition_entity_ids_for_sharding(ids, num_shards=4)
        for a, b in zip(shards_a, shards_b):
            assert a == b


# ---------------------------------------------------------------------------
# compute_anchor_ids
# ---------------------------------------------------------------------------


class TestComputeAnchorIds:
    def _make_cleaned(
        self,
        org_ids: list[str],
        svc_ids: list[str],
        changed: list[tuple[str, str, str]],
    ) -> CleanEntitiesResult:
        org_rows = [_make_entity_row(eid, "organization", [0.5, 0.5]) for eid in org_ids]
        svc_rows = [_make_entity_row(eid, "service", [0.5, 0.5]) for eid in svc_ids]
        org_df = pl.DataFrame(org_rows) if org_rows else pl.DataFrame(
            schema={k: pl.Object for k in _make_entity_row("x", "organization", []).keys()}
        )
        svc_df = pl.DataFrame(svc_rows) if svc_rows else pl.DataFrame(
            schema={k: pl.Object for k in _make_entity_row("x", "service", []).keys()}
        )
        changed_df = pl.DataFrame(
            {"entity_id": [c[0] for c in changed], "entity_type": [c[1] for c in changed], "delta_class": [c[2] for c in changed]},
        ).cast({"entity_id": pl.String, "entity_type": pl.String, "delta_class": pl.String})
        return CleanEntitiesResult(
            denormalized_organization=org_df,
            denormalized_service=svc_df,
            entity_index=pl.DataFrame(),
            entity_delta_summary=pl.DataFrame({"added_count": [0], "changed_count": [0], "removed_count": [0]}),
            removed_entity_ids=pl.DataFrame({"entity_id": pl.Series([], dtype=pl.String)}),
            changed_entities=changed_df,
            no_change=False,
        )

    def test_incremental_returns_changed_ids_only(self) -> None:
        cleaned = self._make_cleaned(
            org_ids=["org-1", "org-2", "org-3"],
            svc_ids=["svc-1", "svc-2"],
            changed=[("org-1", "organization", "changed"), ("svc-2", "service", "added")],
        )
        anchor_ids = compute_anchor_ids(cleaned=cleaned, full_scope_rescore=False)
        assert anchor_ids == frozenset({"org-1", "svc-2"})

    def test_full_scope_rescore_returns_all_entity_ids(self) -> None:
        cleaned = self._make_cleaned(
            org_ids=["org-1", "org-2"],
            svc_ids=["svc-1"],
            changed=[("org-1", "organization", "changed")],
        )
        anchor_ids = compute_anchor_ids(cleaned=cleaned, full_scope_rescore=True)
        assert anchor_ids == frozenset({"org-1", "org-2", "svc-1"})

    def test_removed_delta_class_excluded(self) -> None:
        cleaned = self._make_cleaned(
            org_ids=["org-1", "org-2"],
            svc_ids=[],
            changed=[
                ("org-1", "organization", "changed"),
                ("org-2", "organization", "removed"),
            ],
        )
        anchor_ids = compute_anchor_ids(cleaned=cleaned, full_scope_rescore=False)
        assert anchor_ids == frozenset({"org-1"})

    def test_empty_changed_returns_empty(self) -> None:
        cleaned = self._make_cleaned(
            org_ids=["org-1"],
            svc_ids=[],
            changed=[],
        )
        anchor_ids = compute_anchor_ids(cleaned=cleaned, full_scope_rescore=False)
        assert anchor_ids == frozenset()


# ---------------------------------------------------------------------------
# merge_generate_candidates_results
# ---------------------------------------------------------------------------


class TestMergeGenerateCandidatesResults:
    def test_empty_list_returns_empty(self) -> None:
        result = merge_generate_candidates_results([])
        assert result.candidate_pairs.is_empty()

    def test_all_empty_shards_returns_empty(self) -> None:
        results = [_gen_result([]), _gen_result([])]
        merged = merge_generate_candidates_results(results)
        assert merged.candidate_pairs.is_empty()

    def test_non_overlapping_shards_concatenated(self) -> None:
        r0 = _gen_result([_simple_candidate_record("a__b", "a", "b")])
        r1 = _gen_result([_simple_candidate_record("c__d", "c", "d")])
        merged = merge_generate_candidates_results([r0, r1])
        assert merged.candidate_pairs.height == 2
        keys = set(merged.candidate_pairs.get_column("pair_key").to_list())
        assert keys == {"a__b", "c__d"}

    def test_duplicate_pair_keys_deduplicated(self) -> None:
        record = _simple_candidate_record("a__b", "a", "b")
        r0 = _gen_result([record])
        r1 = _gen_result([record])
        merged = merge_generate_candidates_results([r0, r1])
        assert merged.candidate_pairs.height == 1
        assert merged.candidate_pairs.get_column("pair_key")[0] == "a__b"

    def test_summary_reflects_deduped_count(self) -> None:
        record = _simple_candidate_record("a__b", "a", "b")
        r0 = _gen_result([record, _simple_candidate_record("c__d", "c", "d")])
        r1 = _gen_result([record])  # duplicate of a__b
        merged = merge_generate_candidates_results([r0, r1])
        assert merged.candidate_pairs.height == 2
        assert merged.candidate_summary.get_column("candidate_count")[0] == 2

    def test_output_sorted_by_entity_ids(self) -> None:
        r0 = _gen_result([_simple_candidate_record("m__n", "m", "n")])
        r1 = _gen_result([_simple_candidate_record("a__b", "a", "b")])
        merged = merge_generate_candidates_results([r0, r1])
        keys = merged.candidate_pairs.get_column("pair_key").to_list()
        assert keys == ["a__b", "m__n"]


# ---------------------------------------------------------------------------
# Golden test: sharded generate_candidates == monolithic
# ---------------------------------------------------------------------------


class TestShardsMatchMonolithicGenerate:
    """Verify that generate sharding produces the same candidate pairs as a single run."""

    def _build_org_frame(self, n: int) -> pl.DataFrame:
        import numpy as np

        rng = np.random.default_rng(42)
        rows = []
        for i in range(n):
            vec = rng.standard_normal(8).tolist()
            rows.append(_make_entity_row(f"org-{i:04d}", "organization", vec))
        return pl.DataFrame(rows)

    def _build_svc_frame(self, n: int) -> pl.DataFrame:
        import numpy as np

        rng = np.random.default_rng(99)
        rows = []
        for i in range(n):
            vec = rng.standard_normal(8).tolist()
            rows.append(_make_entity_row(f"svc-{i:04d}", "service", vec))
        return pl.DataFrame(rows)

    def test_org_sharded_matches_monolithic(self) -> None:
        n_entities = 20
        n_shards = 3
        config = _org_config()
        org_df = self._build_org_frame(n_entities)
        svc_df = pl.DataFrame(schema={k: pl.Object if isinstance(v, pl.datatypes.classes.DataTypeClass) else v for k, v in pl.DataFrame([_make_entity_row("x", "service", [0.0])]).schema.items()}).clear()
        all_ids = list(org_df.get_column("entity_id").to_list())
        changed_df = _changed_entities(all_ids, "organization")

        monolithic = generate_candidates(
            denormalized_organization=org_df,
            denormalized_service=svc_df.clear() if not svc_df.is_empty() else svc_df,
            changed_entities=changed_df,
            config=config,
            explicit_backfill=False,
        )

        from hsds_entity_resolution.core.generate_candidates_sharded import partition_entity_ids_for_sharding
        anchor_ids = frozenset(all_ids)
        subsets = partition_entity_ids_for_sharding(anchor_ids, num_shards=n_shards)
        shard_results = [
            generate_candidates(
                denormalized_organization=org_df,
                denormalized_service=svc_df.clear() if not svc_df.is_empty() else svc_df,
                changed_entities=changed_df,
                config=config,
                explicit_backfill=False,
                anchor_ids_subset=subset,
            )
            for subset in subsets
        ]
        merged = merge_generate_candidates_results(shard_results)

        mono_keys = set(monolithic.candidate_pairs.get_column("pair_key").to_list())
        merged_keys = set(merged.candidate_pairs.get_column("pair_key").to_list())
        assert mono_keys == merged_keys, (
            f"Sharded keys differ from monolithic. "
            f"Missing: {mono_keys - merged_keys}, Extra: {merged_keys - mono_keys}"
        )

    def test_service_sharded_matches_monolithic(self) -> None:
        import numpy as np

        n_entities = 16
        n_shards = 4
        config = _svc_config()
        rng = np.random.default_rng(7)
        svc_rows = [
            _make_entity_row(f"svc-{i:04d}", "service", rng.standard_normal(8).tolist())
            for i in range(n_entities)
        ]
        svc_df = pl.DataFrame(svc_rows)
        org_df = pl.DataFrame(schema=svc_df.schema).clear()
        all_ids = list(svc_df.get_column("entity_id").to_list())
        changed_df = _changed_entities(all_ids, "service")

        monolithic = generate_candidates(
            denormalized_organization=org_df,
            denormalized_service=svc_df,
            changed_entities=changed_df,
            config=config,
            explicit_backfill=False,
        )

        from hsds_entity_resolution.core.generate_candidates_sharded import partition_entity_ids_for_sharding
        anchor_ids = frozenset(all_ids)
        subsets = partition_entity_ids_for_sharding(anchor_ids, num_shards=n_shards)
        shard_results = [
            generate_candidates(
                denormalized_organization=org_df,
                denormalized_service=svc_df,
                changed_entities=changed_df,
                config=config,
                explicit_backfill=False,
                anchor_ids_subset=subset,
            )
            for subset in subsets
        ]
        merged = merge_generate_candidates_results(shard_results)

        mono_keys = set(monolithic.candidate_pairs.get_column("pair_key").to_list())
        merged_keys = set(merged.candidate_pairs.get_column("pair_key").to_list())
        assert mono_keys == merged_keys

    def test_empty_anchor_subset_returns_empty(self) -> None:
        import numpy as np

        n_entities = 10
        config = _org_config()
        rng = np.random.default_rng(11)
        org_rows = [
            _make_entity_row(f"org-{i:04d}", "organization", rng.standard_normal(8).tolist())
            for i in range(n_entities)
        ]
        org_df = pl.DataFrame(org_rows)
        svc_df = org_df.clear()
        changed_df = _changed_entities(list(org_df.get_column("entity_id").to_list()), "organization")

        result = generate_candidates(
            denormalized_organization=org_df,
            denormalized_service=svc_df,
            changed_entities=changed_df,
            config=config,
            explicit_backfill=False,
            anchor_ids_subset=frozenset(),
        )
        assert result.candidate_pairs.is_empty()


# ---------------------------------------------------------------------------
# Memory-safe single-shard chunking parity / caps
# ---------------------------------------------------------------------------


class TestGenerateChunkingParity:
    """Chunked generate path should match baseline unless contact caps are set."""

    def _frames(self, n: int = 12) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        import numpy as np

        rng = np.random.default_rng(17)
        rows = [
            _make_entity_row(f"org-{i:04d}", "organization", rng.standard_normal(8).tolist())
            for i in range(n)
        ]
        # Shared phone to exercise contact-overlap expansion.
        for row in rows:
            row["phones"] = ["555-0100"]
            row["locations"] = [
                {
                    "address_1": "100 Main St",
                    "city": "Columbus",
                    "state": "OH",
                    "postal_code": "43215",
                }
            ]
        org_df = pl.DataFrame(rows)
        svc_df = org_df.clear()
        changed = _changed_entities(list(org_df.get_column("entity_id").to_list()), "organization")
        return org_df, svc_df, changed

    def test_anchor_chunking_matches_baseline(self) -> None:
        org_df, svc_df, changed = self._frames()
        baseline_config = _org_config()
        chunked_payload = baseline_config.model_dump()
        chunked_payload["chunking"] = {"generate_anchor_chunk_size": 3}
        chunked_config = EntityResolutionRunConfig.model_validate(chunked_payload)

        baseline = generate_candidates(
            denormalized_organization=org_df,
            denormalized_service=svc_df,
            changed_entities=changed,
            config=baseline_config,
            explicit_backfill=True,
        )
        chunked = generate_candidates(
            denormalized_organization=org_df,
            denormalized_service=svc_df,
            changed_entities=changed,
            config=chunked_config,
            explicit_backfill=True,
        )
        assert set(baseline.candidate_pairs.get_column("pair_key").to_list()) == set(
            chunked.candidate_pairs.get_column("pair_key").to_list()
        )

    def test_contact_overlap_pair_cap_limits_expansion(self) -> None:
        org_df, svc_df, changed = self._frames(n=10)
        baseline_config = _org_config()
        capped_payload = baseline_config.model_dump()
        capped_payload["chunking"] = {
            "generate_anchor_chunk_size": 4,
            "max_contact_overlap_pairs_per_anchor": 2,
        }
        capped_config = EntityResolutionRunConfig.model_validate(capped_payload)

        baseline = generate_candidates(
            denormalized_organization=org_df,
            denormalized_service=svc_df,
            changed_entities=changed,
            config=baseline_config,
            explicit_backfill=True,
        )
        capped = generate_candidates(
            denormalized_organization=org_df,
            denormalized_service=svc_df,
            changed_entities=changed,
            config=capped_config,
            explicit_backfill=True,
        )
        assert capped.candidate_pairs.height < baseline.candidate_pairs.height
        assert capped.candidate_pairs.height > 0
