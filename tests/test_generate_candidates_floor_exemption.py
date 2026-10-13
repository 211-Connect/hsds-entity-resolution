"""Tests for the floor exemption: a caller hook admits chosen pairs below the floor (ISS-2196)."""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl
import pytest

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.generate_candidates import generate_candidates
from hsds_entity_resolution.core.generate_candidates_sharded import (
    merge_generate_candidates_results,
)
from hsds_entity_resolution.core.pipeline import run_incremental, run_incremental_until_candidates

# ---------------------------------------------------------------------------
# Pinned fixture: admission with the exemption off is byte-identical to 2.4.0
# ---------------------------------------------------------------------------

# sha256 of candidate_pairs + excluded_pairs (ndjson) for ``_pinned_services()`` under
# ``_pinned_config()``, recorded on hsds-entity-resolution 2.4.0 (main 680b8d8) before the
# floor exemption existed.
_PINNED_SHA256 = "20fa7c9fc0deb24b15eb2ae1121259d0954ddd4b4434b893538c279313394c88"


def _pinned_services() -> pl.DataFrame:
    """Forty services with seeded embeddings, shared phones and two organizations.

    The embeddings are clustered so that several anchors hit the per-anchor cap, some
    pairs share a phone below the floor, and the exclusion below vetoes a few pairs.
    """
    rng = np.random.default_rng(2196)
    centres = rng.normal(size=(4, 16))
    rows: list[dict[str, Any]] = []
    for index in range(40):
        vector = centres[index % 4] + rng.normal(scale=0.35, size=16)
        rows.append(
            {
                "entity_id": f"svc-{index:02d}",
                "entity_type": "service",
                "source_schema": "SOURCE_A" if index % 3 else "SOURCE_B",
                "name": f"Program {index % 7}",
                "description": f"Description {index}",
                "emails": [],
                "phones": [f"555010{index % 5:04d}"] if index % 2 else [],
                "websites": [],
                "locations": [],
                "taxonomies": [],
                "identifiers": [],
                "services_rollup": [],
                "organization_name": f"Org {index % 2}",
                "organization_id": f"org-{index % 2}",
                "embedding_vector": [float(value) for value in vector],
                "content_hash": f"hash-{index:02d}",
            }
        )
    return pl.DataFrame(rows)


def _pinned_config(**blocking: Any) -> EntityResolutionRunConfig:
    raw = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-exemption", scope_id="scope-exemption", entity_type="service"
    ).model_dump()
    raw["blocking"]["similarity_threshold"] = 0.8
    raw["blocking"]["max_candidates_per_entity"] = 4
    raw["blocking"].update(blocking)
    return EntityResolutionRunConfig.model_validate(raw)


def _exclude_every_seventh(a: Mapping[str, Any], b: Mapping[str, Any]) -> str | None:
    numbers = int(str(a["entity_id"])[-2:]) + int(str(b["entity_id"])[-2:])
    return "every_seventh" if numbers % 7 == 0 else None


def _pinned_digest(result: Any) -> str:
    payload = result.candidate_pairs.write_ndjson() + result.excluded_pairs.write_ndjson()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_pinned(config: EntityResolutionRunConfig, **hooks: Any) -> Any:
    services = _pinned_services()
    return generate_candidates(
        denormalized_organization=services.head(0),
        denormalized_service=services,
        changed_entities=pl.DataFrame(
            {"entity_id": [], "entity_type": [], "delta_class": []},
            schema={"entity_id": pl.String, "entity_type": pl.String, "delta_class": pl.String},
        ),
        config=config,
        explicit_backfill=True,
        structural_exclusion=_exclude_every_seventh,
        **hooks,
    )


def test_the_pinned_fixture_admits_as_2_4_0_did() -> None:
    result = _run_pinned(_pinned_config())

    assert result.candidate_pairs.height > 0
    assert result.excluded_pairs.height > 0
    assert _pinned_digest(result) == _PINNED_SHA256
    assert result.floor_exempt_counts == {}


def _every_pair_is_exempt(_a: Mapping[str, Any], _b: Mapping[str, Any]) -> str | None:
    return "always"


@pytest.mark.parametrize(
    ("blocking", "hook"),
    [
        pytest.param({}, _every_pair_is_exempt, id="hook-without-exemption-floor"),
        pytest.param({"exemption_floor": 0.5}, None, id="exemption-floor-without-hook"),
        pytest.param({"exemption_floor": 0.8}, _every_pair_is_exempt, id="empty-band"),
    ],
)
def test_without_a_band_and_a_hook_admission_is_unchanged(
    blocking: dict[str, Any], hook: Any
) -> None:
    result = _run_pinned(_pinned_config(**blocking), floor_exemption=hook)

    assert _pinned_digest(result) == _PINNED_SHA256
    assert result.floor_exempt_counts == {}


# ---------------------------------------------------------------------------
# The band [exemption_floor, floor)
# ---------------------------------------------------------------------------

_FLOOR = 0.95
_EXEMPTION_FLOOR = 0.90


def _at(similarity: float) -> list[float]:
    """A unit vector whose cosine with ``[1, 0]`` is ``similarity``."""
    return [similarity, math.sqrt(1.0 - similarity**2)]


def _service(
    entity_id: str, *, similarity: float, organization_id: str, phones: list[str] | None = None
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "entity_type": "service",
        "source_schema": "SOURCE_A",
        "name": f"Service {entity_id}",
        "description": "",
        "emails": [],
        "phones": phones or [],
        "websites": [],
        "locations": [],
        "taxonomies": [],
        "identifiers": [],
        "services_rollup": [],
        "organization_name": f"Organization {organization_id}",
        "organization_id": organization_id,
        "embedding_vector": _at(similarity),
        "content_hash": f"hash-{entity_id}",
    }


def _band_services(*extra: dict[str, Any]) -> pl.DataFrame:
    """Anchor ``svc-a`` and one record at each position around the band."""
    return pl.DataFrame(
        [
            _service("svc-a", similarity=1.0, organization_id="org-1"),
            _service("svc-b", similarity=0.97, organization_id="org-2"),  # above the floor
            _service("svc-c", similarity=0.92, organization_id="org-1"),  # band, same org
            _service("svc-d", similarity=0.92, organization_id="org-3"),  # band, other org
            _service("svc-e", similarity=0.85, organization_id="org-1"),  # below the band
            *extra,
        ]
    )


def _same_organization(a: Mapping[str, Any], b: Mapping[str, Any]) -> str | None:
    if a["organization_id"] and a["organization_id"] == b["organization_id"]:
        return "same_organization_id"
    return None


def _band_config(*, max_candidates_per_entity: int = 50) -> EntityResolutionRunConfig:
    raw = EntityResolutionRunConfig.defaults_for_entity_type(
        team_id="team-exemption", scope_id="scope-exemption", entity_type="service"
    ).model_dump()
    raw["blocking"]["similarity_threshold"] = _FLOOR
    raw["blocking"]["exemption_floor"] = _EXEMPTION_FLOOR
    raw["blocking"]["max_candidates_per_entity"] = max_candidates_per_entity
    return EntityResolutionRunConfig.model_validate(raw)


def _run_band(
    services: pl.DataFrame,
    *,
    config: EntityResolutionRunConfig | None = None,
    floor_exemption: Any = _same_organization,
    structural_exclusion: Any = None,
    anchor: str = "svc-a",
) -> Any:
    return generate_candidates(
        denormalized_organization=services.head(0),
        denormalized_service=services,
        changed_entities=pl.DataFrame(
            {"entity_id": [anchor], "entity_type": ["service"], "delta_class": ["added"]}
        ),
        config=config or _band_config(),
        explicit_backfill=False,
        structural_exclusion=structural_exclusion,
        floor_exemption=floor_exemption,
    )


def _pairs(result: Any) -> dict[str, dict[str, Any]]:
    return {row["pair_key"]: row for row in result.candidate_pairs.iter_rows(named=True)}


def test_an_exempt_pair_in_the_band_is_admitted_with_its_reason() -> None:
    result = _run_band(_band_services())

    pair = _pairs(result)["svc-a__svc-c"]
    assert pair["blocking_rule_id"] == "floor_exempt"
    assert pair["candidate_reason_codes"] == ["floor_exempt:same_organization_id"]
    assert pair["embedding_similarity"] == pytest.approx(0.92)
    assert result.floor_exempt_counts == {"same_organization_id": 1}


def test_the_pair_above_the_floor_is_admitted_as_before() -> None:
    pair = _pairs(_run_band(_band_services()))["svc-a__svc-b"]

    assert pair["blocking_rule_id"] == "embedding_floor"
    assert pair["candidate_reason_codes"] == ["embedding_floor"]


def test_a_band_pair_the_exemption_does_not_name_is_not_admitted() -> None:
    result = _run_band(_band_services())

    assert "svc-a__svc-d" not in _pairs(result)
    assert result.excluded_pairs.is_empty()


def test_a_pair_below_the_exemption_floor_is_never_admitted_or_asked_about() -> None:
    asked: list[tuple[str, str]] = []

    def recording(a: Mapping[str, Any], b: Mapping[str, Any]) -> str | None:
        asked.append((str(a["entity_id"]), str(b["entity_id"])))
        return "always"

    result = _run_band(_band_services(), floor_exemption=recording)

    assert set(_pairs(result)) == {"svc-a__svc-b", "svc-a__svc-c", "svc-a__svc-d"}
    assert sorted(asked) == [("svc-a", "svc-c"), ("svc-a", "svc-d")]


def test_the_exemption_sees_the_canonical_pair_whichever_record_is_the_anchor() -> None:
    asked: list[tuple[str, str]] = []

    def recording(a: Mapping[str, Any], b: Mapping[str, Any]) -> str | None:
        asked.append((str(a["entity_id"]), str(b["entity_id"])))
        return None

    _run_band(_band_services(), floor_exemption=recording, anchor="svc-c")

    assert ("svc-a", "svc-c") in asked
    assert ("svc-c", "svc-a") not in asked


def test_the_structural_exclusion_vetoes_an_exempt_pair() -> None:
    def exclude_c(a: Mapping[str, Any], b: Mapping[str, Any]) -> str | None:
        return "different_by_construction" if b["entity_id"] == "svc-c" else None

    result = _run_band(_band_services(), structural_exclusion=exclude_c)

    assert "svc-a__svc-c" not in _pairs(result)
    excluded = result.excluded_pairs.row(0, named=True)
    assert excluded["pair_key"] == "svc-a__svc-c"
    assert excluded["exclusion_reason"] == "different_by_construction"
    assert result.floor_exempt_counts == {}


def test_exempt_pairs_come_after_every_pair_above_the_floor_under_the_cap() -> None:
    """The walk is descending: the cap is spent on the floor first, then the band."""
    one = _run_band(_band_services(), config=_band_config(max_candidates_per_entity=1))
    two = _run_band(_band_services(), config=_band_config(max_candidates_per_entity=2))

    assert set(_pairs(one)) == {"svc-a__svc-b"}
    assert set(_pairs(two)) == {"svc-a__svc-b", "svc-a__svc-c"}


def test_exempt_pairs_count_against_the_cap() -> None:
    services = _band_services(_service("svc-f", similarity=0.93, organization_id="org-1"))

    result = _run_band(services, config=_band_config(max_candidates_per_entity=2))

    assert set(_pairs(result)) == {"svc-a__svc-b", "svc-a__svc-f"}
    assert result.floor_exempt_counts == {"same_organization_id": 1}


def test_an_exempt_pair_that_also_shares_a_key_is_named_by_the_key_and_not_counted() -> None:
    services = pl.DataFrame(
        [
            _service("svc-a", similarity=1.0, organization_id="org-1", phones=["5550100199"]),
            _service("svc-c", similarity=0.92, organization_id="org-1", phones=["5550100199"]),
        ]
    )

    pair = _pairs(_run_band(services))["svc-a__svc-c"]
    result = _run_band(services)

    assert pair["blocking_rule_id"] == "informative_key:phone"
    assert pair["candidate_reason_codes"] == [
        "floor_exempt:same_organization_id",
        "informative_key_phone",
    ]
    assert result.floor_exempt_counts == {}


def test_sharded_merge_counts_each_exempt_pair_once() -> None:
    both_anchors = _band_services()
    shard_a = _run_band(both_anchors, anchor="svc-a")
    shard_c = _run_band(both_anchors, anchor="svc-c")

    merged = merge_generate_candidates_results([shard_a, shard_c])

    # Both shards admit svc-a__svc-c (each anchor walks to the other): counted once.
    assert shard_a.floor_exempt_counts == {"same_organization_id": 1}
    assert shard_c.floor_exempt_counts == {"same_organization_id": 1}
    assert merged.floor_exempt_counts == {"same_organization_id": 1}


def test_the_overview_log_reports_the_exemption(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="hsds_entity_resolution.core.generate_candidates"):
        _run_band(_band_services())

    overview = next(
        r.message for r in caplog.records if "generate_candidates_overview" in r.message
    )
    assert "exemption_floor=0.9" in overview
    assert "floor_exempt={'service': {'same_organization_id': 1}}" in overview


def test_an_anchor_capped_on_exempt_pairs_alone_has_no_pair_above_the_floor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    services = pl.DataFrame(
        [
            _service("svc-a", similarity=1.0, organization_id="org-1"),
            _service("svc-c", similarity=0.92, organization_id="org-1"),
        ]
    )
    with caplog.at_level(logging.INFO, logger="hsds_entity_resolution.core.generate_candidates"):
        result = _run_band(services, config=_band_config(max_candidates_per_entity=1))

    overview = next(
        r.message for r in caplog.records if "generate_candidates_overview" in r.message
    )
    assert set(_pairs(result)) == {"svc-a__svc-c"}
    assert "anchors_with_no_above_floor=1 (100.0%)" in overview
    assert "anchors_at_cap=1 (100.0%)" in overview


def test_an_exemption_floor_above_the_floor_is_rejected() -> None:
    raw = _band_config().model_dump()
    raw["blocking"]["exemption_floor"] = 0.96

    with pytest.raises(ValueError, match="exemption_floor"):
        EntityResolutionRunConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# The pipeline threads the exemption through
# ---------------------------------------------------------------------------


def test_run_incremental_until_candidates_passes_the_exemption() -> None:
    def candidates(floor_exemption: Any) -> pl.DataFrame:
        return run_incremental_until_candidates(
            organization_entities=pl.DataFrame(),
            service_entities=_band_services(),
            previous_entity_index=pl.DataFrame(),
            previous_pair_state_index=pl.DataFrame(),
            config=_band_config(),
            explicit_backfill=True,
            floor_exemption=floor_exemption,
        ).candidates.candidate_pairs

    assert "svc-a__svc-c" not in candidates(None).get_column("pair_key").to_list()
    assert "svc-a__svc-c" in candidates(_same_organization).get_column("pair_key").to_list()


def test_run_incremental_passes_the_exemption_and_reports_its_counts() -> None:
    def run(floor_exemption: Any) -> Any:
        return run_incremental(
            organization_entities=pl.DataFrame(),
            service_entities=_band_services(),
            previous_entity_index=pl.DataFrame(),
            previous_pair_state_index=pl.DataFrame(),
            config=_band_config(),
            explicit_backfill=True,
            floor_exemption=floor_exemption,
        )

    without = run(None)
    exempt = run(_same_organization)

    assert "svc-a__svc-c" not in without.candidate_pairs.get_column("pair_key").to_list()
    assert without.floor_exempt_counts == {}
    assert "svc-a__svc-c" in exempt.candidate_pairs.get_column("pair_key").to_list()
    # With every record an anchor, svc-a__svc-c is the only one-organization pair in the
    # band (svc-a__svc-e is below it; every other org-1 pair is above the floor).
    assert exempt.floor_exempt_counts == {"same_organization_id": 1}
