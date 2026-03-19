"""Smoke tests for exported type modules and schema constants."""

from __future__ import annotations

from hsds_entity_resolution.types.artifact_rows import CandidatePairRow, ScoredPairRow
from hsds_entity_resolution.types.rows import CleanEntityRow, RawEntityRowInput


def test_artifact_row_typeddicts_are_declared() -> None:
    """TypedDict declarations should expose expected fields."""
    assert "pair_key" in CandidatePairRow.__annotations__
    assert "predicted_duplicate" in ScoredPairRow.__annotations__
    assert "pair_outcome" in ScoredPairRow.__annotations__


def test_entity_row_typeddicts_are_declared() -> None:
    """Entity row TypedDict declarations should expose expected fields."""
    assert "entity_id" in RawEntityRowInput.__annotations__
    assert "content_hash" in CleanEntityRow.__annotations__
