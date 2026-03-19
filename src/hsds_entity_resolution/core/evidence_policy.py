"""Shared evidence policy for reason emission and mitigation counting."""

from __future__ import annotations

import polars as pl

from hsds_entity_resolution.types.frames import CONTRIBUTING_REASONS_SCHEMA

EVIDENCE_POLICY_NAME = "weighted_contribution_positive"


def is_contributing_evidence(*, raw_contribution: float, weighted_contribution: float) -> bool:
    """Return whether one reason row counts as meaningful evidence.

    Policy: evidence is contributing only when weighted contribution is positive.
    """
    del raw_contribution
    return weighted_contribution > 0.0


def contributing_evidence_filter() -> pl.Expr:
    """Return Polars filter expression matching the evidence policy."""
    return pl.col("weighted_contribution") > 0.0


def count_contributing_reasons(*, pair_reasons: pl.DataFrame) -> pl.DataFrame:
    """Return per-pair contributing reason counts under the shared policy."""
    required_columns = {"pair_key", "weighted_contribution"}
    if pair_reasons.is_empty() or not required_columns.issubset(pair_reasons.columns):
        return pl.DataFrame(schema=CONTRIBUTING_REASONS_SCHEMA)
    return (
        pair_reasons.filter(contributing_evidence_filter())
        .group_by("pair_key")
        .len()
        .rename({"len": "reason_count"})
    )
