"""Shared type contracts for NLP scoring policies."""

from __future__ import annotations

from dataclasses import dataclass

from hsds_entity_resolution.config import EntityResolutionRunConfig


@dataclass(frozen=True)
class NlpSafeguardContext:
    """Context available to NLP safeguard policies."""

    left_name: str
    right_name: str
    config: EntityResolutionRunConfig


@dataclass(frozen=True)
class NlpSafeguardOutcome:
    """Decision payload returned by a safeguard policy."""

    veto: bool = False
    penalty: float = 0.0
