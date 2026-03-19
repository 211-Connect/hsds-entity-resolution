"""Canonical scalar and identity types for the entity-resolution component."""

from __future__ import annotations

from typing import Literal, NewType

EntityId = NewType("EntityId", str)
PairKey = NewType("PairKey", str)
TeamId = NewType("TeamId", str)
ScopeId = NewType("ScopeId", str)
PolicyVersion = NewType("PolicyVersion", str)
ModelVersion = NewType("ModelVersion", str)

EntityType = Literal["organization", "service"]
NullableEntityType = EntityType | None
