"""Coverage tests for Dagster component integration wrappers."""

from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from dagster import AssetsDefinition
from dagster.components import ComponentLoadContext

from hsds_entity_resolution.dagster.components.entity_resolution_component import (
    EntityResolutionComponent,
    _deep_merge,
    _ensure_frame,
)
from hsds_entity_resolution.definitions import defs


def test_component_build_defs_and_execute_asset_function() -> None:
    """Component should build definitions and run the wrapped multi-asset function."""
    component = EntityResolutionComponent()
    context = cast(ComponentLoadContext, None)
    definitions = component.build_defs(context)
    assert definitions.assets is not None
    asset_def = cast(AssetsDefinition, list(definitions.assets)[0])
    outputs = asset_def.op.compute_fn.decorated_fn(
        organization_entities=pl.DataFrame(
            {
                "entity_id": ["org-a", "org-b"],
                "source_schema": ["il211", "il211"],
                "name": ["Alpha", "Alpha Services"],
                "description": ["A", "A services"],
                "emails": [["a@example.org"], ["a@example.org"]],
                "phones": [["555-0000"], ["555-0000"]],
                "websites": [["example.org"], ["example.org"]],
                "locations": [[], []],
                "taxonomies": [[], []],
                "identifiers": [[], []],
                "services_rollup": [[], []],
                "embedding_vector": [[1.0, 0.0], [0.99, 0.01]],
            }
        ),
        service_entities=pl.DataFrame(),
        previous_entity_index=pl.DataFrame(),
        previous_pair_state_index=pl.DataFrame(),
    )
    assert len(outputs) == 16
    assert isinstance(outputs[0], pl.DataFrame)


def test_component_helpers_and_overrides() -> None:
    """Helper utilities should coerce input and merge overrides deterministically."""
    component = EntityResolutionComponent(
        constants_overrides={"blocking": {"similarity_threshold": 0.9}}
    )
    run_config = component._resolved_run_config()
    assert run_config.blocking.similarity_threshold == 0.9

    merged = _deep_merge({"a": {"b": 1}}, {"a": {"c": 2}})
    assert merged == {"a": {"b": 1, "c": 2}}

    assert isinstance(_ensure_frame([{"x": 1}]), pl.DataFrame)
    assert isinstance(_ensure_frame(None), pl.DataFrame)
    assert isinstance(_ensure_frame(pl.DataFrame()), pl.DataFrame)

    with pytest.raises(TypeError):
        _ = _ensure_frame(3.14)


def test_project_definitions_factory_returns_definitions() -> None:
    """Definitions factory should load project defs folder without raising."""
    loaded = defs()
    assert loaded is not None
