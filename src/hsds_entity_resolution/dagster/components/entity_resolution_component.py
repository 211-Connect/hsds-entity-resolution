"""Dagster component that will host HSDS entity resolution building blocks."""

import dagster as dg


class EntityResolutionComponent(dg.Component, dg.Model, dg.Resolvable):
    """Type-safe component entry point for reusable entity resolution definitions."""

    # Add fields here to define constructor params in Python and
    # YAML schema fields via Resolvable.

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:
        """Build Dagster definitions for this component instance."""
        del context  # placeholder until component logic is implemented
        return dg.Definitions()
