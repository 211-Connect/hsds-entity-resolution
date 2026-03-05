import dagster as dg


class EntityResolutionComponent(dg.Component, dg.Model, dg.Resolvable):
    """COMPONENT SUMMARY HERE.

    COMPONENT DESCRIPTION HERE.
    """

    # Add fields here to define constructor params in Python and
    # YAML schema fields via Resolvable.

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:
        # Add definition construction logic here.
        return dg.Definitions()
