"""Dagster definitions entrypoint for loading project defs from the `defs` folder."""

from pathlib import Path

from dagster import definitions, load_from_defs_folder


@definitions
def defs():
    """Return Dagster definitions discovered under this package."""
    return load_from_defs_folder(path_within_project=Path(__file__).parent)
