"""Pytest bootstrap: project import path and the ``--reference-judge`` option."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--reference-judge``."""
    parser.addoption(
        "--reference-judge",
        action="store_true",
        default=False,
        help=(
            "Run every run_incremental call with the ReferenceJudge as its default judge, "
            "to show the judge stage leaves every other output unchanged."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Make the ReferenceJudge the default judge when ``--reference-judge`` is passed.

    Tests import ``run_incremental`` by name, so the default is set on the function
    object itself rather than by patching a module attribute.
    """
    if not config.getoption("--reference-judge"):
        return
    from hsds_entity_resolution.core.pipeline import run_incremental
    from hsds_entity_resolution.judge import ReferenceJudge

    defaults = run_incremental.__kwdefaults__
    assert defaults is not None and "judge" in defaults
    defaults["judge"] = ReferenceJudge()
