# hsds_entity_resolution

`hsds_entity_resolution` helps community organizations deduplicate HSDS data and orchestrate
continual checks that support long-running community data sharing partnerships.

## Project goals

- Improve entity matching quality across partner-provided HSDS datasets
- Reduce duplicate records that block trusted cross-organization coordination
- Run repeatable validation and quality checks as data pipelines evolve
- Support sustainable, long-term community data sharing operations

## Tooling

- **Dagster (`dagster`, `dg`)**: pipeline orchestration, definitions, and local development UI
- **Polars**: in-memory frames for every pipeline stage; the engine reads and writes no database
- **Pydantic v2**: typed data models and validation for HSDS entities and pipeline I/O
- **Ruff**: Python formatting and linting for fast local feedback
- **Pyright**: static type checking for `src/` and `tests/`
- **Codacy CLI (`.codacy/cli.sh`)**: static analysis and security scanning (Pylint, Semgrep,
  Lizard, Trivy)
- **uv**: dependency and virtual environment management

## Component Package Layout

Reusable Dagster components live in:

- `src/hsds_entity_resolution/dagster/components/`

Core library code should live outside the Dagster adapter layer:

- `src/hsds_entity_resolution/core/`
- `src/hsds_entity_resolution/types/`
- `src/hsds_entity_resolution/config/`

The canonical public component entry point is:

- `hsds_entity_resolution.dagster.components.EntityResolutionComponent`

This module is exported through the Dagster registry entry-point group:

- `dagster_dg_cli.registry_modules`

## Getting started

### Install dependencies

Ensure [`uv`](https://docs.astral.sh/uv/) is installed following the
[official documentation](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
```

### Run the tests

```bash
uv run pytest
```

The suite needs nothing beyond `uv sync`: no database, no credentials, no network.

### Develop the component library

```bash
dg dev
```

`dg dev` loads `hsds_entity_resolution.dagster.definitions`, which intentionally has no
jobs or assets of its own. The engine is a library: a host Dagster project loads its
own entities, calls `run_incremental`, and persists the returned artifacts wherever it
keeps them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for pull request requirements, quality checks, and review
expectations.

## Using This In Another Dagster Repo

1. Publish or install this package (for example: `pip install hsds-entity-resolution`).
2. Confirm discovery in the target environment:

```bash
dg list components --package hsds_entity_resolution
```

3. Use the component key in YAML:

```yaml
type: hsds_entity_resolution.dagster.components.EntityResolutionComponent
attributes: {}
```

## Publishing

> **PyPI README:** The file displayed on the PyPI package page is [`README.pypi.md`](README.pypi.md),
> not this file. Edit that file when updating the consumer-facing documentation.
> This `README.md` is the contributor/developer reference and is not included in the published package.

This package is set up to publish to PyPI from GitHub Actions via Trusted Publishing.

### PyPI Trusted Publisher settings

For the pending or normal PyPI publisher, use:

- PyPI project name: `hsds-entity-resolution`
- Owner: `211-Connect`
- Repository name: `hsds-entity-resolution`
- Workflow name: `publish.yml`
- Environment name: `pypi`

The repository name field should be only the repository name, not `owner/repo`.

From 2.0.0 the project publishes as `hsds-entity-resolution`; releases up to 1.2.0
were published as `hsds-record-matcher`. A pending Trusted Publisher for the new
project name must exist on PyPI before the first 2.x release can publish.

The distribution name on PyPI is independent from the import path in Python:

- Install name: `hsds-entity-resolution`
- Import path: `hsds_entity_resolution`

### Release flow

1. Update `version` in `pyproject.toml`.
2. Merge or push that change to `main`.
3. GitHub Actions will build the wheel and sdist, validate them with `twine check`, and publish to PyPI through the `pypi` environment if the version changed.

You can also run the publish workflow manually with `workflow_dispatch`.
