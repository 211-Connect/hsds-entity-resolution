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
- **Pydantic v2**: typed data models and validation for HSDS entities and pipeline I/O
- **Ruff**: Python formatting and linting for fast local feedback
- **Pyright**: static type checking for `src/` and `tests/`
- **Codacy CLI (`.codacy/cli.sh`)**: static analysis and security scanning (Pylint, Semgrep,
  Lizard, Trivy)
- **uv**: dependency and virtual environment management

## Getting started

### Install dependencies

**Option 1: uv**

Ensure [`uv`](https://docs.astral.sh/uv/) is installed following the
[official documentation](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
```

Activate the virtual environment:

| OS | Command |
| --- | --- |
| MacOS | `source .venv/bin/activate` |
| Windows | `.venv\Scripts\activate` |

**Option 2: pip**

```bash
python3 -m venv .venv
source .venv/bin/activate  # MacOS
pip install -e ".[dev]"
```

### Run the project

Start Dagster locally:

```bash
dg dev
```

Then open [http://localhost:3000](http://localhost:3000).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for pull request requirements, quality checks, and review
expectations.
