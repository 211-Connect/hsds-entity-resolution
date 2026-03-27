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
- **dbt (Snowflake adapter)**: SQL management for source denormalization and incremental persistence
- **dagster-dbt**: Dagster integration that invokes dbt staging and mart phases inside jobs
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

### Run the project

This repo has two entry points depending on what you are working on:

| Goal | Command |
| --- | --- |
| Run the IL211 pipeline (jobs, schedules, Snowflake) | `uv run dagster dev -m consumer.definitions` |
| Develop the `EntityResolutionComponent` library | `dg dev` |

**For pipeline development and debugging, always use:**

```bash
uv run dagster dev -m consumer.definitions
```

Then open [http://localhost:3000](http://localhost:3000) and go to
**Deployment → consumer.definitions → Jobs** to find:

- `entity_resolution__il211_regional__organization`
- `entity_resolution__il211_regional__service`

Use the **Launchpad** tab on either job to configure a run (e.g. restrict to one
`source_schema` for faster local testing) and launch it manually.

`dg dev` loads the reusable `EntityResolutionComponent` package — it intentionally
has no jobs or assets of its own and is only useful when working on the component
library itself.

## dbt project (`consumer/dbt/`)

The pipeline uses a dbt project to manage all complex SQL in one place. It runs
in two phases inside every Dagster job:

| Phase | dbt select | What it does |
| --- | --- | --- |
| **Staging** (before Python ER) | `--select staging` | Materializes `stg_service_denormalized` and `stg_organization_denormalized` tables in `DEDUPLICATION.ER_STAGING` from raw HSDS tables in `NORSE_STAGING` |
| **Marts** (after Python ER stages artifacts) | `--select marts` | Incremental merge models upsert artifact staging rows into the final output tables in `DEDUPLICATION.ER_RUNTIME` |

### Do you need to run any dbt commands before startup?

**No.** `dagster dev -m consumer.definitions` starts cleanly — the `DbtCliResource`
is just a resource handle at startup and triggers no dbt execution. dbt parses
and compiles the project automatically when each job phase runs.

No external dbt packages are used, so `dbt deps` is never required. `dbt build`
is not used; Dagster controls execution order via the phased job structure.

### Useful sanity check during development

After editing the dbt project, validate syntax and macro references without
hitting Snowflake:

```bash
cd consumer/dbt
uv run dbt parse --profiles-dir .
```

This confirms all Jinja loops compile, macro calls are valid, and `sources.yml`
references are consistent.

### Required environment variables for dbt

| Variable | Default | Purpose |
| --- | --- | --- |
| `SNOWFLAKE_ACCOUNT` | — | Snowflake account identifier |
| `SNOWFLAKE_USERNAME` | — | Snowflake username |
| `SNOWFLAKE_PASSWORD` | — | Snowflake password (or use `SNOWFLAKE_PRIVATE_KEY_PATH`) |
| `SNOWFLAKE_ROLE` | `SYSADMIN` | Snowflake role |
| `SNOWFLAKE_WAREHOUSE` | — | Snowflake virtual warehouse |
| `ER_TARGET_DATABASE` | `DEDUPLICATION` | Database for runtime and reconciliation tables |
| `ER_RUNTIME_SCHEMA` | `ER_RUNTIME` | Schema for mart output tables |
| `ER_INCREMENTAL_STATE_SCHEMA` | `ER_INCREMENTAL_STATE` | Schema for incremental state tables |
| `ER_STAGING_DATABASE` | `DEDUPLICATION` | Database for persistent staging tables |
| `ER_STAGING_SCHEMA` | `ER_STAGING` | Schema for persistent staging tables |
| `ER_HSDS_DATABASE` | `NORSE_STAGING` | Source HSDS database |

### dbt project structure

```
consumer/dbt/
  dbt_project.yml          — project config, model materialization defaults
  profiles.yml             — Snowflake connection (env-var based; DbtCliResource overrides in prod)
  models/
    sources.yml            — er_staging source definitions with not_null/unique tests
    schema.yml             — schema tests for staging and mart models
    staging/
      stg_service_denormalized.sql       — multi-tenant UNION over target_schemas
      stg_organization_denormalized.sql  — multi-tenant UNION over target_schemas
    marts/
      denormalized_service_cache.sql
      denormalized_organization_cache.sql
      deduplication_run.sql
      duplicate_pairs.sql
      duplicate_pair_scores.sql
      duplicate_reasons.sql
      mitigated_pairs.sql
      duplicate_clusters.sql
      duplicate_cluster_pairs.sql
  macros/
    taxonomy_rollup.sql         — ARRAY_AGG of taxonomy objects for service or org
    location_rollup_service.sql — SAL → LOCATION → ADDRESS for services
    location_rollup_org.sql     — LOCATION.ORGANIZATION_ID → ADDRESS for orgs
    phone_rollup_service.sql    — 3-path phone UNION for services
    phone_rollup_org.sql        — 4-path phone UNION for organizations
    service_rollup.sql          — org's services with nested taxonomy codes
    service_contact_rollup.sql  — service-level email/website rollup to org
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for pull request requirements, quality checks, and review
expectations.

## Additional Docs

- [DEDUPLICATION schema audit](docs/deduplication-schema-audit.md)
- [Review interface write path](docs/review-interface-write-path.md)
- [Training and tuning notes](docs/training-and-tuning.md)

## Using This In Another Dagster Repo

1. Publish or install this package (for example: `pip install hsds_entity_resolution`).
2. Confirm discovery in the target environment:

```bash
dg list components --package hsds_entity_resolution
```

3. Use the component key in YAML:

```yaml
type: hsds_entity_resolution.dagster.components.EntityResolutionComponent
attributes: {}
```
