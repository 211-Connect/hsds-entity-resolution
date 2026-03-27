# DEDUPLICATION Schema Audit and Semantic Reorganization Brief

## Purpose

This document inventories the live `DEDUPLICATION` tables currently used by the
entity-resolution pipeline and review UI, maps them back to code paths across
the active repos, and proposes a more legible semantic schema split.

This is a documentation artifact only. It does not propose or perform schema
moves, data migration, table renames, view creation, or app rewiring.

## Scope

Audited schemas:

- `DEDUPLICATION.COMMON_EXPERIMENT`
- `DEDUPLICATION.COMMON_EXPERIMENT_STAGING`

Codebases mapped:

- [`hsds-entity-resolution`](../README.md)
- [`dedupe-review`](../../dedupe-review/package.json)
- [`prod-dagster-copy/deduplicators`](../../prod-dagster-copy/deduplicators/README.md) as reference documentation

## Audit Inputs

This brief is based on current truth gathered on March 26, 2026 from:

- live Snowflake inventory queries against `DEDUPLICATION.INFORMATION_SCHEMA.TABLES`
- live Snowflake column metadata queries against `DEDUPLICATION.INFORMATION_SCHEMA.COLUMNS`
- live run and usage spot checks in `DEDUPLICATION.COMMON_EXPERIMENT`
- active code paths in the repos above

## Executive Summary

The current `COMMON_EXPERIMENT` schema blends five different concerns:

1. core ER runtime outputs
2. incremental reconciliation state
3. reviewer workflow state
4. tenant catalog/config sync state
5. staging inputs

That mixed ownership makes the schema harder to navigate than it needs to be.
The cleanest semantic split is:

- `er_runtime`
- `er_reconciliation`
- `review_workflow`
- `tenant_catalog`
- `er_staging`

The code already mostly follows those boundaries, even though the database
schema does not.

## Current Live Observations

As of March 26, 2026, live usage in `COMMON_EXPERIMENT` is concentrated in the
service pipeline for team `IL211`.

Recent runs:

| Run ID | Entity Type | Team | Job Name | Status | Created At |
| --- | --- | --- | --- | --- | --- |
| `er-c697c19fed804e939eefdddf721b18e6` | `service` | `IL211` | `hsds_entity_resolution_consumer` | `completed` | `2026-03-24 13:32:06` |
| `er-131e6d2dd78a46cdba60e99ed4f37e6c` | `service` | `IL211` | `hsds_entity_resolution_consumer` | `completed` | `2026-03-20 17:08:29` |
| `er-af3e97d85a594d2b88d6d8afbe369d4c` | `service` | `IL211` | `hsds_entity_resolution_consumer` | `completed` | `2026-03-20 15:53:06` |

High-signal row counts:

- `DUPLICATE_PAIR_SCORES`: `329,086`
- `DUPLICATE_REASONS`: `1,348,017`
- `DUPLICATE_PAIRS`: `117,516`
- `DUPLICATE_CLUSTER_PAIRS`: `115,464`
- `ER_PAIR_STATE_INDEX`: `110,677`
- `ER_ENTITY_INDEX`: `55,248`
- `DENORMALIZED_SERVICE_CACHE`: `15,027`
- `DENORMALIZED_ORGANIZATION_CACHE`: `0`
- reviewer workflow tables: all `0`

Important live-state conclusions:

- `COMMON_EXPERIMENT` is currently populated mostly by service ER data.
- Human workflow tables exist in the same schema but are currently empty.
- `DUPLICATE_PAIRS` is the durable pair identity table.
- `DUPLICATE_PAIR_SCORES` is the run-scoped scoring and review queue table.
- `COMMON_EXPERIMENT_STAGING` currently contains only two denormalized staging tables.

## Proposed Semantic Schema Split

### `er_runtime`

Tables:

- `DEDUPLICATION_RUN`
- `DENORMALIZED_ORGANIZATION_CACHE`
- `DENORMALIZED_SERVICE_CACHE`
- `DUPLICATE_PAIRS`
- `DUPLICATE_PAIR_SCORES`
- `DUPLICATE_REASONS`
- `MITIGATED_PAIRS`
- `DUPLICATE_CLUSTERS`
- `DUPLICATE_CLUSTER_PAIRS`

Why:

- primary ER outputs
- queue inputs for the review UI
- main persistence contract documented by dbt models and the older consumer spec

### `er_reconciliation`

Tables:

- `ER_ENTITY_INDEX`
- `ER_PAIR_STATE_INDEX`
- `ER_PAIR_ID_REMAP`
- `ER_REMOVED_ENTITY_IDS`
- `ER_REMOVED_PAIR_IDS`
- `ER_RUN_STATE`

Why:

- host-owned incremental state and cleanup metadata
- operational internals, not reviewer-facing state

### `review_workflow`

Tables:

- `DATA_QUALITY_FLAGS`
- `TENANT_PUBLISH_PREFERENCES`
- `REVIEWERS`
- `REVIEWER_TEAMS`
- `REVIEWER_TENANT_PERMISSIONS`
- `TEAMS`

Why:

- human review activity
- reviewer access and permission state
- publication preference decisions

### `tenant_catalog`

Tables:

- `DEDUPE_TENANTS`
- `DEDUPE_TENANT_SCHEMAS`
- `DEDUPE_SYNC_STATE`

Why:

- tenant identity
- tenant-to-source-schema mapping
- sync status for the catalog feed

### `er_staging`

Tables:

- `STG_ORGANIZATION_DENORMALIZED`
- `STG_SERVICE_DENORMALIZED`

Why:

- raw or pre-runtime input shape
- should remain isolated from operational and workflow tables

## Repo Hookup Map

### `hsds-entity-resolution`

Core compute and artifact generation:

- [`src/hsds_entity_resolution/core/pipeline.py`](../src/hsds_entity_resolution/core/pipeline.py)
- [`src/hsds_entity_resolution/core/prepare_persistence_artifacts.py`](../src/hsds_entity_resolution/core/prepare_persistence_artifacts.py)
- [`src/hsds_entity_resolution/dagster/components/entity_resolution_component.py`](../src/hsds_entity_resolution/dagster/components/entity_resolution_component.py)

Current persistence and staging hookup:

- dbt staging models:
  [`consumer/dbt/models/staging/stg_service_denormalized.sql`](../consumer/dbt/models/staging/stg_service_denormalized.sql),
  [`consumer/dbt/models/staging/stg_organization_denormalized.sql`](../consumer/dbt/models/staging/stg_organization_denormalized.sql)
- dbt mart merges:
  [`consumer/dbt/models/marts/`](../consumer/dbt/models/marts)
- cleanup SQL executor:
  [`consumer/consumer_adapter/persistence_executor.py`](../consumer/consumer_adapter/persistence_executor.py)
- reconciliation publisher:
  [`consumer/consumer_adapter/reconciliation_publisher.py`](../consumer/consumer_adapter/reconciliation_publisher.py)

Read-back for training:

- [`src/hsds_entity_resolution/core/training_feature_store.py`](../src/hsds_entity_resolution/core/training_feature_store.py)

### `dedupe-review`

Snowflake access and compiled query path:

- [`src/server/db/snowflake.ts`](../../dedupe-review/src/server/db/snowflake.ts)
- [`src/server/db/kysely.ts`](../../dedupe-review/src/server/db/kysely.ts)

Review queue reads and review writes:

- [`src/server/api/routers/review-query-builder.ts`](../../dedupe-review/src/server/api/routers/review-query-builder.ts)
- [`src/server/api/routers/review.ts`](../../dedupe-review/src/server/api/routers/review.ts)

Flag workflow:

- [`src/server/api/routers/flag.ts`](../../dedupe-review/src/server/api/routers/flag.ts)

Reviewer access sync:

- [`src/server/auth/sync-reviewer-access.ts`](../../dedupe-review/src/server/auth/sync-reviewer-access.ts)

Tenant catalog sync:

- [`scripts/sync-readers-config.mjs`](../../dedupe-review/scripts/sync-readers-config.mjs)
- [`src/server/tenant-catalog.ts`](../../dedupe-review/src/server/tenant-catalog.ts)

### `prod-dagster-copy/deduplicators`

Reference documentation and older persistence contract:

- [`CONSUMER_ADAPTER_TECHNICAL_SPEC.md`](../../prod-dagster-copy/deduplicators/CONSUMER_ADAPTER_TECHNICAL_SPEC.md)
- [`DEDUPE_CODEBASE_DISSECTION_REPORT.md`](../../prod-dagster-copy/deduplicators/DEDUPE_CODEBASE_DISSECTION_REPORT.md)
- [`ENTITY_RESOLUTION_REWRITE_RFC.md`](../../prod-dagster-copy/deduplicators/ENTITY_RESOLUTION_REWRITE_RFC.md)

## Table Inventory

### `er_runtime`

| Table | Current Schema | Live Rows | Purpose | Natural Key | Scope | Primary Owner | Producers | Consumers | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `DEDUPLICATION_RUN` | `COMMON_EXPERIMENT` | 3 | One row per ER run with config and outcome metadata | `ID` | run-scoped | pipeline | [`consumer/dbt/models/marts/deduplication_run.sql`](../consumer/dbt/models/marts/deduplication_run.sql) | [`../../dedupe-review/src/server/api/routers/team.ts`](../../dedupe-review/src/server/api/routers/team.ts), [`../../dedupe-review/src/server/api/routers/review.ts`](../../dedupe-review/src/server/api/routers/review.ts), [`../src/hsds_entity_resolution/core/training_feature_store.py`](../src/hsds_entity_resolution/core/training_feature_store.py) | Recent live rows are all `service` / `IL211` |
| `DENORMALIZED_ORGANIZATION_CACHE` | `COMMON_EXPERIMENT` | 0 | Durable org entity cache used for review display and training | `ID` | durable | pipeline | [`consumer/dbt/models/marts/denormalized_organization_cache.sql`](../consumer/dbt/models/marts/denormalized_organization_cache.sql) | [`../../dedupe-review/src/server/api/routers/review.ts`](../../dedupe-review/src/server/api/routers/review.ts), [`../src/hsds_entity_resolution/core/training_feature_store.py`](../src/hsds_entity_resolution/core/training_feature_store.py) | Present but currently empty in live schema |
| `DENORMALIZED_SERVICE_CACHE` | `COMMON_EXPERIMENT` | 15027 | Durable service entity cache used for review display and training | `ID` | durable | pipeline | [`consumer/dbt/models/marts/denormalized_service_cache.sql`](../consumer/dbt/models/marts/denormalized_service_cache.sql) | [`../../dedupe-review/src/server/api/routers/review.ts`](../../dedupe-review/src/server/api/routers/review.ts), [`../src/hsds_entity_resolution/core/training_feature_store.py`](../src/hsds_entity_resolution/core/training_feature_store.py) | Main populated cache today |
| `DUPLICATE_PAIRS` | `COMMON_EXPERIMENT` | 117516 | Durable pair identity table for entity A/B linkage and factual/global state | `ID`, effectively stable pair key | durable | pipeline | [`consumer/dbt/models/marts/duplicate_pairs.sql`](../consumer/dbt/models/marts/duplicate_pairs.sql) | [`../../dedupe-review/src/server/api/routers/review-query-builder.ts`](../../dedupe-review/src/server/api/routers/review-query-builder.ts), [`../../dedupe-review/src/server/api/routers/flag.ts`](../../dedupe-review/src/server/api/routers/flag.ts), [`../src/hsds_entity_resolution/core/training_feature_store.py`](../src/hsds_entity_resolution/core/training_feature_store.py) | Distinct from run-scoped score history |
| `DUPLICATE_PAIR_SCORES` | `COMMON_EXPERIMENT` | 329086 | Run-scoped score snapshot and reviewer decision carrier | `(DEDUPLICATION_RUN_ID, DUPLICATE_PAIR_ID)` | run-scoped | shared: pipeline writes, frontend updates review fields | [`consumer/dbt/models/marts/duplicate_pair_scores.sql`](../consumer/dbt/models/marts/duplicate_pair_scores.sql) | [`../../dedupe-review/src/server/api/routers/review-query-builder.ts`](../../dedupe-review/src/server/api/routers/review-query-builder.ts), [`../../dedupe-review/src/server/api/routers/review.ts`](../../dedupe-review/src/server/api/routers/review.ts), [`../../dedupe-review/src/server/api/routers/flag.ts`](../../dedupe-review/src/server/api/routers/flag.ts), [`../src/hsds_entity_resolution/core/training_feature_store.py`](../src/hsds_entity_resolution/core/training_feature_store.py) | Main queue source and review write target |
| `DUPLICATE_REASONS` | `COMMON_EXPERIMENT` | 1348017 | Signal-level explainability rows for each pair | `ID` | run-scoped | pipeline | [`consumer/dbt/models/marts/duplicate_reasons.sql`](../consumer/dbt/models/marts/duplicate_reasons.sql) | [`../../dedupe-review/src/server/api/routers/review-query-builder.ts`](../../dedupe-review/src/server/api/routers/review-query-builder.ts), [`../../dedupe-review/src/server/api/routers/review.ts`](../../dedupe-review/src/server/api/routers/review.ts), [`../src/hsds_entity_resolution/core/training_feature_store.py`](../src/hsds_entity_resolution/core/training_feature_store.py) | Largest table by row count |
| `MITIGATED_PAIRS` | `COMMON_EXPERIMENT` | 0 | Records pairs suppressed by mitigation logic | `ID` or `(DEDUPLICATION_RUN_ID, DUPLICATE_PAIR_ID, MITIGATION_REASON)` | run-scoped | pipeline | [`consumer/dbt/models/marts/mitigated_pairs.sql`](../consumer/dbt/models/marts/mitigated_pairs.sql) | [`../../dedupe-review/src/server/api/routers/review-query-builder.ts`](../../dedupe-review/src/server/api/routers/review-query-builder.ts), [`../src/hsds_entity_resolution/core/training_feature_store.py`](../src/hsds_entity_resolution/core/training_feature_store.py) | Table exists but is empty in current live snapshot |
| `DUPLICATE_CLUSTERS` | `COMMON_EXPERIMENT` | 3339 | Cluster-level aggregation over duplicate graph | `ID` | run-scoped | pipeline | [`consumer/dbt/models/marts/duplicate_clusters.sql`](../consumer/dbt/models/marts/duplicate_clusters.sql), [`../consumer/consumer_adapter/persistence_executor.py`](../consumer/consumer_adapter/persistence_executor.py) | no active consumer path found in `dedupe-review`; reference only in specs and exports | Operational runtime output, currently not used by UI |
| `DUPLICATE_CLUSTER_PAIRS` | `COMMON_EXPERIMENT` | 115464 | Cluster membership bridge between clusters and pairs | `(DUPLICATE_CLUSTER_ID, DUPLICATE_PAIR_ID)` | run-scoped | pipeline | [`consumer/dbt/models/marts/duplicate_cluster_pairs.sql`](../consumer/dbt/models/marts/duplicate_cluster_pairs.sql) | no active consumer path found in `dedupe-review`; reference only in specs and exports | Supports clustering but not current UI queries |

### `er_reconciliation`

| Table | Current Schema | Live Rows | Purpose | Natural Key | Scope | Primary Owner | Producers | Consumers | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `ER_ENTITY_INDEX` | `COMMON_EXPERIMENT` | 55248 | Incremental entity fingerprint and active-state index | `(ENTITY_ID, SCOPE_ID, ENTITY_TYPE)` | scope-durable | pipeline host adapter | [`../consumer/consumer_adapter/reconciliation_publisher.py`](../consumer/consumer_adapter/reconciliation_publisher.py) | internal ER runs via source loading and cleanup flow | Operational state, not surfaced in UI |
| `ER_PAIR_STATE_INDEX` | `COMMON_EXPERIMENT` | 110677 | Incremental pair retention and last-seen index | `PAIR_KEY` plus scope | scope-durable | pipeline host adapter | [`../consumer/consumer_adapter/reconciliation_publisher.py`](../consumer/consumer_adapter/reconciliation_publisher.py) | internal ER runs | Core reconciliation state |
| `ER_PAIR_ID_REMAP` | `COMMON_EXPERIMENT` | 0 | Tracks old-to-new pair key remaps after identity changes | `(OLD_PAIR_KEY, NEW_PAIR_KEY, RUN_ID)` | run-scoped | pipeline host adapter | [`../consumer/consumer_adapter/reconciliation_publisher.py`](../consumer/consumer_adapter/reconciliation_publisher.py) | internal ER runs | Present but empty in live snapshot |
| `ER_REMOVED_ENTITY_IDS` | `COMMON_EXPERIMENT` | 0 | Tracks entities removed from active scope | `(ENTITY_ID, RUN_ID)` | run-scoped | pipeline host adapter | [`../consumer/consumer_adapter/reconciliation_publisher.py`](../consumer/consumer_adapter/reconciliation_publisher.py) | internal cleanup workflows | Present but empty in live snapshot |
| `ER_REMOVED_PAIR_IDS` | `COMMON_EXPERIMENT` | 0 | Tracks pairs removed by cleanup logic | `(PAIR_KEY, RUN_ID)` | run-scoped | pipeline host adapter | [`../consumer/consumer_adapter/reconciliation_publisher.py`](../consumer/consumer_adapter/reconciliation_publisher.py) | [`../consumer/consumer_adapter/persistence_executor.py`](../consumer/consumer_adapter/persistence_executor.py) | Feeds delete cascade and stale-pair cleanup |
| `ER_RUN_STATE` | `COMMON_EXPERIMENT` | 2 | Stores policy/model tuple for incremental scope state | `(SCOPE_ID, ENTITY_TYPE)` or latest `RUN_ID` | scope-durable | pipeline host adapter | [`../consumer/consumer_adapter/reconciliation_publisher.py`](../consumer/consumer_adapter/reconciliation_publisher.py) | internal ER runs | Small operational metadata table |

### `review_workflow`

| Table | Current Schema | Live Rows | Purpose | Natural Key | Scope | Primary Owner | Producers | Consumers | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `DATA_QUALITY_FLAGS` | `COMMON_EXPERIMENT` | 0 | Reviewer-created flags attached to a pair or side of a pair | `ID` | workflow-durable | frontend | [`../../dedupe-review/src/server/api/routers/flag.ts`](../../dedupe-review/src/server/api/routers/flag.ts) | [`../../dedupe-review/src/server/api/routers/flag.ts`](../../dedupe-review/src/server/api/routers/flag.ts) | Human exception/reporting state |
| `TENANT_PUBLISH_PREFERENCES` | `COMMON_EXPERIMENT` | 0 | Winner-selection decisions for tenant publication | `ID`; effective uniqueness around pair, tenant, run | workflow-durable | frontend | [`../../dedupe-review/src/server/api/routers/review.ts`](../../dedupe-review/src/server/api/routers/review.ts) | [`../../dedupe-review/src/server/api/routers/review.ts`](../../dedupe-review/src/server/api/routers/review.ts) | Reviewer decision layer, currently empty |
| `REVIEWERS` | `COMMON_EXPERIMENT` | 0 | Reviewer identity registry | `REVIEWER_ID` | durable | frontend | [`../../dedupe-review/src/server/auth/sync-reviewer-access.ts`](../../dedupe-review/src/server/auth/sync-reviewer-access.ts) | same file, plus session/bootstrap paths via Snowflake types | App-owned access state |
| `REVIEWER_TEAMS` | `COMMON_EXPERIMENT` | 0 | Reviewer-to-team membership | `(REVIEWER_ID, TEAM_ID)` | durable | frontend | [`../../dedupe-review/src/server/auth/sync-reviewer-access.ts`](../../dedupe-review/src/server/auth/sync-reviewer-access.ts) | same file | Currently synced from whitelist config |
| `REVIEWER_TENANT_PERMISSIONS` | `COMMON_EXPERIMENT` | 0 | Reviewer-to-tenant publish permission table | `(REVIEWER_ID, TENANT_ID)` | durable | frontend | no active write path found in current repo | no active read path found in current repo | Current app permissions come from Auth0/session claims instead |
| `TEAMS` | `COMMON_EXPERIMENT` | 0 | Team registry for valid reviewer/team relationships | `ID` | durable | likely frontend or external seed | no active write path found in current repo | config comments and tests reference it indirectly | Table exists but appears unwired in active runtime code |

### `tenant_catalog`

| Table | Current Schema | Live Rows | Purpose | Natural Key | Scope | Primary Owner | Producers | Consumers | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `DEDUPE_TENANTS` | `COMMON_EXPERIMENT` | 59 | Canonical tenant list for dedupe UI and target schema mapping | `TENANT_ID` | durable | frontend support script | [`../../dedupe-review/scripts/sync-readers-config.mjs`](../../dedupe-review/scripts/sync-readers-config.mjs) | [`../../dedupe-review/src/server/api/routers/review.ts`](../../dedupe-review/src/server/api/routers/review.ts), [`../../dedupe-review/src/server/tenant-catalog.ts`](../../dedupe-review/src/server/tenant-catalog.ts) | Populated and actively used |
| `DEDUPE_TENANT_SCHEMAS` | `COMMON_EXPERIMENT` | 66 | Mapping from tenant to contributing source schemas | `ID`; effective uniqueness around `(TENANT_ID, SCHEMA_NAME)` | durable | frontend support script | [`../../dedupe-review/scripts/sync-readers-config.mjs`](../../dedupe-review/scripts/sync-readers-config.mjs) | [`../../dedupe-review/src/server/api/routers/review.ts`](../../dedupe-review/src/server/api/routers/review.ts), [`../../dedupe-review/src/server/tenant-catalog.ts`](../../dedupe-review/src/server/tenant-catalog.ts) | Connects `TARGET_SCHEMAS` to tenant identities |
| `DEDUPE_SYNC_STATE` | `COMMON_EXPERIMENT` | 1 | Sync status and fingerprint for tenant catalog load | `SYNC_KEY` | durable | frontend support script | [`../../dedupe-review/scripts/sync-readers-config.mjs`](../../dedupe-review/scripts/sync-readers-config.mjs) | [`../../dedupe-review/src/server/tenant-catalog.ts`](../../dedupe-review/src/server/tenant-catalog.ts) | Operational support table for readers config sync |

### `er_staging`

| Table | Current Schema | Live Rows | Purpose | Natural Key | Scope | Primary Owner | Producers | Consumers | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `STG_ORGANIZATION_DENORMALIZED` | `COMMON_EXPERIMENT_STAGING` | 0 | Pre-ER denormalized organization input built from HSDS source schemas | `ENTITY_ID` | ephemeral or refreshable | pipeline | [`../consumer/dbt/models/staging/stg_organization_denormalized.sql`](../consumer/dbt/models/staging/stg_organization_denormalized.sql) | Python source loading in the consumer adapter | Live table exists but is currently empty |
| `STG_SERVICE_DENORMALIZED` | `COMMON_EXPERIMENT_STAGING` | 0 | Pre-ER denormalized service input built from HSDS source schemas | `ENTITY_ID` | ephemeral or refreshable | pipeline | [`../consumer/dbt/models/staging/stg_service_denormalized.sql`](../consumer/dbt/models/staging/stg_service_denormalized.sql) | Python source loading in the consumer adapter | Live table exists but is currently empty |

## Known Mismatches and Hookup Risks

### 1. `dedupe-review` default schema mismatch

The main app env defaults to `COMMON`:

- [`../../dedupe-review/src/env.js`](../../dedupe-review/src/env.js)

But the reader catalog sync script defaults to `COMMON_EXPERIMENT`:

- [`../../dedupe-review/scripts/sync-readers-config.mjs`](../../dedupe-review/scripts/sync-readers-config.mjs)

That means a missing or inconsistent `SNOWFLAKE_SCHEMA` value can point
different parts of the app at different schemas.

### 2. dbt staging naming mismatch

The `hsds-entity-resolution` dbt project and comments still describe staging as
`DEDUPLICATION.STAGING`:

- [`../consumer/dbt/dbt_project.yml`](../consumer/dbt/dbt_project.yml)
- [`../consumer/dbt/models/sources.yml`](../consumer/dbt/models/sources.yml)
- [`../consumer/dbt/models/staging/stg_service_denormalized.sql`](../consumer/dbt/models/staging/stg_service_denormalized.sql)
- [`../consumer/dbt/models/staging/stg_organization_denormalized.sql`](../consumer/dbt/models/staging/stg_organization_denormalized.sql)

But the live audit target requested here shows physical staging tables in
`COMMON_EXPERIMENT_STAGING`.

This should be treated as a real hookup-risk item until runtime variables and
docs are aligned.

### 3. `REVIEWER_TENANT_PERMISSIONS` is modeled but not actively wired

The table exists in Snowflake and in the app's type model, but active runtime
authorization currently comes from Auth0 claims and reviewer whitelist config,
not from this table.

### 4. `TEAMS` exists without an obvious active write path

Current runtime code expects team identifiers to exist and match config, but no
active producer was found in the audited repos.

### 5. Current persistence truth is split across code and older docs

The older reference spec remains useful:

- [`../../prod-dagster-copy/deduplicators/CONSUMER_ADAPTER_TECHNICAL_SPEC.md`](../../prod-dagster-copy/deduplicators/CONSUMER_ADAPTER_TECHNICAL_SPEC.md)

But current runtime persistence is now more concretely represented by:

- `hsds-entity-resolution` dbt mart models
- `hsds-entity-resolution` consumer adapter cleanup logic
- `hsds-entity-resolution` reconciliation publisher

## What This Split Improves

- Faster navigation by ownership instead of by historical accretion
- Clearer separation between machine-generated ER outputs and human workflow state
- Easier onboarding for engineers who need to know where to look for:
  pipeline outputs, reconciliation state, review decisions, or tenant config
- Safer future refactor planning because the semantic boundaries are explicit

## Explicitly Out of Scope

- moving any physical tables
- creating views or synonyms
- renaming schemas or tables
- changing dbt configs
- updating app connection defaults
- backfilling or migrating data
- modifying permissions or grants

Those are the next phase once this taxonomy is accepted.

## Validation Checklist

- Every live table from `COMMON_EXPERIMENT` and `COMMON_EXPERIMENT_STAGING` appears exactly once in the inventory.
- Every table is assigned one proposed target schema and one primary owner class.
- Every table is mapped to at least one producer or consumer path, or explicitly marked as unwired.
- The live observations section reflects the current March 26, 2026 Snowflake snapshot.
- Current state, proposed organization, and future migration work are kept separate.
