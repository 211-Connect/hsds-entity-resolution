# Review Write Path to Snowflake

## Purpose

This document explains, in implementation-level detail, what changes in Snowflake when a reviewer leaves a review on a duplicate pair in the dedupe-review UI.

It is intended for external data orchestration engineers who want to read review decisions from Snowflake and use them to drive fine-tuning, evaluation, or ML retraining pipelines.

## Executive Summary

When a user reviews a pair in the UI, the app writes exactly one Snowflake row in exactly one table:

- `DUPLICATE_PAIR_SCORES`

For that row, the app updates exactly these columns:

- `IS_DUPLICATE`
- `REVIEWED_BY`
- `REVIEWED_AT`

The row is identified by:

- `ID = score_id`
- `TEAM_ID = selected team`

No other review-decision table is updated as part of the standard confirm/decline action.

In particular:

- `DUPLICATE_PAIRS.IS_DUPLICATE` is not updated by this review UI flow
- `DUPLICATE_REASONS` is not updated
- `DEDUPLICATION_RUN` is not updated
- `MITIGATED_PAIRS` is not updated
- `DATA_QUALITY_FLAGS` is not updated unless the user separately creates a flag

## Important Schema Note

There is a schema-documentation mismatch in this repository:

- The runtime application code clearly treats `DUPLICATE_PAIR_SCORES` as a pair-level scoring table with columns such as `DUPLICATE_PAIR_ID`, `PREDICTED_DUPLICATE`, `CONFIDENCE_SCORE`, `IS_DUPLICATE`, `REVIEWED_BY`, and `REVIEWED_AT`.
- The checked-in `_example_snowflake_rows/_describe_table/duplicate_pair_scores.csv` looks more like an older cluster-level shape.

For this document, "what the app actually uses" is based on:

- the live SQL in [`src/server/api/routers/review.ts`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review.ts)
- the shared queue query builder in [`src/server/api/routers/review-query-builder.ts`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review-query-builder.ts)
- the typed runtime schema in [`src/lib/types/database.ts`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/lib/types/database.ts)

If orchestration code is being built outside this app repo, confirm the production Snowflake schema before hard-coding assumptions.

## End-to-End Review Process

### 1. How a pair gets shown to a reviewer

The review queue is built from `DUPLICATE_PAIR_SCORES` joined to `DUPLICATE_PAIRS`, with `MITIGATED_PAIRS` used as an exclusion list.

The queue logic filters to rows where:

- `dps.TEAM_ID = <current team>`
- `dps.DEDUPLICATION_RUN_ID = <selected run>`
- `dps.ENTITY_TYPE = <organization|service>`
- `dps.PREDICTED_DUPLICATE = TRUE`
- `mp.ID IS NULL`

By default, the "next unreviewed" queue also requires:

- `dps.IS_DUPLICATE IS NULL`

That means the UI considers a pair "unreviewed" when the team-specific score row still has no decision in `DUPLICATE_PAIR_SCORES.IS_DUPLICATE`.

Relevant code:

- [`src/server/api/routers/review-query-builder.ts:86`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review-query-builder.ts#L86)
- [`src/server/api/routers/review-query-builder.ts:114`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review-query-builder.ts#L114)
- [`src/server/api/routers/review-query-builder.ts:134`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review-query-builder.ts#L134)

### 2. What the UI displays before a decision is made

For each pair, the app reads:

- pair-level score state from `DUPLICATE_PAIR_SCORES`
- entity linkage and factual/global duplicate state from `DUPLICATE_PAIRS`
- feature/reason evidence from `DUPLICATE_REASONS`
- entity details from either `DENORMALIZED_ORGANIZATION_CACHE` or `DENORMALIZED_SERVICE_CACHE`

The displayed review object distinguishes two concepts:

- `is_duplicate_opinionated`: the team-specific review decision from `DUPLICATE_PAIR_SCORES.IS_DUPLICATE`
- `is_duplicate_factual`: the global/factual field from `DUPLICATE_PAIRS.IS_DUPLICATE`

That distinction matters downstream: the UI writes the opinionated/team-level field, not the global/factual field.

Relevant code:

- [`src/server/api/routers/review-query-builder.ts:144`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review-query-builder.ts#L144)
- [`src/lib/types/review.ts:72`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/lib/types/review.ts#L72)

### 3. What happens immediately when the user clicks Confirm or Decline

The user can click:

- `Confirm Duplicate`
- `Decline Duplicate`

In the browser, the app performs an optimistic update first. Before Snowflake is updated, the current in-memory pair is changed locally to:

- set `is_duplicate_opinionated` to `true` or `false`
- set `reviewed_by` to the authenticated user's `reviewer_id`
- set `reviewed_at` to the browser's current timestamp string
- enqueue the review into a local pending queue for background submission

This is UI state only. It is not yet durable in Snowflake.

Relevant code:

- [`src/lib/jotai/atoms.ts:240`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/lib/jotai/atoms.ts#L240)
- [`src/components/review/actions/action-button-bar.tsx:20`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/components/review/actions/action-button-bar.tsx#L20)

### 4. How the browser sends the review to the backend

The hook `useReviewSubmitter` watches the local pending queue and sends reviews to the backend mutation `api.review.submit`.

Important behavior:

- submissions are asynchronous
- multiple queued reviews can be submitted in parallel
- if the same `score_id` is queued multiple times, the latest queued decision wins within that client-side batch
- on failure, the queue is retained and retried after a 5-second delay

This means the UI can briefly show a review decision before Snowflake has actually been updated.

Relevant code:

- [`src/hooks/useReviewSubmitter.tsx:23`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/hooks/useReviewSubmitter.tsx#L23)

## The Actual Snowflake Mutation

The standard review action executes this SQL:

```sql
UPDATE DUPLICATE_PAIR_SCORES
SET IS_DUPLICATE = ?,
    REVIEWED_BY = ?,
    REVIEWED_AT = CURRENT_TIMESTAMP()
WHERE ID = ? AND TEAM_ID = ?
```

Bound values are:

- `IS_DUPLICATE = input.is_duplicate_opinionated`
- `REVIEWED_BY = ctx.session.user.reviewer_id`
- `ID = input.score_id`
- `TEAM_ID = input.team_id`

Relevant code:

- [`src/server/api/routers/review.ts:311`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review.ts#L311)

## Exactly What Changes in Snowflake

### Table: `DUPLICATE_PAIR_SCORES`

One existing row is updated.

Row identity used by the app:

- `ID`
- `TEAM_ID`

Columns that change:

| Column | New value | Meaning |
| --- | --- | --- |
| `IS_DUPLICATE` | `true` or `false` | Team-specific review label for this scored pair |
| `REVIEWED_BY` | authenticated `reviewer_id` | Reviewer identity captured from the session |
| `REVIEWED_AT` | `CURRENT_TIMESTAMP()` | Server-side review timestamp |

Columns read but not changed during submit:

| Column | Why it matters |
| --- | --- |
| `DUPLICATE_PAIR_ID` | links the team-specific score row back to the shared pair |
| `DEDUPLICATION_RUN_ID` | used immediately after update to recompute progress |
| `ENTITY_TYPE` | used immediately after update to recompute progress |
| `TEAM_ID` | scopes the review row to a team |
| `PREDICTED_DUPLICATE` | determines queue eligibility |
| `CONFIDENCE_SCORE` | determines queue ordering |

### Table: `DUPLICATE_PAIRS`

No row is changed by the standard review submit flow.

Columns read from this table during review:

| Column | Why it matters |
| --- | --- |
| `ID` | joined from `DUPLICATE_PAIR_SCORES.DUPLICATE_PAIR_ID` |
| `ENTITY_A_ID` | used to load entity A details |
| `ENTITY_B_ID` | used to load entity B details |
| `IS_DUPLICATE` | exposed in the UI as `is_duplicate_factual` |

This is the key distinction for downstream systems:

- `DUPLICATE_PAIR_SCORES.IS_DUPLICATE` = team-level review decision written by this UI
- `DUPLICATE_PAIRS.IS_DUPLICATE` = separate global/factual field, only read here

### Table: `DUPLICATE_REASONS`

No row is changed.

This table is read to explain why the pair was scored as a likely duplicate. It can be useful as feature provenance when building retraining datasets.

Columns read by the UI:

- `DUPLICATE_PAIR_ID`
- `MATCH_TYPE`
- `WEIGHTED_CONTRIBUTION`
- `MATCHED_VALUE`
- `SIMILARITY_SCORE`

### Table: `DENORMALIZED_ORGANIZATION_CACHE`

No row is changed.

Used only when reviewing organization pairs.

The app reads entity attributes such as:

- `ID`
- `NAME`
- `ALTERNATE_NAME`
- `DESCRIPTION`
- `EMAIL`
- `WEBSITES`
- `IDENTIFIERS`
- `LOCATIONS`
- `PHONES`
- `TAXONOMIES`
- `SERVICES`
- lineage fields such as `SOURCE_SCHEMA`, `ORIGINAL_ID`, `RESOURCE_WRITER_NAME`, `ASSURED_DATE`, `ASSURER_EMAIL`

### Table: `DENORMALIZED_SERVICE_CACHE`

No row is changed.

Used only when reviewing service pairs.

The app reads entity attributes such as:

- `ID`
- `ORGANIZATION_ID`
- `ORGANIZATION_NAME`
- `NAME`
- `ALTERNATE_NAME`
- `DESCRIPTION`
- `SHORT_DESCRIPTION`
- `APPLICATION_PROCESS`
- `FEES_DESCRIPTION`
- `ELIGIBILITY_DESCRIPTION`
- `WEBSITES`
- `LOCATIONS`
- `PHONES`
- `TAXONOMIES`
- lineage fields such as `SOURCE_SCHEMA`, `ORIGINAL_ID`, `RESOURCE_WRITER_NAME`, `ASSURED_DATE`, `ASSURER_EMAIL`

### Table: `MITIGATED_PAIRS`

No row is changed by review submit.

This table affects visibility, not review persistence. Any pair present here is excluded from the review queue because the queue builder requires `mp.ID IS NULL`.

### Table: `DEDUPLICATION_RUN`

No row is changed.

The review flow uses the run context to scope queue membership and progress, but the submit action does not mutate the run record.

### Table: `REVIEWERS`

No row is changed during a review submission.

The value written into `DUPLICATE_PAIR_SCORES.REVIEWED_BY` comes from the authenticated session's `reviewer_id`, not from a write into `REVIEWERS` at submit time.

### Table: `REVIEWER_TEAMS`

No row is changed during a review submission.

This table is used for access/scoping elsewhere, not as part of the actual review write.

### Table: `DATA_QUALITY_FLAGS`

No row is changed by a normal confirm/decline review.

This table is only written if the reviewer separately creates a flag. That is a different mutation path from the actual review decision.

Relevant code:

- [`src/server/api/routers/flag.ts:35`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/flag.ts#L35)

## Review State Machine From a Data Perspective

For one `DUPLICATE_PAIR_SCORES` row, the main persisted state transition is:

### Before review

Typical values:

- `IS_DUPLICATE = NULL`
- `REVIEWED_BY = NULL`
- `REVIEWED_AT = NULL`

This row is eligible for the default unreviewed queue if it also satisfies:

- `PREDICTED_DUPLICATE = TRUE`
- matching `TEAM_ID`
- matching `DEDUPLICATION_RUN_ID`
- matching `ENTITY_TYPE`
- not present in `MITIGATED_PAIRS`

### After confirm duplicate

Persisted values become:

- `IS_DUPLICATE = TRUE`
- `REVIEWED_BY = <reviewer_id>`
- `REVIEWED_AT = <server timestamp>`

### After decline duplicate

Persisted values become:

- `IS_DUPLICATE = FALSE`
- `REVIEWED_BY = <reviewer_id>`
- `REVIEWED_AT = <server timestamp>`

### If the same pair is reviewed again later

The same row is updated again, not versioned.

That means:

- the prior decision is overwritten
- the prior reviewer is overwritten
- the prior review timestamp is overwritten
- this app does not maintain an audit/history table for review changes

For ML pipeline design, this is important: the database currently stores the latest state, not a change log.

## How Progress Is Calculated

After each successful submit, the backend recomputes progress using counts over the same queue filter logic.

The important part is:

```sql
COUNT(*) AS TOTAL_PAIRS,
COUNT(dps.IS_DUPLICATE) AS REVIEWED_PAIRS
```

Implications:

- any non-null `IS_DUPLICATE` counts as reviewed
- both `TRUE` and `FALSE` are considered completed reviews
- "reviewed" means "decision present", not "confirmed duplicate"

Relevant code:

- [`src/server/api/routers/review.ts:554`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review.ts#L554)

## Recommended Read Model for Training Pipelines

If you want to build a supervised dataset from reviewer actions, the safest primary source is:

- `DUPLICATE_PAIR_SCORES`

Recommended minimum extraction fields:

| Source table | Fields |
| --- | --- |
| `DUPLICATE_PAIR_SCORES` | `ID`, `DUPLICATE_PAIR_ID`, `DEDUPLICATION_RUN_ID`, `TEAM_ID`, `ENTITY_TYPE`, `PREDICTED_DUPLICATE`, `CONFIDENCE_SCORE`, `DETERMINISTIC_SECTION_SCORE`, `NLP_SECTION_SCORE`, `ML_SECTION_SCORE`, `RAW_DETERMINISTIC_SCORE`, `RAW_NLP_SCORE`, `RAW_ML_SCORE`, `EMBEDDING_SIMILARITY`, `IS_DUPLICATE`, `REVIEWED_BY`, `REVIEWED_AT` |
| `DUPLICATE_PAIRS` | `ID`, `ENTITY_A_ID`, `ENTITY_B_ID`, `IS_DUPLICATE` |
| `DUPLICATE_REASONS` | `DUPLICATE_PAIR_ID`, `MATCH_TYPE`, `WEIGHTED_CONTRIBUTION`, `MATCHED_VALUE`, `SIMILARITY_SCORE` |
| `DEDUPLICATION_RUN` | run metadata such as model versions, thresholds, weights, `JOB_NAME`, `TEAM_ID`, `CREATED_AT`, `COMPLETED_AT` |
| `DENORMALIZED_ORGANIZATION_CACHE` or `DENORMALIZED_SERVICE_CACHE` | entity text/content fields needed for feature regeneration or prompt construction |

Recommended label interpretation:

- use `DUPLICATE_PAIR_SCORES.IS_DUPLICATE` as the human review label produced by this UI
- treat `NULL` as unlabeled
- treat `TRUE` as positive label
- treat `FALSE` as negative label

Recommended filters:

- `PREDICTED_DUPLICATE = TRUE` if you want to mirror exactly what the current review UI exposes
- `REVIEWED_AT IS NOT NULL` or `IS_DUPLICATE IS NOT NULL` if you only want completed reviews

Recommended cautions:

- do not use browser-side optimistic state as truth; only Snowflake writes are durable
- do not assume one pair can have only one human label across all teams unless you explicitly de-duplicate by business rules
- do not confuse `DUPLICATE_PAIR_SCORES.IS_DUPLICATE` with `DUPLICATE_PAIRS.IS_DUPLICATE`
- do not assume review history exists; this schema stores current/latest state only
- a flagged pair is not the same thing as a reviewed pair; `DATA_QUALITY_FLAGS` is a separate signal
- mitigated pairs may never appear to reviewers even if scored, because they are excluded from the queue

## Example Durable Label Query

This query shape matches the current app's notion of a persisted team review label:

```sql
SELECT
  dps.ID AS score_id,
  dps.DUPLICATE_PAIR_ID,
  dps.DEDUPLICATION_RUN_ID,
  dps.TEAM_ID,
  dps.ENTITY_TYPE,
  dps.CONFIDENCE_SCORE,
  dps.PREDICTED_DUPLICATE,
  dps.IS_DUPLICATE AS team_review_label,
  dps.REVIEWED_BY,
  dps.REVIEWED_AT,
  dp.ENTITY_A_ID,
  dp.ENTITY_B_ID,
  dp.IS_DUPLICATE AS factual_duplicate_label
FROM DUPLICATE_PAIR_SCORES dps
JOIN DUPLICATE_PAIRS dp
  ON dps.DUPLICATE_PAIR_ID = dp.ID
WHERE dps.IS_DUPLICATE IS NOT NULL;
```

## Example Feature-Enriched Training Extraction

```sql
SELECT
  dps.ID AS score_id,
  dps.DUPLICATE_PAIR_ID,
  dps.DEDUPLICATION_RUN_ID,
  dps.TEAM_ID,
  dps.ENTITY_TYPE,
  dps.CONFIDENCE_SCORE,
  dps.DETERMINISTIC_SECTION_SCORE,
  dps.NLP_SECTION_SCORE,
  dps.ML_SECTION_SCORE,
  dps.RAW_DETERMINISTIC_SCORE,
  dps.RAW_NLP_SCORE,
  dps.RAW_ML_SCORE,
  dps.EMBEDDING_SIMILARITY,
  dps.IS_DUPLICATE AS team_review_label,
  dps.REVIEWED_BY,
  dps.REVIEWED_AT,
  dp.ENTITY_A_ID,
  dp.ENTITY_B_ID,
  dp.IS_DUPLICATE AS factual_duplicate_label
FROM DUPLICATE_PAIR_SCORES dps
JOIN DUPLICATE_PAIRS dp
  ON dps.DUPLICATE_PAIR_ID = dp.ID
WHERE dps.IS_DUPLICATE IS NOT NULL
  AND dps.PREDICTED_DUPLICATE = TRUE;
```

Then join:

- `DUPLICATE_REASONS` on `DUPLICATE_PAIR_ID`
- organization/service cache tables on `ENTITY_A_ID` and `ENTITY_B_ID`
- `DEDUPLICATION_RUN` on `DEDUPLICATION_RUN_ID`

## Non-Obvious Implications for Orchestration

1. The review label is team-scoped.

The write is constrained by both `ID` and `TEAM_ID`. Downstream pipelines should preserve team context unless there is an explicit normalization rule.

2. The latest review overwrites the previous one.

If a user re-reviews a pair, the prior label is lost unless some external CDC/audit process captured it.

3. Reviewed does not mean positive duplicate.

`FALSE` is still a valuable reviewed label and should remain in the training set as a negative example.

4. The queue excludes mitigated pairs.

A missing review does not necessarily mean "not yet reached"; the pair may be filtered out by mitigation or other queue filters.

5. The app already separates model evidence from human label.

`DUPLICATE_REASONS` and score fields hold model evidence; `DUPLICATE_PAIR_SCORES.IS_DUPLICATE` holds the human label written by this UI.

## Code References

- Review submit mutation: [`src/server/api/routers/review.ts:311`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review.ts#L311)
- Queue filter logic: [`src/server/api/routers/review-query-builder.ts:86`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review-query-builder.ts#L86)
- Queue item projection: [`src/server/api/routers/review-query-builder.ts:114`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review-query-builder.ts#L114)
- Pair detail projection: [`src/server/api/routers/review-query-builder.ts:134`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/server/api/routers/review-query-builder.ts#L134)
- Optimistic client-side review state: [`src/lib/jotai/atoms.ts:240`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/lib/jotai/atoms.ts#L240)
- Background submission behavior: [`src/hooks/useReviewSubmitter.tsx:23`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/hooks/useReviewSubmitter.tsx#L23)
- Runtime DB type definitions: [`src/lib/types/database.ts:33`](/Users/davidbotos/Desktop/9-BearHug_Product/5-Dedupe/dedupe-review/src/lib/types/database.ts#L33)
