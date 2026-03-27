# Pipeline Tuning & Training Data Infrastructure

This document covers the strategy for improving entity resolution accuracy and
the data infrastructure built to support it. The two concerns are tightly
linked: you cannot tune parameters responsibly without labeled ground truth,
and labeled ground truth is only useful if it is stable and ML-contamination-free.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Training Data Schema — `DEDUPLICATION.TRAINING_DATA`](#2-training-data-schema)
3. [Cutting a Training Dataset](#3-cutting-a-training-dataset)
4. [Pipeline Tuning Plan](#4-pipeline-tuning-plan)
5. [Parameter Reference](#5-parameter-reference)
6. [How Everything Fits Together](#6-how-everything-fits-together)

---

## 1. Problem Statement
The pipeline has two accuracy failure modes that require different treatments:

**False positives** — pairs predicted as duplicates that are distinct entities.
Most often caused by high NLP similarity alone (names sound alike) without any
supporting deterministic evidence (no shared email, phone, or address).

**False negatives** — true duplicates that are not being surfaced. Most often
caused by overly aggressive blocking (cosine similarity threshold drops the
pair before scoring) or mitigation vetoing a pair that had strong deterministic
evidence but low embedding similarity.

These goals are in tension. Lowering thresholds to catch more true duplicates
also admits more false positives. The strategy below resolves this tension
through a deliberate, measured sequence: build ground truth first, diagnose
before tuning, change one variable at a time.

---

## 2. Training Data Schema

**Location:** `scripts/create_training_data_schema.sql`
**Snowflake schema:** `DEDUPLICATION.TRAINING_DATA`

Run to create the schema:

```bash
set -a; source .env; set +a
SNOWSQL_PWD="$SNOWFLAKE_PASSWORD" snowsql --noup \
  -a "$SNOWFLAKE_ACCOUNT" -u "$SNOWFLAKE_USERNAME" \
  -r "$SNOWFLAKE_ROLE" -w "$SNOWFLAKE_WAREHOUSE" \
  -f scripts/create_training_data_schema.sql < /dev/null
```

### Design Principles

The schema is built around three guarantees:

1. **Immutability.** Once written, no row is updated in place. Entity content
   and pipeline scores are frozen at the moment a pair is submitted for review.
   Upstream data changes or re-runs never corrupt previously labeled examples.

2. **Full fidelity.** Every field that influenced scoring — raw deterministic
   scores, NLP similarity, embedding vector, signals array, and the complete
   `EntityResolutionRunConfig` — is stored alongside the label. Features are
   materialized offline from immutable snapshots and reasons using the same
   production extraction path, then stored as typed pair-feature rows.

3. **ML contamination isolation.** The `CONFIDENCE_SCORE` stored in
   `TRAINING_PAIR` includes the old ML model's contribution. Training a new
   model on that score creates a circular dependency. A separate
   `BASELINE_CONFIDENCE_SCORE` is stored at insertion time, computed from
   deterministic + NLP sections only. All ML training work uses baseline scores.

### Tables

#### `ENTITY_SNAPSHOT`

The immutable shared identity / provenance spine. Keyed by `(ENTITY_ID,
CONTENT_HASH)` using the same SHA-256 hash the pipeline computes in
`clean_entities`. Identical entity states share one row; a changed entity gets
a new row without touching the old one.

#### `ORGANIZATION_SNAPSHOT` and `SERVICE_SNAPSHOT`

Typed child tables keyed by `SNAPSHOT_ID`. These store the frozen entity
payload used to reconstruct the production feature extractor inputs for each
entity type. The raw nested fields (`LOCATIONS`, `TAXONOMIES`, `PHONES`,
`WEBSITES`, `IDENTIFIERS`, `SERVICES`, `EMBEDDING_VECTOR`) remain available for
auditability and exact feature reconstruction, but the snapshot payload is now
decomposed by entity type rather than stored in one generic table.

#### `TRAINING_PAIR`

An immutable record of one candidate pair comparison. Stores:

- References to both `ENTITY_SNAPSHOT` rows (`ENTITY_A_SNAPSHOT_ID`,
  `ENTITY_B_SNAPSHOT_ID`)
- **Baseline scores (use for ML training):**
  - `BASELINE_CONFIDENCE_SCORE` — renormalized det+NLP score with ML excluded
  - `BASELINE_PREDICTED_DUPLICATE`, `BASELINE_PREDICTED_MAYBE`
- **Full pipeline scores (audit/traceability only):**
  - `CONFIDENCE_SCORE` — ML-inclusive; reflects the original pipeline decision
  - `DETERMINISTIC_SECTION_SCORE`, `NLP_SECTION_SCORE`, `ML_SECTION_SCORE`
  - `RAW_DETERMINISTIC_SCORE`, `RAW_NLP_SCORE`, `RAW_ML_SCORE`
  - `EMBEDDING_SIMILARITY`
- `PIPELINE_SIGNALS` — array of per-signal objects (match type, matched value,
  both entity values, raw contribution, weighted contribution, signal weight)
- `PIPELINE_CONFIG_SNAPSHOT` — full `EntityResolutionRunConfig` as JSON
- `TEAM_ID`, `SCOPE_ID` — client / scope selectors used by the offline feature
  materialization jobs
- `SOURCE_PAIR_ID`, `SOURCE_SCORE_ID` — durable links back to the reviewed
  `COMMON_EXPERIMENT` artifacts the UI actually updates
- `WAS_MITIGATED`, `MITIGATION_REASON`
- `PAIR_CANONICAL_KEY` — `ENTITY_A_ID || '|' || ENTITY_B_ID`, always `A < B`

#### `ORGANIZATION_PAIR_FEATURES` and `SERVICE_PAIR_FEATURES`

The canonical ML input store. These tables are populated by a separate offline
job that reads reviewed pairs, reconstructs the exact production payload used
by ML inference, and persists one immutable typed feature row per
`TRAINING_PAIR_ID` and `FEATURE_SCHEMA_VERSION`.

Important rules:

- These tables store the exact final model input columns defined in
  `src/hsds_entity_resolution/core/ml_inference.py`.
- Override-derived features such as `fuzzy_name`, `shared_address`, and
  `shared_phone` are persisted exactly as production constructs them.
- `ML_SECTION_SCORE`, `RAW_ML_SCORE`, and any new model output are never stored
  here. Those remain audit-only in `TRAINING_PAIR`.

The baseline score formula:

```
BASELINE_CONFIDENCE_SCORE =
    (DETERMINISTIC_SECTION_SCORE + NLP_SECTION_SCORE)
  / (det_section_weight + nlp_section_weight)
```

Both weights come from `PIPELINE_CONFIG_SNAPSHOT` at insertion time.

#### `PAIR_REVIEW`

Append-only review ledger derived downstream from the frontend review UI. The
canonical first write still happens in
`DEDUPLICATION.COMMON_EXPERIMENT.DUPLICATE_PAIR_SCORES` where the UI updates
`IS_DUPLICATE`, `REVIEWED_BY`, and `REVIEWED_AT` on a score row identified by
`ID = score_id` and `TEAM_ID`.

`PAIR_REVIEW` is populated later by the training-feature materializer so the
training schema keeps a stable, append-only label history without requiring the
frontend to write directly into `TRAINING_DATA`. The derived row retains the
upstream lineage in `SOURCE_SCORE_ID` and `REVIEW_SOURCE`.

No row is ever updated in place by dataset tooling. If a later materialization
observes a changed upstream decision for the same score row, the old derived
review is marked inactive and a new active row is inserted so the full
decision chain remains navigable.

| `REVIEW_DECISION` | Meaning |
|---|---|
| `TRUE_DUPLICATE` | Reviewer is confident these are the same real-world entity. Safe as a positive training label. |
| `FALSE_POSITIVE` | Reviewer is confident these are distinct. Safe as a negative training label. |
| `UNSURE` | Reviewer could not determine. Do not use as a definitive training label; track to identify data gaps. |
| `SKIP` | Reviewer deferred. Always excluded from training datasets. |

`REVIEWER_CONFIDENCE` (`HIGH` / `MEDIUM` / `LOW`) records how certain the
reviewer was. `LOW`-confidence reviews should be flagged for second-pass
adjudication before entering a training dataset.

#### `TRAINING_DATASET`

A named, versioned collection header (e.g., `"il211-org-baseline"` / `"1.0.0"`).
Materialized label counts (`TRUE_DUPLICATE_COUNT`, `FALSE_POSITIVE_COUNT`,
`UNSURE_COUNT`) are stored for quick cardinality checks.

#### `TRAINING_DATASET_MEMBER`

Maps pairs to a dataset and **freezes the label at dataset-cut time**.
Subsequent review changes do not affect an already-cut dataset. `LABEL_SOURCE`
records how the label was derived:

| `LABEL_SOURCE` | Meaning |
|---|---|
| `SINGLE_REVIEW` | One active review drove the label; `REVIEW_ID` points to it. |
| `CONSENSUS` | Multiple active reviews agree; `REVIEW_ID` is `NULL`. |
| `ADJUDICATED` | Disagreement resolved by an authoritative reviewer. |

### Views

| View | Purpose |
|---|---|
| `V_ACTIVE_PAIR_LABELS` | Latest active review per pair. Foundation for all label-dependent queries. |
| `V_MATERIALIZED_PAIR_FEATURES_INDEX` | Thin union view over the typed pair-feature tables. The default dataset-cutting and offline-job lookup surface. |
| `V_TRAINING_PAIRS_FULL` | Full entity A + entity B snapshots + active label + all scores in one flat row. Audit and provenance surface, not the canonical ML feature store. |
| `V_REVIEW_DISAGREEMENTS` | Pairs where multiple active reviews conflict. Must be adjudicated before dataset inclusion. |
| `V_UNLABELED_PAIRS` | Pairs awaiting first review, ordered by `BASELINE_CONFIDENCE_SCORE DESC` so reviewers work through the clearest cases first. |

### Population Flow

The real end-to-end flow is:

1. `entity_resolution__...` writes pipeline outputs into
   `DEDUPLICATION.COMMON_EXPERIMENT`, including `DUPLICATE_PAIRS`,
   `DUPLICATE_PAIR_SCORES`, `DUPLICATE_REASONS`, and the denormalized cache.
2. The frontend review UI reads from those `COMMON_EXPERIMENT` tables and, on
   confirm/decline, updates exactly one existing `DUPLICATE_PAIR_SCORES` row:
   `IS_DUPLICATE`, `REVIEWED_BY`, and `REVIEWED_AT`.
3. `training_features__...` later reads the reviewed score rows from
   `COMMON_EXPERIMENT`, joins back to the pair, reasons, run config, and
   denormalized cache, then freezes the result into `TRAINING_DATA`:
   - `ENTITY_SNAPSHOT` + `ORGANIZATION_SNAPSHOT` / `SERVICE_SNAPSHOT`
   - `TRAINING_PAIR`
   - derived `PAIR_REVIEW`
   - typed pair-feature rows in `ORGANIZATION_PAIR_FEATURES` or
     `SERVICE_PAIR_FEATURES`

---

## 3. Cutting a Training Dataset

**Locations:** `scripts/materialize_training_features.py`,
`scripts/cut_training_dataset.py`

```bash
# Always preview before writing
uv run python scripts/cut_training_dataset.py \
    --name "il211-org-baseline" \
    --version "1.0.0" \
    --entity-type organization \
    --created-by "you@example.com" \
    --dry-run

# Write when the plan looks right
uv run python scripts/cut_training_dataset.py \
    --name "il211-org-baseline" \
    --version "1.0.0" \
    --entity-type organization \
    --created-by "you@example.com"
```

### Materialize Features First

Before cutting a dataset, materialize the reviewed pairs into typed feature
rows:

```bash
uv run python scripts/materialize_training_features.py \
    --team-id "IL211" \
    --entity-type organization \
    --run-selection latest
```

The materializer reads reviewed score rows from
`DEDUPLICATION.COMMON_EXPERIMENT.DUPLICATE_PAIR_SCORES`, joins the shared pair
and entity cache data, then freezes those reviewed examples into
`TRAINING_DATA`. As part of that downstream freeze step it also derives
`PAIR_REVIEW` rows so dataset cutting can continue to operate on a stable
append-only review ledger, then writes one immutable typed feature row per
`TRAINING_PAIR_ID` and `FEATURE_SCHEMA_VERSION`.

### Label Resolution

The script considers all active (`IS_ACTIVE = TRUE`) `PAIR_REVIEW` rows per pair:

- **One active review** → `SINGLE_REVIEW`; uses that review's decision.
- **Multiple active reviews, all agree** → `CONSENSUS`; `REVIEW_ID` is `NULL`.
- **Multiple active reviews, disagree** → `DISAGREEMENT`; excluded by default
  until adjudicated.
- **`SKIP` decision** → always excluded.
- **`UNSURE` decision** → excluded by default; include with `--include-unsure`.

### Key Flags

| Flag | Purpose |
|---|---|
| `--entity-type organization\|service` | Restrict to one entity type. Omit for mixed. |
| `--policy-version <ver>` | Only include pairs scored under a specific pipeline policy version. |
| `--min-confidence HIGH\|MEDIUM\|LOW` | Exclude reviews below this confidence level. |
| `--include-unsure` | Include `UNSURE` labels (excluded by default). |
| `--dry-run` | Show plan without writing anything. |
| `--yes` / `-y` | Skip confirmation prompt for scripted use. |

### What Gets Written

In a single atomic transaction:

1. One `TRAINING_DATASET` row (header + materialized counts).
2. One `TRAINING_DATASET_MEMBER` row per included pair, with the label frozen
   from `V_ACTIVE_PAIR_LABELS` at the moment the script ran.

By default the cutter now requires a stored typed pair-feature row and applies
stratified negative sampling by baseline score band:

- 40% `duplicate`
- 40% `maybe`
- 20% `below_maybe`

All reviewed `TRUE_DUPLICATE` pairs are included. Reviewed negatives are
band-balanced by default so weak-evidence negatives do not dominate the
training set.

Subsequent review changes do not affect a dataset once cut. To capture updated
labels, cut a new version with an incremented `--version`.

---

## 4. Pipeline Tuning Plan

### Phase 0 — Build Ground Truth First

**Do not touch any config until this phase is complete.**

A stratified sample of labeled pairs is the foundation for every tuning
decision. Without it, you cannot measure whether a change improved or degraded
accuracy.

**Step 1 — Sample across score bands.** Run the following against
`DEDUPLICATION.COMMON_EXPERIMENT` after a completed pipeline run:

```sql
SELECT
    dp.ENTITY_A_ID,
    dp.ENTITY_B_ID,
    dp.ENTITY_TYPE,
    dps.CONFIDENCE_SCORE,
    dps.DETERMINISTIC_SECTION_SCORE,
    dps.NLP_SECTION_SCORE,
    dps.ML_SECTION_SCORE,
    dps.EMBEDDING_SIMILARITY,
    dps.PREDICTED_DUPLICATE,
    CASE
        WHEN dps.CONFIDENCE_SCORE >= 0.82 THEN 'duplicate'
        WHEN dps.CONFIDENCE_SCORE >= 0.68 THEN 'maybe'
        ELSE 'below_maybe'
    END AS score_band
FROM DEDUPLICATION.COMMON_EXPERIMENT.DUPLICATE_PAIRS dp
JOIN DEDUPLICATION.COMMON_EXPERIMENT.DUPLICATE_PAIR_SCORES dps
    ON dp.ID = dps.DUPLICATE_PAIR_ID
WHERE dps.REVIEWED_AT IS NULL
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY dp.ENTITY_TYPE,
    CASE
        WHEN dps.CONFIDENCE_SCORE >= 0.82 THEN 'duplicate'
        WHEN dps.CONFIDENCE_SCORE >= 0.68 THEN 'maybe'
        ELSE 'below_maybe'
    END
    ORDER BY RANDOM()
) <= 50
ORDER BY dp.ENTITY_TYPE, score_band, dps.CONFIDENCE_SCORE DESC;
```

This yields ~300 pairs: 50 per band per entity type. Aim for at least 100
labeled pairs before drawing any conclusions.

**Step 2 — Near-miss sample.** To catch false negatives, also sample pairs
that were generated as candidates but scored below `maybe_threshold`. These
require temporarily lowering `maybe_threshold` to expose them, or querying the
`ER_PAIR_STATE_INDEX` directly.

**Step 3 — Human labeling.** For each pair, a subject-matter expert labels it
using the review interface. The UI writes that decision to the existing
`DEDUPLICATION.COMMON_EXPERIMENT.DUPLICATE_PAIR_SCORES` row by updating
`IS_DUPLICATE`, `REVIEWED_BY`, and `REVIEWED_AT`. The later
`training_features__...` job then derives `TRAINING_DATA.PAIR_REVIEW` from
those reviewed score rows. Include enough context (entity names, addresses,
contact fields) in the review UI so that decisions do not require domain
knowledge beyond what is in the data.

**Target before moving to Phase 1:** at least 50 labeled pairs per entity type,
covering all three score bands.

---

### Phase 1 — Measure the Baseline

Run the audit script immediately after completing Phase 0, before any config
changes. Record these numbers as the baseline to compare against:

```bash
uv run python scripts/audit_er_run.py --run-id er-<your-run-id> --entity-type organization
uv run python scripts/audit_er_run.py --run-id er-<your-run-id> --entity-type service
```

| Metric to record | Query / source |
|---|---|
| Pairs in `duplicate` band | `audit_er_run.py` Section 2 |
| Pairs in `maybe` band | `audit_er_run.py` Section 2 |
| % of `duplicate` pairs with 0 reasons | `audit_er_run.py` Section 2 (red flag) |
| ML gate pass rate for 2+ signal pairs | `audit_er_run.py` Section 3a |
| Signal × band distribution | `audit_er_run.py` Section 3b |

Also run this signal-contribution query:

```sql
SELECT
    dr.MATCH_TYPE,
    dps.PREDICTED_DUPLICATE,
    COUNT(*) AS pair_count,
    AVG(dps.CONFIDENCE_SCORE) AS avg_score
FROM DEDUPLICATION.COMMON_EXPERIMENT.DUPLICATE_REASONS dr
JOIN DEDUPLICATION.COMMON_EXPERIMENT.DUPLICATE_PAIR_SCORES dps
    ON dr.DUPLICATE_PAIR_ID = dps.DUPLICATE_PAIR_ID
GROUP BY 1, 2
ORDER BY 2 DESC, 3 DESC;
```

If `name` is the dominant signal and `email` / `phone` are rare, you likely
have an NLP-driven false-positive problem.

---

### Phase 2 — Diagnose False Positives and False Negatives

Using the labeled corpus from Phase 0:

**False positive autopsy** — for each pair labeled `FALSE_POSITIVE`, query its
section scores and signal count:

```sql
SELECT
    dps.CONFIDENCE_SCORE,
    dps.DETERMINISTIC_SECTION_SCORE,
    dps.NLP_SECTION_SCORE,
    dps.ML_SECTION_SCORE,
    dps.EMBEDDING_SIMILARITY,
    ARRAY_AGG(dr.MATCH_TYPE) AS signals_present,
    COUNT(dr.MATCH_TYPE) AS signal_count
FROM DEDUPLICATION.COMMON_EXPERIMENT.DUPLICATE_PAIR_SCORES dps
LEFT JOIN DEDUPLICATION.COMMON_EXPERIMENT.DUPLICATE_REASONS dr
    ON dps.DUPLICATE_PAIR_ID = dr.DUPLICATE_PAIR_ID
WHERE dps.DUPLICATE_PAIR_ID IN (<false positive pair IDs>)
GROUP BY 1, 2, 3, 4, 5;
```

Look for the pattern: high `NLP_SECTION_SCORE` with low `DETERMINISTIC_SECTION_SCORE`
indicates false positives driven by name similarity alone.

**False negative autopsy** — for each known true duplicate not surfaced,
determine which stage dropped it:

| Drop stage | Diagnosis | Fix |
|---|---|---|
| Not a candidate | Blocking was too aggressive | Lower `similarity_threshold`; check `overlap_prefilter_channels` |
| Candidate but below `maybe_threshold` | Scoring weights did not recognize the evidence | Adjust section weights or signal weights |
| Scored above `maybe_threshold` but vetoed | Mitigation fired; check `MITIGATED_PAIRS` | Lower `min_embedding_similarity` or raise `min_reason_count_for_keep` |

```sql
SELECT mp.ENTITY_A_ID, mp.ENTITY_B_ID, mp.MITIGATION_REASON, mp.EVIDENCE
FROM DEDUPLICATION.COMMON_EXPERIMENT.MITIGATED_PAIRS mp
WHERE mp.DUPLICATE_PAIR_ID IN (<false negative pair IDs>);
```

---

### Phase 3 — One Variable at a Time

Never change more than one parameter between runs. After each change:

1. Re-run the pipeline on the same dataset.
2. Re-run `scripts/audit_er_run.py`.
3. Score the new output against your labeled corpus.
4. Record precision and recall changes.

Apply changes in this order — safest first:

#### 3.1 — Tighten NLP standalone threshold *(precision, low risk)*

The `standalone_fuzzy_threshold` controls how confident a name match must be
when there is *zero* deterministic evidence. Raising it cuts false positives
from name similarity alone without touching pairs that also have a shared email
or phone.

| Parameter | Org default | Service default | Try |
|---|---|---|---|
| `scoring.nlp.standalone_fuzzy_threshold` | `0.94` | `0.92` | `0.96` / `0.94` |

Verify that pairs dropping out of the review queue are all false positives in
your labeled corpus before moving on.

#### 3.2 — Reweight deterministic signals *(precision + recall, medium risk)*

Email and phone matches are nearly always true duplicates. Raise their weights;
reduce `shared_domain` weight (domain alone is less reliable). Weights are
normalized within the deterministic section, so relative changes matter.

```python
# Organization example — weights must sum to 1.0 after section normalization
"shared_email":      {"weight": 0.35},  # was 0.30
"shared_phone":      {"weight": 0.25},  # was 0.20
"shared_domain":     {"weight": 0.10},  # was 0.15
"shared_address":    {"weight": 0.20},  # unchanged
"shared_identifier": {"weight": 0.10},  # was 0.15
```

#### 3.3 — Expand the review queue *(recall, safe)*

Lower `maybe_threshold` to widen the human review queue. This surfaces more
true duplicates for steward review *without* auto-classifying anything. Monitor
queue volume — if it doubles, ensure steward capacity before proceeding.

| Parameter | Org default | Service default | Try |
|---|---|---|---|
| `scoring.maybe_threshold` | `0.68` | `0.62` | `0.62` / `0.55` |

#### 3.4 — Adjust mitigation for high-deterministic pairs *(recall, medium risk)*

The mitigation veto fires when `embedding_similarity < min_embedding_similarity`
AND `reason_count < min_reason_count_for_keep`. If true duplicates with a
shared email are being vetoed due to low embedding similarity, raise
`min_reason_count_for_keep` from `1` to `2`. This makes the veto stricter
(requires *fewer* reasons before vetoing) only for pairs with fewer signals,
protecting pairs with strong deterministic evidence.

| Parameter | Default | Try |
|---|---|---|
| `mitigation.min_embedding_similarity` | `0.65` | `0.60` |
| `mitigation.min_reason_count_for_keep` | `1` | `2` |

#### 3.5 — Lower blocking threshold *(recall, highest risk)*

Do this last. Lowering `similarity_threshold` increases candidate pairs,
catching true duplicates with more divergent text. It also multiplies compute
cost and exposes the downstream scoring to more noise.

| Parameter | Default | Try |
|---|---|---|
| `blocking.similarity_threshold` | `0.75` | `0.70` |

---

### Phase 4 — Validate and Lock

Once a configuration meaningfully improves precision and recall against the
labeled corpus:

1. **Bump `policy_version`** in `MetadataConfig` (e.g., `"hsds-er-v2"`). This
   triggers an automatic force-rescore on the next run — all existing pairs are
   re-evaluated with the new weights.

2. **Run a full pipeline pass** and re-run `scripts/audit_er_run.py` to confirm
   all 14 integrity checks pass.

3. **Extend the ground-truth corpus.** Label 50–100 new pairs from the re-run,
   particularly from score bands that shifted the most. Cut a new
   `TRAINING_DATASET` version to capture the updated labels.

4. **Repeat.** Tuning is iterative. Each round of labeling narrows the gap
   between the pipeline's predictions and ground truth.

---

## 5. Parameter Reference

All parameters live in `src/hsds_entity_resolution/config/entity_resolution_run_config.py`
and are passed via `constants_overrides` in the Dagster component's YAML config.

### Blocking

| Parameter | Default | Controls |
|---|---|---|
| `blocking.similarity_threshold` | `0.75` | Cosine similarity floor for candidate generation. Lower = more candidates = higher recall at higher compute cost. |
| `blocking.max_candidates_per_entity` | `50` | Max neighbors per anchor. |
| `blocking.overlap_prefilter_channels` | `[email, phone, website, taxonomy, location]` | Which contact channels a pair must share to pass blocking. Remove a channel to be less strict. |

### Scoring (section weights must sum to 1.0)

| Parameter | Org | Service | Controls |
|---|---|---|---|
| `scoring.deterministic_section_weight` | `0.45` | `0.40` | Weight of deterministic section in final score. |
| `scoring.nlp_section_weight` | `0.35` | `0.40` | Weight of NLP section. |
| `scoring.ml_section_weight` | `0.20` | `0.20` | Weight of ML section. |
| `scoring.duplicate_threshold` | `0.82` | `0.70` | Score floor for auto-duplicate classification. |
| `scoring.maybe_threshold` | `0.68` | `0.62` | Score floor for human review queue. |
| `scoring.min_reason_count_for_keep` | `1` | `1` | Minimum contributing signals to survive mitigation. |

### Deterministic Signals

| Signal | Org weight | Service weight |
|---|---|---|
| `shared_email` | `0.30` | `0.22` |
| `shared_phone` | `0.20` | `0.18` |
| `shared_domain` | `0.15` | `0.08` |
| `shared_address` | `0.20` | `0.12` |
| `shared_identifier` | `0.15` | `0.10` |

Each signal also has an `enabled: bool` flag.

### NLP

| Parameter | Org | Service | Controls |
|---|---|---|---|
| `scoring.nlp.fuzzy_algorithm` | `sequence_matcher` | `sequence_matcher` | Algorithm: `sequence_matcher`, `token_sort_ratio`, `jaro_winkler`. |
| `scoring.nlp.fuzzy_threshold` | `0.88` | `0.86` | Minimum raw name similarity before NLP section contributes anything. |
| `scoring.nlp.standalone_fuzzy_threshold` | `0.94` | `0.92` | Higher threshold required when `det_score = 0`. **Primary lever against NLP-driven false positives.** |
| `scoring.nlp.number_mismatch_veto_enabled` | `True` | `True` | Veto NLP contribution when names contain different digit sequences. |

### ML Gate

| Parameter | Default | Controls |
|---|---|---|
| `scoring.ml.ml_enabled` | `True` | Master switch for ML section. |
| `scoring.ml.ml_gate_threshold` | `0.55` (org) / `0.50` (svc) | Minimum pre-ML score to forward a pair to the ML endpoint. |

### Mitigation

| Parameter | Default | Controls |
|---|---|---|
| `mitigation.enabled` | `True` | Master switch. |
| `mitigation.min_embedding_similarity` | `0.65` | Below this AND low reason count → veto duplicate prediction. |
| `mitigation.require_reason_match` | `True` | When `False`, any pair below `min_embedding_similarity` is vetoed regardless of reason count. |

---

## 6. How Everything Fits Together

```
Pipeline run
    │
    ▼
DUPLICATE_PAIRS + DUPLICATE_PAIR_SCORES  (DEDUPLICATION.COMMON_EXPERIMENT)
    │
    │  Review interface updates IS_DUPLICATE / REVIEWED_BY / REVIEWED_AT
    ▼
Reviewed score rows in COMMON_EXPERIMENT
    │
    │  training_features__... freezes reviewed pairs downstream
    ▼
ENTITY_SNAPSHOT + TRAINING_PAIR + PAIR_REVIEW
    │
    │  Offline materialize reviewed typed pair-feature rows
    ▼
ORGANIZATION_PAIR_FEATURES / SERVICE_PAIR_FEATURES
    │
    │  scripts/cut_training_dataset.py
    ▼
TRAINING_DATASET + TRAINING_DATASET_MEMBER
    │
    │  Train / tune from typed features + frozen labels
    ▼
New ML model / updated config weights
    │
    │  Bump policy_version → force-rescore → new pipeline run
    ▼
Improved DUPLICATE_PAIRS results
```

### Score column rules of thumb

| Column | Use for |
|---|---|
| `BASELINE_CONFIDENCE_SCORE` | ML training features, review queue ordering |
| `RAW_DETERMINISTIC_SCORE` | Feature engineering for new ML model |
| `RAW_NLP_SCORE` | Feature engineering for new ML model |
| `EMBEDDING_SIMILARITY` | Feature engineering for new ML model |
| `PIPELINE_SIGNALS` | Audit trail for signal derivation; the typed pair-feature tables persist the final ML input columns |
| `CONFIDENCE_SCORE` | Auditing pipeline decisions; do NOT use for training |
| `ML_SECTION_SCORE` | Auditing old model contribution; do NOT use for training |

### Scripts

| Script | Purpose |
|---|---|
| `scripts/audit_er_run.py` | Post-run integrity checks and score-band diagnostics for a specific run ID |
| `scripts/materialize_training_features.py` | Offline reviewed-pair feature materializer that writes typed ML input rows |
| `scripts/cut_training_dataset.py` | Cut a named, versioned training dataset from the current reviewed labels |
| `scripts/create_training_data_schema.sql` | DDL to create the `DEDUPLICATION.TRAINING_DATA` schema |
| `scripts/truncate_common_experiment.sql` | Reset `DEDUPLICATION.COMMON_EXPERIMENT` for a fresh pipeline run |
