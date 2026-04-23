# hsds-record-matcher

Reusable [Dagster](https://dagster.io) component for incremental entity resolution on
[HSDS](https://docs.openreferral.org/en/latest/) (Human Services Data Specification) records.

**Install:**

```bash
pip install hsds-record-matcher
```

---

## What this package does

Community 211 directories and social-services platforms often ingest the same organizations
and services from multiple data partners. Over time those records diverge — different spellings,
phone formats, partial addresses — making it hard to know which rows describe the same real-world
entity.

`hsds-record-matcher` provides a single Dagster component, `EntityResolutionComponent`, that
runs an incremental seven-stage entity-resolution pipeline on your HSDS data:

| Stage | What happens |
|---|---|
| **Clean entities** | Normalize contact fields, compute content hashes, detect adds/changes/removals since the last run |
| **Generate candidates** | Block on overlap signals (email, phone, domain, taxonomy, location) to produce candidate pairs |
| **Score candidates** | Weighted combination of deterministic overlap signals, NLP fuzzy name/description matching, and optional ML scoring |
| **Apply mitigation** | Carry forward stable pairs, retire pairs for removed entities, detect pair identity continuity |
| **Cluster pairs** | Greedy correlation clustering groups high-confidence duplicate pairs into clusters |
| **Materialize review queue** | Pairs that score above the *maybe* threshold but below *duplicate* are surfaced for human review |
| **Prepare artifacts** | Package all stage outputs into a typed bundle ready for downstream persistence |

The component is **storage-agnostic**: it accepts and emits Polars DataFrames, so you wire it
into whatever persistence layer your Dagster project uses.

---

## Prerequisites

- Python 3.10–3.13
- A [Dagster](https://dagster.io) project managed with the
  [`dg` CLI](https://docs.dagster.io/guides/build/projects-and-components/components/)
- Four upstream Dagster assets that supply denormalized HSDS entity data (see [Input assets](#input-assets))

---

## Quickstart

### 1. Discover the component

```bash
dg list components --package hsds_entity_resolution
```

### 2. Add to your component YAML

```yaml
type: hsds_entity_resolution.dagster.components.EntityResolutionComponent
attributes:
  team_id: my_team
  scope_id: regional
  entity_type: organization
```

That's enough to wire the component into your Dagster asset graph. The four required upstream
assets default to the keys `organization_entities`, `service_entities`,
`previous_entity_index`, and `previous_pair_state_index` — override them with the
`*_asset_key` attributes if your project uses different names.

---

## Input assets

The component depends on four upstream assets. Each must yield a Polars `DataFrame`
(or be coercible to one).

### `organization_entities` and `service_entities`

Denormalized HSDS entity records. Required columns:

| Column | Type | Description |
|---|---|---|
| `entity_id` | `str` | Stable unique identifier for this record |
| `source_schema` | `str` | Tenant or source identifier (e.g. `il211_regional`) |
| `name` | `str` | Entity name used for NLP scoring |
| `description` | `str` | Entity description used for NLP scoring |
| `emails` | `list[str]` | Normalized email addresses |
| `phones` | `list[str]` | Normalized phone numbers |
| `websites` | `list[str]` | Normalized domain names |
| `locations` | `list[dict]` | Structured address/location objects |
| `taxonomies` | `list[dict]` | AIRS taxonomy code objects |
| `identifiers` | `list[dict]` | External identifier objects |
| `embedding_vector` | `list[float]` | Pre-computed text embedding (e.g. 384-dim) |

Optional passthrough columns (`display_name`, `display_description`, `alternate_name`,
`short_description`, `application_process`, `fees_description`, `eligibility_description`,
`resource_writer_name`, `assured_date`, `assurer_email`, `original_id`,
`organization_original_id`, `organization_name`, `organization_id`,
`services_rollup`) are forwarded unchanged to the normalized entity cache and
persistence artifacts.

### `previous_entity_index`

The `normalized_organization` or `normalized_service` output from the *previous* run of this
component. Pass an empty `pl.DataFrame()` on first run.

### `previous_pair_state_index`

The `pair_state_index` output from the *previous* run of this component. Pass an empty
`pl.DataFrame()` on first run. The component uses this to carry forward stable pair scores
and detect pair identity continuity across incremental runs.

---

## Component attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `team_id` | `str` | `"hsds"` | Team identifier written into run metadata |
| `scope_id` | `str` | `"default"` | Deployment or region identifier |
| `entity_type` | `"organization" \| "service"` | `"organization"` | Which entity type this instance processes |
| `policy_version` | `str` | `"hsds-er-v1"` | Scoring policy version tag |
| `model_version` | `str` | `"embedding-only-v1"` | ML model version tag |
| `explicit_backfill` | `bool` | `False` | Force a full re-run even when no entity changes are detected |
| `organization_entities_asset_key` | `str` | `"organization_entities"` | Asset key for upstream org entities |
| `service_entities_asset_key` | `str` | `"service_entities"` | Asset key for upstream service entities |
| `previous_entity_index_asset_key` | `str` | `"previous_entity_index"` | Asset key for previous-run entity cache |
| `previous_pair_state_index_asset_key` | `str` | `"previous_pair_state_index"` | Asset key for previous-run pair state |
| `output_asset_prefix` | `list[str]` | `[]` | Prefix prepended to all output asset keys |
| `constants_overrides` | `dict` | `{}` | Deep-merge override for scoring constants (see [Tuning](#tuning-scoring-constants)) |

---

## Output assets

The component emits fifteen assets. Using `output_asset_prefix: ["my_prefix"]` namespaces
all keys under `my_prefix/`.

| Asset key | Description |
|---|---|
| `normalized_organization` | Cleaned, normalized organization records with content hashes |
| `normalized_service` | Cleaned, normalized service records with content hashes |
| `entity_delta_summary` | Single-row summary of added / changed / removed entity counts |
| `removed_entity_ids` | Entity IDs that were present last run but are absent this run |
| `candidate_pairs` | Raw blocked candidate pairs before scoring |
| `scored_pairs` | Candidate pairs with composite scores and tier labels (`duplicate`, `maybe`, `non_duplicate`) |
| `pair_reasons` | Per-feature evidence rows for every scored pair |
| `mitigation_events` | Pairs reassigned due to entity changes between runs |
| `removed_pair_ids` | Pair IDs retired this run (entity removed or score dropped) |
| `pair_id_remap` | Maps old pair IDs to new ones when pair identity shifts |
| `clusters` | Entity groups identified as duplicates via correlation clustering |
| `cluster_pairs` | Edges within each cluster |
| `pair_state_index` | Full pair state — feed this back as `previous_pair_state_index` next run |
| `review_queue_items` | Borderline pairs above the *maybe* threshold, ready for human review |
| `run_summary` | Single-row run metrics (candidate count, duplicate count, cluster count, etc.) |
| `persistence_artifact_bundle` | Dict containing all stage outputs packaged for bulk persistence |

---

## Tuning scoring constants

The default thresholds are calibrated for HSDS data. Override any constant via
`constants_overrides` without subclassing:

```yaml
type: hsds_entity_resolution.dagster.components.EntityResolutionComponent
attributes:
  team_id: my_team
  scope_id: regional
  entity_type: organization
  constants_overrides:
    scoring:
      duplicate_threshold: 0.85
      maybe_threshold: 0.70
      nlp:
        fuzzy_threshold: 0.90
```

Key thresholds:

| Constant | Default (org) | Default (service) | Description |
|---|---|---|---|
| `scoring.duplicate_threshold` | `0.82` | `0.70` | Minimum score to auto-cluster as duplicate |
| `scoring.maybe_threshold` | `0.68` | `0.62` | Minimum score to send to review queue |
| `scoring.deterministic_section_weight` | `0.45` | `0.40` | Weight of overlap-signal section |
| `scoring.nlp_section_weight` | `0.35` | `0.40` | Weight of NLP fuzzy-match section |
| `scoring.ml_section_weight` | `0.20` | `0.20` | Weight of ML section (disabled by default) |
| `scoring.nlp.fuzzy_threshold` | `0.88` | `0.86` | Minimum name similarity to count as NLP match |
| `blocking.similarity_threshold` | `0.75` | `0.75` | Minimum embedding cosine similarity for blocking |
| `blocking.max_candidates_per_entity` | `50` | `125` | Maximum candidate pairs per entity |

---

## Multiple instances

Deploy one component instance per entity type and scope:

```yaml
# organizations
type: hsds_entity_resolution.dagster.components.EntityResolutionComponent
attributes:
  team_id: my_team
  scope_id: regional
  entity_type: organization
  output_asset_prefix: ["org_er"]

---

# services
type: hsds_entity_resolution.dagster.components.EntityResolutionComponent
attributes:
  team_id: my_team
  scope_id: regional
  entity_type: service
  output_asset_prefix: ["svc_er"]
```

---

## Links

- **Source:** [github.com/211-Connect/hsds-entity-resolution](https://github.com/211-Connect/hsds-entity-resolution)
- **Issues:** [github.com/211-Connect/hsds-entity-resolution/issues](https://github.com/211-Connect/hsds-entity-resolution/issues)
- **PyPI:** [pypi.org/project/hsds-record-matcher](https://pypi.org/project/hsds-record-matcher/)
