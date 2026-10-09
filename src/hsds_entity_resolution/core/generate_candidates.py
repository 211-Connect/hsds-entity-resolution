"""Candidate generation stage with canonical pair identity guarantees.

Admission follows one generic rule (see :mod:`hsds_entity_resolution.core.admission`):

* **Embedding floor** — for each changed anchor, the most similar records are walked in
  descending cosine order down to ``blocking.similarity_threshold`` and admitted, up to
  ``blocking.max_candidates_per_entity`` per anchor.
* **Informative Keys** — every record that shares a value of an informative key field
  with an anchor is admitted regardless of similarity (inverted indexes, bounded by the
  optional ``chunking`` fanout caps).

A caller-supplied Structural Exclusion can veto any pair from either path; excluded
pairs are returned with their reason. Admission never decides a score: scoring picks
its pair rule from the entity type and the two source schemas alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.admission import (
    EMBEDDING_FLOOR_REASON_CODE,
    EMBEDDING_FLOOR_RULE_ID,
    INFORMATIVE_KEY_RULE_PREFIX,
    KEY_FIELDS,
    InformativeKeyTable,
    KeyField,
    StructuralExclusion,
    informative_key_reason_code,
    informative_key_rule_id,
    is_informative,
    key_values,
)
from hsds_entity_resolution.core.dataframe_utils import frame_with_schema
from hsds_entity_resolution.observability import IncrementalProgressLogger
from hsds_entity_resolution.types.contracts import GenerateCandidatesResult
from hsds_entity_resolution.types.frames import CANDIDATE_PAIR_SCHEMA, EXCLUDED_PAIR_SCHEMA

KeyIndexes = dict[KeyField, dict[str, set[str]]]


@dataclass(frozen=True)
class AdmissionInputs:
    """The caller's per-run admission inputs, threaded through candidate generation."""

    informative_keys: InformativeKeyTable | None
    structural_exclusion: StructuralExclusion | None


@dataclass(frozen=True)
class BlockingOverview:
    """Summary diagnostics for one entity type's blocking pass."""

    entity_type: str
    changed_anchors: int
    active_anchors: int
    anchors_with_above_threshold: int
    anchors_with_retained_candidates: int
    anchors_at_candidate_cap: int
    above_threshold_examined: int
    excluded: int
    key_admitted: int
    pairs_kept: int
    truncated_above_threshold: int
    last_retained_similarity_sum: float
    last_retained_similarity_count: int


@dataclass
class BlockingState:
    """Mutable candidate and diagnostics state for one entity type's pass."""

    pairs_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    excluded_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    above_threshold: int = 0
    key_hits: dict[str, int] = field(default_factory=lambda: {key: 0 for key in KEY_FIELDS})
    anchors_with_above_threshold: int = 0
    anchors_with_retained: int = 0
    anchors_at_cap: int = 0
    active_anchors: int = 0
    truncated_above_threshold: int = 0
    last_retained_similarity_sum: float = 0.0
    last_retained_similarity_count: int = 0


@dataclass(frozen=True)
class AnchorProcessingResult:
    """Outcome of expanding one changed anchor along the embedding floor."""

    saw_above_threshold: bool
    selected: int
    last_retained_similarity: float | None
    truncated_above_threshold: int


@dataclass(frozen=True)
class EntityMatrix:
    """Entity rows plus their L2-normalised embedding matrix."""

    rows: list[dict[str, Any]]
    normalized: np.ndarray
    id_to_idx: dict[str, int]


def generate_candidates(
    *,
    denormalized_organization: pl.DataFrame,
    denormalized_service: pl.DataFrame,
    changed_entities: pl.DataFrame,
    config: EntityResolutionRunConfig,
    explicit_backfill: bool,
    force_rescore: bool = False,
    progress_logger: IncrementalProgressLogger | None = None,
    anchor_ids_subset: frozenset[str] | None = None,
    informative_keys: InformativeKeyTable | None = None,
    structural_exclusion: StructuralExclusion | None = None,
) -> GenerateCandidatesResult:
    """Generate candidate pairs by embedding floor and Informative Keys.

    When ``anchor_ids_subset`` is provided (generate-sharding mode), only entities whose
    ID is in the subset are used as anchors. The full entity matrix is still available
    for similarity look-ups; the union of results from all subsets, after deduplication
    by ``pair_key``, equals the result of a monolithic call.

    Args:
        denormalized_organization: Cleaned organization rows with embeddings.
        denormalized_service: Cleaned service rows with embeddings.
        changed_entities: Entity delta rows; added/changed ids become anchors.
        config: Run configuration.
        explicit_backfill: Treat every entity as an anchor.
        force_rescore: Treat every entity as an anchor.
        progress_logger: Optional progress logger.
        anchor_ids_subset: Restrict anchors to these ids (sharding).
        informative_keys: Per-schema key informativeness; ``None`` makes every key
            field informative everywhere.
        structural_exclusion: Optional veto ``(entity_a, entity_b) -> reason | None``.

    Returns:
        Candidate pairs, a one-row summary, and the excluded pairs with reasons.
    """
    full_scope_rescore = explicit_backfill or force_rescore
    delta_entities = changed_entities.filter(pl.col("delta_class").is_in(["added", "changed"]))
    if delta_entities.is_empty() and not full_scope_rescore:
        return _empty_result()
    admission = AdmissionInputs(
        informative_keys=informative_keys, structural_exclusion=structural_exclusion
    )
    outputs = [
        _generate_for_entity_type(
            frame=frame,
            changed_entities=delta_entities,
            entity_type=entity_type,
            config=config,
            full_scope_rescore=full_scope_rescore,
            admission=admission,
            progress_logger=progress_logger,
            anchor_ids_subset=anchor_ids_subset,
        )
        for entity_type, frame in (
            ("organization", denormalized_organization),
            ("service", denormalized_service),
        )
    ]
    candidate_pairs = pl.concat([pairs for pairs, _, _ in outputs], how="diagonal_relaxed")
    excluded_pairs = pl.concat([excluded for _, excluded, _ in outputs], how="diagonal_relaxed")
    _log_generate_candidates_overview(
        overviews=[overview for _, _, overview in outputs],
        candidate_pair_count=candidate_pairs.height,
        config=config,
    )
    summary = pl.DataFrame(
        {
            "candidate_count": [candidate_pairs.height],
            "raw_candidate_count": [candidate_pairs.height],
        }
    )
    return GenerateCandidatesResult(
        candidate_pairs=candidate_pairs,
        candidate_summary=summary,
        excluded_pairs=excluded_pairs,
    )


def _generate_for_entity_type(
    *,
    frame: pl.DataFrame,
    changed_entities: pl.DataFrame,
    entity_type: str,
    config: EntityResolutionRunConfig,
    full_scope_rescore: bool,
    admission: AdmissionInputs,
    progress_logger: IncrementalProgressLogger | None = None,
    anchor_ids_subset: frozenset[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, BlockingOverview]:
    """Generate candidates for one entity type."""
    type_frame = frame.filter(pl.col("entity_type") == entity_type)
    if type_frame.is_empty():
        return _empty_candidate_frame(), _empty_excluded_frame(), _empty_overview(entity_type)
    _assert_unique_entity_rows(type_frame=type_frame, entity_type=entity_type)
    changed_ids = set(
        changed_entities.filter(pl.col("entity_type") == entity_type)
        .get_column("entity_id")
        .to_list()
    )
    if full_scope_rescore:
        changed_ids = set(type_frame.get_column("entity_id").to_list())
    if anchor_ids_subset is not None:
        changed_ids &= anchor_ids_subset
    if not changed_ids:
        return _empty_candidate_frame(), _empty_excluded_frame(), _empty_overview(entity_type)
    matrix = _build_entity_matrix(type_frame=type_frame, entity_type=entity_type)
    state = BlockingState()
    sorted_changed_ids = sorted(changed_ids)
    _collect_embedding_candidates(
        matrix=matrix,
        entity_type=entity_type,
        sorted_changed_ids=sorted_changed_ids,
        config=config,
        admission=admission,
        state=state,
        progress_logger=progress_logger,
    )
    key_admitted = _collect_informative_key_candidates(
        matrix=matrix,
        entity_type=entity_type,
        sorted_changed_ids=sorted_changed_ids,
        config=config,
        admission=admission,
        state=state,
    )
    overview = _overview_from_state(
        state=state,
        entity_type=entity_type,
        changed_anchors=len(sorted_changed_ids),
        key_admitted=key_admitted,
    )
    _log_blocking_summary(overview=overview, state=state, threshold=_floor(config))
    return _candidate_frame(state), _excluded_frame(state), overview


def _assert_unique_entity_rows(*, type_frame: pl.DataFrame, entity_type: str) -> None:
    """Fail loudly when the caller passes the same entity twice."""
    duplicate_rows = (
        type_frame.group_by(["source_schema", "entity_id"])
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") > 1)
        .sort("row_count", descending=True)
    )
    if duplicate_rows.height == 0:
        return
    examples = duplicate_rows.head(5).select(["source_schema", "entity_id", "row_count"])
    example_text = "; ".join(
        f"{row['source_schema']}/{row['entity_id']} ({row['row_count']} rows)"
        for row in examples.iter_rows(named=True)
    )
    extra = duplicate_rows.height - min(duplicate_rows.height, 5)
    suffix = f" (+{extra} more)" if extra > 0 else ""
    extra_row_count = int(duplicate_rows.select((pl.col("row_count") - 1).sum()).item())
    raise ValueError(
        f"Duplicate {entity_type} entity rows before candidate generation: "
        f"{duplicate_rows.height} entity_ids have extra rows "
        f"({extra_row_count} extra rows total). "
        f"Examples: {example_text}{suffix}. "
        "Fix the duplicate rows in the caller's entity input; "
        "generate_candidates does not dedupe silently."
    )


def _build_entity_matrix(*, type_frame: pl.DataFrame, entity_type: str) -> EntityMatrix:
    """Normalise the embedding matrix and drop vectors from the Python rows.

    Extracting ``embedding_vector`` straight from the Polars column avoids materialising
    N × D Python floats inside ``to_dicts()`` rows (about 450 MB for 54 K × 1024).
    """
    raw_embeddings = type_frame.get_column("embedding_vector").to_list()
    embedding_dim = len(raw_embeddings[0]) if raw_embeddings and raw_embeddings[0] else 0
    fingerprints = {
        tuple(round(float(value), 3) for value in embedding[:8])
        for embedding in raw_embeddings
        if len(embedding) >= 8
    }
    fingerprint_total = sum(1 for embedding in raw_embeddings if len(embedding) >= 8)
    matrix = np.array(raw_embeddings, dtype=np.float32)
    del raw_embeddings
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, a_min=1e-8, a_max=None)
    del matrix
    schemas = sorted(type_frame.get_column("source_schema").cast(pl.String).unique().to_list())
    rows = type_frame.drop("embedding_vector").to_dicts()
    _log_entity_sample(
        entity_rows=rows,
        entity_type=entity_type,
        embedding_dim=embedding_dim,
        embedding_unique_count=len(fingerprints),
        embedding_total_count=fingerprint_total,
        schemas=schemas,
    )
    return EntityMatrix(
        rows=rows,
        normalized=normalized,
        id_to_idx={row["entity_id"]: idx for idx, row in enumerate(rows)},
    )


def _floor(config: EntityResolutionRunConfig) -> float:
    """Return the single embedding floor."""
    return config.blocking.similarity_threshold


def _anchor_chunks(*, sorted_ids: list[str], chunk_size: int | None) -> list[list[str]]:
    """Split anchors into memory-bounded chunks (one chunk when unset)."""
    if not sorted_ids:
        return []
    if chunk_size is None:
        return [sorted_ids]
    return [
        sorted_ids[start : start + chunk_size] for start in range(0, len(sorted_ids), chunk_size)
    ]


# ---------------------------------------------------------------------------
# Path A: embedding floor
# ---------------------------------------------------------------------------


def _collect_embedding_candidates(
    *,
    matrix: EntityMatrix,
    entity_type: str,
    sorted_changed_ids: list[str],
    config: EntityResolutionRunConfig,
    admission: AdmissionInputs,
    state: BlockingState,
    progress_logger: IncrementalProgressLogger | None,
) -> None:
    """Admit each anchor's most similar records down to the embedding floor."""
    stage_name = f"generate_candidates.{entity_type}.anchors"
    total = len(sorted_changed_ids)
    if progress_logger is not None:
        progress_logger.stage_started(stage=stage_name, total=total)
    processed = 0
    chunks = _anchor_chunks(
        sorted_ids=sorted_changed_ids, chunk_size=config.chunking.generate_anchor_chunk_size
    )
    for chunk in chunks:
        if len(chunks) > 1:
            logging.getLogger(__name__).info(
                "ℹ️ generate_anchor_chunk entity_type=%s anchors=%d pairs_so_far=%d rss_gb=%.2f",
                entity_type,
                len(chunk),
                len(state.pairs_by_key),
                _rss_gb(),
            )
        for entity_id in chunk:
            processed += 1
            if entity_id in matrix.id_to_idx:
                _expand_anchor(
                    matrix=matrix,
                    anchor_idx=matrix.id_to_idx[entity_id],
                    config=config,
                    admission=admission,
                    state=state,
                )
            if progress_logger is not None:
                progress_logger.stage_advanced(stage=stage_name, processed=processed, total=total)
    if progress_logger is not None:
        progress_logger.stage_completed(
            stage=stage_name, detail={"candidate_pairs": len(state.pairs_by_key)}
        )


def _expand_anchor(
    *,
    matrix: EntityMatrix,
    anchor_idx: int,
    config: EntityResolutionRunConfig,
    admission: AdmissionInputs,
    state: BlockingState,
) -> None:
    """Expand one anchor along the embedding floor and fold the result into ``state``."""
    state.active_anchors += 1
    similarities = matrix.normalized @ matrix.normalized[anchor_idx]
    top_indices = np.argsort(similarities)[::-1].tolist()
    result = _collect_anchor_candidates(
        matrix=matrix,
        anchor_idx=anchor_idx,
        similarities=similarities,
        top_indices=top_indices,
        threshold=_floor(config),
        max_per_entity=config.blocking.max_candidates_per_entity,
        admission=admission,
        state=state,
    )
    if result.saw_above_threshold:
        state.anchors_with_above_threshold += 1
    if result.selected > 0:
        state.anchors_with_retained += 1
        if result.last_retained_similarity is not None:
            state.last_retained_similarity_sum += result.last_retained_similarity
            state.last_retained_similarity_count += 1
    if result.selected >= config.blocking.max_candidates_per_entity:
        state.anchors_at_cap += 1
    state.truncated_above_threshold += result.truncated_above_threshold


def _collect_anchor_candidates(
    *,
    matrix: EntityMatrix,
    anchor_idx: int,
    similarities: np.ndarray,
    top_indices: list[int],
    threshold: float,
    max_per_entity: int,
    admission: AdmissionInputs,
    state: BlockingState,
) -> AnchorProcessingResult:
    """Walk one anchor's candidates in descending similarity down to the floor."""
    anchor = matrix.rows[anchor_idx]
    selected = 0
    saw_above_threshold = False
    last_retained_similarity: float | None = None
    for position, candidate_idx in enumerate(top_indices):
        if candidate_idx == anchor_idx:
            continue
        similarity = float(similarities[candidate_idx])
        if similarity < threshold:
            break
        saw_above_threshold = True
        state.above_threshold += 1
        record = _to_candidate_record(
            anchor=anchor,
            candidate=matrix.rows[candidate_idx],
            similarity=similarity,
            reason_codes=[EMBEDDING_FLOOR_REASON_CODE],
            blocking_rule_id=EMBEDDING_FLOOR_RULE_ID,
        )
        if not _admit(
            record=record,
            entities=(anchor, matrix.rows[candidate_idx]),
            admission=admission,
            state=state,
        ):
            continue
        selected += 1
        last_retained_similarity = similarity
        if selected >= max_per_entity:
            return AnchorProcessingResult(
                saw_above_threshold=True,
                selected=selected,
                last_retained_similarity=last_retained_similarity,
                truncated_above_threshold=_count_truncated_above_threshold(
                    similarities=similarities,
                    top_indices=top_indices,
                    anchor_idx=anchor_idx,
                    start_position=position + 1,
                    threshold=threshold,
                ),
            )
    return AnchorProcessingResult(
        saw_above_threshold=saw_above_threshold,
        selected=selected,
        last_retained_similarity=last_retained_similarity,
        truncated_above_threshold=0,
    )


def _count_truncated_above_threshold(
    *,
    similarities: np.ndarray,
    top_indices: list[int],
    anchor_idx: int,
    start_position: int,
    threshold: float,
) -> int:
    """Count above-floor candidates skipped after hitting the per-anchor cap."""
    truncated = 0
    for candidate_idx in top_indices[start_position:]:
        if candidate_idx == anchor_idx:
            continue
        if float(similarities[candidate_idx]) < threshold:
            break
        truncated += 1
    return truncated


# ---------------------------------------------------------------------------
# Path B: Informative Keys
# ---------------------------------------------------------------------------


def _collect_informative_key_candidates(
    *,
    matrix: EntityMatrix,
    entity_type: str,
    sorted_changed_ids: list[str],
    config: EntityResolutionRunConfig,
    admission: AdmissionInputs,
    state: BlockingState,
) -> int:
    """Admit every record sharing an informative key value with an anchor."""
    indexes = _build_key_indexes(rows=matrix.rows, informative_keys=admission.informative_keys)
    max_pairs_per_anchor = config.chunking.max_contact_overlap_pairs_per_anchor
    max_fanout = config.chunking.max_contact_index_fanout
    admitted = 0
    anchors_hit_pair_cap = 0
    index_keys_truncated = 0
    for anchor_id in sorted_changed_ids:
        anchor_idx = matrix.id_to_idx.get(anchor_id)
        if anchor_idx is None:
            continue
        anchor = matrix.rows[anchor_idx]
        shared, truncated = _shared_keys_by_candidate(
            anchor=anchor, indexes=indexes, max_fanout=max_fanout
        )
        index_keys_truncated += truncated
        count, hit_cap = _admit_key_candidates(
            matrix=matrix,
            anchor_idx=anchor_idx,
            shared=shared,
            max_pairs_per_anchor=max_pairs_per_anchor,
            admission=admission,
            state=state,
        )
        admitted += count
        anchors_hit_pair_cap += int(hit_cap)
    _log_key_caps(
        entity_type=entity_type,
        anchors_hit_pair_cap=anchors_hit_pair_cap,
        index_keys_truncated=index_keys_truncated,
        max_pairs_per_anchor=max_pairs_per_anchor,
        max_fanout=max_fanout,
    )
    return admitted


def _build_key_indexes(
    *, rows: list[dict[str, Any]], informative_keys: InformativeKeyTable | None
) -> KeyIndexes:
    """Build one inverted index per key field over records where the field is informative.

    A record contributes a field's values only when the field is informative in its
    schema, so a lookup can only match two records for which the field is a key.
    """
    indexes: KeyIndexes = {key: {} for key in KEY_FIELDS}
    for row in rows:
        schema = str(row.get("source_schema") or "")
        entity_id = str(row["entity_id"])
        for key in KEY_FIELDS:
            if not is_informative(informative_keys, schema=schema, field=key):
                continue
            for value in key_values(row, key):
                indexes[key].setdefault(value, set()).add(entity_id)
    return indexes


def _shared_keys_by_candidate(
    *,
    anchor: dict[str, Any],
    indexes: KeyIndexes,
    max_fanout: int | None,
) -> tuple[dict[str, set[KeyField]], int]:
    """Return candidate id → key fields shared with the anchor, and truncated index keys.

    With ``max_fanout`` set, a value shared by more records than the cap is truncated to a
    deterministic (sorted) subset.
    """
    shared: dict[str, set[KeyField]] = {}
    truncated = 0
    anchor_id = str(anchor["entity_id"])
    for key, index in indexes.items():
        for value in key_values(anchor, key):
            matches = index.get(value, set())
            if anchor_id not in matches:
                continue
            if max_fanout is not None and len(matches) > max_fanout:
                truncated += 1
                matches = set(sorted(matches)[:max_fanout])
            for candidate_id in matches:
                if candidate_id != anchor_id:
                    shared.setdefault(candidate_id, set()).add(key)
    return shared, truncated


def _admit_key_candidates(
    *,
    matrix: EntityMatrix,
    anchor_idx: int,
    shared: dict[str, set[KeyField]],
    max_pairs_per_anchor: int | None,
    admission: AdmissionInputs,
    state: BlockingState,
) -> tuple[int, bool]:
    """Admit one anchor's key-sharing candidates; return (admitted, hit the per-anchor cap)."""
    anchor = matrix.rows[anchor_idx]
    admitted = 0
    for considered, (candidate_id, keys) in enumerate(sorted(shared.items())):
        if max_pairs_per_anchor is not None and considered >= max_pairs_per_anchor:
            return admitted, True
        candidate_idx = matrix.id_to_idx[candidate_id]
        candidate = matrix.rows[candidate_idx]
        for key in keys:
            state.key_hits[key] += 1
        record = _to_candidate_record(
            anchor=anchor,
            candidate=candidate,
            similarity=float(matrix.normalized[candidate_idx] @ matrix.normalized[anchor_idx]),
            reason_codes=[informative_key_reason_code(key) for key in keys],
            blocking_rule_id=informative_key_rule_id(keys),
        )
        if _admit(record=record, entities=(anchor, candidate), admission=admission, state=state):
            admitted += 1
    return admitted, False


# ---------------------------------------------------------------------------
# Shared admission step
# ---------------------------------------------------------------------------


def _admit(
    *,
    record: dict[str, Any],
    entities: tuple[dict[str, Any], dict[str, Any]],
    admission: AdmissionInputs,
    state: BlockingState,
) -> bool:
    """Apply the Structural Exclusion, then merge the record into the candidates.

    Returns:
        ``True`` when the pair is (or already was) a candidate; ``False`` when excluded.
    """
    pair_key = record["pair_key"]
    if pair_key in state.excluded_by_key:
        return False
    if pair_key not in state.pairs_by_key and admission.structural_exclusion is not None:
        reason = admission.structural_exclusion(*_canonical_entities(record, entities))
        if reason:
            state.excluded_by_key[pair_key] = _to_excluded_record(record=record, reason=reason)
            return False
    _merge_candidate_record(pairs_by_key=state.pairs_by_key, record=record)
    return True


def _canonical_entities(
    record: dict[str, Any], entities: tuple[dict[str, Any], dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Order the two entities as (entity_a, entity_b) of the canonical pair."""
    first, second = entities
    if str(first["entity_id"]) == record["entity_a_id"]:
        return first, second
    return second, first


def _merge_candidate_record(
    *, pairs_by_key: dict[str, dict[str, Any]], record: dict[str, Any]
) -> None:
    """Insert a record, or union reason codes into the existing one.

    An Informative Key admission names the pair's ``blocking_rule_id`` over the embedding
    floor, so the rule id always says which key admitted a pair when one did.
    """
    existing = pairs_by_key.get(record["pair_key"])
    if existing is None:
        pairs_by_key[record["pair_key"]] = record
        return
    existing["candidate_reason_codes"] = sorted(
        set(existing["candidate_reason_codes"]).union(record["candidate_reason_codes"])
    )
    if str(record["blocking_rule_id"]).startswith(INFORMATIVE_KEY_RULE_PREFIX):
        existing["blocking_rule_id"] = record["blocking_rule_id"]


def _to_candidate_record(
    *,
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    similarity: float,
    reason_codes: list[str],
    blocking_rule_id: str,
) -> dict[str, Any]:
    """Build one canonical candidate record with provenance fields."""
    entity_a_id, entity_b_id = _canonical_pair(anchor["entity_id"], candidate["entity_id"])
    first, second = (
        (anchor, candidate) if anchor["entity_id"] == entity_a_id else (candidate, anchor)
    )
    return {
        "pair_key": f"{entity_a_id}__{entity_b_id}",
        "entity_a_id": entity_a_id,
        "entity_b_id": entity_b_id,
        "entity_type": anchor["entity_type"],
        "embedding_similarity": similarity,
        "candidate_reason_codes": sorted(set(reason_codes)),
        "source_schema_a": first["source_schema"],
        "source_schema_b": second["source_schema"],
        "blocking_rule_id": blocking_rule_id,
    }


def _to_excluded_record(*, record: dict[str, Any], reason: str) -> dict[str, Any]:
    """Project a candidate record onto the excluded-pairs shape."""
    excluded = {key: record[key] for key in EXCLUDED_PAIR_SCHEMA if key in record}
    excluded["exclusion_reason"] = reason
    return excluded


def _canonical_pair(entity_a_id: str, entity_b_id: str) -> tuple[str, str]:
    """Return lexicographically ordered pair IDs."""
    if entity_a_id < entity_b_id:
        return entity_a_id, entity_b_id
    return entity_b_id, entity_a_id


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def _candidate_frame(state: BlockingState) -> pl.DataFrame:
    """Return the candidate pairs collected in ``state``."""
    if not state.pairs_by_key:
        return _empty_candidate_frame()
    return frame_with_schema(list(state.pairs_by_key.values()), CANDIDATE_PAIR_SCHEMA).sort(
        ["entity_a_id", "entity_b_id"]
    )


def _excluded_frame(state: BlockingState) -> pl.DataFrame:
    """Return the excluded pairs collected in ``state``."""
    if not state.excluded_by_key:
        return _empty_excluded_frame()
    return frame_with_schema(list(state.excluded_by_key.values()), EXCLUDED_PAIR_SCHEMA).sort(
        ["entity_a_id", "entity_b_id"]
    )


def _empty_result() -> GenerateCandidatesResult:
    """Return empty candidate stage outputs."""
    return GenerateCandidatesResult(
        candidate_pairs=_empty_candidate_frame(),
        candidate_summary=pl.DataFrame({"candidate_count": [0], "raw_candidate_count": [0]}),
        excluded_pairs=_empty_excluded_frame(),
    )


def _empty_candidate_frame() -> pl.DataFrame:
    """Return canonical empty candidate-pairs frame."""
    return pl.DataFrame(schema=CANDIDATE_PAIR_SCHEMA)


def _empty_excluded_frame() -> pl.DataFrame:
    """Return canonical empty excluded-pairs frame."""
    return pl.DataFrame(schema=EXCLUDED_PAIR_SCHEMA)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _overview_from_state(
    *, state: BlockingState, entity_type: str, changed_anchors: int, key_admitted: int
) -> BlockingOverview:
    """Freeze ``state`` into an overview."""
    return BlockingOverview(
        entity_type=entity_type,
        changed_anchors=changed_anchors,
        active_anchors=state.active_anchors,
        anchors_with_above_threshold=state.anchors_with_above_threshold,
        anchors_with_retained_candidates=state.anchors_with_retained,
        anchors_at_candidate_cap=state.anchors_at_cap,
        above_threshold_examined=state.above_threshold,
        excluded=len(state.excluded_by_key),
        key_admitted=key_admitted,
        pairs_kept=len(state.pairs_by_key),
        truncated_above_threshold=state.truncated_above_threshold,
        last_retained_similarity_sum=state.last_retained_similarity_sum,
        last_retained_similarity_count=state.last_retained_similarity_count,
    )


def _empty_overview(entity_type: str) -> BlockingOverview:
    """Return zeroed blocking diagnostics for one entity type."""
    return _overview_from_state(
        state=BlockingState(), entity_type=entity_type, changed_anchors=0, key_admitted=0
    )


def _rss_gb() -> float:
    """Return current process RSS in GB, or 0.0 when unavailable."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**3)
    except Exception:  # noqa: BLE001
        return 0.0


def _log_key_caps(
    *,
    entity_type: str,
    anchors_hit_pair_cap: int,
    index_keys_truncated: int,
    max_pairs_per_anchor: int | None,
    max_fanout: int | None,
) -> None:
    """Warn when the optional Informative Key caps truncated admission."""
    if not anchors_hit_pair_cap and not index_keys_truncated:
        return
    logging.getLogger(__name__).warning(
        "⚠️ informative_key_caps_hit entity_type=%s anchors_pair_cap=%d "
        "index_keys_truncated=%d max_pairs_per_anchor=%s max_index_fanout=%s",
        entity_type,
        anchors_hit_pair_cap,
        index_keys_truncated,
        max_pairs_per_anchor,
        max_fanout,
    )


def _log_entity_sample(
    *,
    entity_rows: list[dict[str, Any]],
    entity_type: str,
    embedding_dim: int,
    embedding_unique_count: int,
    embedding_total_count: int,
    schemas: list[str],
) -> None:
    """Emit a DEBUG snapshot of the first 3 entity rows to verify denormalized fields."""
    lines = "\n".join(
        f"  [{i}] id={str(r.get('entity_id') or '?')[:20]}"
        f" schema={r.get('source_schema') or '?'}"
        f" tax={len(r.get('taxonomies') or [])}"
        f" loc={len(r.get('locations') or [])}"
        f" phones={len(r.get('phones') or [])}"
        f" websites={len(r.get('websites') or [])}"
        f" emails={len(r.get('emails') or [])}"
        f" vec_len={embedding_dim}"
        for i, r in enumerate(entity_rows[:3])
    )
    shared_pct = (
        round(100.0 * (1 - embedding_unique_count / embedding_total_count), 1)
        if embedding_total_count
        else 0.0
    )
    logging.getLogger(__name__).debug(
        "🗂 entity_sample entity_type=%s total=%d schemas=%s"
        " unique_embeddings=%d/%d (%.1f%% share a vector)\n%s",
        entity_type,
        len(entity_rows),
        schemas,
        embedding_unique_count,
        embedding_total_count,
        shared_pct,
        lines,
    )


def _log_blocking_summary(
    *, overview: BlockingOverview, state: BlockingState, threshold: float
) -> None:
    """Emit a single DEBUG summary of one entity type's blocking pass."""
    rule_counts: dict[str, int] = {}
    for record in state.pairs_by_key.values():
        rule_counts[record["blocking_rule_id"]] = rule_counts.get(record["blocking_rule_id"], 0) + 1
    logging.getLogger(__name__).debug(
        "🧮 blocking_summary entity_type=%s floor=%s above_floor=%d excluded=%d"
        " key_admitted=%d pairs_kept=%d key_hits=%s rule_counts=%s",
        overview.entity_type,
        threshold,
        overview.above_threshold_examined,
        overview.excluded,
        overview.key_admitted,
        overview.pairs_kept,
        state.key_hits,
        dict(sorted(rule_counts.items())),
    )


def _log_generate_candidates_overview(
    *,
    overviews: list[BlockingOverview],
    candidate_pair_count: int,
    config: EntityResolutionRunConfig,
) -> None:
    """Emit one INFO-level overview for coarse admission tuning."""
    active = sum(overview.active_anchors for overview in overviews)
    above = sum(overview.anchors_with_above_threshold for overview in overviews)
    at_cap = sum(overview.anchors_at_candidate_cap for overview in overviews)
    truncated = sum(overview.truncated_above_threshold for overview in overviews)
    last_sum = sum(overview.last_retained_similarity_sum for overview in overviews)
    last_count = sum(overview.last_retained_similarity_count for overview in overviews)
    logging.getLogger(__name__).info(
        "ℹ️ generate_candidates_overview floor=%.3f max_candidates_per_entity=%d"
        " candidate_pairs=%d active_anchors=%d anchors_with_no_above_floor=%d (%.1f%%)"
        " anchors_at_cap=%d (%.1f%%) above_floor_truncated=%d"
        " avg_last_retained_similarity=%.4f (n=%d) key_admitted=%d excluded=%d"
        " chunking=[generate_anchor_chunk_size=%s max_contact_overlap_pairs_per_anchor=%s"
        " max_contact_index_fanout=%s] per_type=[%s]",
        _floor(config),
        config.blocking.max_candidates_per_entity,
        candidate_pair_count,
        active,
        max(0, active - above),
        _percent(max(0, active - above), active),
        at_cap,
        _percent(at_cap, active),
        truncated,
        last_sum / last_count if last_count else 0.0,
        last_count,
        sum(overview.key_admitted for overview in overviews),
        sum(overview.excluded for overview in overviews),
        config.chunking.generate_anchor_chunk_size,
        config.chunking.max_contact_overlap_pairs_per_anchor,
        config.chunking.max_contact_index_fanout,
        ", ".join(
            f"{overview.entity_type}: active={overview.active_anchors}"
            f" kept={overview.pairs_kept} at_cap={overview.anchors_at_candidate_cap}"
            f" excluded={overview.excluded}"
            for overview in overviews
        ),
    )


def _percent(numerator: int, denominator: int) -> float:
    """Return a rounded percentage without division-by-zero."""
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 1)
