"""Correlation clustering stage for mitigated scored entity pairs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import polars as pl

from hsds_entity_resolution.config import EntityResolutionRunConfig
from hsds_entity_resolution.core.dataframe_utils import frame_with_schema, to_dataframe
from hsds_entity_resolution.types.contracts import ClusterPairsResult
from hsds_entity_resolution.types.frames import CLUSTER_PAIRS_SCHEMA, CLUSTERS_SCHEMA

_EPSILON = 1e-12


@dataclass
class _UnionFind:
    """Simple disjoint-set helper for positive-edge components."""

    parent: dict[str, str]

    def find(self, node: str) -> str:
        """Return canonical representative for one node."""
        parent = self.parent.setdefault(node, node)
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        """Union two nodes into one component."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_pairs(
    *,
    finalized_scored_pairs: pl.DataFrame | pl.LazyFrame,
    removed_pair_ids: pl.DataFrame | pl.LazyFrame,
    config: EntityResolutionRunConfig,
) -> ClusterPairsResult:
    """Build correlation clusters and bridge rows from finalized scored pairs.

    Clustering covers all *review-eligible* pairs — both the ``duplicate`` band
    (score ≥ duplicate_threshold) and the ``maybe`` band (score ≥ maybe_threshold).
    Edge weights are signed relative to ``maybe_threshold`` so that every
    review-eligible pair produces a non-negative edge and therefore a cluster
    of at least two nodes.

    Risk scoring inside ``_cluster_rows`` still uses ``duplicate_threshold`` as its
    sign pivot, meaning clusters that contain only ``maybe``-band links surface with
    a non-zero ``cluster_risk_score`` and can be deprioritised or reviewed more carefully.
    """
    finalized = to_dataframe(finalized_scored_pairs)
    removed = to_dataframe(removed_pair_ids)
    if finalized.is_empty():
        return ClusterPairsResult(
            clusters=_empty_clusters_frame(),
            cluster_pairs=_empty_cluster_pairs_frame(),
        )
    removed_keys = (
        set(removed.get_column("pair_key").to_list()) if "pair_key" in removed.columns else set()
    )
    # Restrict to review-eligible pairs only (duplicate + maybe bands).
    # below_maybe pairs are noise at this stage and must not distort cluster edges.
    eligible_filter = (
        pl.col("review_eligible")
        if "review_eligible" in finalized.columns
        else pl.col("predicted_duplicate")
    )
    active = finalized.filter(eligible_filter)
    if removed_keys:
        active = active.filter(~pl.col("pair_key").is_in(sorted(removed_keys)))
    if active.is_empty():
        return ClusterPairsResult(
            clusters=_empty_clusters_frame(),
            cluster_pairs=_empty_cluster_pairs_frame(),
        )
    # Use maybe_threshold as the edge-weight sign pivot so that every
    # review-eligible pair produces a non-negative edge and its two nodes
    # always end up in the same connected component.
    eligible_edge_weights = _edge_weights_from_rows(
        active=active, threshold=config.scoring.maybe_threshold
    )
    # Full edge weights include non-eligible pairs (negative edges) so that the
    # CC solver can resolve contradictory triangles — e.g. A≈B, A≈C, but B≠C.
    # Without the negative B–C edge the solver would merge all three into one
    # cluster, which contradicts the scoring signal on that pair.
    all_active = finalized
    if removed_keys:
        all_active = all_active.filter(~pl.col("pair_key").is_in(sorted(removed_keys)))
    full_edge_weights = _edge_weights_from_rows(
        active=all_active, threshold=config.scoring.maybe_threshold
    )
    components = _positive_components(
        edges=eligible_edge_weights,
        min_edge_weight=config.clustering.min_edge_weight,
    )
    cluster_members: list[list[str]] = []
    for component_nodes in components:
        assignments = _solve_component(
            nodes=component_nodes,
            edge_weights=full_edge_weights,
            max_iter=config.clustering.max_iter,
        )
        cluster_members.extend(
            sorted(members)
            for members in assignments.values()
            if len(members) >= config.clustering.min_cluster_size
        )
    if not cluster_members:
        return ClusterPairsResult(
            clusters=_empty_clusters_frame(),
            cluster_pairs=_empty_cluster_pairs_frame(),
        )
    sorted_clusters = sorted(cluster_members, key=lambda members: (len(members), members))
    cluster_ids = {tuple(members): _cluster_id_for_members(members) for members in sorted_clusters}
    entity_to_cluster: dict[str, str] = {}
    for members in sorted_clusters:
        cluster_id = cluster_ids[tuple(members)]
        for entity_id in members:
            entity_to_cluster[entity_id] = cluster_id
    cluster_rows = _cluster_rows(
        active=active,
        clusters=sorted_clusters,
        cluster_ids=cluster_ids,
        config=config,
    )
    cluster_pair_rows = _cluster_pair_rows(
        eligible=active,
        edge_weights=eligible_edge_weights,
        entity_to_cluster=entity_to_cluster,
        algorithm=config.clustering.algorithm,
    )
    return ClusterPairsResult(
        clusters=frame_with_schema(cluster_rows, CLUSTERS_SCHEMA),
        cluster_pairs=frame_with_schema(cluster_pair_rows, CLUSTER_PAIRS_SCHEMA),
    )


def _edge_weights_from_rows(
    *, active: pl.DataFrame, threshold: float
) -> dict[tuple[str, str], float]:
    """Map canonical pair key to signed correlation-clustering edge weight."""
    weights: dict[tuple[str, str], float] = {}
    for row in active.select(["entity_a_id", "entity_b_id", "final_score"]).to_dicts():
        left = str(row["entity_a_id"])
        right = str(row["entity_b_id"])
        key = _canonical_pair(left, right)
        score = float(row["final_score"])
        if score >= threshold:
            denominator = max(1.0 - threshold, _EPSILON)
            weight = (score - threshold) / denominator
        else:
            denominator = max(threshold, _EPSILON)
            weight = -(threshold - score) / denominator
        weights[key] = _clamp(weight, lower=-1.0, upper=1.0)
    return weights


def _positive_components(
    *, edges: dict[tuple[str, str], float], min_edge_weight: float
) -> list[list[str]]:
    """Build sorted connected components over non-negative/positive edge links."""
    nodes = sorted({node for edge in edges for node in edge})
    union_find = _UnionFind(parent={node: node for node in nodes})
    for (left, right), weight in edges.items():
        if weight >= min_edge_weight:
            union_find.union(left, right)
    buckets: dict[str, list[str]] = {}
    for node in nodes:
        root = union_find.find(node)
        buckets.setdefault(root, []).append(node)
    return sorted((sorted(members) for members in buckets.values()), key=lambda members: members)


def _solve_component(
    *,
    nodes: list[str],
    edge_weights: dict[tuple[str, str], float],
    max_iter: int,
) -> dict[int, set[str]]:
    """Run deterministic greedy local-search correlation clustering on one component."""
    assignments: dict[str, int] = {node: index for index, node in enumerate(nodes, start=1)}
    clusters: dict[int, set[str]] = {assignments[node]: {node} for node in nodes}
    next_cluster_id = len(nodes) + 1
    for _ in range(max_iter):
        moved = False
        for node in nodes:
            current_cluster_id = assignments[node]
            current_members = clusters[current_cluster_id]
            loss_from_current = _sum_weights(node=node, others=current_members, edges=edge_weights)
            best_delta = 0.0
            best_target_id = current_cluster_id
            for target_id in sorted(clusters):
                if target_id == current_cluster_id:
                    continue
                gain_to_target = _sum_weights(
                    node=node,
                    others=clusters[target_id],
                    edges=edge_weights,
                )
                delta = gain_to_target - loss_from_current
                if delta > best_delta + _EPSILON:
                    best_delta = delta
                    best_target_id = target_id
            if -loss_from_current > best_delta + _EPSILON:
                best_delta = -loss_from_current
                best_target_id = -1
            if best_delta <= _EPSILON:
                continue
            current_members.remove(node)
            if not current_members:
                del clusters[current_cluster_id]
            if best_target_id == -1:
                target_id = next_cluster_id
                next_cluster_id += 1
                clusters[target_id] = {node}
                assignments[node] = target_id
            else:
                clusters[best_target_id].add(node)
                assignments[node] = best_target_id
            moved = True
        if not moved:
            break
    return clusters


def _sum_weights(
    *,
    node: str,
    others: set[str],
    edges: dict[tuple[str, str], float],
) -> float:
    """Return sum of known edges from one node to a set of peers."""
    total = 0.0
    for other in others:
        if other == node:
            continue
        total += edges.get(_canonical_pair(node, other), 0.0)
    return total


def _cluster_id_for_members(members: list[str]) -> str:
    """Build stable cluster ID from sorted cluster members."""
    joined = "|".join(sorted(members))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"ccv1::{digest}"


def _cluster_rows(
    *,
    active: pl.DataFrame,
    clusters: list[list[str]],
    cluster_ids: dict[tuple[str, ...], str],
    config: EntityResolutionRunConfig,
) -> list[dict[str, object]]:
    """Build cluster summary rows with objective and risk metrics.

    ``active`` contains all review-eligible pairs (duplicate + maybe bands).
    The objective and risk scores are signed against ``duplicate_threshold`` so
    that clusters with only ``maybe``-band links receive a non-zero risk score,
    making them visually distinct from high-confidence clusters in the review UI.
    """
    rows: list[dict[str, object]] = []
    for members in clusters:
        cluster_id = cluster_ids[tuple(members)]
        member_set = set(members)
        subset = active.filter(
            pl.col("entity_a_id").is_in(member_set) & pl.col("entity_b_id").is_in(member_set)
        )
        # Risk metrics signed against duplicate_threshold: maybe-band pairs
        # contribute negative signed weights, raising cluster_risk_score.
        signed_weights = [
            _signed_weight(
                score=float(score),
                threshold=config.scoring.duplicate_threshold,
            )
            for score in subset.get_column("final_score").to_list()
        ]
        objective_score = float(sum(signed_weights))
        positive_edge_sum = float(sum(value for value in signed_weights if value > 0.0))
        negative_edge_penalty = float(sum(abs(value) for value in signed_weights if value < 0.0))
        cluster_size = len(members)
        cluster_risk_score = negative_edge_penalty / max(cluster_size, 1)
        avg_score = _safe_float(subset.get_column("final_score").mean())
        max_score = _safe_float(subset.get_column("final_score").max())
        min_score = _safe_float(subset.get_column("final_score").min())
        rows.append(
            {
                "cluster_id": cluster_id,
                "entity_type": config.metadata.entity_type,
                "cluster_size": cluster_size,
                "pair_count": subset.height,
                "avg_confidence_score": avg_score,
                "max_confidence_score": max_score,
                "min_confidence_score": min_score,
                "objective_score": objective_score,
                "positive_edge_sum": positive_edge_sum,
                "negative_edge_penalty": negative_edge_penalty,
                "cluster_risk_score": cluster_risk_score,
                "algorithm_version": config.clustering.algorithm,
            }
        )
    return rows


def _cluster_pair_rows(
    *,
    eligible: pl.DataFrame,
    edge_weights: dict[tuple[str, str], float],
    entity_to_cluster: dict[str, str],
    algorithm: str,
) -> list[dict[str, object]]:
    """Build cluster/pair bridge rows from all review-eligible pairs in a cluster."""
    rows: list[dict[str, object]] = []
    for row in eligible.sort("pair_key").to_dicts():
        pair_key = str(row["pair_key"])
        left = str(row["entity_a_id"])
        right = str(row["entity_b_id"])
        cluster_id = entity_to_cluster.get(left)
        if cluster_id is None or cluster_id != entity_to_cluster.get(right):
            continue
        assignment_score = edge_weights.get(_canonical_pair(left, right), 0.0)
        rows.append(
            {
                "cluster_id": cluster_id,
                "pair_key": pair_key,
                "is_reviewed": None,
                "review_decision": None,
                "assignment_score": float(assignment_score),
                "assignment_method": algorithm,
            }
        )
    return rows


def _signed_weight(*, score: float, threshold: float) -> float:
    """Convert confidence score to signed edge weight in [-1, 1]."""
    if score >= threshold:
        denominator = max(1.0 - threshold, _EPSILON)
        value = (score - threshold) / denominator
    else:
        denominator = max(threshold, _EPSILON)
        value = -(threshold - score) / denominator
    return _clamp(value, lower=-1.0, upper=1.0)


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    """Return canonical lexicographic pair key tuple."""
    return (left, right) if left <= right else (right, left)


def _clamp(value: float, *, lower: float, upper: float) -> float:
    """Clamp one float to inclusive lower/upper bounds."""
    return max(lower, min(upper, value))


def _safe_float(value: object) -> float | None:
    """Normalize optional numeric value to optional float."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _empty_clusters_frame() -> pl.DataFrame:
    """Return canonical empty cluster summary artifact."""
    return pl.DataFrame(schema=CLUSTERS_SCHEMA)


def _empty_cluster_pairs_frame() -> pl.DataFrame:
    """Return canonical empty cluster/pair bridge artifact."""
    return pl.DataFrame(schema=CLUSTER_PAIRS_SCHEMA)
