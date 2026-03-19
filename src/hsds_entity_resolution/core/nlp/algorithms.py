"""Fuzzy algorithm implementations for NLP scoring."""

from __future__ import annotations

from difflib import SequenceMatcher

from hsds_entity_resolution.core.dataframe_utils import clean_text_scalar


def resolve_fuzzy_similarity(
    *,
    left_name: str,
    right_name: str,
    algorithm: str,
    strict_validation_mode: bool,
) -> float:
    """Resolve configured fuzzy algorithm and compute similarity [0, 1]."""
    normalized_algorithm = clean_text_scalar(algorithm)
    if normalized_algorithm == "sequence_matcher":
        return SequenceMatcher(a=left_name, b=right_name).ratio()
    if normalized_algorithm == "token_sort_ratio":
        return token_sort_ratio_similarity(left_name=left_name, right_name=right_name)
    if normalized_algorithm == "jaro_winkler":
        return jaro_winkler_similarity(left_name=left_name, right_name=right_name)
    if strict_validation_mode:
        message = f"Unsupported fuzzy_algorithm '{algorithm}'"
        raise ValueError(message)
    return SequenceMatcher(a=left_name, b=right_name).ratio()


def token_sort_ratio_similarity(*, left_name: str, right_name: str) -> float:
    """Compute token-sort similarity, robust to token order changes."""
    left_sorted = " ".join(sorted(left_name.split()))
    right_sorted = " ".join(sorted(right_name.split()))
    return SequenceMatcher(a=left_sorted, b=right_sorted).ratio()


def jaro_winkler_similarity(*, left_name: str, right_name: str) -> float:
    """Compute Jaro-Winkler similarity without third-party dependencies."""
    if not left_name or not right_name:
        return 0.0
    if left_name == right_name:
        return 1.0
    left_length = len(left_name)
    right_length = len(right_name)
    max_distance = (max(left_length, right_length) // 2) - 1
    left_matches = [False] * left_length
    right_matches = [False] * right_length
    matches = 0
    for left_index in range(left_length):
        start = max(0, left_index - max_distance)
        end = min(left_index + max_distance + 1, right_length)
        for right_index in range(start, end):
            if right_matches[right_index] or left_name[left_index] != right_name[right_index]:
                continue
            left_matches[left_index] = True
            right_matches[right_index] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    right_cursor = 0
    for left_index in range(left_length):
        if not left_matches[left_index]:
            continue
        while not right_matches[right_cursor]:
            right_cursor += 1
        if left_name[left_index] != right_name[right_cursor]:
            transpositions += 1
        right_cursor += 1
    transposition_score = transpositions / 2
    jaro = (
        (matches / left_length)
        + (matches / right_length)
        + ((matches - transposition_score) / matches)
    ) / 3.0
    prefix_length = 0
    for index in range(min(left_length, right_length, 4)):
        if left_name[index] != right_name[index]:
            break
        prefix_length += 1
    return jaro + (0.1 * prefix_length * (1 - jaro))
