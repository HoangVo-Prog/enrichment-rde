"""Cue-swap gallery construction using external cue-affinity scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from diagnostic.config import stable_seed


@dataclass
class GalleryBuild:
    galleries: dict[str, np.ndarray]
    dense: dict[str, np.ndarray]
    neutral: dict[str, np.ndarray]
    shared_neutral: np.ndarray
    positive_indices: np.ndarray


def stable_topk(indices: np.ndarray, scores: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    if len(indices) < k:
        raise ValueError("not_enough_indices_for_topk")
    local_scores = scores[indices]
    order = np.lexsort((indices, -local_scores if largest else local_scores))
    return indices[order[:k]].astype(np.int64)


def choose_neutral_indices(
    remaining: np.ndarray,
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    k: int,
    rng: np.random.Generator,
    strategy: str,
    pool_factor: int,
    insufficient_reason: str = "not_enough_remaining_for_neutral_fill",
) -> np.ndarray:
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    if len(remaining) < k:
        raise ValueError(insufficient_reason)
    if strategy == "random":
        pool = remaining
    elif strategy == "low_affinity":
        max_affinity = np.maximum(psi_a[remaining], psi_b[remaining])
        order = np.lexsort((remaining, max_affinity))
        pool_size = min(len(remaining), max(k * pool_factor, k))
        pool = remaining[order[:pool_size]]
    else:
        raise ValueError(f"Unsupported neutral strategy: {strategy}")
    return np.sort(rng.choice(pool, size=k, replace=False).astype(np.int64))


def make_gallery(
    positive_indices: np.ndarray,
    candidate_indices: np.ndarray,
    dense_scores: np.ndarray,
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    num_dense: int,
    num_neutral: int,
    rng: np.random.Generator,
    neutral_strategy: str,
    neutral_pool_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense = stable_topk(candidate_indices, dense_scores, num_dense, largest=True)
    dense_set = set(int(index) for index in dense.tolist())
    remaining = np.asarray([int(index) for index in candidate_indices if int(index) not in dense_set], dtype=np.int64)
    neutral = choose_neutral_indices(remaining, psi_a, psi_b, num_neutral, rng, neutral_strategy, neutral_pool_factor)
    gallery = np.concatenate([positive_indices, dense, neutral]).astype(np.int64)
    if len(np.unique(gallery)) != len(gallery):
        raise ValueError("gallery_contains_duplicate_image_ids")
    return gallery, dense, neutral


def validate_shared_neutral_galleries(
    *,
    positive_indices: np.ndarray,
    dense_a: np.ndarray,
    dense_b: np.ndarray,
    shared_neutral: np.ndarray,
    gallery_a: np.ndarray,
    gallery_b: np.ndarray,
    gallery_size: int,
    configured_dense_count: int,
) -> None:
    positive_set = set(int(index) for index in positive_indices.tolist())
    dense_a_set = set(int(index) for index in dense_a.tolist())
    dense_b_set = set(int(index) for index in dense_b.tolist())
    shared_neutral_set = set(int(index) for index in shared_neutral.tolist())

    if len(gallery_a) != gallery_size or len(gallery_b) != gallery_size:
        raise ValueError("constructed_gallery_size_mismatch")
    if len(dense_a) != configured_dense_count or len(dense_b) != configured_dense_count:
        raise ValueError("constructed_dense_count_mismatch")
    if len(np.unique(shared_neutral)) != len(shared_neutral):
        raise ValueError("shared_neutral_contains_duplicate_image_ids")
    if len(np.unique(gallery_a)) != len(gallery_a) or len(np.unique(gallery_b)) != len(gallery_b):
        raise ValueError("gallery_contains_duplicate_image_ids")
    if positive_set & dense_a_set or positive_set & dense_b_set or positive_set & shared_neutral_set:
        raise ValueError("positive_image_in_distractor_set")
    if (dense_a_set | dense_b_set) & shared_neutral_set:
        raise ValueError("dense_image_in_shared_neutral")
    if set(int(index) for index in gallery_a[: len(positive_indices)].tolist()) != positive_set:
        raise ValueError("gallery_a_missing_complete_positive_set")
    if set(int(index) for index in gallery_b[: len(positive_indices)].tolist()) != positive_set:
        raise ValueError("gallery_b_missing_complete_positive_set")
    if int(len(gallery_a) - len(positive_indices)) != int(len(gallery_b) - len(positive_indices)):
        raise ValueError("gallery_distractor_count_mismatch")
    if set(int(index) for index in gallery_a.tolist()) - positive_set - shared_neutral_set != dense_a_set:
        raise ValueError("gallery_a_dense_set_mismatch")
    if set(int(index) for index in gallery_b.tolist()) - positive_set - shared_neutral_set != dense_b_set:
        raise ValueError("gallery_b_dense_set_mismatch")


def construct_cue_swap_galleries(
    pid: int,
    gallery_pids: np.ndarray,
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    gallery_size: int,
    dense_ratio: float,
    lambda_contrast: float,
    seed: int,
    case_id: str,
    query_id: int,
    trial_id: int,
    neutral_strategy: str,
    neutral_pool_factor: int,
) -> tuple[Optional[GalleryBuild], Optional[str]]:
    positive_indices = np.flatnonzero(gallery_pids == int(pid)).astype(np.int64)
    if len(positive_indices) == 0:
        return None, "no_positive_gallery_images"
    if len(positive_indices) > gallery_size:
        return None, "num_positives_exceeds_gallery_size"
    candidate_indices = np.flatnonzero(gallery_pids != int(pid)).astype(np.int64)
    num_distractors = int(gallery_size - len(positive_indices))
    if len(candidate_indices) < num_distractors:
        return None, "not_enough_valid_distractors"

    num_dense = int(round(dense_ratio * num_distractors))
    num_dense = min(max(num_dense, 0), num_distractors)
    num_neutral = num_distractors - num_dense
    score_a_dense = psi_a - lambda_contrast * psi_b
    score_b_dense = psi_b - lambda_contrast * psi_a

    try:
        dense_a = stable_topk(candidate_indices, score_a_dense, num_dense, largest=True)
        dense_b = stable_topk(candidate_indices, score_b_dense, num_dense, largest=True)
        dense_union = set(int(index) for index in dense_a.tolist()) | set(int(index) for index in dense_b.tolist())
        remaining = np.asarray(
            [int(index) for index in candidate_indices.tolist() if int(index) not in dense_union],
            dtype=np.int64,
        )
        shared_neutral = choose_neutral_indices(
            remaining,
            psi_a,
            psi_b,
            num_neutral,
            np.random.default_rng(stable_seed(seed, case_id, query_id, trial_id, "shared_neutral")),
            neutral_strategy,
            neutral_pool_factor,
            insufficient_reason="not_enough_remaining_for_shared_neutral_fill",
        )
        gallery_a = np.concatenate([positive_indices, dense_a, shared_neutral]).astype(np.int64)
        gallery_b = np.concatenate([positive_indices, dense_b, shared_neutral]).astype(np.int64)
        validate_shared_neutral_galleries(
            positive_indices=positive_indices,
            dense_a=dense_a,
            dense_b=dense_b,
            shared_neutral=shared_neutral,
            gallery_a=gallery_a,
            gallery_b=gallery_b,
            gallery_size=gallery_size,
            configured_dense_count=num_dense,
        )
    except ValueError as exc:
        return None, str(exc)

    return GalleryBuild(
        galleries={"a_dense": gallery_a, "b_dense": gallery_b},
        dense={"a_dense": dense_a, "b_dense": dense_b},
        neutral={"shared": shared_neutral, "a_dense": shared_neutral, "b_dense": shared_neutral},
        shared_neutral=shared_neutral,
        positive_indices=positive_indices,
    ), None

