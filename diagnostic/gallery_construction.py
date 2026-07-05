"""Cue-swap gallery construction using external cue-affinity scores."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np

from diagnostic.config import stable_seed


@dataclass
class GalleryBuild:
    galleries: dict[str, np.ndarray]
    dense: dict[str, np.ndarray]
    neutral: dict[str, np.ndarray]


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
) -> np.ndarray:
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    if len(remaining) < k:
        raise ValueError("not_enough_remaining_for_neutral_fill")
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
        gallery_a, dense_a, neutral_a = make_gallery(
            positive_indices,
            candidate_indices,
            score_a_dense,
            psi_a,
            psi_b,
            num_dense,
            num_neutral,
            np.random.default_rng(stable_seed(seed, case_id, query_id, trial_id, "a_dense")),
            neutral_strategy,
            neutral_pool_factor,
        )
        gallery_b, dense_b, neutral_b = make_gallery(
            positive_indices,
            candidate_indices,
            score_b_dense,
            psi_a,
            psi_b,
            num_dense,
            num_neutral,
            np.random.default_rng(stable_seed(seed, case_id, query_id, trial_id, "b_dense")),
            neutral_strategy,
            neutral_pool_factor,
        )
    except ValueError as exc:
        return None, str(exc)

    if len(gallery_a) != gallery_size or len(gallery_b) != gallery_size:
        return None, "constructed_gallery_size_mismatch"
    return GalleryBuild(
        galleries={"a_dense": gallery_a, "b_dense": gallery_b},
        dense={"a_dense": dense_a, "b_dense": dense_b},
        neutral={"a_dense": neutral_a, "b_dense": neutral_b},
    ), None

