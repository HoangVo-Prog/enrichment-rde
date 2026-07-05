"""Hardness-matched control gallery construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class HardnessControlBuild:
    galleries: dict[str, np.ndarray]
    diagnostics: dict[str, float]


def _bin_matched_without_replacement(
    target_scores: np.ndarray,
    candidate_indices: np.ndarray,
    full_scores: np.ndarray,
    num_bins: int | None = None,
) -> np.ndarray:
    """Quantile-bin hardness matching without replacement.

    This follows the AGENTS.md-allowed stratified/bin matching strategy and is
    much faster than per-target Python nearest-neighbor matching for large
    galleries. Within each score bin, candidates closest to the target-bin mean
    are selected deterministically.
    """
    target_scores = np.asarray(target_scores, dtype=float)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    k = int(len(target_scores))
    if k == 0:
        return np.asarray([], dtype=np.int64)
    if len(candidate_indices) < k:
        raise ValueError("not_enough_hardness_candidates")

    candidate_scores = np.asarray(full_scores[candidate_indices], dtype=float)
    if num_bins is None:
        num_bins = int(np.clip(np.sqrt(len(candidate_indices)), 8, 64))

    combined_scores = np.concatenate([target_scores, candidate_scores])
    edges = np.quantile(combined_scores, np.linspace(0.0, 1.0, num_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        order = np.lexsort((candidate_indices, np.abs(candidate_scores - float(np.mean(target_scores)))))
        return candidate_indices[order[:k]].astype(np.int64)

    target_bins = np.searchsorted(edges[1:-1], target_scores, side="right")
    candidate_bins = np.searchsorted(edges[1:-1], candidate_scores, side="right")

    selected: list[int] = []
    selected_set: set[int] = set()
    deficits: list[float] = []
    for bin_id in range(len(edges) - 1):
        target_in_bin = target_scores[target_bins == bin_id]
        need = int(len(target_in_bin))
        if need == 0:
            continue
        pool_mask = candidate_bins == bin_id
        pool = candidate_indices[pool_mask]
        pool_scores = candidate_scores[pool_mask]
        target_center = float(np.mean(target_in_bin))
        if len(pool) >= need:
            order = np.lexsort((pool, np.abs(pool_scores - target_center)))
            chosen = pool[order[:need]]
            selected.extend(int(index) for index in chosen.tolist())
            selected_set.update(int(index) for index in chosen.tolist())
        else:
            selected.extend(int(index) for index in pool.tolist())
            selected_set.update(int(index) for index in pool.tolist())
            deficits.extend(float(value) for value in target_in_bin[len(pool):].tolist())

    if len(selected) < k:
        remaining = np.asarray(
            [int(index) for index in candidate_indices.tolist() if int(index) not in selected_set],
            dtype=np.int64,
        )
        if len(remaining) < k - len(selected):
            raise ValueError("not_enough_hardness_candidates")
        target_center = float(np.mean(deficits if deficits else target_scores))
        remaining_scores = full_scores[remaining]
        order = np.lexsort((remaining, np.abs(remaining_scores - target_center)))
        fill = remaining[order[: k - len(selected)]]
        selected.extend(int(index) for index in fill.tolist())

    selected_array = np.asarray(selected[:k], dtype=np.int64)
    if len(np.unique(selected_array)) != k:
        raise ValueError("hardness_control_contains_duplicate_image_ids")
    return selected_array


def _build_one_control(
    gallery_type: str,
    pid: int,
    cue_gallery: np.ndarray,
    positive_indices: np.ndarray,
    gallery_pids: np.ndarray,
    full_scores: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    cue_distractors = cue_gallery[gallery_pids[cue_gallery] != int(pid)].astype(np.int64)
    if len(cue_distractors) == 0:
        raise ValueError("cue_gallery_has_no_distractors")
    cue_set = set(int(index) for index in cue_gallery.tolist())
    candidate_pool = np.asarray(
        [index for index, gallery_pid in enumerate(gallery_pids) if int(gallery_pid) != int(pid) and index not in cue_set],
        dtype=np.int64,
    )
    if len(candidate_pool) < len(cue_distractors):
        candidate_pool = np.asarray(
            [index for index, gallery_pid in enumerate(gallery_pids) if int(gallery_pid) != int(pid)],
            dtype=np.int64,
        )
    if len(candidate_pool) < len(cue_distractors):
        raise ValueError("not_enough_hardness_candidates")
    target_order = np.lexsort((cue_distractors, full_scores[cue_distractors]))
    sorted_targets = cue_distractors[target_order]
    matched = _bin_matched_without_replacement(full_scores[sorted_targets], candidate_pool, full_scores)
    gallery = np.concatenate([positive_indices, matched]).astype(np.int64)
    if len(np.unique(gallery)) != len(gallery):
        raise ValueError("hardness_control_contains_duplicate_image_ids")
    prefix = "hm_a" if gallery_type == "a_dense" else "hm_b"
    return gallery, {
        f"{prefix}_mean_score_cue_subset": float(np.mean(full_scores[cue_distractors])),
        f"{prefix}_mean_score_control_subset": float(np.mean(full_scores[matched])),
        f"{prefix}_std_score_cue_subset": float(np.std(full_scores[cue_distractors])),
        f"{prefix}_std_score_control_subset": float(np.std(full_scores[matched])),
    }


def construct_hardness_matched_controls(
    pid: int,
    cue_galleries: dict[str, np.ndarray],
    gallery_pids: np.ndarray,
    full_scores: np.ndarray,
) -> tuple[Optional[HardnessControlBuild], Optional[str]]:
    positive_indices = np.flatnonzero(gallery_pids == int(pid)).astype(np.int64)
    try:
        hm_a, diag_a = _build_one_control("a_dense", pid, cue_galleries["a_dense"], positive_indices, gallery_pids, full_scores)
        hm_b, diag_b = _build_one_control("b_dense", pid, cue_galleries["b_dense"], positive_indices, gallery_pids, full_scores)
    except ValueError as exc:
        return None, str(exc)
    diagnostics = {**diag_a, **diag_b}
    return HardnessControlBuild(galleries={"hm_a": hm_a, "hm_b": hm_b}, diagnostics=diagnostics), None
