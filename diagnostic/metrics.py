"""Retrieval and diagnostic metric utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def stable_descending_order(scores: np.ndarray) -> np.ndarray:
    return np.lexsort((np.arange(len(scores)), -scores))


def retrieval_metrics(scores: np.ndarray, is_positive: np.ndarray) -> Dict[str, float]:
    if int(is_positive.sum()) == 0:
        raise ValueError("retrieval_metrics called with no positive gallery item")
    order = stable_descending_order(scores)
    matches = is_positive[order].astype(np.int64)
    positive_ranks = np.flatnonzero(matches) + 1
    cumulative = np.cumsum(matches)
    precisions = cumulative[positive_ranks - 1] / positive_ranks
    return {
        "R1": float(matches[:1].any() * 100.0),
        "R5": float(matches[:5].any() * 100.0),
        "R10": float(matches[:10].any() * 100.0),
        "AP": float(precisions.mean() * 100.0),
        "best_positive_rank": float(positive_ranks.min()),
    }


def retriever_hardness_stats(scores: np.ndarray, is_positive: np.ndarray) -> Dict[str, float]:
    if int(is_positive.sum()) == 0:
        raise ValueError("retriever_hardness_stats called with no positive gallery item")
    scores = np.asarray(scores, dtype=float)
    is_positive = np.asarray(is_positive, dtype=bool)
    positive_scores = scores[is_positive]
    negative_scores = scores[~is_positive]
    best_positive_score = float(np.max(positive_scores))
    max_negative_score = float(np.max(negative_scores)) if len(negative_scores) else float("nan")
    margin = best_positive_score - max_negative_score if np.isfinite(max_negative_score) else float("nan")
    return {
        "best_positive_score": best_positive_score,
        "max_negative_score": max_negative_score,
        "positive_negative_margin": float(margin),
    }


def query_negative_score_scale(
    full_retriever_scores: np.ndarray,
    full_gallery_pids: np.ndarray,
    query_pid: int,
) -> tuple[float, float]:
    negative_scores = np.asarray(full_retriever_scores, dtype=float)[
        np.asarray(full_gallery_pids) != int(query_pid)
    ]
    negative_score_scale = float(np.std(negative_scores, ddof=0)) if len(negative_scores) else float("nan")
    safe_scale = max(negative_score_scale, 1e-12) if np.isfinite(negative_score_scale) else float("nan")
    return negative_score_scale, float(safe_scale)


def paired_hardness_gaps(
    cue_metrics: Dict[str, Dict[str, float]],
    hm_metrics: Dict[str, Dict[str, float]],
    safe_scale: float,
    tight_hardness_z_tolerance: float,
) -> Dict[str, float | bool]:
    a_signed_max_negative_gap = float(
        cue_metrics["a_dense"]["max_negative_score"] - hm_metrics["hm_a"]["max_negative_score"]
    )
    b_signed_max_negative_gap = float(
        cue_metrics["b_dense"]["max_negative_score"] - hm_metrics["hm_b"]["max_negative_score"]
    )
    a_signed_margin_gap = float(
        cue_metrics["a_dense"]["positive_negative_margin"] - hm_metrics["hm_a"]["positive_negative_margin"]
    )
    b_signed_margin_gap = float(
        cue_metrics["b_dense"]["positive_negative_margin"] - hm_metrics["hm_b"]["positive_negative_margin"]
    )
    mean_signed_max_negative_gap = 0.5 * (a_signed_max_negative_gap + b_signed_max_negative_gap)
    mean_signed_margin_gap = 0.5 * (a_signed_margin_gap + b_signed_margin_gap)
    mean_abs_max_negative_gap = 0.5 * (
        abs(a_signed_max_negative_gap) + abs(b_signed_max_negative_gap)
    )

    if np.isfinite(safe_scale) and safe_scale > 0.0:
        a_normalized_max_negative_gap = float(a_signed_max_negative_gap / safe_scale)
        b_normalized_max_negative_gap = float(b_signed_max_negative_gap / safe_scale)
    else:
        a_normalized_max_negative_gap = float("nan")
        b_normalized_max_negative_gap = float("nan")
    max_abs_normalized = max(
        abs(a_normalized_max_negative_gap),
        abs(b_normalized_max_negative_gap),
    )
    mean_normalized_max_negative_gap = 0.5 * (
        a_normalized_max_negative_gap + b_normalized_max_negative_gap
    )
    tight_hardness_match = bool(
        np.isfinite(a_normalized_max_negative_gap)
        and np.isfinite(b_normalized_max_negative_gap)
        and abs(a_normalized_max_negative_gap) <= float(tight_hardness_z_tolerance)
        and abs(b_normalized_max_negative_gap) <= float(tight_hardness_z_tolerance)
    )
    return {
        "a_signed_max_negative_gap": float(a_signed_max_negative_gap),
        "b_signed_max_negative_gap": float(b_signed_max_negative_gap),
        "a_signed_margin_gap": float(a_signed_margin_gap),
        "b_signed_margin_gap": float(b_signed_margin_gap),
        "mean_signed_max_negative_gap": float(mean_signed_max_negative_gap),
        "mean_signed_margin_gap": float(mean_signed_margin_gap),
        "mean_abs_max_negative_gap": float(mean_abs_max_negative_gap),
        "a_normalized_max_negative_gap": float(a_normalized_max_negative_gap),
        "b_normalized_max_negative_gap": float(b_normalized_max_negative_gap),
        "mean_normalized_max_negative_gap": float(mean_normalized_max_negative_gap),
        "max_abs_normalized_max_negative_gap": float(max_abs_normalized),
        "tight_hardness_match": tight_hardness_match,
    }


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def density_over_indices(
    psi: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    tau_density: float,
) -> Tuple[float, float]:
    if len(indices) == 0:
        return float("nan"), float("nan")
    values = psi[indices]
    hard = float(np.mean(values > threshold))
    soft = float(np.mean(sigmoid_np((values - threshold) / tau_density)))
    return hard, soft


def gallery_distractor_indices(gallery_indices: np.ndarray, gallery_pids: np.ndarray, pid: int) -> np.ndarray:
    return gallery_indices[gallery_pids[gallery_indices] != int(pid)]


def cue_shift(
    gallery_a: np.ndarray,
    gallery_b: np.ndarray,
    gallery_pids: np.ndarray,
    pid: int,
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    tau_density: float,
) -> Dict[str, float]:
    distractors_a = gallery_distractor_indices(gallery_a, gallery_pids, pid)
    distractors_b = gallery_distractor_indices(gallery_b, gallery_pids, pid)
    hard_a_a, soft_a_a = density_over_indices(psi_a, distractors_a, threshold_a, tau_density)
    hard_a_b, soft_a_b = density_over_indices(psi_a, distractors_b, threshold_a, tau_density)
    hard_b_a, soft_b_a = density_over_indices(psi_b, distractors_a, threshold_b, tau_density)
    hard_b_b, soft_b_b = density_over_indices(psi_b, distractors_b, threshold_b, tau_density)
    shift = 0.5 * ((soft_a_a - soft_a_b) + (soft_b_b - soft_b_a))
    return {
        "cue_shift": float(shift),
        "D_hard_a_a_dense": hard_a_a,
        "D_soft_a_a_dense": soft_a_a,
        "D_hard_a_b_dense": hard_a_b,
        "D_soft_a_b_dense": soft_a_b,
        "D_hard_b_a_dense": hard_b_a,
        "D_soft_b_a_dense": soft_b_a,
        "D_hard_b_b_dense": hard_b_b,
        "D_soft_b_b_dense": soft_b_b,
    }


def paired_metrics(metrics_a: Dict[str, float], metrics_b: Dict[str, float]) -> Dict[str, float]:
    return {
        "r1_flip": float(metrics_a["R1"] != metrics_b["R1"]),
        "rank_shift": float(abs(metrics_a["best_positive_rank"] - metrics_b["best_positive_rank"])),
        "ap_delta": float(metrics_a["AP"] - metrics_b["AP"]),
    }

