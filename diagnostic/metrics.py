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

