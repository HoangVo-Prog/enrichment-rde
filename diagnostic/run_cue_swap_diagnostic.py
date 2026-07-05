"""Run the generalized cue-swap diagnostic.

This is a matched empirical probe, not a universal theorem about all TBPS
retrievers. Cue-biased galleries are constructed with an off-the-shelf CLIP
cue-affinity scorer, while retrieval and hardness matching use the evaluated
frozen TBPS retriever.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "2024-CVPR-RDE"))

from diagnostic.audit import log_validity_warnings
from diagnostic.bootstrap import bootstrap_count_summary, cluster_bootstrap_ci
from diagnostic.clip_cue_scorer import OffTheShelfCLIPCueScorer
from diagnostic.config import (
    jsonable,
    load_repo_args,
    parse_args,
    resolve_device,
    set_deterministic,
    validate_args,
)
from diagnostic.constants import (
    CASE_QUERY_CLUSTER_COLS,
    OUTPUT_FILES,
    PAIRED_DELTA_COLUMNS,
    PRIMARY_BOOTSTRAP_UNIT_FOR_BOTH,
    SUMMARY_CI_COLUMNS,
    SUMMARY_CI_METRICS,
    UNIQUE_QUERY_CLUSTER_COLS,
)
from diagnostic.controls import construct_hardness_matched_controls
from diagnostic.cue_cases import load_cases, select_queries_for_cases
from diagnostic.data_loading import QueryRecord, load_split, load_split_metadata
from diagnostic.embeddings import extract_retriever_embeddings
from diagnostic.gallery_construction import construct_cue_swap_galleries
from diagnostic.metrics import (
    cue_shift,
    density_over_indices,
    gallery_distractor_indices,
    paired_metrics,
    retrieval_metrics,
)
from diagnostic.outputs import (
    write_case_candidates,
    write_config_used,
    write_run_outputs,
    write_thresholds,
    write_whole_test,
    summarize_outputs,
)
from diagnostic.retriever_loading import load_retriever
from diagnostic.scoring import (
    compute_full_test_metrics,
    full_query_scores,
)


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cue_swap_diagnostic")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(output_dir / "cue_swap_diagnostic.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def unique_cues(cases: list[Mapping[str, Any]]) -> list[str]:
    cues = sorted({str(case["cue_a"]) for case in cases} | {str(case["cue_b"]) for case in cases})
    if not cues:
        raise RuntimeError("No cues found in cue cases")
    return cues


def format_counts(counts: Mapping[str, int], limit: int = 8) -> str:
    items = Counter({str(key): int(value) for key, value in counts.items()})
    if not items:
        return "none"
    shown = [f"{key}={value}" for key, value in items.most_common(limit)]
    hidden = len(items) - len(shown)
    if hidden > 0:
        shown.append(f"... {hidden} more")
    return ", ".join(shown)


def describe_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "max": np.nan,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def format_description(stats: Mapping[str, float | int]) -> str:
    if int(stats.get("count", 0)) == 0:
        return "count=0"
    return (
        f"count={int(stats['count'])} mean={float(stats['mean']):.4f} "
        f"min={float(stats['min']):.4f} p25={float(stats['p25']):.4f} "
        f"p50={float(stats['p50']):.4f} p75={float(stats['p75']):.4f} "
        f"p90={float(stats['p90']):.4f} max={float(stats['max']):.4f}"
    )


def gallery_result_row(
    *,
    args: Any,
    score_mode: str,
    query: QueryRecord,
    query_text: str,
    case_id: str,
    cue_a: str,
    cue_b: str,
    trial_id: int,
    construction_type: str,
    gallery_type: str,
    gallery_indices: np.ndarray,
    scores: np.ndarray,
    gallery_pids: np.ndarray,
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
) -> tuple[dict[str, Any], dict[str, float], np.ndarray]:
    is_positive = gallery_pids[gallery_indices] == int(query.pid)
    metrics = retrieval_metrics(scores, is_positive)
    distractors = gallery_distractor_indices(gallery_indices, gallery_pids, query.pid)
    hard_a, soft_a = density_over_indices(psi_a, distractors, threshold_a, args.tau_density)
    hard_b, soft_b = density_over_indices(psi_b, distractors, threshold_b, args.tau_density)
    distractor_scores = scores[~is_positive]
    row = {
        "dataset": args.dataset,
        "retriever_name": args.retriever_name,
        "cue_scorer": args.cue_scorer,
        "case_id": case_id,
        "query_id": query.query_id,
        "query_text": query_text,
        "pid": query.pid,
        "cue_a": cue_a,
        "cue_b": cue_b,
        "trial_id": trial_id,
        "construction_type": construction_type,
        "gallery_type": gallery_type,
        "gallery_size": int(len(gallery_indices)),
        "num_positives": int(is_positive.sum()),
        "num_distractors": int((~is_positive).sum()),
        "positive_ratio": float(is_positive.mean()),
        **metrics,
        "D_soft_a": soft_a,
        "D_soft_b": soft_b,
        "D_hard_a": hard_a,
        "D_hard_b": hard_b,
        "mean_retriever_score_distractors": float(np.mean(distractor_scores)) if len(distractor_scores) else np.nan,
        "score_mode": score_mode,
        "lambda_global": args.lambda_global,
        "seed": args.seed,
    }
    return row, metrics, is_positive


def precompute_selected_full_scores(
    *,
    cache: Any,
    query_ids: list[int],
    mode: str,
    lambda_global: float,
    device: torch.device,
    logger: logging.Logger,
    max_elements: int = 50_000_000,
    chunk_size: int = 256,
) -> dict[int, np.ndarray]:
    unique_query_ids = sorted(set(int(query_id) for query_id in query_ids))
    if not unique_query_ids:
        return {}
    num_gallery = int(cache.gallery_global.shape[0])
    num_elements = len(unique_query_ids) * num_gallery
    if num_elements > max_elements:
        logger.info(
            "Skipping upfront full-score precompute: unique_queries=%d gallery=%d elements=%d exceeds max_elements=%d; using lazy cache.",
            len(unique_query_ids),
            num_gallery,
            num_elements,
            max_elements,
        )
        return {}

    compute_device = device if device.type == "cuda" else torch.device("cpu")
    logger.info(
        "Precomputing full query-gallery scores unique_queries=%d gallery=%d elements=%d device=%s chunk_size=%d",
        len(unique_query_ids),
        num_gallery,
        num_elements,
        compute_device,
        chunk_size,
    )
    started = time.perf_counter()
    query_ids_tensor = torch.as_tensor(unique_query_ids, dtype=torch.long)
    gallery_global = cache.gallery_global.to(compute_device)
    gallery_grab = cache.gallery_grab.to(compute_device) if cache.gallery_grab is not None else None
    score_cache: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(unique_query_ids), chunk_size):
            end = min(start + chunk_size, len(unique_query_ids))
            chunk_query_ids = query_ids_tensor[start:end]
            global_scores = cache.query_global[chunk_query_ids].to(compute_device) @ gallery_global.T
            if mode == "global":
                scores = global_scores
            else:
                if cache.query_grab is None or gallery_grab is None:
                    raise RuntimeError("GRAB/fusion score precompute requested without GRAB features")
                grab_scores = cache.query_grab[chunk_query_ids].to(compute_device) @ gallery_grab.T
                if mode == "grab":
                    scores = grab_scores
                elif mode == "fusion":
                    scores = lambda_global * global_scores + (1.0 - lambda_global) * grab_scores
                else:
                    raise ValueError(f"Unsupported score mode: {mode}")
            scores_np = scores.cpu().numpy()
            for row_index, query_id in enumerate(unique_query_ids[start:end]):
                score_cache[int(query_id)] = scores_np[row_index]
            logger.info(
                "Full-score precompute progress queries=%d/%d elapsed=%.1fs",
                end,
                len(unique_query_ids),
                time.perf_counter() - started,
            )
    logger.info("Full-score precompute finished in %.2fs", time.perf_counter() - started)
    return score_cache


def requested_bootstrap_units(bootstrap_unit: str) -> list[str]:
    if bootstrap_unit == "both":
        return ["unique_query", "case_query"]
    return [bootstrap_unit]


def primary_bootstrap_unit(bootstrap_unit: str) -> str:
    return PRIMARY_BOOTSTRAP_UNIT_FOR_BOTH if bootstrap_unit == "both" else bootstrap_unit


def cluster_columns_for_unit(bootstrap_unit: str) -> list[str]:
    if bootstrap_unit == "unique_query":
        return UNIQUE_QUERY_CLUSTER_COLS
    if bootstrap_unit == "case_query":
        return CASE_QUERY_CLUSTER_COLS
    raise ValueError(f"Unsupported bootstrap unit: {bootstrap_unit}")


def bootstrap_label(bootstrap_unit: str) -> str:
    if bootstrap_unit == "unique_query":
        return "unique-query cluster bootstrap"
    if bootstrap_unit == "case_query":
        return "case-query-instance cluster bootstrap"
    raise ValueError(f"Unsupported bootstrap unit: {bootstrap_unit}")


def run_standard_repo_evaluation(retriever: Any, split: Any, logger: logging.Logger) -> None:
    from utils.metrics import Evaluator

    logger.info("Running standard RDE test-set retrieval evaluation before diagnostic scoring")
    top1 = Evaluator(split.img_loader, split.txt_loader).eval(retriever.model.eval())
    logger.info("Standard RDE evaluator returned top1=%.4f", float(top1))


def rde_ablation_candidates() -> list[tuple[str, str, float, int]]:
    return [
        ("global", "BGE", 1.0, 0),
        ("grab", "TSE", 0.0, 1),
        ("fusion", "BGE+TSE", 0.5, 2),
    ]


def select_best_rde_ablation(
    *,
    retriever_cache: Any,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    chunk_size: int,
    logger: logging.Logger,
) -> tuple[str, float, dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for mode, label, lambda_global, tie_order in rde_ablation_candidates():
        metrics = compute_full_test_metrics(
            retriever_cache,
            query_pids,
            gallery_pids,
            mode,
            lambda_global,
            chunk_size=chunk_size,
        )
        row = {
            **metrics,
            "mode": mode,
            "ablation_name": label,
            "lambda_global": lambda_global,
            "rSum": float(metrics["R1"] + metrics["R5"] + metrics["R10"]),
            "_tie_order": tie_order,
        }
        rows.append(row)
        logger.info(
            "RDE ablation candidate=%s score_mode=%s lambda_global=%.2f "
            "R1=%.2f R5=%.2f R10=%.2f mAP=%.2f rSum=%.2f",
            label,
            mode,
            lambda_global,
            float(row["R1"]),
            float(row["R5"]),
            float(row["R10"]),
            float(row["mAP"]),
            float(row["rSum"]),
        )
    selected = max(
        rows,
        key=lambda row: (
            float(row["R1"]),
            float(row["R5"]),
            float(row["R10"]),
            float(row["mAP"]),
            float(row["rSum"]),
            int(row["_tie_order"]),
        ),
    )
    logger.info(
        "Selected RDE ablation candidate=%s score_mode=%s lambda_global=%.2f",
        selected["ablation_name"],
        selected["mode"],
        float(selected["lambda_global"]),
    )
    clean_rows = [{key: value for key, value in row.items() if key != "_tie_order"} for row in rows]
    selected_clean = {key: value for key, value in selected.items() if key != "_tie_order"}
    return str(selected["mode"]), float(selected["lambda_global"]), selected_clean, clean_rows


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    set_deterministic(args.seed)

    repo_args = load_repo_args(args)
    split = load_split_metadata(repo_args, args.split) if args.dry_run else load_split(repo_args, args.split)
    logger.info(
        "Loaded dataset=%s split=%s queries=%d gallery=%d",
        args.dataset,
        args.split,
        len(split.query_records),
        len(split.gallery_records),
    )
    logger.info(
        "Run parameters retriever=%s gallery_size=%d dense_ratio=%.3f num_trials=%d "
        "lambda_contrast=%.3f min_pair_cue_shift=%.4f bootstrap_iters=%d",
        args.retriever_name,
        args.gallery_size,
        args.dense_ratio,
        args.num_trials,
        args.lambda_contrast,
        args.min_pair_cue_shift,
        args.bootstrap_iters,
    )

    cases = load_cases(args.cases_file)
    case_candidate_rows = [
        {
            "case_id": case["case_id"],
            "cue_a": case["cue_a"],
            "cue_b": case["cue_b"],
            "num_queries": len(case.get("query_ids", [])),
            "query_ids": case.get("query_ids", ""),
            "meets_min_queries": True,
            "kept_after_support": True,
            "reason": "",
        }
        for case in cases
    ]
    write_case_candidates(args.output_dir, case_candidate_rows)
    candidate_reason_counts = Counter(str(row.get("reason", "") or "kept") for row in case_candidate_rows)
    logger.info(
        "Cue case source=%s candidates=%d retained=%d reason_counts=[%s]",
        args.cases_file,
        len(case_candidate_rows),
        len(cases),
        format_counts(candidate_reason_counts),
    )
    for case in cases[:10]:
        logger.info(
            "Retained case case_id=%s cue_a=%s cue_b=%s support=%s",
            case.get("case_id"),
            case.get("cue_a"),
            case.get("cue_b"),
            case.get("auto_case_support", case.get("num_queries", "manual")),
        )
    if len(cases) > 10:
        logger.info("Retained case log truncated after 10 of %d cases", len(cases))
    if not cases:
        raise RuntimeError("No cue cases are available. See cue_case_candidates.csv.")

    selected_queries, skipped_rows = select_queries_for_cases(
        args.dataset,
        cases,
        split.query_records,
        split.gallery_pids,
        args.max_queries_per_case,
        logger=logger,
    )
    selected_by_case = Counter(str(row["case_id"]) for row in selected_queries)
    logger.info(
        "Selected queries=%d unique_cases=%d expected_query_trials=%d skipped_selection_rows=%d",
        len(selected_queries),
        len(selected_by_case),
        len(selected_queries) * args.num_trials,
        len(skipped_rows),
    )
    logger.info("Top selected-query cases: %s", format_counts(selected_by_case, limit=12))
    if not selected_queries:
        raise RuntimeError("No eligible queries were selected for the retained cue cases.")
    query_lookup = {record.query_id: record for record in split.query_records}

    if args.dry_run:
        write_config_used(
            args.output_dir,
            {
                "args": vars(args),
                "repo_args": repo_args,
                "dry_run": True,
                "cases": cases,
            },
        )
        logger.info("Dry run complete: cases=%d selected_queries=%d", len(cases), len(selected_queries))
        print(f"Dry run complete: cases={len(cases)} selected_queries={len(selected_queries)}")
        return

    device = resolve_device(args.device)
    retriever = load_retriever(
        args.retriever_name,
        repo_args,
        args.retriever_checkpoint,
        split.num_classes,
        device,
        logger,
    )
    run_standard_repo_evaluation(retriever, split, logger)
    retriever_cache = extract_retriever_embeddings(
        retriever,
        split.img_loader,
        split.txt_loader,
        use_grab=retriever.has_grab,
        logger=logger,
    )

    chunk_size = max(1, min(256, int(repo_args.test_batch_size)))
    score_mode, selected_lambda_global, whole_metrics, ablation_rows = select_best_rde_ablation(
        retriever_cache=retriever_cache,
        query_pids=split.query_pids,
        gallery_pids=split.gallery_pids,
        chunk_size=chunk_size,
        logger=logger,
    )
    if args.score_mode != "auto":
        logger.warning(
            "Ignoring explicit --score_mode=%s for RDE diagnostic; using best selected ablation score_mode=%s",
            args.score_mode,
            score_mode,
        )
    args.lambda_global = selected_lambda_global
    whole_metrics.update(
        {
            "dataset": args.dataset,
            "retriever_name": args.retriever_name,
            "score_mode": score_mode,
            "lambda_global": args.lambda_global,
        }
    )
    whole_df = write_whole_test(args.output_dir, [whole_metrics])
    ref_r1 = float(whole_metrics["R1"])
    logger.info(
        "Whole-test sanity metrics mode=%s R1=%.2f R5=%.2f R10=%.2f mAP=%.2f queries=%d",
        score_mode,
        float(whole_metrics["R1"]),
        float(whole_metrics["R5"]),
        float(whole_metrics["R10"]),
        float(whole_metrics["mAP"]),
        int(whole_metrics["num_queries"]),
    )

    cues = unique_cues(cases)
    logger.info("Unique cues for external scorer=%d", len(cues))
    cue_scorer = OffTheShelfCLIPCueScorer(args.clip_model_name, repo_args, device, logger)
    cue_output = cue_scorer.score(cues, split.img_loader, repo_args.text_length, logger)
    thresholds = {
        cue: float(np.quantile(cue_output.affinities[cue], args.cue_threshold_quantile))
        for cue in cues
    }
    for cue in cues[:12]:
        psi = cue_output.affinities[cue]
        logger.info(
            "Cue threshold cue=%s threshold=%.4f quantile=%.2f affinity_stats=[%s]",
            cue,
            thresholds[cue],
            args.cue_threshold_quantile,
            format_description(describe_values([float(value) for value in psi.tolist()])),
        )
    if len(cues) > 12:
        logger.info("Cue threshold log truncated after 12 of %d cues", len(cues))
    write_thresholds(
        args.output_dir,
        [
            {
                "cue": cue,
                "threshold": thresholds[cue],
                "quantile": args.cue_threshold_quantile,
                "cue_scorer": args.cue_scorer,
            }
            for cue in cues
        ],
    )

    write_config_used(
        args.output_dir,
        {
            "args": vars(args),
            "repo_args": repo_args,
            "resolved_score_mode": score_mode,
            "selected_ablation": whole_metrics,
            "ablation_candidates": ablation_rows,
            "bootstrap_unit": args.bootstrap_unit,
            "cue_scorer": {
                "name": args.cue_scorer,
                "clip_model_name": args.clip_model_name,
                "prompt_templates": cue_scorer.prompt_templates,
                "prompts_by_cue": cue_output.prompts_by_cue,
            },
            "claim_scope": (
                "This diagnostic is a matched empirical probe of representative frozen "
                "gallery-agnostic TBPS retrievers under cue-biased gallery perturbations."
            ),
        },
    )

    per_gallery_rows: list[dict[str, Any]] = []
    paired_cue_rows: list[dict[str, Any]] = []
    paired_hm_rows: list[dict[str, Any]] = []
    paired_delta_rows: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []
    constructibility = defaultdict(
        lambda: {
            "attempted_pairs": 0,
            "valid_pairs": 0,
            "candidate_cue_shifts": [],
            "valid_cue_shifts": [],
            "skip_reasons": defaultdict(int),
        }
    )
    full_score_cache: dict[int, np.ndarray] = precompute_selected_full_scores(
        cache=retriever_cache,
        query_ids=[int(row["query_id"]) for row in selected_queries],
        mode=score_mode,
        lambda_global=args.lambda_global,
        device=device,
        logger=logger,
    )
    progress_started = time.perf_counter()
    last_progress_time = progress_started
    last_progress_index = 0

    for selected_index, selected in enumerate(selected_queries, start=1):
        case_id = str(selected["case_id"])
        cue_a = str(selected["cue_a"])
        cue_b = str(selected["cue_b"])
        query_id = int(selected["query_id"])
        query = query_lookup[query_id]
        psi_a = cue_output.affinities[cue_a]
        psi_b = cue_output.affinities[cue_b]
        threshold_a = thresholds[cue_a]
        threshold_b = thresholds[cue_b]
        case_stats = constructibility[case_id]

        for trial_id in range(args.num_trials):
            case_stats["attempted_pairs"] += 1
            build, skip_reason = construct_cue_swap_galleries(
                pid=query.pid,
                gallery_pids=split.gallery_pids,
                psi_a=psi_a,
                psi_b=psi_b,
                gallery_size=args.gallery_size,
                dense_ratio=args.dense_ratio,
                lambda_contrast=args.lambda_contrast,
                seed=args.seed,
                case_id=case_id,
                query_id=query_id,
                trial_id=trial_id,
                neutral_strategy=args.neutral_strategy,
                neutral_pool_factor=args.neutral_pool_factor,
            )
            if build is None:
                case_stats["skip_reasons"][skip_reason] += 1
                skipped_rows.append({"dataset": args.dataset, "case_id": case_id, "query_id": query_id, "trial_id": trial_id, "reason": skip_reason})
                continue

            shift_info = cue_shift(
                build.galleries["a_dense"],
                build.galleries["b_dense"],
                split.gallery_pids,
                query.pid,
                psi_a,
                psi_b,
                threshold_a,
                threshold_b,
                args.tau_density,
            )
            case_stats["candidate_cue_shifts"].append(float(shift_info["cue_shift"]))
            if float(shift_info["cue_shift"]) < args.min_pair_cue_shift:
                reason = "below_min_pair_cue_shift"
                case_stats["skip_reasons"][reason] += 1
                skipped_rows.append({"dataset": args.dataset, "case_id": case_id, "query_id": query_id, "trial_id": trial_id, "reason": reason, "cue_shift": shift_info["cue_shift"]})
                continue

            full_scores = full_score_cache.get(query_id)
            if full_scores is None:
                full_scores = full_query_scores(retriever_cache, query_id, score_mode, args.lambda_global)
                full_score_cache[query_id] = full_scores
            hm_build, hm_skip = construct_hardness_matched_controls(
                query.pid,
                build.galleries,
                split.gallery_pids,
                full_scores,
            )
            if hm_build is None:
                case_stats["skip_reasons"][hm_skip] += 1
                skipped_rows.append({"dataset": args.dataset, "case_id": case_id, "query_id": query_id, "trial_id": trial_id, "reason": hm_skip})
                continue

            case_stats["valid_pairs"] += 1
            case_stats["valid_cue_shifts"].append(float(shift_info["cue_shift"]))

            cue_metrics: dict[str, dict[str, float]] = {}
            hm_metrics: dict[str, dict[str, float]] = {}
            for gallery_type in ("a_dense", "b_dense"):
                gallery_indices = build.galleries[gallery_type]
                scores = full_scores[gallery_indices]
                row, metrics, _ = gallery_result_row(
                    args=args,
                    score_mode=score_mode,
                    query=query,
                    query_text=str(selected["query_text"]),
                    case_id=case_id,
                    cue_a=cue_a,
                    cue_b=cue_b,
                    trial_id=trial_id,
                    construction_type="cue_swap",
                    gallery_type=gallery_type,
                    gallery_indices=gallery_indices,
                    scores=scores,
                    gallery_pids=split.gallery_pids,
                    psi_a=psi_a,
                    psi_b=psi_b,
                    threshold_a=threshold_a,
                    threshold_b=threshold_b,
                )
                per_gallery_rows.append(row)
                cue_metrics[gallery_type] = metrics

            for gallery_type in ("hm_a", "hm_b"):
                gallery_indices = hm_build.galleries[gallery_type]
                scores = full_scores[gallery_indices]
                row, metrics, _ = gallery_result_row(
                    args=args,
                    score_mode=score_mode,
                    query=query,
                    query_text=str(selected["query_text"]),
                    case_id=case_id,
                    cue_a=cue_a,
                    cue_b=cue_b,
                    trial_id=trial_id,
                    construction_type="hardness_control",
                    gallery_type=gallery_type,
                    gallery_indices=gallery_indices,
                    scores=scores,
                    gallery_pids=split.gallery_pids,
                    psi_a=psi_a,
                    psi_b=psi_b,
                    threshold_a=threshold_a,
                    threshold_b=threshold_b,
                )
                per_gallery_rows.append(row)
                hm_metrics[gallery_type] = metrics

            cue_pair = paired_metrics(cue_metrics["a_dense"], cue_metrics["b_dense"])
            hm_pair = paired_metrics(hm_metrics["hm_a"], hm_metrics["hm_b"])
            paired_cue_rows.append(
                {
                    "dataset": args.dataset,
                    "retriever_name": args.retriever_name,
                    "cue_scorer": args.cue_scorer,
                    "case_id": case_id,
                    "query_id": query_id,
                    "pid": query.pid,
                    "trial_id": trial_id,
                    "cue_a": cue_a,
                    "cue_b": cue_b,
                    **cue_pair,
                    "cue_shift": shift_info["cue_shift"],
                    "D_soft_a_a_dense": shift_info["D_soft_a_a_dense"],
                    "D_soft_a_b_dense": shift_info["D_soft_a_b_dense"],
                    "D_soft_b_a_dense": shift_info["D_soft_b_a_dense"],
                    "D_soft_b_b_dense": shift_info["D_soft_b_b_dense"],
                    "score_mode": score_mode,
                    "lambda_global": args.lambda_global,
                    "seed": args.seed,
                }
            )
            paired_hm_rows.append(
                {
                    "dataset": args.dataset,
                    "retriever_name": args.retriever_name,
                    "cue_scorer": args.cue_scorer,
                    "case_id": case_id,
                    "query_id": query_id,
                    "pid": query.pid,
                    "trial_id": trial_id,
                    "cue_a": cue_a,
                    "cue_b": cue_b,
                    "hm_r1_flip": hm_pair["r1_flip"],
                    "hm_rank_shift": hm_pair["rank_shift"],
                    "hm_ap_delta": hm_pair["ap_delta"],
                    **hm_build.diagnostics,
                    "score_mode": score_mode,
                    "lambda_global": args.lambda_global,
                    "seed": args.seed,
                }
            )
            paired_delta_rows.append(
                {
                    "dataset": args.dataset,
                    "retriever_name": args.retriever_name,
                    "cue_scorer": args.cue_scorer,
                    "case_id": case_id,
                    "query_id": query_id,
                    "pid": query.pid,
                    "trial_id": trial_id,
                    "cue_a": cue_a,
                    "cue_b": cue_b,
                    "r1_flip": cue_pair["r1_flip"],
                    "hm_r1_flip": hm_pair["r1_flip"],
                    "delta_r1_flip": cue_pair["r1_flip"] - hm_pair["r1_flip"],
                    "rank_shift": cue_pair["rank_shift"],
                    "hm_rank_shift": hm_pair["rank_shift"],
                    "delta_rank_shift": cue_pair["rank_shift"] - hm_pair["rank_shift"],
                    "ap_delta": cue_pair["ap_delta"],
                    "hm_ap_delta": hm_pair["ap_delta"],
                    "delta_ap_delta": cue_pair["ap_delta"] - hm_pair["ap_delta"],
                    "cue_shift": shift_info["cue_shift"],
                    "score_mode": score_mode,
                    "lambda_global": args.lambda_global,
                    "seed": args.seed,
                }
            )
            if args.save_galleries:
                for gallery_type, gallery_indices in {**build.galleries, **hm_build.galleries}.items():
                    gallery_row = {
                        "dataset": args.dataset,
                        "retriever_name": args.retriever_name,
                        "case_id": case_id,
                        "query_id": query_id,
                        "pid": query.pid,
                        "trial_id": trial_id,
                        "gallery_type": gallery_type,
                        "image_ids": gallery_indices.astype(int).tolist(),
                    }
                    if args.save_image_paths:
                        gallery_row["image_paths"] = [split.gallery_paths[int(index)] for index in gallery_indices]
                    gallery_rows.append(gallery_row)

        if selected_index == 1 or selected_index % 50 == 0 or selected_index == len(selected_queries):
            now = time.perf_counter()
            interval_seconds = max(now - last_progress_time, 1e-9)
            interval_queries = selected_index - last_progress_index
            attempted_so_far = sum(int(stats["attempted_pairs"]) for stats in constructibility.values())
            valid_so_far = sum(int(stats["valid_pairs"]) for stats in constructibility.values())
            skip_counts_so_far = Counter()
            all_candidate_shifts: list[float] = []
            for stats in constructibility.values():
                skip_counts_so_far.update({str(key): int(value) for key, value in stats["skip_reasons"].items()})
                all_candidate_shifts.extend(float(value) for value in stats["candidate_cue_shifts"])
            logger.info(
                "Progress selected_queries=%d/%d attempted_pairs=%d valid_pairs=%d "
                "score_cache_queries=%d elapsed=%.1fs interval=%.1fs qps=%.2f "
                "skip_counts=[%s] candidate_cue_shift=[%s]",
                selected_index,
                len(selected_queries),
                attempted_so_far,
                valid_so_far,
                len(full_score_cache),
                now - progress_started,
                interval_seconds,
                interval_queries / interval_seconds,
                format_counts(skip_counts_so_far),
                format_description(describe_values(all_candidate_shifts)),
            )
            last_progress_time = now
            last_progress_index = selected_index

    constructibility_rows = []
    for case in cases:
        case_id = str(case["case_id"])
        stats = constructibility[case_id]
        attempted = int(stats["attempted_pairs"])
        valid = int(stats["valid_pairs"])
        candidate_shift_stats = describe_values([float(value) for value in stats["candidate_cue_shifts"]])
        valid_shift_stats = describe_values([float(value) for value in stats["valid_cue_shifts"]])
        constructibility_rows.append(
            {
                "dataset": args.dataset,
                "retriever_name": args.retriever_name,
                "case_id": case_id,
                "cue_a": case["cue_a"],
                "cue_b": case["cue_b"],
                "attempted_pairs": attempted,
                "valid_pairs": valid,
                "valid_pair_rate": float(valid / attempted) if attempted else 0.0,
                "candidate_cue_shift_count": candidate_shift_stats["count"],
                "candidate_cue_shift_mean": candidate_shift_stats["mean"],
                "candidate_cue_shift_min": candidate_shift_stats["min"],
                "candidate_cue_shift_p50": candidate_shift_stats["p50"],
                "candidate_cue_shift_p75": candidate_shift_stats["p75"],
                "candidate_cue_shift_p90": candidate_shift_stats["p90"],
                "candidate_cue_shift_max": candidate_shift_stats["max"],
                "valid_cue_shift_count": valid_shift_stats["count"],
                "mean_cue_shift": valid_shift_stats["mean"],
                "valid_cue_shift_min": valid_shift_stats["min"],
                "valid_cue_shift_p50": valid_shift_stats["p50"],
                "valid_cue_shift_p75": valid_shift_stats["p75"],
                "valid_cue_shift_p90": valid_shift_stats["p90"],
                "valid_cue_shift_max": valid_shift_stats["max"],
                "skip_reasons_json": jsonable(dict(stats["skip_reasons"])),
            }
        )

    global_skip_counts = Counter()
    global_candidate_shifts: list[float] = []
    global_valid_shifts: list[float] = []
    for stats in constructibility.values():
        global_skip_counts.update({str(key): int(value) for key, value in stats["skip_reasons"].items()})
        global_candidate_shifts.extend(float(value) for value in stats["candidate_cue_shifts"])
        global_valid_shifts.extend(float(value) for value in stats["valid_cue_shifts"])
    logger.info("Final construction skip_counts=[%s]", format_counts(global_skip_counts, limit=12))
    logger.info("Final candidate Cue Shift stats=[%s]", format_description(describe_values(global_candidate_shifts)))
    logger.info("Final valid Cue Shift stats=[%s]", format_description(describe_values(global_valid_shifts)))
    if not paired_delta_rows:
        logger.warning(
            "No valid paired results were produced. Most common skips=[%s]. "
            "If below_min_pair_cue_shift dominates, reduce --min_pair_cue_shift or inspect "
            "cue_case_constructibility.csv candidate_cue_shift_* columns.",
            format_counts(global_skip_counts, limit=5),
        )

    paired_cue_df = pd.DataFrame(paired_cue_rows)
    paired_hm_df = pd.DataFrame(paired_hm_rows)
    paired_delta_df = pd.DataFrame(paired_delta_rows, columns=PAIRED_DELTA_COLUMNS)
    logger.info(
        "Building summary tables from paired rows cue=%d hardness=%d delta=%d",
        len(paired_cue_df),
        len(paired_hm_df),
        len(paired_delta_df),
    )
    summary_started = time.perf_counter()
    summary_by_case, summary_overall, validity_counts = summarize_outputs(
        args.dataset,
        args.retriever_name,
        args.cue_scorer,
        ref_r1,
        len(case_candidate_rows),
        selected_queries,
        paired_cue_df,
        paired_hm_df,
        paired_delta_df,
    )
    logger.info("Summary tables built in %.2fs", time.perf_counter() - summary_started)
    if validity_counts:
        validity_counts[0]["expected_pairs"] = int(len(selected_queries) * args.num_trials)
        validity_counts[0]["valid_pair_rate"] = (
            float(validity_counts[0]["valid_pairs"] / validity_counts[0]["expected_pairs"])
            if validity_counts[0]["expected_pairs"]
            else 0.0
        )
        if not summary_overall.empty:
            summary_overall.loc[0, "valid_pair_rate"] = validity_counts[0]["valid_pair_rate"]

    summary_ci_by_unit: dict[str, pd.DataFrame] = {}
    for unit in requested_bootstrap_units(args.bootstrap_unit):
        cluster_cols = cluster_columns_for_unit(unit)
        count_summary = bootstrap_count_summary(paired_delta_df, cluster_cols)
        logger.info(
            "Starting %s: bootstrap_unit=%s unique_query_count=%d case_query_count=%d "
            "cluster_count=%d trial_count=%d bootstrap_iters=%d bootstrap_seed=%d",
            bootstrap_label(unit),
            unit,
            count_summary["unique_query_count"],
            count_summary["case_query_count"],
            count_summary["cluster_count"],
            count_summary["trial_count"],
            args.bootstrap_iters,
            args.bootstrap_seed,
        )
        bootstrap_started = time.perf_counter()
        summary_ci_by_unit[unit] = cluster_bootstrap_ci(
            paired_delta_df,
            SUMMARY_CI_METRICS,
            cluster_cols=cluster_cols,
            iters=args.bootstrap_iters,
            seed=args.bootstrap_seed,
            bootstrap_unit=unit,
        )
        logger.info(
            "%s finished in %.2fs rows=%d",
            bootstrap_label(unit),
            time.perf_counter() - bootstrap_started,
            len(summary_ci_by_unit[unit]),
        )
    primary_unit = primary_bootstrap_unit(args.bootstrap_unit)
    summary_ci = summary_ci_by_unit.get(
        primary_unit,
        pd.DataFrame(columns=SUMMARY_CI_COLUMNS),
    )
    logger.info("Primary CI output uses bootstrap_unit=%s", primary_unit)
    write_started = time.perf_counter()
    write_run_outputs(
        args.output_dir,
        selected_queries,
        constructibility_rows,
        validity_counts,
        per_gallery_rows,
        paired_cue_rows,
        paired_hm_rows,
        paired_delta_rows,
        summary_by_case,
        summary_overall,
        summary_ci,
        summary_ci_by_unit,
        skipped_rows,
        gallery_rows,
        args.save_galleries,
    )
    logger.info("Output writing finished in %.2fs", time.perf_counter() - write_started)
    logger.info(
        "Output rows selected=%d constructibility=%d per_gallery=%d paired_cue=%d paired_hardness=%d paired_delta=%d skipped=%d galleries=%d",
        len(selected_queries),
        len(constructibility_rows),
        len(per_gallery_rows),
        len(paired_cue_rows),
        len(paired_hm_rows),
        len(paired_delta_rows),
        len(skipped_rows),
        len(gallery_rows),
    )
    logger.info(
        "Key output files: %s, %s, %s, %s",
        args.output_dir / OUTPUT_FILES["constructibility"],
        args.output_dir / OUTPUT_FILES["paired_delta"],
        args.output_dir / OUTPUT_FILES["validity_counts"],
        args.output_dir / OUTPUT_FILES["skipped"],
    )
    log_validity_warnings(summary_overall, logger)

    print("\nWhole-test sanity metrics:")
    print(whole_df.to_string(index=False))
    print("\nSummary overall:")
    print(summary_overall.to_string(index=False))
    print("\nBootstrap CIs:")
    print(summary_ci.to_string(index=False))
    logger.info("Wrote diagnostic outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
