"""Shared constants and stable output schemas for cue-swap diagnostics."""

DATASET_NAMES = ("CUHK-PEDES", "ICFG-PEDES", "RSTPReid")
RETRIEVER_NAMES = ("rde",)
CUE_SCORERS = ("off_the_shelf_clip",)

CUE_GALLERY_TYPES = ("a_dense", "b_dense")
HM_GALLERY_TYPES = ("hm_a", "hm_b")
ALL_GALLERY_TYPES = CUE_GALLERY_TYPES + HM_GALLERY_TYPES

DEFAULT_PROMPT_TEMPLATES = (
    "a photo of a pedestrian with {cue}",
    "a photo of a person wearing {cue}",
    "a person with {cue}",
    "a pedestrian showing {cue}",
)

OUTPUT_FILES = {
    "config": "config_used.json",
    "whole_test": "whole_test_metrics.csv",
    "thresholds": "cue_thresholds.csv",
    "case_candidates": "cue_case_candidates.csv",
    "constructibility": "cue_case_constructibility.csv",
    "selected_queries": "selected_queries.csv",
    "validity_counts": "validity_counts.csv",
    "per_gallery": "per_gallery_results.csv",
    "paired_cue": "paired_cue_swap_results.csv",
    "paired_hm": "paired_hardness_control_results.csv",
    "paired_delta": "paired_delta_results.csv",
    "summary_overall": "summary_overall.csv",
    "summary_by_case": "summary_by_case.csv",
    "summary_ci": "summary_with_ci.csv",
    "summary_ci_unique_query": "summary_with_ci_unique_query.csv",
    "summary_ci_case_query": "summary_with_ci_case_query.csv",
    "hardness_audit_ci": "hardness_audit_with_ci.csv",
    "tight_hardness_summary_ci": "tight_hardness_summary_with_ci.csv",
    "skipped": "skipped_queries.jsonl",
    "galleries": "galleries.jsonl",
}

CASE_QUERY_CLUSTER_COLS = ["dataset", "retriever_name", "case_id", "query_id"]
UNIQUE_QUERY_CLUSTER_COLS = ["dataset", "retriever_name", "query_id"]
BOOTSTRAP_UNITS = ("case_query", "unique_query", "both")
PRIMARY_BOOTSTRAP_UNIT_FOR_BOTH = "unique_query"

SELECTED_QUERY_COLUMNS = [
    "dataset",
    "case_id",
    "query_id",
    "query_text",
    "pid",
    "cue_a",
    "cue_b",
    "selection_method",
]

PER_GALLERY_COLUMNS = [
    "dataset",
    "retriever_name",
    "cue_scorer",
    "case_id",
    "query_id",
    "query_text",
    "pid",
    "cue_a",
    "cue_b",
    "trial_id",
    "construction_type",
    "gallery_type",
    "gallery_size",
    "num_positives",
    "num_distractors",
    "positive_ratio",
    "R1",
    "R5",
    "R10",
    "AP",
    "best_positive_rank",
    "best_positive_score",
    "max_negative_score",
    "positive_negative_margin",
    "negative_score_scale",
    "D_soft_a",
    "D_soft_b",
    "D_hard_a",
    "D_hard_b",
    "mean_retriever_score_distractors",
    "score_mode",
    "lambda_global",
    "seed",
]

PAIRED_CUE_COLUMNS = [
    "dataset",
    "retriever_name",
    "cue_scorer",
    "case_id",
    "query_id",
    "pid",
    "trial_id",
    "cue_a",
    "cue_b",
    "r1_flip",
    "rank_shift",
    "ap_delta",
    "cue_shift",
    "D_soft_a_a_dense",
    "D_soft_a_b_dense",
    "D_soft_b_a_dense",
    "D_soft_b_b_dense",
    "score_mode",
    "lambda_global",
    "seed",
]

PAIRED_HM_COLUMNS = [
    "dataset",
    "retriever_name",
    "cue_scorer",
    "case_id",
    "query_id",
    "pid",
    "trial_id",
    "cue_a",
    "cue_b",
    "hm_r1_flip",
    "hm_rank_shift",
    "hm_ap_delta",
    "hm_a_mean_score_cue_subset",
    "hm_a_mean_score_control_subset",
    "hm_a_std_score_cue_subset",
    "hm_a_std_score_control_subset",
    "hm_b_mean_score_cue_subset",
    "hm_b_mean_score_control_subset",
    "hm_b_std_score_cue_subset",
    "hm_b_std_score_control_subset",
    "score_mode",
    "lambda_global",
    "seed",
]

PAIRED_DELTA_COLUMNS = [
    "dataset",
    "retriever_name",
    "cue_scorer",
    "case_id",
    "query_id",
    "pid",
    "trial_id",
    "cue_a",
    "cue_b",
    "r1_flip",
    "hm_r1_flip",
    "delta_r1_flip",
    "rank_shift",
    "hm_rank_shift",
    "delta_rank_shift",
    "ap_delta",
    "hm_ap_delta",
    "delta_ap_delta",
    "cue_shift",
    "negative_score_scale",
    "a_signed_max_negative_gap",
    "b_signed_max_negative_gap",
    "a_signed_margin_gap",
    "b_signed_margin_gap",
    "mean_signed_max_negative_gap",
    "mean_signed_margin_gap",
    "mean_abs_max_negative_gap",
    "a_normalized_max_negative_gap",
    "b_normalized_max_negative_gap",
    "mean_normalized_max_negative_gap",
    "max_abs_normalized_max_negative_gap",
    "tight_hardness_match",
    "tight_hardness_z_tolerance",
    "score_mode",
    "lambda_global",
    "seed",
]

HARDNESS_AUDIT_METRICS = (
    "mean_signed_max_negative_gap",
    "mean_signed_margin_gap",
    "mean_abs_max_negative_gap",
    "mean_signed_normalized_max_negative_gap",
    "mean_max_abs_normalized_max_negative_gap",
)

TIGHT_HARDNESS_METRICS = (
    "r1_flip",
    "hm_r1_flip",
    "delta_r1_flip",
)

SUMMARY_CI_METRICS = (
    "r1_flip",
    "rank_shift",
    "hm_r1_flip",
    "hm_rank_shift",
    "delta_r1_flip",
    "delta_rank_shift",
    "cue_shift",
)

SUMMARY_CI_COLUMNS = [
    "metric",
    "mean",
    "ci_low",
    "ci_high",
    "bootstrap_iters",
    "bootstrap_unit",
    "cluster_count",
    "unique_query_count",
    "case_query_count",
    "trial_count",
]

HARDNESS_AUDIT_COLUMNS = [
    "retriever",
    *SUMMARY_CI_COLUMNS,
]

TIGHT_HARDNESS_SUMMARY_COLUMNS = [
    "retriever",
    "tight_hardness_z_tolerance",
    "tight_trial_count",
    "tight_unique_query_count",
    "tight_case_query_count",
    "tight_trial_rate",
    *SUMMARY_CI_COLUMNS,
]
