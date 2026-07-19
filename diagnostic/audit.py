"""Audit helpers and warnings for diagnostic validity."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from diagnostic.bootstrap import bootstrap_count_summary, cluster_bootstrap_ci
from diagnostic.constants import (
    HARDNESS_AUDIT_COLUMNS,
    HARDNESS_AUDIT_METRICS,
    SUMMARY_CI_COLUMNS,
    TIGHT_HARDNESS_METRICS,
    TIGHT_HARDNESS_SUMMARY_COLUMNS,
    UNIQUE_QUERY_CLUSTER_COLS,
)


def empty_ci_rows(
    metrics: Sequence[str],
    *,
    bootstrap_iters: int,
    bootstrap_unit: str,
    cluster_count: int = 0,
    unique_query_count: int = 0,
    case_query_count: int = 0,
    trial_count: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "bootstrap_iters": int(bootstrap_iters),
                "bootstrap_unit": bootstrap_unit,
                "cluster_count": int(cluster_count),
                "unique_query_count": int(unique_query_count),
                "case_query_count": int(case_query_count),
                "trial_count": int(trial_count),
            }
            for metric in metrics
        ],
        columns=SUMMARY_CI_COLUMNS,
    )


def _infer_retriever_name(df: pd.DataFrame, retriever_name: str | None) -> str:
    if retriever_name is not None:
        return str(retriever_name)
    if "retriever_name" not in df.columns or df.empty:
        return ""
    values = [str(value) for value in df["retriever_name"].dropna().unique().tolist()]
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        return "mixed"
    return ""


def _attach_retriever_column(ci: pd.DataFrame, retriever_name: str) -> pd.DataFrame:
    ci = ci.reindex(columns=SUMMARY_CI_COLUMNS)
    ci.insert(0, "retriever", retriever_name)
    return ci.reindex(columns=HARDNESS_AUDIT_COLUMNS)


def _ensure_metric_rows(
    ci: pd.DataFrame,
    metrics: Sequence[str],
    *,
    bootstrap_iters: int,
    bootstrap_unit: str,
    counts: dict[str, int],
) -> pd.DataFrame:
    present_metrics = set(ci["metric"].tolist()) if "metric" in ci.columns else set()
    missing_metrics = [metric for metric in metrics if metric not in present_metrics]
    if missing_metrics:
        ci = pd.concat(
            [
                ci,
                empty_ci_rows(
                    missing_metrics,
                    bootstrap_iters=bootstrap_iters,
                    bootstrap_unit=bootstrap_unit,
                    **counts,
                ),
            ],
            ignore_index=True,
        )
    return ci


def _prepare_hardness_audit_frame(paired_delta_df: pd.DataFrame) -> pd.DataFrame:
    df = paired_delta_df.copy()
    if (
        "mean_normalized_max_negative_gap" not in df.columns
        and "a_normalized_max_negative_gap" in df.columns
        and "b_normalized_max_negative_gap" in df.columns
    ):
        a_gap = pd.to_numeric(df["a_normalized_max_negative_gap"], errors="coerce")
        b_gap = pd.to_numeric(df["b_normalized_max_negative_gap"], errors="coerce")
        df["mean_normalized_max_negative_gap"] = 0.5 * (a_gap + b_gap)
    if "mean_normalized_max_negative_gap" in df.columns:
        df["mean_signed_normalized_max_negative_gap"] = pd.to_numeric(
            df["mean_normalized_max_negative_gap"],
            errors="coerce",
        )
    else:
        df["mean_signed_normalized_max_negative_gap"] = np.nan
    if "max_abs_normalized_max_negative_gap" in df.columns:
        df["mean_max_abs_normalized_max_negative_gap"] = pd.to_numeric(
            df["max_abs_normalized_max_negative_gap"],
            errors="coerce",
        )
    else:
        df["mean_max_abs_normalized_max_negative_gap"] = np.nan
    return df


def build_hardness_audit_with_ci(
    paired_delta_df: pd.DataFrame,
    *,
    bootstrap_iters: int,
    bootstrap_seed: int,
    retriever_name: str | None = None,
) -> pd.DataFrame:
    resolved_retriever = _infer_retriever_name(paired_delta_df, retriever_name)
    audit_df = _prepare_hardness_audit_frame(paired_delta_df)
    if audit_df.empty:
        return _attach_retriever_column(
            empty_ci_rows(
                HARDNESS_AUDIT_METRICS,
                bootstrap_iters=bootstrap_iters,
                bootstrap_unit="unique_query",
            ),
            resolved_retriever,
        )
    counts = bootstrap_count_summary(audit_df, UNIQUE_QUERY_CLUSTER_COLS)
    ci = cluster_bootstrap_ci(
        audit_df,
        HARDNESS_AUDIT_METRICS,
        cluster_cols=UNIQUE_QUERY_CLUSTER_COLS,
        iters=bootstrap_iters,
        seed=bootstrap_seed,
        bootstrap_unit="unique_query",
    )
    ci = _ensure_metric_rows(
        ci,
        HARDNESS_AUDIT_METRICS,
        bootstrap_iters=bootstrap_iters,
        bootstrap_unit="unique_query",
        counts=counts,
    )
    return _attach_retriever_column(ci, resolved_retriever)


def build_tight_hardness_summary_with_ci(
    paired_delta_df: pd.DataFrame,
    *,
    retriever_name: str,
    tight_hardness_z_tolerance: float,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    all_valid_trial_count = int(len(paired_delta_df))
    if paired_delta_df.empty or "tight_hardness_match" not in paired_delta_df.columns:
        tight_df = paired_delta_df.iloc[0:0].copy()
    else:
        tight_mask = paired_delta_df["tight_hardness_match"].fillna(False).astype(bool)
        tight_df = paired_delta_df.loc[tight_mask].copy()

    tight_trial_count = int(len(tight_df))
    tight_trial_rate = float(tight_trial_count / all_valid_trial_count) if all_valid_trial_count else 0.0
    counts = bootstrap_count_summary(tight_df, UNIQUE_QUERY_CLUSTER_COLS)
    if tight_df.empty:
        ci = empty_ci_rows(
            TIGHT_HARDNESS_METRICS,
            bootstrap_iters=bootstrap_iters,
            bootstrap_unit="unique_query",
            **counts,
        )
    else:
        ci = cluster_bootstrap_ci(
            tight_df,
            TIGHT_HARDNESS_METRICS,
            cluster_cols=UNIQUE_QUERY_CLUSTER_COLS,
            iters=bootstrap_iters,
            seed=bootstrap_seed,
            bootstrap_unit="unique_query",
        )
        ci = _ensure_metric_rows(
            ci,
            TIGHT_HARDNESS_METRICS,
            bootstrap_iters=bootstrap_iters,
            bootstrap_unit="unique_query",
            counts=counts,
        )

    ci = ci.reindex(columns=SUMMARY_CI_COLUMNS)
    ci.insert(0, "tight_trial_rate", tight_trial_rate)
    ci.insert(0, "tight_case_query_count", counts["case_query_count"])
    ci.insert(0, "tight_unique_query_count", counts["unique_query_count"])
    ci.insert(0, "tight_trial_count", tight_trial_count)
    ci.insert(0, "tight_hardness_z_tolerance", float(tight_hardness_z_tolerance))
    ci.insert(0, "retriever", retriever_name)
    return ci.reindex(columns=TIGHT_HARDNESS_SUMMARY_COLUMNS)


def log_validity_warnings(
    summary_overall: pd.DataFrame,
    logger: logging.Logger,
    min_valid_pair_rate: float = 0.2,
    min_mean_cue_shift: float = 0.01,
) -> None:
    if summary_overall.empty:
        logger.warning("No summary rows were produced.")
        return
    row = summary_overall.iloc[0]
    if float(row.get("valid_pair_rate", 0.0)) < min_valid_pair_rate:
        logger.warning("Low valid-pair rate: %.4f", float(row.get("valid_pair_rate", 0.0)))
    cue_shift = row.get("mean_cue_shift")
    if pd.notna(cue_shift) and float(cue_shift) < min_mean_cue_shift:
        logger.warning("Weak mean Cue Shift: %.4f", float(cue_shift))

