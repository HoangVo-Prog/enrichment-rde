"""Cluster bootstrap confidence intervals over query clusters."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from diagnostic.constants import SUMMARY_CI_COLUMNS


def bootstrap_count_summary(df: pd.DataFrame, cluster_cols: Sequence[str]) -> dict[str, int]:
    if df.empty:
        return {
            "cluster_count": 0,
            "unique_query_count": 0,
            "case_query_count": 0,
            "trial_count": 0,
        }
    required_for_counts = {"dataset", "retriever_name", "query_id", "case_id"}
    missing = required_for_counts - set(df.columns)
    if missing:
        raise ValueError(f"Bootstrap count summary missing columns: {sorted(missing)}")
    missing_cluster = set(cluster_cols) - set(df.columns)
    if missing_cluster:
        raise ValueError(f"Bootstrap count summary missing cluster columns: {sorted(missing_cluster)}")
    return {
        "cluster_count": int(df[list(cluster_cols)].drop_duplicates().shape[0]),
        "unique_query_count": int(
            df[["dataset", "retriever_name", "query_id"]].drop_duplicates().shape[0]
        ),
        "case_query_count": int(
            df[["dataset", "retriever_name", "case_id", "query_id"]].drop_duplicates().shape[0]
        ),
        "trial_count": int(len(df)),
    }


def cluster_row_counts(df: pd.DataFrame, cluster_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(cluster_cols) + ["row_count"])
    return df.groupby(list(cluster_cols), dropna=False).size().reset_index(name="row_count")


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    metrics: Sequence[str],
    cluster_cols: Sequence[str],
    iters: int,
    seed: int,
    bootstrap_unit: str,
) -> pd.DataFrame:
    columns = SUMMARY_CI_COLUMNS
    if df.empty:
        return pd.DataFrame(columns=columns)

    missing = set(cluster_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Cluster bootstrap missing cluster columns: {sorted(missing)}")

    count_summary = bootstrap_count_summary(df, cluster_cols)
    cluster_codes = df.groupby(list(cluster_cols), dropna=False).ngroup().to_numpy()
    cluster_count = int(cluster_codes.max()) + 1 if len(cluster_codes) else 0
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        metric_values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(metric_values)
        values = metric_values[valid_mask]
        if values.size == 0:
            rows.append(
                {
                    "metric": metric,
                    "mean": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "bootstrap_iters": iters,
                    "bootstrap_unit": bootstrap_unit,
                    "cluster_count": cluster_count,
                    "unique_query_count": count_summary["unique_query_count"],
                    "case_query_count": count_summary["case_query_count"],
                    "trial_count": int(len(df)),
                }
            )
            continue

        valid_codes = cluster_codes[valid_mask]
        cluster_sums = np.bincount(valid_codes, weights=values, minlength=cluster_count).astype(float)
        cluster_counts = np.bincount(valid_codes, minlength=cluster_count).astype(float)
        sampled = rng.integers(0, cluster_count, size=(iters, cluster_count))
        sampled_sums = cluster_sums[sampled].sum(axis=1)
        sampled_counts = cluster_counts[sampled].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot = sampled_sums / sampled_counts
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "ci_low": float(np.nanpercentile(boot, 2.5)),
                "ci_high": float(np.nanpercentile(boot, 97.5)),
                "bootstrap_iters": iters,
                "bootstrap_unit": bootstrap_unit,
                "cluster_count": cluster_count,
                "unique_query_count": count_summary["unique_query_count"],
                "case_query_count": count_summary["case_query_count"],
                "trial_count": int(len(df)),
            }
        )
    return pd.DataFrame(rows, columns=columns)
