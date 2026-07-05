"""Stable output writers for diagnostic runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from diagnostic.config import jsonable, write_json
from diagnostic.constants import (
    OUTPUT_FILES,
    PAIRED_CUE_COLUMNS,
    PAIRED_DELTA_COLUMNS,
    PAIRED_HM_COLUMNS,
    PER_GALLERY_COLUMNS,
    SELECTED_QUERY_COLUMNS,
    SUMMARY_CI_COLUMNS,
)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(dict(row)), sort_keys=True) + "\n")


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(columns))


def write_config_used(output_dir: Path, payload: Mapping[str, Any]) -> None:
    write_json(output_dir / OUTPUT_FILES["config"], dict(payload))


def write_case_candidates(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pd.DataFrame(list(rows)).to_csv(output_dir / OUTPUT_FILES["case_candidates"], index=False)


def write_thresholds(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pd.DataFrame(list(rows), columns=["cue", "threshold", "quantile", "cue_scorer"]).to_csv(
        output_dir / OUTPUT_FILES["thresholds"], index=False
    )


def write_whole_test(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    df.to_csv(output_dir / OUTPUT_FILES["whole_test"], index=False)
    return df


def write_run_outputs(
    output_dir: Path,
    selected_queries: Sequence[Mapping[str, Any]],
    constructibility_rows: Sequence[Mapping[str, Any]],
    validity_counts: Sequence[Mapping[str, Any]],
    per_gallery_rows: Sequence[Mapping[str, Any]],
    paired_cue_rows: Sequence[Mapping[str, Any]],
    paired_hm_rows: Sequence[Mapping[str, Any]],
    paired_delta_rows: Sequence[Mapping[str, Any]],
    summary_by_case: pd.DataFrame,
    summary_overall: pd.DataFrame,
    summary_ci: pd.DataFrame,
    summary_ci_by_unit: Mapping[str, pd.DataFrame],
    skipped_rows: Sequence[Mapping[str, Any]],
    gallery_rows: Sequence[Mapping[str, Any]],
    save_galleries: bool,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_df = frame(selected_queries, SELECTED_QUERY_COLUMNS)
    per_gallery_df = frame(per_gallery_rows, PER_GALLERY_COLUMNS)
    cue_df = frame(paired_cue_rows, PAIRED_CUE_COLUMNS)
    hm_df = frame(paired_hm_rows, PAIRED_HM_COLUMNS)
    delta_df = frame(paired_delta_rows, PAIRED_DELTA_COLUMNS)
    constructibility_df = pd.DataFrame(list(constructibility_rows))
    validity_df = pd.DataFrame(list(validity_counts))

    selected_df.to_csv(output_dir / OUTPUT_FILES["selected_queries"], index=False)
    constructibility_df.to_csv(output_dir / OUTPUT_FILES["constructibility"], index=False)
    validity_df.to_csv(output_dir / OUTPUT_FILES["validity_counts"], index=False)
    per_gallery_df.to_csv(output_dir / OUTPUT_FILES["per_gallery"], index=False)
    cue_df.to_csv(output_dir / OUTPUT_FILES["paired_cue"], index=False)
    hm_df.to_csv(output_dir / OUTPUT_FILES["paired_hm"], index=False)
    delta_df.to_csv(output_dir / OUTPUT_FILES["paired_delta"], index=False)
    summary_by_case.to_csv(output_dir / OUTPUT_FILES["summary_by_case"], index=False)
    summary_overall.to_csv(output_dir / OUTPUT_FILES["summary_overall"], index=False)
    summary_ci.reindex(columns=SUMMARY_CI_COLUMNS).to_csv(
        output_dir / OUTPUT_FILES["summary_ci"], index=False
    )
    if "unique_query" in summary_ci_by_unit:
        summary_ci_by_unit["unique_query"].reindex(columns=SUMMARY_CI_COLUMNS).to_csv(
            output_dir / OUTPUT_FILES["summary_ci_unique_query"], index=False
        )
    if "case_query" in summary_ci_by_unit:
        summary_ci_by_unit["case_query"].reindex(columns=SUMMARY_CI_COLUMNS).to_csv(
            output_dir / OUTPUT_FILES["summary_ci_case_query"], index=False
        )
    write_jsonl(output_dir / OUTPUT_FILES["skipped"], skipped_rows)
    if save_galleries:
        write_jsonl(output_dir / OUTPUT_FILES["galleries"], gallery_rows)
    else:
        write_jsonl(output_dir / OUTPUT_FILES["galleries"], [])
    return {
        "selected": selected_df,
        "per_gallery": per_gallery_df,
        "paired_cue": cue_df,
        "paired_hm": hm_df,
        "paired_delta": delta_df,
        "summary_by_case": summary_by_case,
        "summary_overall": summary_overall,
        "summary_ci": summary_ci,
    }


def summarize_outputs(
    dataset: str,
    retriever_name: str,
    cue_scorer: str,
    ref_r1: float,
    candidate_case_count: int,
    selected_queries: Sequence[Mapping[str, Any]],
    paired_cue_df: pd.DataFrame,
    paired_hm_df: pd.DataFrame,
    paired_delta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    num_queries = int(pd.DataFrame(list(selected_queries))["query_id"].nunique()) if selected_queries else 0
    valid_pairs = int(len(paired_cue_df))
    expected_pairs = max(1, len(selected_queries)) * (int(paired_cue_df["trial_id"].max()) + 1 if not paired_cue_df.empty else 1)
    valid_pair_rate = float(valid_pairs / expected_pairs) if expected_pairs else 0.0
    validity_counts = [
        {
            "dataset": dataset,
            "retriever_name": retriever_name,
            "cue_scorer": cue_scorer,
            "candidate_cases": int(candidate_case_count),
            "selected_queries": int(len(selected_queries)),
            "unique_queries": num_queries,
            "valid_pairs": valid_pairs,
            "expected_pairs": int(expected_pairs),
            "valid_pair_rate": valid_pair_rate,
        }
    ]

    if paired_delta_df.empty:
        summary_by_case = pd.DataFrame(
            columns=[
                "dataset",
                "retriever_name",
                "case_id",
                "num_queries",
                "num_pairs",
                "mean_cue_shift",
                "r1_flip",
                "rank_shift",
                "hm_r1_flip",
                "hm_rank_shift",
                "delta_r1_flip",
                "delta_rank_shift",
            ]
        )
        summary_overall = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "retriever_name": retriever_name,
                    "cue_scorer": cue_scorer,
                    "ref_R1": ref_r1,
                    "num_cases": 0,
                    "num_queries": num_queries,
                    "num_pairs": 0,
                    "valid_pair_rate": valid_pair_rate,
                    "mean_cue_shift": np.nan,
                    "r1_flip": np.nan,
                    "rank_shift": np.nan,
                    "hm_r1_flip": np.nan,
                    "hm_rank_shift": np.nan,
                    "delta_r1_flip": np.nan,
                    "delta_rank_shift": np.nan,
                }
            ]
        )
        return summary_by_case, summary_overall, validity_counts

    case_rows = []
    for case_id, group in paired_delta_df.groupby("case_id"):
        case_rows.append(
            {
                "dataset": dataset,
                "retriever_name": retriever_name,
                "case_id": case_id,
                "num_queries": int(group["query_id"].nunique()),
                "num_pairs": int(len(group)),
                "mean_cue_shift": float(group["cue_shift"].mean()),
                "r1_flip": float(group["r1_flip"].mean()),
                "rank_shift": float(group["rank_shift"].mean()),
                "hm_r1_flip": float(group["hm_r1_flip"].mean()),
                "hm_rank_shift": float(group["hm_rank_shift"].mean()),
                "delta_r1_flip": float(group["delta_r1_flip"].mean()),
                "delta_rank_shift": float(group["delta_rank_shift"].mean()),
            }
        )
    summary_by_case = pd.DataFrame(case_rows).sort_values("case_id")
    summary_overall = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "retriever_name": retriever_name,
                "cue_scorer": cue_scorer,
                "ref_R1": ref_r1,
                "num_cases": int(paired_delta_df["case_id"].nunique()),
                "num_queries": int(paired_delta_df["query_id"].nunique()),
                "num_pairs": int(len(paired_delta_df)),
                "valid_pair_rate": valid_pair_rate,
                "mean_cue_shift": float(paired_delta_df["cue_shift"].mean()),
                "r1_flip": float(paired_delta_df["r1_flip"].mean()),
                "rank_shift": float(paired_delta_df["rank_shift"].mean()),
                "hm_r1_flip": float(paired_delta_df["hm_r1_flip"].mean()),
                "hm_rank_shift": float(paired_delta_df["hm_rank_shift"].mean()),
                "delta_r1_flip": float(paired_delta_df["delta_r1_flip"].mean()),
                "delta_rank_shift": float(paired_delta_df["delta_rank_shift"].mean()),
            }
        ]
    )
    return summary_by_case, summary_overall, validity_counts
