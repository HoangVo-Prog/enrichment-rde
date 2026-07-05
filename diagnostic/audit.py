"""Audit helpers and warnings for diagnostic validity."""

from __future__ import annotations

import logging

import pandas as pd


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

