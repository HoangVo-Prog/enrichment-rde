"""Manual and automatic cue-case handling without retrieval-outcome filtering."""

from __future__ import annotations

import json
import logging
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from diagnostic.cue_ontology import CueSpec, contains_normalized_phrase, detect_cues_in_query, normalize_text
from diagnostic.data_loading import QueryRecord


class CaseValidationError(ValueError):
    """Raised when a cue-case file is malformed."""


def load_cases(cases_file: Path) -> list[dict[str, Any]]:
    import yaml

    with cases_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, list) or not data:
        raise CaseValidationError("Cue case file must contain a non-empty list")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_case in enumerate(data):
        if not isinstance(raw_case, dict):
            raise CaseValidationError(f"Case at index {index} must be a mapping")
        case = dict(raw_case)
        prefix = f"Case at index {index}"
        for field in ("case_id", "cue_a", "cue_b"):
            if not isinstance(case.get(field), str) or not str(case[field]).strip():
                raise CaseValidationError(f"{prefix} missing non-empty string field '{field}'")
            case[field] = str(case[field]).strip()
        if case["case_id"] in seen:
            raise CaseValidationError(f"Duplicate case_id '{case['case_id']}'")
        seen.add(case["case_id"])

        if "query_include_all" in case:
            values = case["query_include_all"]
            if not isinstance(values, list) or not values:
                raise CaseValidationError(f"{prefix} field 'query_include_all' must be a non-empty list")
            if not all(isinstance(value, str) and value.strip() for value in values):
                raise CaseValidationError(f"{prefix} field 'query_include_all' must contain non-empty strings")
            case["query_include_all"] = [value.strip() for value in values]
        if "query_ids" in case:
            values = case["query_ids"]
            if not isinstance(values, list) or not values or not all(isinstance(value, int) for value in values):
                raise CaseValidationError(f"{prefix} field 'query_ids' must be a non-empty integer list")
        if "query_regex" in case:
            if not isinstance(case["query_regex"], str) or not case["query_regex"].strip():
                raise CaseValidationError(f"{prefix} field 'query_regex' must be a non-empty string")
            try:
                re.compile(case["query_regex"], flags=re.IGNORECASE)
            except re.error as exc:
                raise CaseValidationError(f"{prefix} has invalid query_regex: {exc}") from exc
        for field in ("max_queries", "min_queries"):
            if field in case and (not isinstance(case[field], int) or case[field] <= 0):
                raise CaseValidationError(f"{prefix} field '{field}' must be a positive integer")
        cases.append(case)
    return cases


def case_needles(case: Mapping[str, Any]) -> list[str]:
    if "query_include_all" in case:
        needles = [normalize_text(value) for value in case["query_include_all"]]
    elif "query_regex" in case:
        needles = []
    else:
        needles = normalize_text(f"{case['cue_a']} {case['cue_b']}").split()
    return [needle for needle in needles if needle]


def generate_auto_cases(
    query_records: Sequence[QueryRecord],
    gallery_pids: np.ndarray,
    cue_specs: Sequence[CueSpec],
    min_queries: int,
    max_cases: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[str]]]:
    pids_with_gallery = set(int(pid) for pid in gallery_pids.tolist())
    detected_by_query = {
        record.query_id: detect_cues_in_query(record.text, cue_specs)
        for record in query_records
    }
    pair_to_query_ids: dict[tuple[str, str], list[int]] = {}
    for record in query_records:
        if int(record.pid) not in pids_with_gallery:
            continue
        cues = sorted(set(detected_by_query.get(record.query_id, [])))
        for cue_a, cue_b in combinations(cues, 2):
            pair_to_query_ids.setdefault(tuple(sorted((cue_a, cue_b))), []).append(record.query_id)

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for (cue_a, cue_b), query_ids in pair_to_query_ids.items():
        query_ids = sorted(set(int(query_id) for query_id in query_ids))
        case_id = f"{_slug(cue_a)}__{_slug(cue_b)}"
        supported = len(query_ids) >= min_queries
        rows.append(
            {
                "case_id": case_id,
                "cue_a": cue_a,
                "cue_b": cue_b,
                "num_queries": len(query_ids),
                "query_ids": json.dumps(query_ids),
                "meets_min_queries": supported,
                "kept_after_support": supported,
                "reason": "" if supported else "below_min_queries_per_auto_case",
            }
        )
        if supported:
            cases.append(
                {
                    "case_id": case_id,
                    "cue_a": cue_a,
                    "cue_b": cue_b,
                    "query_ids": query_ids,
                    "min_queries": min_queries,
                    "selection_method": "pre_registered_auto",
                    "auto_case_support": len(query_ids),
                }
            )
    cases.sort(key=lambda case: (-int(case["auto_case_support"]), str(case["case_id"])))
    if max_cases is not None:
        kept = {str(case["case_id"]) for case in cases[:max_cases]}
        cases = cases[:max_cases]
        for row in rows:
            if row["kept_after_support"] and row["case_id"] not in kept:
                row["kept_after_support"] = False
                row["reason"] = "capped_by_max_auto_cases"
    rows.sort(key=lambda row: (-int(row["num_queries"]), str(row["case_id"])))
    return cases, rows, detected_by_query


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", normalize_text(value))
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "cue"


def _positive_lookahead_fragments(pattern: str) -> list[str] | None:
    """Return fragments from generated (?=.*fragment) case regexes."""
    fragments: list[str] = []
    index = 0
    prefix = "(?=.*"
    while index < len(pattern):
        if not pattern.startswith(prefix, index):
            return None
        start = index + len(prefix)
        depth = 1
        escaped = False
        in_char_class = False
        cursor = start
        while cursor < len(pattern):
            char = pattern[cursor]
            if escaped:
                escaped = False
                cursor += 1
                continue
            if char == "\\":
                escaped = True
                cursor += 1
                continue
            if char == "[" and not in_char_class:
                in_char_class = True
                cursor += 1
                continue
            if char == "]" and in_char_class:
                in_char_class = False
                cursor += 1
                continue
            if not in_char_class:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        fragments.append(pattern[start:cursor])
                        index = cursor + 1
                        break
            cursor += 1
        else:
            return None
    return fragments or None


def select_queries_for_cases(
    dataset_name: str,
    cases: Sequence[Mapping[str, Any]],
    query_records: Sequence[QueryRecord],
    gallery_pids: np.ndarray,
    max_queries_per_case: int | None,
    logger: logging.Logger | None = None,
    progress_interval: int = 250,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    query_by_id = {record.query_id: record for record in query_records}
    normalized_by_query_id = {record.query_id: normalize_text(record.text) for record in query_records}
    pids_with_gallery = set(int(pid) for pid in gallery_pids.tolist())
    fragment_match_cache: dict[str, set[int]] = {}
    selection_started = time.perf_counter()
    last_progress_time = selection_started

    def matching_records_for_fragments(fragments: list[str]) -> list[QueryRecord] | None:
        matching_sets: list[set[int]] = []
        for fragment in fragments:
            if fragment not in fragment_match_cache:
                try:
                    compiled = re.compile(fragment, flags=re.IGNORECASE)
                except re.error:
                    return None
                fragment_match_cache[fragment] = {
                    record.query_id
                    for record in query_records
                    if compiled.search(normalized_by_query_id[record.query_id]) is not None
                }
            matching_sets.append(fragment_match_cache[fragment])
        if not matching_sets:
            return None
        candidate_ids = set.intersection(*matching_sets)
        return [record for record in query_records if record.query_id in candidate_ids]

    total_cases = len(cases)
    for case_index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        regex_pattern = str(case["query_regex"]) if "query_regex" in case else None
        regex_fragments = _positive_lookahead_fragments(regex_pattern) if regex_pattern is not None else None
        if "query_ids" in case:
            candidates = []
            for query_id in case["query_ids"]:
                record = query_by_id.get(int(query_id))
                if record is None:
                    skipped.append({"dataset": dataset_name, "case_id": case_id, "query_id": int(query_id), "reason": "query_id_not_in_split"})
                    continue
                candidates.append(record)
            selection_method = str(case.get("selection_method", "query_ids"))
        else:
            needles = case_needles(case)
            candidates = []
            source_records = (
                matching_records_for_fragments(regex_fragments)
                if regex_fragments is not None
                else None
            )
            regex = (
                re.compile(regex_pattern, flags=re.IGNORECASE)
                if regex_pattern is not None and source_records is None
                else None
            )
            records_to_scan = source_records if source_records is not None else query_records
            for record in records_to_scan:
                normalized = normalized_by_query_id[record.query_id]
                if not all(contains_normalized_phrase(normalized, needle) for needle in needles):
                    continue
                if source_records is None and regex is not None and regex.search(normalized) is None:
                    continue
                candidates.append(record)
            selection_method = "query_text_filter"

        valid = []
        for record in candidates:
            if int(record.pid) not in pids_with_gallery:
                skipped.append({"dataset": dataset_name, "case_id": case_id, "query_id": record.query_id, "reason": "query_pid_has_no_gallery_positive"})
                continue
            valid.append(record)
        limit = case.get("max_queries", max_queries_per_case)
        if limit is not None:
            valid = valid[: int(limit)]
        min_queries = int(case.get("min_queries", 1))
        if len(valid) < min_queries:
            skipped.append({"dataset": dataset_name, "case_id": case_id, "query_id": "", "reason": "case_below_min_queries_after_validation", "num_valid": len(valid), "min_queries": min_queries})
        for record in valid:
            selected.append(
                {
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "query_id": record.query_id,
                    "query_text": record.text,
                    "pid": record.pid,
                    "cue_a": case["cue_a"],
                    "cue_b": case["cue_b"],
                    "selection_method": selection_method,
                }
            )
        if logger is not None and (
            case_index == 1
            or case_index == total_cases
            or (progress_interval > 0 and case_index % progress_interval == 0)
        ):
            now = time.perf_counter()
            logger.info(
                "Query selection progress cases=%d/%d selected_queries=%d skipped_rows=%d "
                "cached_regex_fragments=%d elapsed=%.1fs interval=%.1fs",
                case_index,
                total_cases,
                len(selected),
                len(skipped),
                len(fragment_match_cache),
                now - selection_started,
                now - last_progress_time,
            )
            last_progress_time = now
    return selected, skipped
