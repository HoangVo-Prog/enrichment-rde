"""Shared gallery-conditioned query evaluation protocol.

This module is intentionally repo-neutral. Thin scripts in each repository pass a
small spec that tells the runner where the local model, dataset, and evaluator
modules live; the protocol, output schema, deterministic assignment, and summary
math stay identical across hosts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


PROTOCOL_NAME = "gallery_conditioned_query"
PROTOCOL_VERSION = "gallery_conditioned_query_v1"
DEFAULT_SPLIT_SEEDS = [20260711, 20260712, 20260713, 20260714, 20260715]
DEFAULT_BOOTSTRAP_SEED = 20260801
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DATASET_CHOICES = ("CUHK-PEDES", "ICFG-PEDES", "RSTPReid")
DATASET_NAME_ALIASES = {
    "cuhk-pedes": "CUHK-PEDES",
    "cuhk_pedes": "CUHK-PEDES",
    "cuhkpedes": "CUHK-PEDES",
    "icfg-pedes": "ICFG-PEDES",
    "icfg_pedes": "ICFG-PEDES",
    "icfgpedes": "ICFG-PEDES",
    "rstpreid": "RSTPReid",
    "rstp-reid": "RSTPReid",
    "rstp_reid": "RSTPReid",
}
DATASET_LAYOUTS = {
    "CUHK-PEDES": {
        "dataset_dir": "CUHK-PEDES",
        "annotation": "reid_raw.json",
        "image_dir": "imgs",
        "image_key": "file_path",
    },
    "ICFG-PEDES": {
        "dataset_dir": "ICFG-PEDES",
        "annotation": "ICFG-PEDES.json",
        "image_dir": "imgs",
        "image_key": "file_path",
    },
    "RSTPReid": {
        "dataset_dir": "RSTPReid",
        "annotation": "data_captions.json",
        "image_dir": "imgs",
        "image_key": "img_path",
    },
}


GATE_CONFIG_KEYS = {
    "target_enrichment",
    "enrichment_start",
    "enrichment_space",
    "top_m",
    "topm_rank_space",
    "topm_rank_lambda",
    "extractor_mode",
    "num_parts",
    "target_relative_space",
    "target_relative_num_clusters",
    "target_relative_cluster_method",
    "evidence_token_budget",
    "evidence_projection",
    "context_module",
    "mixer_dim",
    "mixer_depth",
    "mixer_hidden_part",
    "mixer_hidden_rank",
    "mixer_hidden_channel",
    "mixer_hidden_readout",
    "context_pooling",
    "mixer_context_pooling",
    "residual_gate",
    "gate_mode",
    "enrich_gamma",
    "residual_gate_hidden_dim",
    "lambda_ret",
    "tau",
    "recompute_level",
    "recompute_interval",
    "pool_interval",
}


@dataclass(frozen=True)
class RepoSpec:
    repository_name: str
    repo_kind: str
    code_root: Path
    default_host_model: str
    metrics_module: str
    logger_name: str
    supports_host_model: bool = False


@dataclass
class FeatureBundle:
    qids: Any
    gids: Any
    host_text: Any
    alt_text: Optional[Any]
    host_image: Any
    alt_image: Optional[Any]
    raw_target_cache: Dict[str, Any]
    query_ids: List[str]
    image_ids: List[str]
    text_batch_sizes: List[int]


def _repo_root_from_spec(spec: RepoSpec) -> Path:
    if spec.repository_name == "enrichment-rde":
        return spec.code_root.parent
    return spec.code_root


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if torch is not None and torch.is_tensor(value):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return [_json_safe(v) for v in value.detach().cpu().tolist()]
    if np is not None and isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[gallery-conditioned {}] {}".format(timestamp, message), flush=True)


def _should_log_progress(index: int, total: Optional[int], interval: int) -> bool:
    if index <= 1:
        return True
    if interval > 0 and index % interval == 0:
        return True
    return total is not None and index >= total


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _atomic_write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    text = "".join(json.dumps(_json_safe(row), sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _atomic_write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                safe_row = {}
                for key in keys:
                    value = _json_safe(row.get(key))
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, sort_keys=True)
                    safe_row[key] = value
                writer.writerow(safe_row)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _write_table(path: Path, rows: Sequence[Dict[str, Any]], fallbacks: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        frame = pd.DataFrame([_json_safe(row) for row in rows])
        tmp_path = path.with_name(path.name + ".tmp")
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception as error:
        fallback_path = path.with_suffix(path.suffix + ".csv.gz")
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fd, tmp_name = tempfile.mkstemp(prefix=fallback_path.name + ".", dir=str(fallback_path.parent))
        os.close(fd)
        try:
            with gzip.open(tmp_name, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    safe_row = {}
                    for key in keys:
                        value = _json_safe(row.get(key))
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value, sort_keys=True)
                        safe_row[key] = value
                    writer.writerow(safe_row)
            os.replace(tmp_name, fallback_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        fallbacks[str(path.name)] = (
            "Parquet unavailable; wrote {} because: {}".format(
                fallback_path.name,
                error,
            )
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _git_info(repo_root: Path) -> Dict[str, Any]:
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    status = _run_git(repo_root, "status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status or "",
    }


def _set_if_missing(args: SimpleNamespace, name: str, value: Any) -> None:
    if not hasattr(args, name):
        setattr(args, name, value)


def _load_train_defaults() -> Dict[str, Any]:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        get_args = importlib.import_module("utils.options").get_args
        return vars(get_args())
    finally:
        sys.argv = original_argv


def _load_config(config_path: Path) -> Tuple[SimpleNamespace, Dict[str, Any]]:
    load_train_configs = importlib.import_module("utils.iotools").load_train_configs
    defaults = _load_train_defaults()
    config = dict(load_train_configs(str(config_path)))
    merged = dict(defaults)
    merged.update(config)
    return SimpleNamespace(**merged), config


def _namespace_from_mapping(values: Dict[str, Any]) -> SimpleNamespace:
    defaults = _load_train_defaults()
    merged = dict(defaults)
    merged.update(dict(values))
    return SimpleNamespace(**merged)


def _load_host_args(base_checkpoint: Path) -> Tuple[SimpleNamespace, Dict[str, Any], str]:
    defaults = _load_train_defaults()
    metadata_values: Dict[str, Any] = {}
    metadata_source = "repository_defaults"
    checkpoint = _torch_load_checkpoint(base_checkpoint)
    if isinstance(checkpoint, dict):
        for key in ("args", "config", "config_args", "train_args", "model_args"):
            value = checkpoint.get(key)
            if isinstance(value, SimpleNamespace):
                metadata_values = dict(vars(value))
            elif isinstance(value, dict):
                metadata_values = dict(value)
            elif hasattr(value, "items"):
                metadata_values = dict(value.items())
            elif hasattr(value, "__dict__"):
                metadata_values = dict(vars(value))
            else:
                continue
            metadata_source = "checkpoint:{}".format(key)
            break

    merged = dict(defaults)
    merged.update(metadata_values)
    return SimpleNamespace(**merged), metadata_values, metadata_source


def _canonical_dataset_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in DATASET_CHOICES:
        return text
    return DATASET_NAME_ALIASES.get(text.lower())


def _dataset_validation_error(
    reason: str,
    selected_dataset: str,
    provided_root_dir: Path,
    resolved_dataset_dir: Path,
    expected_annotation: Path,
    expected_image_root: Path,
) -> ValueError:
    return ValueError(
        "\n".join(
            [
                "Dataset validation failed: {}".format(reason),
                "Selected dataset: {}".format(selected_dataset),
                "Provided root directory: {}".format(provided_root_dir),
                "Resolved dataset directory: {}".format(resolved_dataset_dir),
                "Expected annotation locations: {}".format(expected_annotation),
                "Expected image directory: {}".format(expected_image_root),
            ]
        )
    )


def _validate_gate_dataset_compatibility(gate_config_values: Dict[str, Any], requested_dataset: str) -> None:
    for key in ("dataset_name", "dataset", "data"):
        if key not in gate_config_values:
            continue
        configured = _canonical_dataset_name(gate_config_values.get(key))
        if configured is None:
            continue
        if configured != requested_dataset:
            raise ValueError(
                "GATE config dataset conflict: --data is {}, but --config contains {}={} "
                "({}). Use a GATE checkpoint/config trained for the requested dataset.".format(
                    requested_dataset,
                    key,
                    gate_config_values.get(key),
                    configured,
                )
            )


def _split_counts_from_annotation(annotation_path: Path, split: str) -> Tuple[int, int]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, list):
        raise ValueError("annotation JSON must contain a list of records")
    rows = [row for row in annotations if isinstance(row, dict) and str(row.get("split")) == split]
    gallery_count = len(rows)
    query_count = 0
    for row in rows:
        captions = row.get("captions")
        if isinstance(captions, list):
            query_count += len(captions)
    return query_count, gallery_count


def _resolve_dataset_root(requested_dataset: str, root_dir: Path, split: str) -> Dict[str, Any]:
    root_dir = root_dir.expanduser().resolve()
    layout = DATASET_LAYOUTS[requested_dataset]
    dataset_dir_name = layout["dataset_dir"]

    if not root_dir.exists():
        candidate = root_dir / dataset_dir_name
        raise _dataset_validation_error(
            "provided --root-dir does not exist",
            requested_dataset,
            root_dir,
            candidate,
            candidate / layout["annotation"],
            candidate / layout["image_dir"],
        )
    if not root_dir.is_dir():
        candidate = root_dir / dataset_dir_name
        raise _dataset_validation_error(
            "provided --root-dir is not a directory",
            requested_dataset,
            root_dir,
            candidate,
            candidate / layout["annotation"],
            candidate / layout["image_dir"],
        )

    parent_convention_dir = root_dir / dataset_dir_name
    parent_annotation = parent_convention_dir / layout["annotation"]
    parent_image_root = parent_convention_dir / layout["image_dir"]
    direct_annotation = root_dir / layout["annotation"]
    direct_image_root = root_dir / layout["image_dir"]

    if parent_annotation.is_file() and parent_image_root.is_dir():
        loader_root = root_dir
        dataset_dir = parent_convention_dir
        annotation_path = parent_annotation
        image_root = parent_image_root
        dataset_dir_override = None
        root_convention = "parent_contains_dataset_dir"
    elif direct_annotation.is_file() and direct_image_root.is_dir():
        loader_root = root_dir.parent
        dataset_dir = root_dir
        annotation_path = direct_annotation
        image_root = direct_image_root
        dataset_dir_override = str(dataset_dir) if root_dir.name != dataset_dir_name else None
        root_convention = "direct_dataset_dir"
    else:
        raise _dataset_validation_error(
            "expected annotation file and image directory were not found",
            requested_dataset,
            root_dir,
            parent_convention_dir,
            parent_annotation,
            parent_image_root,
        )

    try:
        query_count, gallery_count = _split_counts_from_annotation(annotation_path, split)
    except Exception as error:
        raise _dataset_validation_error(
            "could not read {} split from annotation: {}".format(split, error),
            requested_dataset,
            root_dir,
            dataset_dir,
            annotation_path,
            image_root,
        ) from error
    if query_count <= 0 or gallery_count <= 0:
        raise _dataset_validation_error(
            "{} split has no valid queries or gallery images".format(split),
            requested_dataset,
            root_dir,
            dataset_dir,
            annotation_path,
            image_root,
        )

    return {
        "requested_dataset": requested_dataset,
        "resolved_dataset_name": requested_dataset,
        "provided_root_dir": str(root_dir),
        "resolved_root_dir": str(loader_root.resolve()),
        "resolved_dataset_dir": str(dataset_dir.resolve()),
        "resolved_annotation_path": str(annotation_path.resolve()),
        "resolved_image_root": str(image_root.resolve()),
        "dataset_split": split,
        "number_of_queries": int(query_count),
        "number_of_gallery_images": int(gallery_count),
        "root_convention": root_convention,
        "dataset_dir_override": dataset_dir_override,
    }


def _apply_gate_config_overlay(host_args: SimpleNamespace, gate_args: SimpleNamespace, gate_config_values: Dict[str, Any]) -> Tuple[SimpleNamespace, List[str]]:
    applied_keys: List[str] = []
    gate_keys_from_file = set(gate_config_values.keys())
    for key in sorted(GATE_CONFIG_KEYS):
        if key in gate_keys_from_file and hasattr(gate_args, key):
            setattr(host_args, key, getattr(gate_args, key))
            applied_keys.append(key)

    if "gate_mode" in gate_keys_from_file and hasattr(gate_args, "gate_mode"):
        host_args.residual_gate = getattr(gate_args, "gate_mode")
        applied_keys.append("residual_gate<-gate_mode")
    if "residual_gate" in gate_keys_from_file and hasattr(gate_args, "residual_gate"):
        host_args.gate_mode = getattr(gate_args, "residual_gate")
        applied_keys.append("gate_mode<-residual_gate")
    if "mixer_context_pooling" in gate_keys_from_file and hasattr(gate_args, "mixer_context_pooling"):
        host_args.context_pooling = getattr(gate_args, "mixer_context_pooling")
        applied_keys.append("context_pooling<-mixer_context_pooling")
    if "context_pooling" in gate_keys_from_file and hasattr(gate_args, "context_pooling"):
        host_args.mixer_context_pooling = getattr(gate_args, "context_pooling")
        applied_keys.append("mixer_context_pooling<-context_pooling")

    host_args.target_enrichment = True
    if "target_enrichment" in applied_keys:
        applied_keys.append("target_enrichment forced True for evaluation")
    else:
        applied_keys.append("target_enrichment=True")
    return host_args, applied_keys


def _ensure_eval_defaults(args: SimpleNamespace, spec: RepoSpec, cli_args: argparse.Namespace) -> None:
    _set_if_missing(args, "test_batch_size", 512)
    _set_if_missing(args, "batch_size", getattr(args, "test_batch_size", 512))
    _set_if_missing(args, "num_workers", 4)
    _set_if_missing(args, "seed", 1)
    _set_if_missing(args, "deterministic", True)
    _set_if_missing(args, "deterministic_warn_only", False)
    _set_if_missing(args, "training", False)
    _set_if_missing(args, "distributed", False)
    _set_if_missing(args, "local_rank", 0)
    _set_if_missing(args, "pretrain_choice", "ViT-B/16")
    _set_if_missing(args, "img_size", (384, 128))
    _set_if_missing(args, "stride_size", 16)
    _set_if_missing(args, "temperature", 0.02)
    _set_if_missing(args, "tau", 0.015)
    _set_if_missing(args, "target_enrichment", True)
    _set_if_missing(args, "enrichment_space", "global")
    _set_if_missing(args, "top_m", 64)
    _set_if_missing(args, "topm_rank_space", "host_global")
    _set_if_missing(args, "topm_rank_lambda", 0.5)
    _set_if_missing(args, "extractor_mode", "global,horizontal")
    _set_if_missing(args, "num_parts", 6)
    _set_if_missing(args, "target_relative_space", "host_global")
    _set_if_missing(args, "target_relative_num_clusters", 16)
    _set_if_missing(args, "target_relative_cluster_method", "kmeans")
    _set_if_missing(args, "evidence_projection", "auto")
    _set_if_missing(args, "context_module", "mixer")
    _set_if_missing(args, "mixer_dim", 256)
    _set_if_missing(args, "mixer_depth", 2)
    _set_if_missing(args, "mixer_hidden_part", 32)
    _set_if_missing(args, "mixer_hidden_rank", 64)
    _set_if_missing(args, "mixer_hidden_channel", 512)
    _set_if_missing(args, "mixer_hidden_readout", 128)
    _set_if_missing(args, "context_pooling", "mlp")
    _set_if_missing(args, "residual_gate", "residual")
    _set_if_missing(args, "enrich_gamma", None)
    _set_if_missing(args, "residual_gate_hidden_dim", 128)
    _set_if_missing(args, "lambda_ret", 1.0)
    _set_if_missing(args, "eval_score_chunk_size", 0)
    _set_if_missing(args, "target_cache_batch_size", getattr(args, "test_batch_size", 512))
    _set_if_missing(args, "target_query_batch_size", getattr(args, "test_batch_size", 512))

    dataset_resolution = _resolve_dataset_root(
        cli_args.data,
        Path(cli_args.root_dir),
        cli_args.split,
    )
    args.dataset_name = cli_args.data
    args.data = cli_args.data
    args.root_dir = dataset_resolution["resolved_root_dir"]
    args.requested_dataset = dataset_resolution["requested_dataset"]
    args.resolved_dataset_name = dataset_resolution["resolved_dataset_name"]
    args.provided_root_dir = dataset_resolution["provided_root_dir"]
    args.resolved_root_dir = dataset_resolution["resolved_root_dir"]
    args.resolved_dataset_dir = dataset_resolution["resolved_dataset_dir"]
    args.resolved_annotation_path = dataset_resolution["resolved_annotation_path"]
    args.resolved_image_root = dataset_resolution["resolved_image_root"]
    args.dataset_split = dataset_resolution["dataset_split"]
    args.number_of_queries = dataset_resolution["number_of_queries"]
    args.number_of_gallery_images = dataset_resolution["number_of_gallery_images"]
    args.dataset_root_convention = dataset_resolution["root_convention"]
    args.dataset_dir_override = dataset_resolution["dataset_dir_override"]
    args._dataset_resolution = dataset_resolution

    if isinstance(args.img_size, list):
        args.img_size = tuple(args.img_size)
    args.training = False
    args.distributed = False
    args.target_enrichment = True
    args.num_workers = int(cli_args.num_workers)
    args.test_batch_size = int(cli_args.query_batch_size)
    args.batch_size = int(cli_args.query_batch_size)
    args.target_query_batch_size = int(cli_args.query_batch_size)
    args.eval_score_chunk_size = int(cli_args.gallery_chunk_size)
    if spec.repo_kind == "prototype" and cli_args.host_model == "clip":
        args.only_global = True
        args.return_all = False
        args.topm_rank_space = "host_global"
        if getattr(args, "enrichment_space", "global") == "grab":
            args.enrichment_space = "global"
    if spec.repo_kind == "adapter":
        args.only_global = True
        if getattr(args, "enrichment_space", "global") == "grab":
            raise ValueError("DM-Adapter host in enrichment-rde-adapter exposes only global target enrichment")
        if getattr(args, "topm_rank_space", "host_global") == "hybrid_global_grab":
            raise ValueError("DM-Adapter host in enrichment-rde-adapter exposes only global top-M ranking")


def _configure_reproducibility(seed: int, deterministic: bool = True, warn_only: bool = False) -> None:
    import numpy as np
    import torch

    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(warn_only))
        except TypeError:
            torch.use_deterministic_algorithms(bool(deterministic))


def _resolve_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def _torch_load_checkpoint(path: Path) -> Any:
    import torch

    try:
        return torch.load(str(path), map_location=torch.device("cpu"))
    except RuntimeError:
        return torch.jit.load(str(path), map_location=torch.device("cpu")).state_dict()


def _unwrap_state_dict(checkpoint: Any) -> Tuple[Dict[str, Any], Optional[str], List[str]]:
    if isinstance(checkpoint, dict):
        top_level_keys = [str(key) for key in checkpoint.keys()]
        for key in ("model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, key, top_level_keys
        return checkpoint, None, top_level_keys
    raise TypeError("Checkpoint must be a state dict or contain a model/state_dict entry")


def _strip_module_prefix(key: str) -> Tuple[str, Optional[str]]:
    if key.startswith("module."):
        return key[len("module."):], "module."
    return key, None


def _infer_num_classes_from_state_dict(state_dict: Dict[str, Any], configured: Optional[int] = None) -> int:
    if configured is not None:
        return int(configured)
    for key, value in state_dict.items():
        norm_key, _ = _strip_module_prefix(str(key))
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != 2:
            continue
        if norm_key.endswith("classifier_global.weight"):
            return max(0, int(shape[0]) - 1)
        if norm_key.endswith("classifier.weight"):
            return int(shape[0])
    return 0


def _load_selected_state(
    model: Any,
    checkpoint_path: Path,
    role: str,
    include_model_key: Callable[[str], bool],
    allow_unexpected_key: Callable[[str], bool],
) -> Dict[str, Any]:
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    state_dict, state_dict_key, top_level_keys = _unwrap_state_dict(checkpoint)
    model_state = model.state_dict()
    loaded_keys: List[str] = []
    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    shape_mismatches: List[Dict[str, Any]] = []
    prefix_transformations: List[Dict[str, str]] = []
    mapped: Dict[str, Any] = {}

    normalized_items: List[Tuple[str, str, Any]] = []
    for raw_key, value in state_dict.items():
        key, stripped = _strip_module_prefix(str(raw_key))
        if stripped is not None:
            prefix_transformations.append({"from": str(raw_key), "to": key})
        normalized_items.append((str(raw_key), key, value))

    has_prefixed_gate = any(key.startswith("target_enricher.") for _, key, _ in normalized_items)
    for raw_key, key, value in normalized_items:
        candidate_keys = [key]
        if role == "gate" and not has_prefixed_gate:
            candidate_keys.insert(0, "target_enricher." + key)
        target_key = None
        for candidate in candidate_keys:
            if candidate in model_state and include_model_key(candidate):
                target_key = candidate
                break
        if target_key is None:
            suffix_matches = [
                model_key
                for model_key in model_state.keys()
                if include_model_key(model_key) and model_key.endswith(key)
            ]
            if len(suffix_matches) == 1:
                target_key = suffix_matches[0]
                prefix_transformations.append({"from": key, "to": target_key, "mode": "suffix_match"})
        if target_key is None:
            if allow_unexpected_key(key):
                continue
            unexpected_keys.append(key)
            continue
        if tuple(getattr(model_state[target_key], "shape", ())) != tuple(getattr(value, "shape", ())):
            shape_mismatches.append(
                {
                    "checkpoint_key": raw_key,
                    "model_key": target_key,
                    "checkpoint_shape": list(getattr(value, "shape", ())),
                    "model_shape": list(getattr(model_state[target_key], "shape", ())),
                }
            )
            continue
        mapped[target_key] = value
        loaded_keys.append(target_key)
        if target_key != key:
            prefix_transformations.append({"from": key, "to": target_key})

    for key in model_state.keys():
        if include_model_key(key) and key not in mapped:
            missing_keys.append(key)

    report = {
        "role": role,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "top_level_keys": top_level_keys,
        "state_dict_key_used": state_dict_key,
        "loaded_key_count": len(loaded_keys),
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "prefix_transformations": prefix_transformations,
        "strict": True,
    }
    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(
            "Strict {} checkpoint load failed for {}: {} missing, {} unexpected, {} shape mismatches".format(
                role,
                checkpoint_path,
                len(missing_keys),
                len(unexpected_keys),
                len(shape_mismatches),
            )
        )

    next_state = dict(model_state)
    next_state.update(mapped)
    model.load_state_dict(next_state, strict=True)
    return report


def _load_base_and_gate(model: Any, base_checkpoint: Path, gate_checkpoint: Path) -> List[Dict[str, Any]]:
    def base_include(key: str) -> bool:
        return "target_enricher" not in key

    def base_allow_unexpected(key: str) -> bool:
        return "target_enricher" in key

    def gate_include(key: str) -> bool:
        return key.startswith("target_enricher.")

    def gate_allow_unexpected(key: str) -> bool:
        if key.startswith("target_enricher."):
            return False
        return True

    reports = [
        _load_selected_state(model, base_checkpoint, "base", base_include, base_allow_unexpected),
        _load_selected_state(model, gate_checkpoint, "gate", gate_include, gate_allow_unexpected),
    ]
    return reports


def _freeze_and_eval(model: Any) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError("Evaluation model still has trainable parameters: {}".format(", ".join(trainable[:20])))


def _build_eval_loaders(args: SimpleNamespace, split: str) -> Tuple[Any, Any]:
    build_dataloader = importlib.import_module("datasets").build_dataloader
    dataset_class = None
    old_dataset_dir = None
    dataset_dir_override = getattr(args, "dataset_dir_override", None)
    if dataset_dir_override:
        try:
            build_module = importlib.import_module("datasets.build")
            factory = getattr(build_module, "__factory", None)
            if isinstance(factory, dict):
                dataset_class = factory.get(getattr(args, "dataset_name"))
                if dataset_class is not None:
                    old_dataset_dir = getattr(dataset_class, "dataset_dir", None)
                    dataset_class.dataset_dir = str(dataset_dir_override)
        except Exception as error:
            raise RuntimeError(
                "Could not apply direct-dataset --root-dir override for {}: {}".format(
                    getattr(args, "dataset_name", "unknown"),
                    error,
                )
            ) from error
    old_training = getattr(args, "training", False)
    old_val_dataset = getattr(args, "val_dataset", None)
    old_batch_size = getattr(args, "batch_size", None)
    try:
        if split == "test":
            args.training = False
            result = build_dataloader(args)
        else:
            args.training = True
            args.val_dataset = split
            args.batch_size = getattr(args, "test_batch_size", args.batch_size)
            result = build_dataloader(args)
        if not isinstance(result, (list, tuple)):
            raise TypeError("build_dataloader returned non-sequence {}".format(type(result)))
        if split == "test":
            return result[0], result[1]
        return result[1], result[2]
    finally:
        args.training = old_training
        if old_val_dataset is not None:
            args.val_dataset = old_val_dataset
        if old_batch_size is not None:
            args.batch_size = old_batch_size
        if dataset_class is not None and old_dataset_dir is not None:
            dataset_class.dataset_dir = old_dataset_dir


def _validate_loaded_eval_loaders(img_loader: Any, txt_loader: Any, args: SimpleNamespace) -> None:
    image_dataset = getattr(img_loader, "dataset", None)
    text_dataset = getattr(txt_loader, "dataset", None)
    image_count = len(image_dataset) if image_dataset is not None and hasattr(image_dataset, "__len__") else 0
    query_count = len(text_dataset) if text_dataset is not None and hasattr(text_dataset, "__len__") else 0
    if image_count <= 0 or query_count <= 0:
        resolution = getattr(args, "_dataset_resolution", {})
        raise ValueError(
            "\n".join(
                [
                    "Dataset loader validation failed: official loader returned an empty split",
                    "Selected dataset: {}".format(getattr(args, "dataset_name", None)),
                    "Provided root directory: {}".format(resolution.get("provided_root_dir")),
                    "Resolved dataset directory: {}".format(resolution.get("resolved_dataset_dir")),
                    "Expected annotation locations: {}".format(resolution.get("resolved_annotation_path")),
                    "Expected image directory: {}".format(resolution.get("resolved_image_root")),
                ]
            )
        )
    resolution = getattr(args, "_dataset_resolution", {})
    expected_root = resolution.get("resolved_image_root")
    img_paths = getattr(image_dataset, "img_paths", None)
    if expected_root and img_paths:
        expected_path = Path(expected_root).resolve()
        first_path = Path(str(img_paths[0])).expanduser().resolve()
        try:
            first_path.relative_to(expected_path)
        except ValueError as error:
            raise ValueError(
                "\n".join(
                    [
                        "Dataset loader validation failed: loaded image path is outside the selected dataset image root",
                        "Selected dataset: {}".format(getattr(args, "dataset_name", None)),
                        "Provided root directory: {}".format(resolution.get("provided_root_dir")),
                        "Resolved dataset directory: {}".format(resolution.get("resolved_dataset_dir")),
                        "Expected annotation locations: {}".format(resolution.get("resolved_annotation_path")),
                        "Expected image directory: {}".format(resolution.get("resolved_image_root")),
                        "First loaded image path: {}".format(first_path),
                    ]
                )
            ) from error


def _core_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _needs_alt_features(spec: RepoSpec, args: SimpleNamespace) -> bool:
    if spec.repo_kind == "rde":
        return True
    if spec.repo_kind == "irra":
        return True
    if spec.repo_kind == "prototype":
        return (
            not bool(getattr(args, "only_global", False))
            or str(getattr(args, "enrichment_space", "global")).lower() in {"grab", "tse"}
            or str(getattr(args, "topm_rank_space", "host_global")).lower() in {"hybrid_global_grab", "hybrid_global_tse"}
        )
    return False


def _active_space(spec: RepoSpec, args: SimpleNamespace) -> str:
    space = str(getattr(args, "enrichment_space", "global")).lower()
    if spec.repo_kind == "irra":
        if space in {"grab", "retrieval"}:
            return "retrieval"
        return "global"
    if space == "tse":
        return "grab"
    return space


def _active_query_features(spec: RepoSpec, args: SimpleNamespace, host_text: Any, alt_text: Optional[Any]) -> Any:
    space = _active_space(spec, args)
    if space in {"grab", "retrieval"}:
        if alt_text is None:
            raise ValueError("Active enrichment space {} requires alternate text features".format(space))
        return alt_text
    return host_text


def _needs_alt_for_topm(args: SimpleNamespace) -> bool:
    return str(getattr(args, "topm_rank_space", "host_global")).lower() in {
        "hybrid_global_grab",
        "hybrid_global_tse",
        "hybrid_global_retrieval",
    }


def _canonical_image_ids(img_loader: Any, args: SimpleNamespace, repo_root: Path) -> List[str]:
    dataset = getattr(img_loader, "dataset", None)
    paths = list(getattr(dataset, "img_paths", []))
    if not paths:
        return ["image_{:06d}".format(index) for index in range(len(dataset))]

    bases = []
    for base in [getattr(args, "root_dir", None), repo_root]:
        if base:
            bases.append(Path(base).expanduser())
            if not Path(base).is_absolute():
                bases.append((repo_root / base).resolve())
    ids = []
    for index, path in enumerate(paths):
        path_obj = Path(path)
        if not path_obj.is_absolute():
            ids.append(path_obj.as_posix())
            continue
        chosen = None
        for base in bases:
            try:
                rel = path_obj.resolve().relative_to(base.resolve())
                chosen = rel.as_posix()
                break
            except Exception:
                continue
        ids.append(chosen or "{}:{:06d}".format(path_obj.name, index))
    return ids


def _query_ids(qids: Any) -> List[str]:
    qid_list = [int(v) for v in qids.detach().cpu().view(-1).tolist()]
    return ["query_{:06d}_pid_{}".format(index, pid) for index, pid in enumerate(qid_list)]


def _extract_features(
    spec: RepoSpec,
    model: Any,
    img_loader: Any,
    txt_loader: Any,
    args: SimpleNamespace,
    repo_root: Path,
    log_interval: int = 50,
) -> FeatureBundle:
    import torch

    model.eval()
    core = _core_model(model)
    device = next(core.parameters()).device
    include_alt = _needs_alt_features(spec, args)
    text_total = len(txt_loader) if hasattr(txt_loader, "__len__") else None
    image_total = len(img_loader) if hasattr(img_loader, "__len__") else None
    _log(
        "Encoding text features: batches={} include_alt={}".format(
            text_total if text_total is not None else "unknown",
            include_alt,
        )
    )

    qids = []
    host_text = []
    alt_text = []
    batch_sizes = []
    with torch.inference_mode():
        for batch_index, (pid, caption) in enumerate(txt_loader, start=1):
            caption = caption.to(device)
            if hasattr(core, "encode_eval_text_bundle") and spec.repo_kind != "rde":
                bundle = core.encode_eval_text_bundle(caption, include_grab=include_alt)
                host = bundle["host_features"]
                alt = bundle.get("grab_features")
            elif spec.repo_kind == "irra":
                host = core.encode_clip_global_text(caption)
                alt = core.encode_retrieval_text(caption) if include_alt else None
            else:
                host = core.encode_text(caption)
                alt = core.encode_text_grab(caption) if include_alt else None
            flat_pid = pid.view(-1)
            qids.append(flat_pid.cpu())
            host_text.append(host.detach().cpu())
            if include_alt:
                if alt is None:
                    raise ValueError("Alternate text feature branch was requested but not returned")
                alt_text.append(alt.detach().cpu())
            batch_sizes.append(int(flat_pid.numel()))
            if _should_log_progress(batch_index, text_total, log_interval):
                _log(
                    "Encoded text batch {}/{} (queries={})".format(
                        batch_index,
                        text_total if text_total is not None else "?",
                        sum(batch_sizes),
                    )
                )

    qids_t = torch.cat(qids, dim=0).cpu()
    host_text_t = torch.cat(host_text, dim=0).cpu()
    alt_text_t = torch.cat(alt_text, dim=0).cpu() if include_alt else None
    _log("Text feature extraction complete: queries={}".format(int(qids_t.numel())))

    gids = []
    host_image = []
    alt_image = []
    cache_chunks: List[Dict[str, Any]] = []
    image_seen = 0
    _log(
        "Encoding image/gallery features: batches={} include_alt={}".format(
            image_total if image_total is not None else "unknown",
            include_alt,
        )
    )
    with torch.inference_mode():
        for batch_index, (pid, image) in enumerate(img_loader, start=1):
            image = image.to(device)
            if hasattr(core, "encode_eval_image_bundle") and spec.repo_kind != "rde":
                bundle = core.encode_eval_image_bundle(
                    image,
                    include_grab=include_alt,
                    cache_target=True,
                    cache_prototypes=True,
                )
                host = bundle["host_features"]
                alt = bundle.get("grab_features")
                cache = bundle["target_cache"]
            elif spec.repo_kind == "irra":
                host = core.encode_clip_global_image(image)
                alt = core.encode_retrieval_image(image) if include_alt else None
                cache = core.encode_target_image_cache(image)
            else:
                host = core.encode_image(image)
                alt = core.encode_image_grab(image) if include_alt else None
                cache = core.encode_target_image_cache(image)
            gids.append(pid.view(-1).cpu())
            host_image.append(host.detach().cpu())
            if include_alt:
                if alt is None:
                    raise ValueError("Alternate image feature branch was requested but not returned")
                alt_image.append(alt.detach().cpu())
            cache_chunks.append({key: value.detach().cpu() for key, value in cache.items()})
            image_seen += int(pid.view(-1).numel())
            if _should_log_progress(batch_index, image_total, log_interval):
                _log(
                    "Encoded image batch {}/{} (gallery_images={})".format(
                        batch_index,
                        image_total if image_total is not None else "?",
                        image_seen,
                    )
                )

    gids_t = torch.cat(gids, dim=0).cpu()
    host_image_t = torch.cat(host_image, dim=0).cpu()
    alt_image_t = torch.cat(alt_image, dim=0).cpu() if include_alt else None
    raw_cache: Dict[str, Any] = {}
    for key in cache_chunks[0].keys():
        raw_cache[key] = torch.cat([chunk[key] for chunk in cache_chunks], dim=0).cpu()
    _log("Image feature extraction complete: gallery_images={}".format(int(gids_t.numel())))

    return FeatureBundle(
        qids=qids_t,
        gids=gids_t,
        host_text=host_text_t,
        alt_text=alt_text_t,
        host_image=host_image_t,
        alt_image=alt_image_t,
        raw_target_cache=raw_cache,
        query_ids=_query_ids(qids_t),
        image_ids=_canonical_image_ids(img_loader, args, repo_root),
        text_batch_sizes=batch_sizes,
    )


def _subset_and_finalize_cache(core: Any, raw_cache: Dict[str, Any], gids: Any, indices: Sequence[int], device: Any) -> Dict[str, Any]:
    import torch

    index_tensor = torch.as_tensor(list(indices), dtype=torch.long)
    cache: Dict[str, Any] = {}
    pool_size = int(gids.numel())
    for key, value in raw_cache.items():
        if torch.is_tensor(value) and value.shape[0] == pool_size:
            cache[key] = value.index_select(0, index_tensor).to(device)
        else:
            cache[key] = value.to(device) if torch.is_tensor(value) else value
    cache["pids"] = gids.index_select(0, index_tensor).to(device)
    if hasattr(core, "finalize_target_cache"):
        cache = core.finalize_target_cache(cache)
    return cache


def _cache_retrieval_features(cache: Dict[str, Any]) -> Any:
    import torch.nn.functional as F

    return F.normalize(cache["retrieval_features"].detach().cpu(), p=2, dim=1)


def _torch_normalize(tensor: Any) -> Any:
    import torch.nn.functional as F

    return F.normalize(tensor.float(), p=2, dim=-1)


def _score_matrix(query_features: Any, gallery_features: Any, chunk_size: int = 0) -> Any:
    if chunk_size is None or int(chunk_size) <= 0:
        return query_features @ gallery_features.t()
    chunks = []
    chunk_size = int(chunk_size)
    for start in range(0, query_features.shape[0], chunk_size):
        chunks.append(query_features[start:start + chunk_size] @ gallery_features.t())
    import torch

    return torch.cat(chunks, dim=0)


def _score_tensor_to_list(scores: Any) -> List[float]:
    return [float(v) for v in scores.detach().cpu().view(-1).tolist()]


def _scale_scores_like(scores: Any, reference: Any, eps: float = 1e-12) -> Any:
    score_min = scores.min(dim=1, keepdim=True).values
    score_max = scores.max(dim=1, keepdim=True).values
    ref_min = reference.min(dim=1, keepdim=True).values
    ref_max = reference.max(dim=1, keepdim=True).values
    score_range = (score_max - score_min).clamp_min(eps)
    ref_range = ref_max - ref_min
    return (scores - score_min) / score_range * ref_range + ref_min


def _format_lambda(value: float) -> str:
    if abs(float(value) - round(float(value))) < 1e-12:
        return str(int(round(float(value))))
    return "{:.2f}".format(float(value)).rstrip("0").rstrip(".")


def _prototype_lambdas() -> List[float]:
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _global_grab_lambdas() -> List[float]:
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.68, 0.32]


def _metric_task_name(task: str) -> str:
    task = str(task).replace("+", "_plus_")
    task = task.replace("(", "_").replace(")", "")
    task = task.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_/-]+", "_", task).strip("_")


def _task_metric_key(task: str, metric: str) -> str:
    return "eval/{}-t2i/{}".format(_metric_task_name(task), metric)


def _task_proto_lambda(task: str) -> Optional[float]:
    match = re.search(r"\(([-+]?\d*\.?\d+)\)$", str(task))
    return float(match.group(1)) if match else None


def _official_base_tasks(spec: RepoSpec, args: SimpleNamespace, sims_global: Any, sims_alt: Optional[Any]) -> List[Dict[str, Any]]:
    if spec.repo_kind == "rde":
        if sims_alt is None:
            raise ValueError("RDE official evaluation requires TSE/retrieval score components")
        return [
            {"task": "BGE", "scores": sims_global, "base_task": "BGE"},
            {"task": "TSE", "scores": sims_alt, "base_task": "TSE"},
            {"task": "BGE+TSE", "scores": (sims_global + sims_alt) / 2.0, "base_task": "BGE+TSE"},
        ]
    if spec.repo_kind == "irra":
        if sims_alt is None:
            raise ValueError("IRRA official evaluation requires retrieval score components")
        return [
            {"task": "global", "scores": sims_global, "base_task": "global"},
            {"task": "retrieval", "scores": sims_alt, "base_task": "retrieval"},
        ]
    if spec.repo_kind == "adapter":
        return [{"task": "global", "scores": sims_global, "base_task": "global"}]

    base_tasks = [{"task": "global", "scores": sims_global, "base_task": "global"}]
    if bool(getattr(args, "only_global", False)):
        return base_tasks
    if sims_alt is None:
        raise ValueError("Prototype official non-global evaluation requires GRAB/TSE score components")
    base_tasks.append({"task": "grab", "scores": sims_alt, "base_task": "grab"})
    scaled_alt = _scale_scores_like(sims_alt, sims_global)
    for lambda_value in _global_grab_lambdas():
        task = "global+grab({})".format(_format_lambda(lambda_value))
        base_tasks.append(
            {
                "task": task,
                "scores": float(lambda_value) * sims_global + (1.0 - float(lambda_value)) * scaled_alt,
                "base_task": task,
            }
        )
    return base_tasks


def _official_target_tasks(
    spec: RepoSpec,
    base_tasks: Sequence[Dict[str, Any]],
    sims_target: Any,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    if spec.repo_kind == "irra":
        base_reference = base_tasks[0] if base_tasks else {"task": "global", "scores": sims_target}
        tasks.append(
            {
                "task": "target+proto(1)",
                "scores": sims_target,
                "base_task": base_reference["task"],
                "base_scores": base_reference["scores"],
                "proto_lambda": 1.0,
            }
        )

    for proto_lambda in _prototype_lambdas():
        proto_value = _format_lambda(proto_lambda)
        for base in base_tasks:
            task = "{}+proto({})".format(base["task"], proto_value)
            if abs(proto_lambda) < 1e-12:
                scores = base["scores"]
            elif abs(proto_lambda - 1.0) < 1e-12 and spec.repo_kind in {"prototype", "adapter"}:
                scores = sims_target
            else:
                scaled_base = _scale_scores_like(base["scores"], sims_target)
                scores = (1.0 - float(proto_lambda)) * scaled_base + float(proto_lambda) * sims_target
            tasks.append(
                {
                    "task": task,
                    "scores": scores,
                    "base_task": base["task"],
                    "base_scores": base["scores"],
                    "proto_lambda": float(proto_lambda),
                }
            )
    return tasks


def _official_score_tasks(
    spec: RepoSpec,
    args: SimpleNamespace,
    sims_global: Any,
    sims_alt: Optional[Any],
    sims_target: Optional[Any] = None,
    include_base: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_tasks = _official_base_tasks(spec, args, sims_global, sims_alt)
    output = list(base_tasks) if include_base else []
    if sims_target is not None:
        output.extend(_official_target_tasks(spec, base_tasks, sims_target))
    return output, base_tasks


def _context_base_tasks(
    spec: RepoSpec,
    args: SimpleNamespace,
    host_query: Any,
    host_gallery: Any,
    alt_query: Optional[Any],
    alt_gallery: Optional[Any],
) -> List[Dict[str, Any]]:
    import torch.nn.functional as F

    sims_global = F.normalize(host_query.float(), p=2, dim=1) @ F.normalize(host_gallery.float(), p=2, dim=1).t()
    sims_alt = None
    if alt_query is not None and alt_gallery is not None:
        sims_alt = F.normalize(alt_query.float(), p=2, dim=1) @ F.normalize(alt_gallery.float(), p=2, dim=1).t()
    return _official_base_tasks(spec, args, sims_global, sims_alt)


def _scores_by_task(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(task["task"]): task for task in tasks}


def _score_ablation_description(spec: RepoSpec, args: SimpleNamespace) -> Dict[str, Any]:
    if spec.repo_kind == "rde":
        base_components = ["BGE", "TSE", "BGE+TSE=(BGE+TSE)/2"]
    elif spec.repo_kind == "irra":
        base_components = ["global", "retrieval"]
    elif spec.repo_kind == "adapter":
        base_components = ["global"]
    else:
        base_components = ["global"]
        if not bool(getattr(args, "only_global", False)):
            base_components.extend(
                ["grab", "global+grab(lambda)=lambda*global+(1-lambda)*scale(grab_like_global)"]
            )
    return {
        "base_components": base_components,
        "proto_lambdas": _prototype_lambdas(),
        "global_grab_lambdas": _global_grab_lambdas() if spec.repo_kind == "prototype" and not bool(getattr(args, "only_global", False)) else [],
        "scale_formula": "scale(scores_like_reference)=rowwise_minmax(scores) mapped to rowwise min/max(reference)",
        "proto_fusion_formula": "fused=(1-lambda)*scale(base_like_target)+lambda*target",
        "lambda_zero_context_canonicalization": "lambda=0 uses the exact base score tensor in decomposed context/control scoring so matched and mismatched tensors are exactly identical on the same gallery; this preserves official ranks/AP because the evaluator's scaled lambda=0 row is a positive row-wise affine transform of base",
        "official_best_policy": "select the best existing full-gallery +proto(lambda) ablation row by official R@1; do not select by Delta_ctx",
    }


def _best_task_from_metrics(
    spec: RepoSpec,
    task_metrics: Dict[str, Dict[str, float]],
    prefer_ablation: bool = False,
) -> Optional[str]:
    best_task = None
    best_r1 = None
    for task, metrics in task_metrics.items():
        if prefer_ablation and "+proto(" not in task:
            continue
        r1 = metrics.get("R1")
        if r1 is None:
            continue
        if best_r1 is None:
            best_task = task
            best_r1 = float(r1)
            continue
        if spec.repo_kind == "rde":
            is_better = float(r1) >= best_r1
        else:
            is_better = float(r1) > best_r1
        if is_better:
            best_task = task
            best_r1 = float(r1)
    return best_task


def _select_task_metrics(task_metrics: Dict[str, Dict[str, float]], task: Optional[str]) -> Dict[str, float]:
    if task and task in task_metrics:
        return task_metrics[task]
    if task_metrics:
        return next(iter(task_metrics.values()))
    return {}


def _compute_top_indices(
    spec: RepoSpec,
    core: Any,
    args: SimpleNamespace,
    target_cache: Dict[str, Any],
    query_features: Any,
    host_text_features: Any,
    alt_text_features: Optional[Any],
) -> Any:
    enricher = getattr(core, "target_enricher", None)
    if enricher is None:
        raise ValueError("Model has no target_enricher module")
    q = _torch_normalize(query_features.to(next(core.parameters()).device))
    host_text = _torch_normalize(host_text_features.to(next(core.parameters()).device))
    host_images = _torch_normalize(target_cache["host_image_features"])
    retrieval_images = _torch_normalize(target_cache["retrieval_features"])
    grab_text = None
    if alt_text_features is not None and (_needs_alt_for_topm(args) or _active_space(spec, args) in {"grab", "retrieval"}):
        grab_text = _torch_normalize(alt_text_features.to(next(core.parameters()).device))
    space = _active_space(spec, args)
    with __import__("torch").inference_mode():
        if hasattr(enricher, "_top_indices"):
            return enricher._top_indices(
                query_features=q,
                host_text_features=host_text,
                host_image_features=host_images,
                retrieval_features=retrieval_images,
                pool_cache=target_cache,
                space=space,
                grab_text_features=grab_text,
            ).detach().cpu()
        if hasattr(enricher, "_select_top_indices"):
            return enricher._select_top_indices(
                q,
                host_text,
                host_images,
                retrieval_images,
                target_cache,
                grab_text,
            ).detach().cpu()
    raise ValueError("Target enricher does not expose a top-M selection helper")


def _enrich_query_features(
    spec: RepoSpec,
    core: Any,
    args: SimpleNamespace,
    target_cache: Dict[str, Any],
    query_features: Any,
    host_text_features: Any,
    alt_text_features: Optional[Any],
    top_indices: Any,
) -> Any:
    import torch

    device = next(core.parameters()).device
    cache = dict(target_cache)
    cache["top_indices"] = top_indices.to(device)
    query = query_features.to(device)
    host = host_text_features.to(device)
    alt = alt_text_features.to(device) if alt_text_features is not None and (_needs_alt_for_topm(args) or _active_space(spec, args) in {"grab", "retrieval"}) else None
    with torch.inference_mode():
        try:
            enriched = core.enrich_text_features(
                query,
                host,
                cache,
                grab_text_features=alt,
            )
        except TypeError:
            enriched = core.enrich_text_features(
                query,
                host,
                cache,
                alt_text_features=alt,
            )
    return enriched.detach().cpu()


def _enrich_all_queries(
    spec: RepoSpec,
    core: Any,
    args: SimpleNamespace,
    target_cache: Dict[str, Any],
    active_queries: Any,
    host_text: Any,
    alt_text: Optional[Any],
    batch_size: int,
    stage_name: str = "all queries",
    log_interval: int = 50,
) -> Tuple[Any, List[Any]]:
    import torch

    chunks = []
    top_chunks = []
    total_queries = int(active_queries.shape[0])
    step = max(1, int(batch_size))
    total_batches = int(math.ceil(total_queries / float(step))) if total_queries else 0
    _log(
        "Enriching {}: queries={} batch_size={} batches={}".format(
            stage_name,
            total_queries,
            step,
            total_batches,
        )
    )
    for batch_index, start in enumerate(range(0, total_queries, step), start=1):
        end = min(total_queries, start + step)
        active = active_queries[start:end]
        host = host_text[start:end]
        alt = alt_text[start:end] if alt_text is not None else None
        top_indices = _compute_top_indices(spec, core, args, target_cache, active, host, alt)
        enriched = _enrich_query_features(spec, core, args, target_cache, active, host, alt, top_indices)
        chunks.append(enriched)
        top_chunks.append(top_indices)
        if _should_log_progress(batch_index, total_batches, log_interval):
            _log(
                "Enriched {} batch {}/{} (queries_done={}/{})".format(
                    stage_name,
                    batch_index,
                    total_batches,
                    end,
                    total_queries,
                )
            )
    return torch.cat(chunks, dim=0).cpu(), top_chunks


def _ranked_metrics_for_query(scores: Sequence[float], query_pid: int, gallery_pids: Sequence[int], image_ids: Sequence[str]) -> Dict[str, Any]:
    import torch

    order = [int(idx) for idx in torch.argsort(torch.as_tensor(scores, dtype=torch.float32), descending=True).tolist()]
    positives = [idx for idx, pid in enumerate(gallery_pids) if int(pid) == int(query_pid)]
    positive_set = set(positives)
    if not positive_set:
        raise ValueError("Query has no positive in evaluation gallery")
    hits = []
    ranks = []
    for rank_index, gallery_index in enumerate(order, start=1):
        is_hit = gallery_index in positive_set
        hits.append(is_hit)
        if is_hit:
            ranks.append(rank_index)
    cum_hits = 0
    ap_sum = 0.0
    for rank_index, is_hit in enumerate(hits, start=1):
        if is_hit:
            cum_hits += 1
            ap_sum += cum_hits / float(rank_index)
    ap = ap_sum / float(len(positive_set))
    last_positive_rank = max(ranks)
    min_positive_rank = min(ranks)
    pos_scores = [float(scores[idx]) for idx in positives]
    neg_scores = [float(scores[idx]) for idx, pid in enumerate(gallery_pids) if int(pid) != int(query_pid)]
    highest_positive = max(pos_scores)
    highest_negative = max(neg_scores) if neg_scores else None
    return {
        "best_positive_rank": int(min_positive_rank),
        "r1": 1.0 if min_positive_rank <= 1 else 0.0,
        "r5": 1.0 if min_positive_rank <= 5 else 0.0,
        "r10": 1.0 if min_positive_rank <= 10 else 0.0,
        "ap": float(ap),
        "minp": float(len(positive_set) / float(last_positive_rank)),
        "highest_positive_score": float(highest_positive),
        "highest_non_positive_score": float(highest_negative) if highest_negative is not None else None,
        "positive_minus_hardest_negative_margin": (
            float(highest_positive - highest_negative) if highest_negative is not None else None
        ),
    }


def _assert_metric_dict_equal(left: Dict[str, Any], right: Dict[str, Any], label: str) -> None:
    keys = [
        "best_positive_rank",
        "r1",
        "r5",
        "r10",
        "ap",
        "minp",
        "highest_positive_score",
        "highest_non_positive_score",
        "positive_minus_hardest_negative_margin",
    ]
    for key in keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None or right_value is None:
            if left_value != right_value:
                raise AssertionError("{} lambda=0 metric {} mismatch: {} != {}".format(label, key, left_value, right_value))
            continue
        if isinstance(left_value, float) or isinstance(right_value, float):
            if float(left_value) != float(right_value):
                raise AssertionError("{} lambda=0 metric {} mismatch: {} != {}".format(label, key, left_value, right_value))
        elif left_value != right_value:
            raise AssertionError("{} lambda=0 metric {} mismatch: {} != {}".format(label, key, left_value, right_value))


def _assert_lambda_zero_pair(
    label: str,
    matched_scores: Any,
    mismatched_scores: Any,
    query_pid: int,
    gallery_pids: Sequence[int],
    gallery_image_ids: Sequence[str],
) -> None:
    import torch

    matched_cpu = matched_scores.detach().cpu()
    mismatched_cpu = mismatched_scores.detach().cpu()
    if not torch.equal(matched_cpu, mismatched_cpu):
        max_abs = float((matched_cpu - mismatched_cpu).abs().max().item())
        raise AssertionError("{} lambda=0 score tensor mismatch; max_abs_diff={}".format(label, max_abs))
    matched_list = _score_tensor_to_list(matched_cpu)
    mismatched_list = _score_tensor_to_list(mismatched_cpu)
    matched_metrics = _ranked_metrics_for_query(matched_list, query_pid, gallery_pids, gallery_image_ids)
    mismatched_metrics = _ranked_metrics_for_query(mismatched_list, query_pid, gallery_pids, gallery_image_ids)
    _assert_metric_dict_equal(matched_metrics, mismatched_metrics, label)


def _matrix_metrics(scores: Any, qids: Any, gids: Any, image_ids: Sequence[str]) -> Dict[str, float]:
    rows = []
    scores_cpu = scores.detach().cpu()
    qid_list = [int(v) for v in qids.detach().cpu().view(-1).tolist()]
    gid_list = [int(v) for v in gids.detach().cpu().view(-1).tolist()]
    for query_index, query_pid in enumerate(qid_list):
        rows.append(
            _ranked_metrics_for_query(
                [float(v) for v in scores_cpu[query_index].tolist()],
                query_pid,
                gid_list,
                image_ids,
            )
        )
    return {
        "R1": _mean(row["r1"] for row in rows),
        "R5": _mean(row["r5"] for row in rows),
        "R10": _mean(row["r10"] for row in rows),
        "mAP": _mean(row["ap"] for row in rows),
        "mINP": _mean(row["minp"] for row in rows),
    }


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / float(len(filtered))


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    values = sorted(float(v) for v in values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    mu = sum(values) / float(len(values))
    return math.sqrt(sum((float(v) - mu) ** 2 for v in values) / float(len(values)))


def _ks_statistic(a_values: Sequence[float], b_values: Sequence[float]) -> Optional[float]:
    if not a_values or not b_values:
        return None
    a = sorted(float(v) for v in a_values)
    b = sorted(float(v) for v in b_values)
    i = j = 0
    max_diff = 0.0
    values = sorted(set(a + b))
    for value in values:
        while i < len(a) and a[i] <= value:
            i += 1
        while j < len(b) and b[j] <= value:
            j += 1
        max_diff = max(max_diff, abs(i / float(len(a)) - j / float(len(b))))
    return max_diff


def _assignment_hash(dataset: str, host_model: str, query_id: str, seed: int, image_id_a: str, image_id_b: str) -> Tuple[str, int]:
    left, right = sorted([str(image_id_a), str(image_id_b)])
    material = "|".join([PROTOCOL_VERSION, dataset, host_model, query_id, str(seed), left, right])
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    assignment_bit = int(digest[:2], 16) & 1
    return digest, assignment_bit


def _build_contexts_for_query(
    dataset: str,
    host_model: str,
    query_id: str,
    query_pid: int,
    seed: int,
    base_scores: Sequence[float],
    gallery_pids: Sequence[int],
    image_ids: Sequence[str],
) -> Tuple[List[int], List[int], List[Dict[str, Any]], Dict[str, Any]]:
    positives = [idx for idx, pid in enumerate(gallery_pids) if int(pid) == int(query_pid)]
    negatives = [idx for idx, pid in enumerate(gallery_pids) if int(pid) != int(query_pid)]
    ranked_negatives = sorted(negatives, key=lambda idx: (-float(base_scores[idx]), str(image_ids[idx])))

    a_indices = list(positives)
    b_indices = list(positives)
    manifest_rows: List[Dict[str, Any]] = []
    for idx in positives:
        manifest_rows.append(
            {
                "dataset": dataset,
                "host_model": host_model,
                "query_id": query_id,
                "split_seed": int(seed),
                "image_id": image_ids[idx],
                "context": "shared_positive",
                "is_positive": True,
                "base_rank_among_negatives": None,
                "base_score": float(base_scores[idx]),
                "adjacent_pair_index": None,
                "assignment_hash": None,
                "assignment_bit": None,
            }
        )

    paired_diffs = []
    for pair_start in range(0, len(ranked_negatives) - 1, 2):
        first = ranked_negatives[pair_start]
        second = ranked_negatives[pair_start + 1]
        digest, bit = _assignment_hash(dataset, host_model, query_id, seed, image_ids[first], image_ids[second])
        if bit == 0:
            a_idx, b_idx = first, second
        else:
            a_idx, b_idx = second, first
        pair_index = pair_start // 2
        a_indices.append(a_idx)
        b_indices.append(b_idx)
        paired_diffs.append(abs(float(base_scores[first]) - float(base_scores[second])))
        for context_name, idx in (("A", a_idx), ("B", b_idx)):
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "host_model": host_model,
                    "query_id": query_id,
                    "split_seed": int(seed),
                    "image_id": image_ids[idx],
                    "context": context_name,
                    "is_positive": False,
                    "base_rank_among_negatives": int(ranked_negatives.index(idx) + 1),
                    "base_score": float(base_scores[idx]),
                    "adjacent_pair_index": int(pair_index),
                    "assignment_hash": digest,
                    "assignment_bit": int(bit),
                }
            )

    omitted_count = 0
    if len(ranked_negatives) % 2 == 1:
        omitted = ranked_negatives[-1]
        omitted_count = 1
        manifest_rows.append(
            {
                "dataset": dataset,
                "host_model": host_model,
                "query_id": query_id,
                "split_seed": int(seed),
                "image_id": image_ids[omitted],
                "context": "omitted_remainder",
                "is_positive": False,
                "base_rank_among_negatives": int(len(ranked_negatives)),
                "base_score": float(base_scores[omitted]),
                "adjacent_pair_index": None,
                "assignment_hash": None,
                "assignment_bit": None,
                "omission_reason": "odd_negative_remainder",
            }
        )

    a_set = set(a_indices)
    b_set = set(b_indices)
    if set(positives) - a_set or set(positives) - b_set:
        raise AssertionError("A/B contexts do not contain the same positives")
    if len([idx for idx in a_indices if idx not in positives]) != len([idx for idx in b_indices if idx not in positives]):
        raise AssertionError("A/B contexts do not contain equal negative counts")
    if set(idx for idx in a_indices if idx not in positives) & set(idx for idx in b_indices if idx not in positives):
        raise AssertionError("A/B negative sets overlap")
    if len(a_indices) != len(set(a_indices)) or len(b_indices) != len(set(b_indices)):
        raise AssertionError("Duplicate image within a context")

    a_scores = [float(base_scores[idx]) for idx in a_indices]
    b_scores = [float(base_scores[idx]) for idx in b_indices]
    balance = {
        "gallery_cardinality": int(len(a_indices)),
        "positive_cardinality": int(len(positives)),
        "non_positive_cardinality": int(len(a_indices) - len(positives)),
        "A_base_score_mean": _mean(a_scores),
        "B_base_score_mean": _mean(b_scores),
        "A_base_score_median": _median(a_scores),
        "B_base_score_median": _median(b_scores),
        "A_base_score_std": _std(a_scores),
        "B_base_score_std": _std(b_scores),
        "paired_adjacent_score_abs_diff_mean": _mean(paired_diffs),
        "paired_adjacent_score_abs_diff_max": max(paired_diffs) if paired_diffs else None,
        "A_B_base_score_ks": _ks_statistic(a_scores, b_scores),
        "omitted_odd_remainder_count": int(omitted_count),
    }
    return a_indices, b_indices, manifest_rows, balance


def _topm_audit(
    top_indices: Any,
    context_indices: Sequence[int],
    query_pid: int,
    gallery_pids: Sequence[int],
    image_ids: Sequence[str],
    full_base_scores: Sequence[float],
) -> Dict[str, Any]:
    selected_local = [int(v) for v in top_indices.detach().cpu().view(-1).tolist()]
    selected_full = [context_indices[idx] for idx in selected_local]
    selected_ids = [image_ids[idx] for idx in selected_full]
    selected_pids = [int(gallery_pids[idx]) for idx in selected_full]
    selected_scores = [float(full_base_scores[idx]) for idx in selected_full]
    return {
        "topm_ids": selected_ids,
        "topm_positive_count": int(sum(1 for pid in selected_pids if pid == int(query_pid))),
        "topm_has_positive": bool(any(pid == int(query_pid) for pid in selected_pids)),
        "topm_actual_selected_count": int(len(selected_ids)),
        "topm_selected_base_score_min": min(selected_scores) if selected_scores else None,
        "topm_selected_base_score_max": max(selected_scores) if selected_scores else None,
        "topm_selected_base_score_mean": _mean(selected_scores),
        "topm_selected_base_score_median": _median(selected_scores),
    }


def _audit_overlap(audit_a: Dict[str, Any], audit_b: Dict[str, Any]) -> Dict[str, Any]:
    set_a = set(audit_a["topm_ids"])
    set_b = set(audit_b["topm_ids"])
    union = set_a | set_b
    inter = set_a & set_b
    return {
        "topm_overlap_count": int(len(inter)),
        "topm_jaccard": float(len(inter) / len(union)) if union else 0.0,
    }


def _row_with_prefixed_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {"{}_{}".format(prefix, key): value for key, value in metrics.items()}


def _context_direction_row(
    dataset: str,
    host_model: str,
    score_task: str,
    score_base_task: Optional[str],
    score_proto_lambda: Optional[float],
    is_official_best_score_task: bool,
    query_id: str,
    query_pid: int,
    seed: int,
    direction: str,
    context_indices: Sequence[int],
    base_scores: Sequence[float],
    matched_scores: Sequence[float],
    mismatched_scores: Sequence[float],
    gids: Sequence[int],
    image_ids: Sequence[str],
    balance: Dict[str, Any],
    query_measurements: Dict[str, Any],
    audit_a: Dict[str, Any],
    audit_b: Dict[str, Any],
) -> Dict[str, Any]:
    gallery_pids = [int(gids[idx]) for idx in context_indices]
    gallery_image_ids = [image_ids[idx] for idx in context_indices]
    row = {
        "dataset": dataset,
        "host_model": host_model,
        "score_task": str(score_task),
        "score_base_task": score_base_task,
        "score_proto_lambda": score_proto_lambda,
        "is_official_best_score_task": bool(is_official_best_score_task),
        "query_id": query_id,
        "query_pid": int(query_pid),
        "split_seed": int(seed),
        "evaluation_gallery": direction,
        "gallery_cardinality": int(len(context_indices)),
    }
    row.update(_row_with_prefixed_metrics("base", _ranked_metrics_for_query(base_scores, query_pid, gallery_pids, gallery_image_ids)))
    row.update(_row_with_prefixed_metrics("matched", _ranked_metrics_for_query(matched_scores, query_pid, gallery_pids, gallery_image_ids)))
    row.update(_row_with_prefixed_metrics("mismatched", _ranked_metrics_for_query(mismatched_scores, query_pid, gallery_pids, gallery_image_ids)))
    row.update(balance)
    row.update(query_measurements)
    row.update({"top_m_configured": audit_a.get("top_m_configured")})
    row.update({f"A_{key}": value for key, value in audit_a.items()})
    row.update({f"B_{key}": value for key, value in audit_b.items()})
    row.update(_audit_overlap(audit_a, audit_b))
    return row


def _union_control_row(
    dataset: str,
    host_model: str,
    score_task: str,
    score_base_task: Optional[str],
    score_proto_lambda: Optional[float],
    is_official_best_score_task: bool,
    query_id: str,
    query_pid: int,
    seed: int,
    union_indices: Sequence[int],
    base_scores: Sequence[float],
    enriched_a_scores: Sequence[float],
    enriched_b_scores: Sequence[float],
    gids: Sequence[int],
    image_ids: Sequence[str],
    query_measurements: Dict[str, Any],
) -> Dict[str, Any]:
    gallery_pids = [int(gids[idx]) for idx in union_indices]
    gallery_image_ids = [image_ids[idx] for idx in union_indices]
    row = {
        "dataset": dataset,
        "host_model": host_model,
        "score_task": str(score_task),
        "score_base_task": score_base_task,
        "score_proto_lambda": score_proto_lambda,
        "is_official_best_score_task": bool(is_official_best_score_task),
        "query_id": query_id,
        "query_pid": int(query_pid),
        "split_seed": int(seed),
        "evaluation_gallery": "A_union_B",
        "gallery_cardinality": int(len(union_indices)),
    }
    row.update(_row_with_prefixed_metrics("base", _ranked_metrics_for_query(base_scores, query_pid, gallery_pids, gallery_image_ids)))
    row.update(_row_with_prefixed_metrics("enriched_A", _ranked_metrics_for_query(enriched_a_scores, query_pid, gallery_pids, gallery_image_ids)))
    row.update(_row_with_prefixed_metrics("enriched_B", _ranked_metrics_for_query(enriched_b_scores, query_pid, gallery_pids, gallery_image_ids)))
    row.update(query_measurements)
    return row


def _context_negative_ids(indices: Sequence[int], query_pid: int, gids: Sequence[int], image_ids: Sequence[str]) -> List[str]:
    return [image_ids[idx] for idx in indices if int(gids[idx]) != int(query_pid)]


def _topm_negative_records(
    audit: Dict[str, Any],
    query_pid: int,
    context_indices: Sequence[int],
    gids: Sequence[int],
    image_ids: Sequence[str],
    full_base_scores: Sequence[float],
    limit: int,
) -> List[Dict[str, Any]]:
    image_id_to_index = {image_ids[idx]: idx for idx in context_indices}
    records = []
    for image_id in audit.get("topm_ids", []):
        full_index = image_id_to_index.get(image_id)
        if full_index is None or int(gids[full_index]) == int(query_pid):
            continue
        records.append(
            {
                "image_id": image_id,
                "pid": int(gids[full_index]),
                "full_index": int(full_index),
                "base_score": float(full_base_scores[full_index]),
            }
        )
        if len(records) >= int(limit):
            break
    return records


def _hardest_negative_record(
    scores: Sequence[float],
    context_indices: Sequence[int],
    query_pid: int,
    gids: Sequence[int],
    image_ids: Sequence[str],
    enrichment_indices: Sequence[int],
) -> Optional[Dict[str, Any]]:
    enrichment_set = set(int(idx) for idx in enrichment_indices)
    best_local = None
    best_score = None
    for local_index, full_index in enumerate(context_indices):
        if int(gids[full_index]) == int(query_pid):
            continue
        score = float(scores[local_index])
        if best_score is None or score > best_score:
            best_local = int(local_index)
            best_score = score
    if best_local is None:
        return None
    full_index = int(context_indices[best_local])
    return {
        "image_id": image_ids[full_index],
        "pid": int(gids[full_index]),
        "full_index": full_index,
        "score": float(best_score),
        "belongs_to_enrichment_context": bool(full_index in enrichment_set),
    }


def _cell_metrics_for_audit(
    scores: Sequence[float],
    context_indices: Sequence[int],
    query_pid: int,
    gids: Sequence[int],
    image_ids: Sequence[str],
    enrichment_indices: Sequence[int],
) -> Dict[str, Any]:
    gallery_pids = [int(gids[idx]) for idx in context_indices]
    gallery_image_ids = [image_ids[idx] for idx in context_indices]
    metrics = _ranked_metrics_for_query(scores, query_pid, gallery_pids, gallery_image_ids)
    return {
        "best_positive_rank": metrics.get("best_positive_rank"),
        "ap": metrics.get("ap"),
        "highest_positive_score": metrics.get("highest_positive_score"),
        "highest_non_positive_score": metrics.get("highest_non_positive_score"),
        "positive_minus_hardest_negative_margin": metrics.get("positive_minus_hardest_negative_margin"),
        "hardest_negative": _hardest_negative_record(
            scores,
            context_indices,
            query_pid,
            gids,
            image_ids,
            enrichment_indices,
        ),
    }


def _query_measurements(active_query: Any, enriched_a: Any, enriched_b: Any) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    active = F.normalize(active_query.detach().cpu().float(), p=2, dim=1)
    a = enriched_a.detach().cpu().float()
    b = enriched_b.detach().cpu().float()
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return {
        "d_AB": float(1.0 - F.cosine_similarity(a_norm, b_norm, dim=1).mean().item()),
        "base_enriched_A_cosine": float(F.cosine_similarity(active, a_norm, dim=1).mean().item()),
        "base_enriched_B_cosine": float(F.cosine_similarity(active, b_norm, dim=1).mean().item()),
        "enriched_A_minus_base_l2": float((a_norm - active).norm(dim=1).mean().item()),
        "enriched_B_minus_base_l2": float((b_norm - active).norm(dim=1).mean().item()),
        "base_query_norm_before_final_normalization": float(active_query.detach().cpu().float().norm(dim=1).mean().item()),
        "base_query_norm_after_final_normalization": float(active.norm(dim=1).mean().item()),
        "enriched_A_norm_before_final_normalization": float(a.norm(dim=1).mean().item()),
        "enriched_A_norm_after_final_normalization": float(a_norm.norm(dim=1).mean().item()),
        "enriched_B_norm_before_final_normalization": float(b.norm(dim=1).mean().item()),
        "enriched_B_norm_after_final_normalization": float(b_norm.norm(dim=1).mean().item()),
        "residual_gate_value": None,
    }


def _summary_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"metrics_unit": "fraction"}
    for metric_name in ("r1", "r5", "r10", "ap", "minp"):
        metric_label = "mAP" if metric_name == "ap" else ("mINP" if metric_name == "minp" else metric_name.upper())
        base = _mean(row.get(f"base_{metric_name}") for row in rows)
        matched = _mean(row.get(f"matched_{metric_name}") for row in rows)
        mismatched = _mean(row.get(f"mismatched_{metric_name}") for row in rows)
        summary[f"base_context_{metric_label}"] = base
        summary[f"matched_{metric_label}"] = matched
        summary[f"mismatched_{metric_label}"] = mismatched
        summary[f"delta_ctx_{metric_label}"] = (
            matched - mismatched if matched is not None and mismatched is not None else None
        )
        summary[f"matched_gain_over_base_{metric_label}"] = (
            matched - base if matched is not None and base is not None else None
        )
        summary[f"mismatched_gain_over_base_{metric_label}"] = (
            mismatched - base if mismatched is not None and base is not None else None
        )

    better = tied = worse = 0
    matched_only = mismatched_only = 0
    margin_diffs = []
    for row in rows:
        matched_rank = row.get("matched_best_positive_rank")
        mismatched_rank = row.get("mismatched_best_positive_rank")
        if matched_rank is not None and mismatched_rank is not None:
            if matched_rank < mismatched_rank:
                better += 1
            elif matched_rank == mismatched_rank:
                tied += 1
            else:
                worse += 1
        if row.get("matched_r1") == 1.0 and row.get("mismatched_r1") == 0.0:
            matched_only += 1
        if row.get("matched_r1") == 0.0 and row.get("mismatched_r1") == 1.0:
            mismatched_only += 1
        if row.get("matched_positive_minus_hardest_negative_margin") is not None and row.get("mismatched_positive_minus_hardest_negative_margin") is not None:
            margin_diffs.append(
                row["matched_positive_minus_hardest_negative_margin"]
                - row["mismatched_positive_minus_hardest_negative_margin"]
            )
    total = max(1, len(rows))
    summary.update(
        {
            "matched_better_rate_by_best_positive_rank": better / float(total),
            "tied_rate_by_best_positive_rank": tied / float(total),
            "mismatched_better_rate_by_best_positive_rank": worse / float(total),
            "matched_only_top1_success_rate": matched_only / float(total),
            "mismatched_only_top1_success_rate": mismatched_only / float(total),
            "net_top1_matched_advantage": (matched_only - mismatched_only) / float(total),
            "positive_margin_difference_matched_minus_mismatched_mean": _mean(margin_diffs),
            "mean_gallery_induced_query_displacement": _mean(row.get("d_AB") for row in rows),
            "median_gallery_induced_query_displacement": _median([row["d_AB"] for row in rows if row.get("d_AB") is not None]),
            "topm_positive_coverage_A": _mean(1.0 if row.get("A_topm_has_positive") else 0.0 for row in rows),
            "topm_positive_coverage_B": _mean(1.0 if row.get("B_topm_has_positive") else 0.0 for row in rows),
            "topm_overlap_count_mean": _mean(row.get("topm_overlap_count") for row in rows),
            "topm_jaccard_mean": _mean(row.get("topm_jaccard") for row in rows),
            "balance_A_B_base_score_ks_mean": _mean(row.get("A_B_base_score_ks") for row in rows),
            "paired_adjacent_score_abs_diff_mean": _mean(row.get("paired_adjacent_score_abs_diff_mean") for row in rows),
            "omitted_odd_remainder_count_total": int(sum(int(row.get("omitted_odd_remainder_count") or 0) for row in rows) / 2),
            "row_count": int(len(rows)),
            "query_count": int(len(set(row["query_id"] for row in rows))),
            "query_seed_direction_count": int(len(rows)),
        }
    )
    return summary


def _bootstrap_ci(rows: Sequence[Dict[str, Any]], seed: int, resamples: int, log_interval: int = 50) -> Dict[str, Any]:
    import numpy as np

    if not rows or resamples <= 0:
        return {}
    started = time.time()
    by_query: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    query_ids = sorted(by_query)
    query_count = len(query_ids)
    _log(
        "Bootstrap CI: query-cluster resampling queries={} resamples={}".format(
            query_count,
            int(resamples),
        )
    )
    rng = np.random.default_rng(int(seed))

    counts = np.zeros(query_count, dtype=np.float64)
    sums: Dict[str, Any] = {
        "base_r1": np.zeros(query_count, dtype=np.float64),
        "matched_r1": np.zeros(query_count, dtype=np.float64),
        "mismatched_r1": np.zeros(query_count, dtype=np.float64),
        "base_ap": np.zeros(query_count, dtype=np.float64),
        "matched_ap": np.zeros(query_count, dtype=np.float64),
        "mismatched_ap": np.zeros(query_count, dtype=np.float64),
        "d_AB": np.zeros(query_count, dtype=np.float64),
    }
    for index, query_id in enumerate(query_ids):
        query_rows = by_query[query_id]
        counts[index] = float(len(query_rows))
        for row in query_rows:
            for key, array in sums.items():
                value = row.get(key)
                if value is not None:
                    array[index] += float(value)

    values = {
        "delta_ctx_R1": [],
        "delta_ctx_mAP": [],
        "matched_gain_over_base_R1": [],
        "matched_gain_over_base_mAP": [],
        "mean_gallery_induced_query_displacement": [],
    }
    total_resamples = int(resamples)
    chunk_size = min(max(int(log_interval), 1), 256)
    done = 0
    while done < total_resamples:
        current = min(chunk_size, total_resamples - done)
        sampled = rng.integers(0, query_count, size=(current, query_count))
        sampled_count = counts[sampled].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            base_r1 = sums["base_r1"][sampled].sum(axis=1) / sampled_count
            matched_r1 = sums["matched_r1"][sampled].sum(axis=1) / sampled_count
            mismatched_r1 = sums["mismatched_r1"][sampled].sum(axis=1) / sampled_count
            base_ap = sums["base_ap"][sampled].sum(axis=1) / sampled_count
            matched_ap = sums["matched_ap"][sampled].sum(axis=1) / sampled_count
            mismatched_ap = sums["mismatched_ap"][sampled].sum(axis=1) / sampled_count
            displacement = sums["d_AB"][sampled].sum(axis=1) / sampled_count
        values["delta_ctx_R1"].extend((matched_r1 - mismatched_r1).tolist())
        values["delta_ctx_mAP"].extend((matched_ap - mismatched_ap).tolist())
        values["matched_gain_over_base_R1"].extend((matched_r1 - base_r1).tolist())
        values["matched_gain_over_base_mAP"].extend((matched_ap - base_ap).tolist())
        values["mean_gallery_induced_query_displacement"].extend(displacement.tolist())
        done += current
        if _should_log_progress(done, total_resamples, max(chunk_size, int(log_interval))):
            _log("Bootstrap CI progress: {}/{} resamples".format(done, total_resamples))

    ci = {}
    for name, items in values.items():
        if not items:
            ci[name] = None
            continue
        finite_items = [float(item) for item in items if math.isfinite(float(item))]
        if not finite_items:
            ci[name] = None
            continue
        ci[name] = {
            "lower_2p5": float(np.percentile(finite_items, 2.5)),
            "upper_97p5": float(np.percentile(finite_items, 97.5)),
            "resamples": int(resamples),
            "cluster": "query_id",
        }
    _log("Bootstrap CI complete in {:.1f}s".format(time.time() - started))
    return ci


def _per_seed_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for seed in sorted(set(int(row["split_seed"]) for row in rows)):
        seed_rows = [row for row in rows if int(row["split_seed"]) == seed]
        summary = _summary_from_rows(seed_rows)
        summary["split_seed"] = seed
        output.append(summary)
    return output


def _score_task_summaries(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for task in sorted(set(str(row.get("score_task", "")) for row in rows)):
        task_rows = [row for row in rows if str(row.get("score_task", "")) == task]
        summary = _summary_from_rows(task_rows)
        summary["score_task"] = task
        summary["score_base_task"] = task_rows[0].get("score_base_task") if task_rows else None
        summary["score_proto_lambda"] = task_rows[0].get("score_proto_lambda") if task_rows else None
        output[task] = summary
    return output


def _union_summary_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"metrics_unit": "fraction"}
    for metric_name in ("r1", "r5", "r10", "ap", "minp"):
        metric_label = "mAP" if metric_name == "ap" else ("mINP" if metric_name == "minp" else metric_name.upper())
        base = _mean(row.get(f"base_{metric_name}") for row in rows)
        enriched_a = _mean(row.get(f"enriched_A_{metric_name}") for row in rows)
        enriched_b = _mean(row.get(f"enriched_B_{metric_name}") for row in rows)
        summary[f"union_base_{metric_label}"] = base
        summary[f"union_enriched_A_{metric_label}"] = enriched_a
        summary[f"union_enriched_B_{metric_label}"] = enriched_b
        summary[f"union_delta_A_minus_B_{metric_label}"] = (
            enriched_a - enriched_b if enriched_a is not None and enriched_b is not None else None
        )
    margin_diffs = []
    for row in rows:
        left = row.get("enriched_A_positive_minus_hardest_negative_margin")
        right = row.get("enriched_B_positive_minus_hardest_negative_margin")
        if left is not None and right is not None:
            margin_diffs.append(float(left) - float(right))
    summary.update(
        {
            "union_positive_margin_difference_A_minus_B_mean": _mean(margin_diffs),
            "mean_gallery_induced_query_displacement": _mean(row.get("d_AB") for row in rows),
            "median_gallery_induced_query_displacement": _median([row["d_AB"] for row in rows if row.get("d_AB") is not None]),
            "row_count": int(len(rows)),
            "query_count": int(len(set(row["query_id"] for row in rows))),
            "query_seed_count": int(len(rows)),
        }
    )
    return summary


def _union_score_task_summaries(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for task in sorted(set(str(row.get("score_task", "")) for row in rows)):
        task_rows = [row for row in rows if str(row.get("score_task", "")) == task]
        summary = _union_summary_from_rows(task_rows)
        summary["score_task"] = task
        summary["score_base_task"] = task_rows[0].get("score_base_task") if task_rows else None
        summary["score_proto_lambda"] = task_rows[0].get("score_proto_lambda") if task_rows else None
        output[task] = summary
    return output


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, subvalue in value.items():
            _flatten(f"{prefix}{key}.", subvalue, out)
    else:
        out[prefix[:-1]] = value


def _summary_csv_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    _flatten("", summary, row)
    return row


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return "{:.2f}".format(100.0 * float(value))


def _format_ci(ci: Optional[Dict[str, Any]]) -> str:
    if not ci:
        return ""
    return " [{:.2f}, {:.2f}]".format(100.0 * ci["lower_2p5"], 100.0 * ci["upper_97p5"])


def _summary_row_tex(dataset: str, host_model: str, summary: Dict[str, Any], ci: Dict[str, Any]) -> str:
    return (
        "\\begin{{tabular}}{{llrrrrrrrrr}}\n"
        "\\toprule\n"
        "Dataset & Host & Base R@1 & Matched R@1 & Mismatched R@1 & $\\Delta_{{ctx}}$ R@1 "
        "& Base mAP & Matched mAP & Mismatched mAP & $\\Delta_{{ctx}}$ mAP & Mean $d_{{AB}}$ \\\\\n"
        "\\midrule\n"
        "{} & {} & {} & {} & {} & {}{} & {} & {} & {} & {}{} & {:.4f} \\\\\n"
        "\\bottomrule\n"
        "\\end{{tabular}}\n"
    ).format(
        dataset,
        host_model,
        _format_percent(summary.get("base_context_R1")),
        _format_percent(summary.get("matched_R1")),
        _format_percent(summary.get("mismatched_R1")),
        _format_percent(summary.get("delta_ctx_R1")),
        _format_ci(ci.get("delta_ctx_R1")),
        _format_percent(summary.get("base_context_mAP")),
        _format_percent(summary.get("matched_mAP")),
        _format_percent(summary.get("mismatched_mAP")),
        _format_percent(summary.get("delta_ctx_mAP")),
        _format_ci(ci.get("delta_ctx_mAP")),
        float(summary.get("mean_gallery_induced_query_displacement") or 0.0),
    )


def _console_report(
    spec: RepoSpec,
    dataset: str,
    host_model: str,
    base_hash: str,
    gate_hash: str,
    gate_config_hash: str,
    counts: Dict[str, Any],
    full_gallery: Dict[str, Any],
    summary: Dict[str, Any],
    union_summary: Dict[str, Any],
    ci: Dict[str, Any],
    output_dir: Path,
) -> str:
    lines = [
        "Repository / host / dataset: {} / {} / {}".format(spec.repository_name, host_model, dataset),
        "Checkpoint and config hashes: base={} gate={} gate_config={}".format(
            base_hash[:12],
            gate_hash[:12],
            gate_config_hash[:12],
        ),
        "Official query count / included / excluded: {} / {} / {}".format(
            counts.get("official_query_count"),
            counts.get("included_query_count"),
            counts.get("excluded_query_count"),
        ),
        "Full-gallery reproduction: {}".format(json.dumps(_json_safe(full_gallery), sort_keys=True)[:1000]),
        "Context size and balance statistics: mean size={}, mean KS={}".format(
            summary.get("gallery_cardinality_mean"),
            summary.get("balance_A_B_base_score_ks_mean"),
        ),
        "Official score setting for headline context metrics: {}".format(summary.get("score_task")),
        "Matched metrics: R@1={} R@5={} R@10={} mAP={}".format(
            _format_percent(summary.get("matched_R1")),
            _format_percent(summary.get("matched_R5")),
            _format_percent(summary.get("matched_R10")),
            _format_percent(summary.get("matched_mAP")),
        ),
        "Mismatched metrics: R@1={} R@5={} R@10={} mAP={}".format(
            _format_percent(summary.get("mismatched_R1")),
            _format_percent(summary.get("mismatched_R5")),
            _format_percent(summary.get("mismatched_R10")),
            _format_percent(summary.get("mismatched_mAP")),
        ),
        "Delta_ctx: R@1={}{} mAP={}{}".format(
            _format_percent(summary.get("delta_ctx_R1")),
            _format_ci(ci.get("delta_ctx_R1")),
            _format_percent(summary.get("delta_ctx_mAP")),
            _format_ci(ci.get("delta_ctx_mAP")),
        ),
        "Union control A-vs-B: R@1 delta={} mAP delta={}".format(
            _format_percent(union_summary.get("union_delta_A_minus_B_R1")),
            _format_percent(union_summary.get("union_delta_A_minus_B_mAP")),
        ),
        "Matched gain over base: R@1={}{} mAP={}{}".format(
            _format_percent(summary.get("matched_gain_over_base_R1")),
            _format_ci(ci.get("matched_gain_over_base_R1")),
            _format_percent(summary.get("matched_gain_over_base_mAP")),
            _format_ci(ci.get("matched_gain_over_base_mAP")),
        ),
        "Mean/median gallery-induced query displacement: {:.6f} / {:.6f}".format(
            float(summary.get("mean_gallery_induced_query_displacement") or 0.0),
            float(summary.get("median_gallery_induced_query_displacement") or 0.0),
        ),
        "Matched-better / tied / mismatched-better rates: {} / {} / {}".format(
            _format_percent(summary.get("matched_better_rate_by_best_positive_rank")),
            _format_percent(summary.get("tied_rate_by_best_positive_rank")),
            _format_percent(summary.get("mismatched_better_rate_by_best_positive_rank")),
        ),
        "Output directory: {}".format(output_dir),
    ]
    return "\n".join(lines) + "\n"


def _package_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    for package in ("torch", "numpy", "pandas", "pyarrow", "yaml"):
        try:
            module = importlib.import_module(package)
            versions[package] = getattr(module, "__version__", "available")
        except Exception:
            versions[package] = None
    try:
        import torch

        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["torch_cuda"] = torch.version.cuda
        versions["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return versions


def _resolved_config_yaml(args: SimpleNamespace) -> str:
    try:
        import yaml

        return yaml.dump(_json_safe(vars(args)), default_flow_style=False, sort_keys=True)
    except Exception:
        return json.dumps(_json_safe(vars(args)), indent=2, sort_keys=True)


def _save_features(path: Path, features: FeatureBundle, active_queries: Any, retrieval_features: Any) -> None:
    import torch

    payload = {
        "qids": features.qids,
        "gids": features.gids,
        "host_text": features.host_text,
        "alt_text": features.alt_text,
        "host_image": features.host_image,
        "alt_image": features.alt_image,
        "active_queries": active_queries,
        "full_retrieval_features": retrieval_features,
        "query_ids": features.query_ids,
        "image_ids": features.image_ids,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)


def _run_official_reproduction(
    spec: RepoSpec,
    model: Any,
    img_loader: Any,
    txt_loader: Any,
    args: SimpleNamespace,
) -> Dict[str, Any]:
    metrics_module = importlib.import_module(spec.metrics_module)
    evaluator_cls = metrics_module.Evaluator
    official: Dict[str, Any] = {}
    try:
        _log("Official reproduction check: running frozen-base evaluator")
        base_eval = evaluator_cls(img_loader, txt_loader, args)
        base_top1 = base_eval.eval(model, i2t_metric=False, use_target_enrichment=False)
        _log("Official reproduction check: frozen-base evaluator complete (best R1={:.4f})".format(float(base_top1)))
        _log("Official reproduction check: running GATE evaluator")
        gate_eval = evaluator_cls(img_loader, txt_loader, args)
        gate_top1 = gate_eval.eval(model, i2t_metric=False, use_target_enrichment=True)
        _log("Official reproduction check: GATE evaluator complete (best R1={:.4f})".format(float(gate_top1)))
        official = {
            "status": "ok",
            "base_best_R1_percent": float(base_top1),
            "gate_best_R1_percent": float(gate_top1),
            "base_metrics_percent": _json_safe(getattr(base_eval, "last_metrics", {})),
            "base_best_task": getattr(base_eval, "last_best_task", None),
            "gate_metrics_percent": _json_safe(getattr(gate_eval, "last_metrics", {})),
            "gate_best_task": getattr(gate_eval, "last_best_task", None),
        }
    except Exception as error:
        official = {"status": "failed", "error": str(error)}
    return official


def _compare_reproduction(
    spec: RepoSpec,
    protocol_base_by_task: Dict[str, Dict[str, float]],
    protocol_gate_by_task: Dict[str, Dict[str, float]],
    official: Dict[str, Any],
) -> Dict[str, Any]:
    if official.get("status") != "ok":
        return {"status": "not_checked", "reason": official.get("error", "official reproduction failed")}
    checks = []
    for group_name, protocol_by_task, official_key in [
        ("base", protocol_base_by_task, "base_metrics_percent"),
        ("gate", protocol_gate_by_task, "gate_metrics_percent"),
    ]:
        official_metrics = official.get(official_key, {})
        for task, protocol_metrics in protocol_by_task.items():
            for metric in ("R1", "R5", "R10", "mAP"):
                official_value = official_metrics.get(_task_metric_key(task, metric))
                if official_value is None or metric not in protocol_metrics:
                    continue
                checks.append(
                    {
                        "name": "{}_{}_{}".format(group_name, task, metric),
                        "score_task": task,
                        "metric": metric,
                        "protocol_percent": 100.0 * protocol_metrics[metric],
                        "official_percent": float(official_value),
                        "abs_diff_pp": abs(100.0 * protocol_metrics[metric] - float(official_value)),
                    }
                )
    if not checks:
        return {"status": "not_checked", "reason": "no comparable official metric keys found"}
    tolerance = 0.02
    failures = [check for check in checks if check["abs_diff_pp"] > tolerance]
    return {
        "status": "failed" if failures else "ok",
        "tolerance_percentage_points": tolerance,
        "checks": checks,
        "failures": failures,
    }


def _parse_args(spec: RepoSpec, argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gallery-conditioned query evaluation for {}".format(spec.repository_name),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-checkpoint", required=True, help="trained frozen TBPS host checkpoint")
    parser.add_argument("--gate-checkpoint", required=True, help="trained GATE checkpoint")
    parser.add_argument("--config", required=True, help="GATE training/evaluation config")
    parser.add_argument(
        "--data",
        required=True,
        choices=list(DATASET_CHOICES),
        help="Dataset used for evaluation.",
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        type=Path,
        help="Root directory of the selected evaluation dataset.",
    )
    parser.add_argument("--output-dir", default="gallery_conditioned_query_eval")
    parser.add_argument("--device", default="auto", help="cuda when available, cpu, cuda, or cuda:N")
    if spec.supports_host_model:
        parser.add_argument("--host-model", choices=("clip", "itself"), default=spec.default_host_model)
    else:
        parser.set_defaults(host_model=spec.default_host_model)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(DEFAULT_SPLIT_SEEDS))
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--gallery-chunk-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50, help="progress log interval for batches and queries")
    parser.add_argument("--audit-query-limit", type=int, default=10, help="number of deterministic first-seed queries to write to structural_audit_examples.jsonl")
    parser.add_argument("--audit-top-k", type=int, default=10, help="maximum number of hardest/top-M negative records per audit example")
    parser.add_argument("--save-context-manifest", action="store_true")
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run_gallery_conditioned_query(spec: RepoSpec, argv: Optional[Sequence[str]] = None) -> None:
    started_at = time.time()
    spec = RepoSpec(
        repository_name=spec.repository_name,
        repo_kind=spec.repo_kind,
        code_root=Path(spec.code_root).resolve(),
        default_host_model=spec.default_host_model,
        metrics_module=spec.metrics_module,
        logger_name=spec.logger_name,
        supports_host_model=spec.supports_host_model,
    )
    if str(spec.code_root) not in sys.path:
        sys.path.insert(0, str(spec.code_root))
    repo_root = _repo_root_from_spec(spec).resolve()
    cli_args = _parse_args(spec, argv)
    log_interval = max(1, int(cli_args.log_interval))
    _log(
        "Starting {} gallery-conditioned query evaluation (host_model={}, data={}, split={})".format(
            spec.repository_name,
            cli_args.host_model,
            cli_args.data,
            cli_args.split,
        )
    )
    output_dir = Path(cli_args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not cli_args.overwrite:
        raise FileExistsError("Output directory exists and is not empty; pass --overwrite: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(cli_args.config).expanduser().resolve()
    base_checkpoint = Path(cli_args.base_checkpoint).expanduser().resolve()
    gate_checkpoint = Path(cli_args.gate_checkpoint).expanduser().resolve()
    _log("Output directory: {}".format(output_dir))
    _log("Checking inputs: base={} gate={} config={}".format(base_checkpoint, gate_checkpoint, config_path))
    for label, path in (("gate config", config_path), ("base checkpoint", base_checkpoint), ("gate checkpoint", gate_checkpoint)):
        if not path.is_file():
            raise FileNotFoundError("{} not found: {}".format(label, path))

    _log("Loading host defaults/checkpoint metadata and GATE config")
    host_args, host_metadata_values, host_settings_source = _load_host_args(base_checkpoint)
    resolved_host_args = _namespace_from_mapping(vars(host_args))
    gate_args, gate_config_values = _load_config(config_path)
    _validate_gate_dataset_compatibility(gate_config_values, cli_args.data)
    args, applied_gate_keys = _apply_gate_config_overlay(host_args, gate_args, gate_config_values)
    args.gate_config_file = str(config_path)
    args.config_file = str(config_path)
    args.base_checkpoint = str(base_checkpoint)
    args.gate_checkpoint = str(gate_checkpoint)
    args.eval_output_dir = str(output_dir)
    _ensure_eval_defaults(args, spec, cli_args)
    _log(
        "Dataset resolved: requested={} loader_root={} dataset_dir={} annotation={} image_root={} queries={} gallery_images={}".format(
            getattr(args, "requested_dataset", None),
            getattr(args, "resolved_root_dir", None),
            getattr(args, "resolved_dataset_dir", None),
            getattr(args, "resolved_annotation_path", None),
            getattr(args, "resolved_image_root", None),
            getattr(args, "number_of_queries", None),
            getattr(args, "number_of_gallery_images", None),
        )
    )
    _configure_reproducibility(
        int(getattr(args, "seed", 1)),
        deterministic=bool(getattr(args, "deterministic", True)),
        warn_only=bool(getattr(args, "deterministic_warn_only", False)),
    )

    import torch
    import torch.nn.functional as F

    device = _resolve_device(cli_args.device)
    _log("Using device: {}".format(device))
    _log("Loading base checkpoint metadata for num_classes inference")
    first_checkpoint = _torch_load_checkpoint(base_checkpoint)
    first_state, _, _ = _unwrap_state_dict(first_checkpoint)
    num_classes = _infer_num_classes_from_state_dict(first_state, getattr(args, "num_classes", None))
    _log("Building model with num_classes={}".format(num_classes))
    build_model = importlib.import_module("model").build_model
    model = build_model(args, num_classes)
    model.to(device)
    _log("Loading base and GATE checkpoint weights")
    checkpoint_reports = _load_base_and_gate(model, base_checkpoint, gate_checkpoint)
    _freeze_and_eval(model)
    _atomic_write_json(output_dir / "checkpoint_load_report.json", checkpoint_reports)
    _atomic_write_text(output_dir / "resolved_config.yaml", _resolved_config_yaml(args))
    _atomic_write_text(output_dir / "resolved_gate_config.yaml", _resolved_config_yaml(gate_args))
    _log("Checkpoint loading complete; model is frozen/eval")

    _log("Building official dataloaders")
    img_loader, txt_loader = _build_eval_loaders(args, cli_args.split)
    _validate_loaded_eval_loaders(img_loader, txt_loader, args)
    _log(
        "Dataloaders ready: image_batches={} text_batches={}".format(
            len(img_loader) if hasattr(img_loader, "__len__") else "unknown",
            len(txt_loader) if hasattr(txt_loader, "__len__") else "unknown",
        )
    )
    core = _core_model(model)
    features = _extract_features(spec, model, img_loader, txt_loader, args, repo_root, log_interval=log_interval)
    _log("Feature extraction complete: queries={} gallery_images={}".format(len(features.query_ids), len(features.image_ids)))
    active_queries = _active_query_features(spec, args, features.host_text, features.alt_text)

    all_gallery_indices = list(range(int(features.gids.numel())))
    _log("Building and finalizing full-gallery target cache")
    full_cache = _subset_and_finalize_cache(core, features.raw_target_cache, features.gids, all_gallery_indices, device)
    full_retrieval = _cache_retrieval_features(full_cache)
    _log("Computing full-gallery official base score components")
    host_queries_norm = F.normalize(features.host_text.float(), p=2, dim=1)
    host_gallery_norm = F.normalize(features.host_image.float(), p=2, dim=1)
    full_global_scores = _score_matrix(host_queries_norm, host_gallery_norm, int(cli_args.gallery_chunk_size))
    full_alt_scores = None
    if features.alt_text is not None and features.alt_image is not None:
        full_alt_scores = _score_matrix(
            F.normalize(features.alt_text.float(), p=2, dim=1),
            F.normalize(features.alt_image.float(), p=2, dim=1),
            int(cli_args.gallery_chunk_size),
        )
    _log("Computing full-gallery GATE-enriched target score component")
    full_enriched, _ = _enrich_all_queries(
        spec,
        core,
        args,
        full_cache,
        active_queries,
        features.host_text,
        features.alt_text,
        int(cli_args.query_batch_size),
        stage_name="full-gallery sanity",
        log_interval=log_interval,
    )
    full_enriched_norm = F.normalize(full_enriched.float(), p=2, dim=1)
    full_target_scores = _score_matrix(full_enriched_norm, full_retrieval, int(cli_args.gallery_chunk_size))
    full_gate_tasks, full_base_tasks = _official_score_tasks(
        spec,
        args,
        full_global_scores,
        full_alt_scores,
        full_target_scores,
        include_base=True,
    )
    protocol_full_base_by_task = {
        task["task"]: _matrix_metrics(task["scores"], features.qids, features.gids, features.image_ids)
        for task in full_base_tasks
    }
    protocol_full_gate_by_task = {
        task["task"]: _matrix_metrics(task["scores"], features.qids, features.gids, features.image_ids)
        for task in full_gate_tasks
    }
    full_base_score_by_task = {task["task"]: task["scores"] for task in full_base_tasks}
    _log(
        "Full-gallery protocol metrics computed: base_tasks={} gate_tasks={}".format(
            len(protocol_full_base_by_task),
            len(protocol_full_gate_by_task),
        )
    )

    official_reproduction = _run_official_reproduction(spec, model, img_loader, txt_loader, args)
    official_base_task = official_reproduction.get("base_best_task")
    if official_base_task not in protocol_full_base_by_task:
        official_base_task = _best_task_from_metrics(spec, protocol_full_base_by_task)
    official_gate_eval_best_task = official_reproduction.get("gate_best_task")
    official_gate_task = _best_task_from_metrics(spec, protocol_full_gate_by_task, prefer_ablation=True)
    if official_gate_task not in protocol_full_gate_by_task:
        official_gate_task = official_gate_eval_best_task if official_gate_eval_best_task in protocol_full_gate_by_task else None
    protocol_full_base = _select_task_metrics(protocol_full_base_by_task, official_base_task)
    protocol_full_gate = _select_task_metrics(protocol_full_gate_by_task, official_gate_task)
    frozen_base_scores = full_base_score_by_task.get(str(official_base_task), next(iter(full_base_score_by_task.values())))
    reproduction_check = _compare_reproduction(
        spec,
        protocol_full_base_by_task,
        protocol_full_gate_by_task,
        official_reproduction,
    )
    if official_reproduction.get("status") == "failed" or reproduction_check.get("status") == "failed":
        _log("Full-gallery reproduction check failed; writing failure summary")
        _atomic_write_json(
            output_dir / "summary.json",
            {
                "status": "failed",
                "reason": "full-gallery reproduction check failed",
                "reproduction_check": reproduction_check,
                "official_reproduction": official_reproduction,
                "protocol_full_base_by_task": protocol_full_base_by_task,
                "protocol_full_gate_by_task": protocol_full_gate_by_task,
                "official_base_task": official_base_task,
                "official_gate_task": official_gate_task,
                "official_gate_eval_best_task": official_gate_eval_best_task,
            },
        )
        raise RuntimeError("Full-gallery reproduction check failed; see summary.json")
    _log(
        "Full-gallery reproduction check status: {}; official_base_task={} official_gate_task={}".format(
            reproduction_check.get("status"),
            official_base_task,
            official_gate_task,
        )
    )

    if cli_args.save_features:
        _log("Saving raw feature tensors")
        _save_features(output_dir / "features.pt", features, active_queries, full_retrieval)

    dataset = str(getattr(args, "dataset_name", "unknown"))
    host_model = str(cli_args.host_model)
    gids_list = [int(v) for v in features.gids.detach().cpu().view(-1).tolist()]
    qids_list = [int(v) for v in features.qids.detach().cpu().view(-1).tolist()]

    per_query_rows: List[Dict[str, Any]] = []
    union_control_rows: List[Dict[str, Any]] = []
    context_manifest_rows: List[Dict[str, Any]] = []
    structural_audit_examples: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    included_queries = set()

    total_queries = len(features.query_ids)
    total_seed_evals = total_queries * len(cli_args.split_seeds)
    processed_seed_evals = 0
    _log(
        "Starting matched/mismatched context protocol: queries={} seeds={} expected query-seed contexts={}".format(
            total_queries,
            len(cli_args.split_seeds),
            total_seed_evals,
        )
    )
    for query_index, (query_id, query_pid) in enumerate(zip(features.query_ids, qids_list)):
        full_base_for_query = [float(v) for v in frozen_base_scores[query_index].detach().cpu().tolist()]
        positive_count = sum(1 for pid in gids_list if int(pid) == int(query_pid))
        if positive_count < 1:
            exclusions.append(
                {
                    "dataset": dataset,
                    "host_model": host_model,
                    "query_id": query_id,
                    "query_pid": int(query_pid),
                    "reason": "no_valid_positive_image",
                }
            )
            continue
        included_queries.add(query_id)
        active_query = active_queries[query_index:query_index + 1]
        host_query = features.host_text[query_index:query_index + 1]
        alt_query = features.alt_text[query_index:query_index + 1] if features.alt_text is not None else None

        for split_seed in cli_args.split_seeds:
            processed_seed_evals += 1
            a_indices, b_indices, manifest_rows, balance = _build_contexts_for_query(
                dataset,
                host_model,
                query_id,
                int(query_pid),
                int(split_seed),
                full_base_for_query,
                gids_list,
                features.image_ids,
            )
            context_manifest_rows.extend(manifest_rows)
            cache_a = _subset_and_finalize_cache(core, features.raw_target_cache, features.gids, a_indices, device)
            cache_b = _subset_and_finalize_cache(core, features.raw_target_cache, features.gids, b_indices, device)
            top_a = _compute_top_indices(spec, core, args, cache_a, active_query, host_query, alt_query)
            top_b = _compute_top_indices(spec, core, args, cache_b, active_query, host_query, alt_query)
            enriched_a = _enrich_query_features(spec, core, args, cache_a, active_query, host_query, alt_query, top_a)
            enriched_b = _enrich_query_features(spec, core, args, cache_b, active_query, host_query, alt_query, top_b)
            retrieval_a = _cache_retrieval_features(cache_a)
            retrieval_b = _cache_retrieval_features(cache_b)
            enriched_a_norm = F.normalize(enriched_a.float(), p=2, dim=1)
            enriched_b_norm = F.normalize(enriched_b.float(), p=2, dim=1)

            index_a = torch.as_tensor(a_indices, dtype=torch.long)
            index_b = torch.as_tensor(b_indices, dtype=torch.long)
            host_gallery_a = features.host_image.index_select(0, index_a)
            host_gallery_b = features.host_image.index_select(0, index_b)
            alt_gallery_a = features.alt_image.index_select(0, index_a) if features.alt_image is not None else None
            alt_gallery_b = features.alt_image.index_select(0, index_b) if features.alt_image is not None else None
            base_tasks_a = _context_base_tasks(spec, args, host_query, host_gallery_a, alt_query, alt_gallery_a)
            base_tasks_b = _context_base_tasks(spec, args, host_query, host_gallery_b, alt_query, alt_gallery_b)
            target_aa_scores = enriched_a_norm @ retrieval_a.t()
            target_ba_scores = enriched_b_norm @ retrieval_a.t()
            target_bb_scores = enriched_b_norm @ retrieval_b.t()
            target_ab_scores = enriched_a_norm @ retrieval_b.t()
            matched_a_tasks = _official_target_tasks(spec, base_tasks_a, target_aa_scores)
            matched_a_by_task = _scores_by_task(matched_a_tasks)
            mismatched_a_by_task = _scores_by_task(_official_target_tasks(spec, base_tasks_a, target_ba_scores))
            matched_b_tasks = _official_target_tasks(spec, base_tasks_b, target_bb_scores)
            matched_b_by_task = _scores_by_task(matched_b_tasks)
            mismatched_b_by_task = _scores_by_task(_official_target_tasks(spec, base_tasks_b, target_ab_scores))
            gallery_pids_a = [int(gids_list[idx]) for idx in a_indices]
            gallery_ids_a = [features.image_ids[idx] for idx in a_indices]
            gallery_pids_b = [int(gids_list[idx]) for idx in b_indices]
            gallery_ids_b = [features.image_ids[idx] for idx in b_indices]

            audit_a = _topm_audit(top_a, a_indices, int(query_pid), gids_list, features.image_ids, full_base_for_query)
            audit_b = _topm_audit(top_b, b_indices, int(query_pid), gids_list, features.image_ids, full_base_for_query)
            audit_a["top_m_configured"] = int(getattr(getattr(core, "target_enricher"), "top_m", getattr(args, "top_m", 0)))
            audit_b["top_m_configured"] = audit_a["top_m_configured"]
            query_measurements = _query_measurements(active_query, enriched_a, enriched_b)

            for task_info in matched_a_tasks:
                task_name = str(task_info["task"])
                mismatch_info = mismatched_a_by_task[task_name]
                if abs(float(task_info.get("proto_lambda") or 0.0)) < 1e-12:
                    _assert_lambda_zero_pair(
                        "{} query={} seed={} gallery=A task={}".format(dataset, query_id, int(split_seed), task_name),
                        task_info["scores"],
                        mismatch_info["scores"],
                        int(query_pid),
                        gallery_pids_a,
                        gallery_ids_a,
                    )
                base_scores_tensor = task_info.get("base_scores")
                if base_scores_tensor is None:
                    base_scores_tensor = base_tasks_a[0]["scores"]
                per_query_rows.append(
                    _context_direction_row(
                        dataset,
                        host_model,
                        task_name,
                        task_info.get("base_task"),
                        task_info.get("proto_lambda"),
                        task_name == official_gate_task,
                        query_id,
                        int(query_pid),
                        int(split_seed),
                        "A",
                        a_indices,
                        _score_tensor_to_list(base_scores_tensor),
                        _score_tensor_to_list(task_info["scores"]),
                        _score_tensor_to_list(mismatch_info["scores"]),
                        gids_list,
                        features.image_ids,
                        balance,
                        query_measurements,
                        audit_a,
                        audit_b,
                    )
                )
            for task_info in matched_b_tasks:
                task_name = str(task_info["task"])
                mismatch_info = mismatched_b_by_task[task_name]
                if abs(float(task_info.get("proto_lambda") or 0.0)) < 1e-12:
                    _assert_lambda_zero_pair(
                        "{} query={} seed={} gallery=B task={}".format(dataset, query_id, int(split_seed), task_name),
                        task_info["scores"],
                        mismatch_info["scores"],
                        int(query_pid),
                        gallery_pids_b,
                        gallery_ids_b,
                    )
                base_scores_tensor = task_info.get("base_scores")
                if base_scores_tensor is None:
                    base_scores_tensor = base_tasks_b[0]["scores"]
                per_query_rows.append(
                    _context_direction_row(
                        dataset,
                        host_model,
                        task_name,
                        task_info.get("base_task"),
                        task_info.get("proto_lambda"),
                        task_name == official_gate_task,
                        query_id,
                        int(query_pid),
                        int(split_seed),
                        "B",
                        b_indices,
                        _score_tensor_to_list(base_scores_tensor),
                        _score_tensor_to_list(task_info["scores"]),
                        _score_tensor_to_list(mismatch_info["scores"]),
                        gids_list,
                        features.image_ids,
                        balance,
                        query_measurements,
                        audit_a,
                        audit_b,
                    )
                )
            union_indices = sorted(set(int(idx) for idx in a_indices) | set(int(idx) for idx in b_indices))
            cache_union = _subset_and_finalize_cache(core, features.raw_target_cache, features.gids, union_indices, device)
            retrieval_union = _cache_retrieval_features(cache_union)
            index_union = torch.as_tensor(union_indices, dtype=torch.long)
            host_gallery_union = features.host_image.index_select(0, index_union)
            alt_gallery_union = features.alt_image.index_select(0, index_union) if features.alt_image is not None else None
            base_tasks_union = _context_base_tasks(spec, args, host_query, host_gallery_union, alt_query, alt_gallery_union)
            target_a_union_scores = enriched_a_norm @ retrieval_union.t()
            target_b_union_scores = enriched_b_norm @ retrieval_union.t()
            union_a_tasks = _official_target_tasks(spec, base_tasks_union, target_a_union_scores)
            union_b_by_task = _scores_by_task(_official_target_tasks(spec, base_tasks_union, target_b_union_scores))
            gallery_pids_union = [int(gids_list[idx]) for idx in union_indices]
            gallery_ids_union = [features.image_ids[idx] for idx in union_indices]
            for task_info in union_a_tasks:
                task_name = str(task_info["task"])
                union_b_info = union_b_by_task[task_name]
                if abs(float(task_info.get("proto_lambda") or 0.0)) < 1e-12:
                    _assert_lambda_zero_pair(
                        "{} query={} seed={} gallery=A_union_B task={}".format(dataset, query_id, int(split_seed), task_name),
                        task_info["scores"],
                        union_b_info["scores"],
                        int(query_pid),
                        gallery_pids_union,
                        gallery_ids_union,
                    )
                base_scores_tensor = task_info.get("base_scores")
                if base_scores_tensor is None:
                    base_scores_tensor = base_tasks_union[0]["scores"]
                union_control_rows.append(
                    _union_control_row(
                        dataset,
                        host_model,
                        task_name,
                        task_info.get("base_task"),
                        task_info.get("proto_lambda"),
                        task_name == official_gate_task,
                        query_id,
                        int(query_pid),
                        int(split_seed),
                        union_indices,
                        _score_tensor_to_list(base_scores_tensor),
                        _score_tensor_to_list(task_info["scores"]),
                        _score_tensor_to_list(union_b_info["scores"]),
                        gids_list,
                        features.image_ids,
                        query_measurements,
                    )
                )
            if (
                int(cli_args.audit_query_limit) > 0
                and len(structural_audit_examples) < int(cli_args.audit_query_limit)
                and int(split_seed) == int(cli_args.split_seeds[0])
            ):
                task_audits: Dict[str, Any] = {}
                audit_task_names = sorted(matched_a_by_task.keys())
                for task_name in audit_task_names:
                    aa_scores = _score_tensor_to_list(matched_a_by_task[task_name]["scores"])
                    ba_scores = _score_tensor_to_list(mismatched_a_by_task[task_name]["scores"])
                    bb_scores = _score_tensor_to_list(matched_b_by_task[task_name]["scores"])
                    ab_scores = _score_tensor_to_list(mismatched_b_by_task[task_name]["scores"])
                    task_audits[task_name] = {
                        "score_proto_lambda": matched_a_by_task[task_name].get("proto_lambda"),
                        "AA": _cell_metrics_for_audit(aa_scores, a_indices, int(query_pid), gids_list, features.image_ids, a_indices),
                        "BA": _cell_metrics_for_audit(ba_scores, a_indices, int(query_pid), gids_list, features.image_ids, b_indices),
                        "BB": _cell_metrics_for_audit(bb_scores, b_indices, int(query_pid), gids_list, features.image_ids, b_indices),
                        "AB": _cell_metrics_for_audit(ab_scores, b_indices, int(query_pid), gids_list, features.image_ids, a_indices),
                    }
                structural_audit_examples.append(
                    {
                        "dataset": dataset,
                        "host_model": host_model,
                        "query_id": query_id,
                        "query_pid": int(query_pid),
                        "split_seed": int(split_seed),
                        "A_negative_ids": _context_negative_ids(a_indices, int(query_pid), gids_list, features.image_ids),
                        "B_negative_ids": _context_negative_ids(b_indices, int(query_pid), gids_list, features.image_ids),
                        "A_topm_ids": audit_a.get("topm_ids", []),
                        "B_topm_ids": audit_b.get("topm_ids", []),
                        "A_hardest_negatives_affecting_enrichment": _topm_negative_records(
                            audit_a,
                            int(query_pid),
                            a_indices,
                            gids_list,
                            features.image_ids,
                            full_base_for_query,
                            int(cli_args.audit_top_k),
                        ),
                        "B_hardest_negatives_affecting_enrichment": _topm_negative_records(
                            audit_b,
                            int(query_pid),
                            b_indices,
                            gids_list,
                            features.image_ids,
                            full_base_for_query,
                            int(cli_args.audit_top_k),
                        ),
                        "score_tasks": task_audits,
                    }
                )
            del cache_a, cache_b, cache_union
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if _should_log_progress(processed_seed_evals, total_seed_evals, log_interval):
                elapsed = time.time() - started_at
                _log(
                    "Context protocol progress: query {}/{} seed_eval {}/{} rows={} elapsed={:.1f}s".format(
                        query_index + 1,
                        total_queries,
                        processed_seed_evals,
                        total_seed_evals,
                        len(per_query_rows),
                        elapsed,
                    )
                )

    if not per_query_rows:
        raise RuntimeError("No query rows were produced; all queries were excluded")

    _log("Aggregating per-query rows and bootstrap confidence intervals")
    score_ablation_summaries = _score_task_summaries(per_query_rows)
    selected_score_task = str(official_gate_task) if official_gate_task in score_ablation_summaries else next(iter(score_ablation_summaries))
    selected_rows = [row for row in per_query_rows if str(row.get("score_task")) == selected_score_task]
    summary = _summary_from_rows(selected_rows)
    summary["gallery_cardinality_mean"] = _mean(row.get("gallery_cardinality") for row in selected_rows)
    summary["score_task"] = selected_score_task
    summary["official_best_score_task"] = official_gate_task
    summary["score_task_selection_policy"] = "official full-gallery +proto(lambda) ablation best R@1; not selected by Delta_ctx"
    union_score_ablation_summaries = _union_score_task_summaries(union_control_rows)
    selected_union_rows = [row for row in union_control_rows if str(row.get("score_task")) == selected_score_task]
    union_summary = _union_summary_from_rows(selected_union_rows) if selected_union_rows else {}
    union_summary["score_task"] = selected_score_task
    union_summary["control_name"] = "A_union_B_candidate_gallery"
    bootstrap_ci = _bootstrap_ci(
        selected_rows,
        int(cli_args.bootstrap_seed),
        int(cli_args.bootstrap_resamples),
        log_interval=log_interval,
    )
    per_seed = _per_seed_summary(selected_rows)
    fallbacks: Dict[str, str] = {}
    _log("Writing output tables and manifests")
    _write_table(output_dir / "per_query_metrics.parquet", per_query_rows, fallbacks)
    _write_table(output_dir / "context_manifest.parquet", context_manifest_rows, fallbacks)
    _write_table(output_dir / "union_control_metrics.parquet", union_control_rows, fallbacks)
    _atomic_write_jsonl(output_dir / "exclusions.jsonl", exclusions)
    _atomic_write_jsonl(output_dir / "structural_audit_examples.jsonl", structural_audit_examples)
    _atomic_write_csv(output_dir / "per_seed_summary.csv", per_seed)
    _atomic_write_csv(output_dir / "per_score_ablation_summary.csv", list(score_ablation_summaries.values()))
    _atomic_write_csv(output_dir / "union_score_ablation_summary.csv", list(union_score_ablation_summaries.values()))

    base_hash = _sha256_file(base_checkpoint)
    gate_hash = _sha256_file(gate_checkpoint)
    gate_config_hash = _sha256_file(config_path)
    counts = {
        "official_query_count": int(len(features.query_ids)),
        "included_query_count": int(len(included_queries)),
        "excluded_query_count": int(len(exclusions)),
        "gallery_image_count": int(len(features.image_ids)),
        "context_manifest_rows": int(len(context_manifest_rows)),
        "per_query_rows": int(len(per_query_rows)),
        "selected_score_task_rows": int(len(selected_rows)),
        "score_ablation_task_count": int(len(score_ablation_summaries)),
        "union_control_rows": int(len(union_control_rows)),
        "selected_union_control_rows": int(len(selected_union_rows)),
        "structural_audit_example_count": int(len(structural_audit_examples)),
    }
    full_gallery = {
        "protocol_base_metrics_fraction": protocol_full_base,
        "protocol_gate_metrics_fraction": protocol_full_gate,
        "protocol_base_metrics_by_task_fraction": protocol_full_base_by_task,
        "protocol_gate_metrics_by_task_fraction": protocol_full_gate_by_task,
        "official_base_task": official_base_task,
        "official_gate_task": official_gate_task,
        "official_gate_eval_best_task": official_gate_eval_best_task,
        "official_reproduction": official_reproduction,
        "reproduction_check": reproduction_check,
    }
    dataset_resolution = getattr(args, "_dataset_resolution", {})
    run_manifest = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "repository_name": spec.repository_name,
        "host_model": host_model,
        "dataset": dataset,
        "split": cli_args.split,
        "requested_dataset": dataset_resolution.get("requested_dataset"),
        "resolved_dataset_name": dataset_resolution.get("resolved_dataset_name"),
        "provided_root_dir": dataset_resolution.get("provided_root_dir"),
        "resolved_root_dir": dataset_resolution.get("resolved_root_dir"),
        "resolved_dataset_dir": dataset_resolution.get("resolved_dataset_dir"),
        "resolved_annotation_path": dataset_resolution.get("resolved_annotation_path"),
        "resolved_image_root": dataset_resolution.get("resolved_image_root"),
        "dataset_split": dataset_resolution.get("dataset_split"),
        "number_of_queries": dataset_resolution.get("number_of_queries"),
        "number_of_gallery_images": dataset_resolution.get("number_of_gallery_images"),
        "input_paths": {
            "base_checkpoint": str(base_checkpoint),
            "gate_checkpoint": str(gate_checkpoint),
            "gate_config": str(config_path),
        },
        "sha256": {
            "base_checkpoint": base_hash,
            "gate_checkpoint": gate_hash,
            "gate_config": gate_config_hash,
        },
        "git": _git_info(repo_root),
        "resolved_command_line_arguments": vars(cli_args),
        "host_settings_source": host_settings_source,
        "host_checkpoint_metadata_keys": sorted(host_metadata_values.keys()),
        "resolved_host_settings": _json_safe(vars(resolved_host_args)),
        "gate_config_keys_from_file": sorted(gate_config_values.keys()),
        "ignored_non_gate_config_keys": sorted(
            key for key in gate_config_values.keys() if key not in GATE_CONFIG_KEYS
        ),
        "gate_config_overlay_policy": "Only allow-listed GATE settings are copied from --config onto the host args.",
        "applied_gate_config_keys": applied_gate_keys,
        "split_seeds": [int(seed) for seed in cli_args.split_seeds],
        "bootstrap_seed": int(cli_args.bootstrap_seed),
        "bootstrap_resamples": int(cli_args.bootstrap_resamples),
        "top_m": int(getattr(getattr(core, "target_enricher"), "top_m", getattr(args, "top_m", 0))),
        "official_base_score_task": official_base_task,
        "official_gate_score_task": official_gate_task,
        "official_gate_eval_best_task": official_gate_eval_best_task,
        "selected_context_summary_score_task": selected_score_task,
        "score_ablation_tasks": list(score_ablation_summaries.keys()),
        "score_ablation_formula": _score_ablation_description(spec, args),
        "structural_audit": {
            "lambda_zero_assertions": "enabled for scores, ranks, AP, positive/hardest-negative scores, and margins on each same-gallery comparison",
            "audit_query_limit": int(cli_args.audit_query_limit),
            "audit_top_k": int(cli_args.audit_top_k),
            "examples_path": str(output_dir / "structural_audit_examples.jsonl"),
        },
        "union_control": {
            "enabled": True,
            "description": "Evaluates enriched_A and enriched_B on the same G_A union G_B candidate gallery for every official score-ablation task.",
            "metrics_path": str(output_dir / "union_control_metrics.parquet"),
            "summary_path": str(output_dir / "union_score_ablation_summary.csv"),
        },
        "enabled_evidence_providers": str(getattr(args, "extractor_mode", "")),
        "clustering_settings": {
            "target_relative_space": getattr(args, "target_relative_space", None),
            "target_relative_num_clusters": getattr(args, "target_relative_num_clusters", None),
            "target_relative_cluster_method": getattr(args, "target_relative_cluster_method", None),
        },
        "versions": _package_versions(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "deterministic_mode": {
            "seed": getattr(args, "seed", None),
            "deterministic": getattr(args, "deterministic", None),
            "deterministic_warn_only": getattr(args, "deterministic_warn_only", None),
        },
        "counts": counts,
        "output_fallbacks": fallbacks,
    }

    summary_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "host_model": host_model,
        "metrics_unit": "fraction",
        "counts": counts,
        "aggregate": summary,
        "score_ablation_summaries": score_ablation_summaries,
        "union_control": {
            "aggregate": union_summary,
            "score_ablation_summaries": union_score_ablation_summaries,
        },
        "bootstrap_ci_95": bootstrap_ci,
        "full_gallery_reproduction": full_gallery,
        "output_fallbacks": fallbacks,
    }
    _atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    _atomic_write_json(output_dir / "summary.json", summary_payload)
    _atomic_write_csv(output_dir / "summary.csv", [_summary_csv_row(summary_payload)])
    _atomic_write_text(output_dir / "summary_row.tex", _summary_row_tex(dataset, host_model, summary, bootstrap_ci))
    console_report = _console_report(
        spec,
        dataset,
        host_model,
        base_hash,
        gate_hash,
        gate_config_hash,
        counts,
        full_gallery,
        summary,
        union_summary,
        bootstrap_ci,
        output_dir,
    )
    _atomic_write_text(output_dir / "console_report.txt", console_report)
    _log("Evaluation complete in {:.1f}s".format(time.time() - started_at))
    print(console_report, flush=True)
