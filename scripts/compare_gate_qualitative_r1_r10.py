#!/usr/bin/env python3
"""Qualitative R1-RK comparison for enrichment-rde GATE runs.

This script evaluates every official full-gallery GATE score-task candidate on
the configured test split, selects the best candidate with a deterministic
retrieval-metric rule, then renders Baseline-vs-GATE qualitative examples.

Example:
    python scripts/compare_gate_qualitative_r1_r10.py \
      --gate_config /path/to/gate_config.yaml \
      --base_checkpoint /path/to/base_retriever.pth \
      --gate_checkpoint /path/to/gate_best.pth \
      --output_dir outputs/qual_rde_gate \
      --dataset RSTPReid \
      --root_dir /path/to/dataset_parent \
      --selection_metric R1 \
      --num_figs 30 \
      --top_k 10 \
      --sort_mode best_r1_green_gap \
      --device cuda \
      --cache_inference
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gallery_conditioned_query_common import (  # noqa: E402
    DATASET_CHOICES,
    GATE_CONFIG_KEYS,
    RepoSpec,
    _active_query_features,
    _apply_gate_config_overlay,
    _atomic_write_csv,
    _atomic_write_json,
    _atomic_write_text,
    _build_eval_loaders,
    _cache_retrieval_features,
    _canonical_dataset_name,
    _compare_reproduction,
    _configure_reproducibility,
    _core_model,
    _ensure_eval_defaults,
    _extract_features,
    _freeze_and_eval,
    _format_lambda,
    _global_grab_lambdas,
    _infer_num_classes_from_state_dict,
    _json_safe,
    _load_base_and_gate,
    _load_config,
    _load_host_args,
    _log,
    _repo_root_from_spec,
    _resolve_device,
    _resolve_dataset_root,
    _resolved_config_yaml,
    _run_official_reproduction,
    _score_matrix,
    _sha256_file,
    _prototype_lambdas,
    _scale_scores_like,
    _subset_and_finalize_cache,
    _torch_load_checkpoint,
    _validate_gate_dataset_compatibility,
    _validate_loaded_eval_loaders,
)


SORT_MODES = (
    "best_r1_green_gap",
    "strict_showcase",
    "best_r1_baseline_not_r1",
    "best_r1_baseline_not_r10",
    "rank_improvement",
    "rr_improvement",
    "best_r1",
    "baseline_fail_best_success",
    "all",
)

SELECTION_METRICS = ("R1", "mAP", "R5", "R10")

CSV_FIELD_ORDER = [
    "query_index",
    "query_pid",
    "caption",
    "selected_combination",
    "baseline_top1_identity",
    "gate_top1_identity",
    "baseline_top1_index",
    "gate_top1_index",
    "baseline_top1_score",
    "gate_top1_score",
    "baseline_r1_correct",
    "gate_r1_correct",
    "baseline_z_topK",
    "gate_z_topK",
    "baseline_green_count@K",
    "gate_green_count@K",
    "green_gap",
    "baseline_first_correct_rank",
    "gate_first_correct_rank",
    "rank_gain",
    "baseline_has_correct_top10",
    "gate_has_correct_top10",
    "rank_improvement",
    "reciprocal_rank_improvement",
    "baseline_discounted_green_score@K",
    "gate_discounted_green_score@K",
    "early_green_gap",
    "showcase_score",
    "baseline_top10_positive_count",
    "gate_top10_positive_count",
    "baseline_top_indices",
    "gate_top_indices",
    "baseline_top_pids",
    "gate_top_pids",
    "baseline_top_scores",
    "gate_top_scores",
    "baseline_top_image_paths",
    "gate_top_image_paths",
]

FIXED_COMBINATION_KEYS = (
    "selected_combination",
    "fixed_combination",
    "inference_combination",
    "score_task",
    "official_gate_score_task",
    "best_task",
)

COMBINATION_LIST_KEYS = (
    "inference_combinations",
    "gate_combinations",
    "combinations",
    "eval_combinations",
    "candidate_combinations",
)

Image = None
ImageDraw = None
ImageFont = None
ImageOps = None
RESAMPLE_LANCZOS = None


def ensure_pillow() -> None:
    global Image, ImageDraw, ImageFont, ImageOps, RESAMPLE_LANCZOS
    if Image is not None:
        return
    try:
        from PIL import Image as pil_image
        from PIL import ImageDraw as pil_image_draw
        from PIL import ImageFont as pil_image_font
        from PIL import ImageOps as pil_image_ops
    except ImportError as error:
        raise ImportError(
            "Pillow is required to render qualitative figures. Install project requirements "
            "or run with --num_figs 0 if you only need CSV/JSON outputs."
        ) from error
    Image = pil_image
    ImageDraw = pil_image_draw
    ImageFont = pil_image_font
    ImageOps = pil_image_ops
    try:
        RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    except AttributeError:
        RESAMPLE_LANCZOS = Image.LANCZOS


@dataclass
class SplitMeta:
    captions: List[str]
    caption_pids: List[int]
    image_paths: List[str]
    image_pids: List[int]
    query_digest: str
    gallery_digest: str


@dataclass
class ScoreComponents:
    qids: torch.Tensor
    gids: torch.Tensor
    captions: List[str]
    image_paths: List[str]
    image_ids: List[str]
    global_scores: torch.Tensor
    alt_scores: Optional[torch.Tensor]
    target_scores: torch.Tensor
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate configured GATE score-task combinations on the complete "
            "test gallery, select the best one, and render R1-RK qualitative "
            "Baseline-vs-GATE comparisons."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_checkpoint", required=True, help="Frozen TBPS/base retriever checkpoint.")
    parser.add_argument("--gate_checkpoint", required=True, help="GATE checkpoint.")
    parser.add_argument("--gate_config", required=True, help="GATE training/evaluation config YAML.")
    parser.add_argument("--output_dir", required=True, help="Directory for CSV/JSON summaries and figures.")
    parser.add_argument("--selection_metric", choices=SELECTION_METRICS, default="R1")
    parser.add_argument("--num_figs", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sort_mode", choices=SORT_MODES, default="best_r1_green_gap")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--cache_inference", action="store_true")
    parser.add_argument("--save_json", action="store_true", help="Also save ranking_results.json and selected_results.json.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Explicit seed override. Defaults to the GATE config/checkpoint args.")
    parser.add_argument(
        "--dataset",
        "--data",
        dest="data",
        choices=list(DATASET_CHOICES),
        default=None,
        help="Explicit dataset override.",
    )
    parser.add_argument("--root_dir", type=Path, default=None, help="Explicit dataset root override.")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.set_defaults(host_model=None)
    parser.add_argument("--query_batch_size", type=int, default=None, help="Override test/query batch size.")
    parser.add_argument("--gallery_chunk_size", type=int, default=0, help="Chunk query scoring to reduce peak memory; 0 disables.")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--img_size", default=None, help='Explicit image-size override, for example "384,128".')
    parser.add_argument("--text_length", type=int, default=None)
    parser.add_argument("--baseline_task", default=None, help="Optional official frozen-base score task to use for Baseline.")
    parser.add_argument(
        "--respect_fixed_combination",
        action="store_true",
        help="If the config names a fixed/selected combination, evaluate only that official score task.",
    )
    parser.add_argument(
        "--continue_on_combination_error",
        action="store_true",
        help="Record failed candidate metric rows instead of aborting immediately.",
    )
    parser.add_argument(
        "--skip_official_check",
        action="store_true",
        help="Skip the expensive official evaluator reproduction check.",
    )
    parser.add_argument("--min_best_green", type=int, default=None)
    parser.add_argument("--max_baseline_green", type=int, default=None)
    parser.add_argument("--baseline_name", default="Frozen TBPS")
    parser.add_argument("--gate_name", default="Frozen TBPS + GATE")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()


def parse_img_size(value: Any) -> Tuple[int, int]:
    if value is None:
        return (384, 128)
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, str):
        parts = [part.strip() for part in value.strip().strip("()[]").split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError("--img_size must look like 'height,width'")
        return (int(parts[0]), int(parts[1]))
    if isinstance(value, Sequence) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError("Could not parse image size from {!r}".format(value))


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.num_figs < 0:
        raise ValueError("--num_figs must be non-negative")
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if args.query_batch_size is not None and args.query_batch_size <= 0:
        raise ValueError("--query_batch_size must be positive")
    if args.gallery_chunk_size < 0:
        raise ValueError("--gallery_chunk_size must be non-negative")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.text_length is not None and args.text_length <= 0:
        raise ValueError("--text_length must be positive")
    if args.seed is not None and (args.seed < 0 or args.seed >= 2**32):
        raise ValueError("--seed must be in [0, 2**32)")
    if args.min_best_green is not None and args.min_best_green < 0:
        raise ValueError("--min_best_green must be non-negative")
    if args.max_baseline_green is not None and args.max_baseline_green < 0:
        raise ValueError("--max_baseline_green must be non-negative")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    parse_img_size(args.img_size) if args.img_size is not None else None


def resolve_showcase_thresholds(args: argparse.Namespace) -> Tuple[int, int]:
    default_min_best_green = 3 if args.top_k <= 5 else 5
    default_max_baseline_green = 2 if args.top_k <= 5 else 3
    min_best_green = default_min_best_green if args.min_best_green is None else int(args.min_best_green)
    max_baseline_green = default_max_baseline_green if args.max_baseline_green is None else int(args.max_baseline_green)
    return min_best_green, max_baseline_green


def _first_config_value(config: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Optional[Any], Optional[str]]:
    for key in keys:
        if key in config and config[key] not in (None, ""):
            return config[key], key
    return None, None


def resolve_dataset_and_root(
    cli_args: argparse.Namespace,
    gate_config_values: Mapping[str, Any],
    config_path: Path,
) -> Tuple[str, Path, List[Dict[str, Any]]]:
    overrides: List[Dict[str, Any]] = []
    raw_dataset, dataset_key = _first_config_value(gate_config_values, ("dataset_name", "dataset", "data"))
    config_dataset = _canonical_dataset_name(raw_dataset)
    if cli_args.data is not None:
        dataset = cli_args.data
        overrides.append({"name": "data", "source": "cli", "value": dataset, "config_value": raw_dataset})
    elif config_dataset is not None:
        dataset = config_dataset
    else:
        raise ValueError("GATE config must provide dataset_name/dataset/data, or pass --data explicitly")

    raw_root, root_key = _first_config_value(gate_config_values, ("root_dir", "data_root", "dataset_root", "root"))
    if cli_args.root_dir is not None:
        root_dir = cli_args.root_dir.expanduser().resolve()
        overrides.append({"name": "root_dir", "source": "cli", "value": str(root_dir), "config_value": raw_root})
    elif raw_root not in (None, ""):
        raw_root_path = Path(str(raw_root)).expanduser()
        if raw_root_path.is_absolute():
            root_dir = raw_root_path.resolve()
        else:
            bases = [Path.cwd(), REPO_ROOT, REPO_ROOT / "2024-CVPR-RDE", REPO_ROOT.parent, config_path.parent]
            candidates: List[Path] = []
            seen = set()
            for base in bases:
                candidate = (base / raw_root_path).resolve()
                if str(candidate) not in seen:
                    candidates.append(candidate)
                    seen.add(str(candidate))
            root_dir = candidates[0]
            for candidate in candidates:
                if not candidate.exists():
                    continue
                try:
                    _resolve_dataset_root(dataset, candidate, cli_args.split)
                    root_dir = candidate
                    break
                except Exception:
                    continue
    else:
        raise ValueError("GATE config must provide root_dir/data_root/dataset_root, or pass --root_dir explicitly")

    if dataset_key:
        overrides.append({"name": "dataset_source_key", "source": "gate_config", "value": dataset_key})
    if root_key:
        overrides.append({"name": "root_source_key", "source": "gate_config", "value": root_key})
    return dataset, root_dir, overrides


def resolve_host_model(cli_args: argparse.Namespace, gate_config_values: Mapping[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    if cli_args.host_model is not None:
        return cli_args.host_model, {"name": "host_model", "source": "cli", "value": cli_args.host_model}
    raw, _ = _first_config_value(gate_config_values, ("host_model", "base_host_model"))
    if raw is not None and str(raw).lower() in {"clip", "itself"}:
        return str(raw).lower(), {"name": "host_model", "source": "gate_config", "value": str(raw).lower()}
    return "itself", None


def maybe_record_override(overrides: List[Dict[str, Any]], name: str, value: Any, config_value: Any) -> None:
    if value is not None:
        overrides.append({"name": name, "source": "cli", "value": _json_safe(value), "config_value": _json_safe(config_value)})


def normalize_rde_loss_names_for_eval(
    args: SimpleNamespace,
    gate_config_values: Mapping[str, Any],
    overrides: List[Dict[str, Any]],
) -> None:
    """Avoid RDE model/build.py's silent exit() on inherited IRRA loss defaults."""
    raw_loss_names = str(getattr(args, "loss_names", "") or "")
    tasks = [task.strip() for task in raw_loss_names.split("+") if task.strip()]
    valid_by_lower = {
        "tal": "TAL",
        "trl": "TRL",
        "infonce": "InfoNCE",
        "sdm": "SDM",
    }
    for task in tasks:
        canonical = valid_by_lower.get(task.lower())
        if canonical is not None:
            if raw_loss_names != canonical:
                overrides.append(
                    {
                        "name": "loss_names",
                        "source": "rde_eval_compatibility",
                        "value": canonical,
                        "config_value": _json_safe(gate_config_values.get("loss_names")),
                        "previous_value": raw_loss_names,
                        "reason": (
                            "RDE model construction accepts TAL/TRL/InfoNCE/SDM and otherwise calls exit(); "
                            "qualitative evaluation uses encoders only, so a compatible RDE loss selector is sufficient."
                        ),
                    }
                )
                _log("RDE eval compatibility: normalized loss_names from {!r} to {!r}".format(raw_loss_names, canonical))
            args.loss_names = canonical
            return
    raise ValueError(
        "RDE backbone requires loss_names containing one of TAL, TRL, InfoNCE, or SDM. "
        "Resolved loss_names={!r}. This usually means the provided base checkpoint/config is not an RDE run; "
        "check --base_checkpoint and --gate_config.".format(raw_loss_names)
    )


def prepare_output_dir(output_dir: Path, overwrite: bool, save_json_rows: bool) -> Path:
    generated = [
        output_dir / "combination_results.csv",
        output_dir / "combination_results.json",
        output_dir / "selected_combination.json",
        output_dir / "ranking_results.csv",
        output_dir / "selected_results.csv",
        output_dir / "summary.json",
        output_dir / "checkpoint_load_report.json",
        output_dir / "resolved_config.yaml",
        output_dir / "resolved_gate_config.yaml",
        output_dir / "skipped_queries.json",
    ]
    if save_json_rows:
        generated.extend([output_dir / "ranking_results.json", output_dir / "selected_results.json"])
    figs_dir = output_dir / "figs"
    collisions = [path for path in generated if path.exists()]
    if figs_dir.is_dir():
        collisions.extend(figs_dir.glob("*.png"))
    if collisions and not overwrite:
        preview = "\n".join("  {}".format(path) for path in collisions[:12])
        raise FileExistsError("Output files already exist; pass --overwrite to replace them:\n" + preview)

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in generated:
            if path.is_file():
                path.unlink()
        if figs_dir.is_dir():
            shutil.rmtree(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cache").mkdir(parents=True, exist_ok=True)
    return figs_dir


def sequence_digest(items: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(str(item).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def split_meta_from_loaders(img_loader: Any, txt_loader: Any) -> SplitMeta:
    image_dataset = getattr(img_loader, "test_img_set", getattr(img_loader, "dataset", None))
    text_dataset = getattr(txt_loader, "test_txt_set", getattr(txt_loader, "dataset", None))
    if image_dataset is None or text_dataset is None:
        raise RuntimeError("Official dataloaders did not expose dataset objects")
    for name in ("image_pids", "img_paths"):
        if not hasattr(image_dataset, name):
            raise RuntimeError("Image dataset does not expose {}".format(name))
    for name in ("caption_pids", "captions"):
        if not hasattr(text_dataset, name):
            raise RuntimeError("Text dataset does not expose {}".format(name))

    captions = [str(caption) for caption in text_dataset.captions]
    caption_pids = [int(pid) for pid in text_dataset.caption_pids]
    image_paths = [str(path) for path in image_dataset.img_paths]
    image_pids = [int(pid) for pid in image_dataset.image_pids]
    return SplitMeta(
        captions=captions,
        caption_pids=caption_pids,
        image_paths=image_paths,
        image_pids=image_pids,
        query_digest=sequence_digest(list(zip(caption_pids, captions))),
        gallery_digest=sequence_digest(list(zip(image_pids, image_paths))),
    )


def ensure_order_integrity(components: ScoreComponents, split_meta: SplitMeta) -> None:
    expected_qids = torch.as_tensor(split_meta.caption_pids, dtype=torch.long)
    expected_gids = torch.as_tensor(split_meta.image_pids, dtype=torch.long)
    if components.qids.shape != expected_qids.shape or not torch.equal(components.qids.cpu().long(), expected_qids):
        raise RuntimeError("Query PID order differs between cached/inferred scores and official dataloader metadata")
    if components.gids.shape != expected_gids.shape or not torch.equal(components.gids.cpu().long(), expected_gids):
        raise RuntimeError("Gallery PID order differs between cached/inferred scores and official dataloader metadata")
    if list(components.captions) != list(split_meta.captions):
        raise RuntimeError("Caption order differs between cached/inferred scores and official dataloader metadata")
    if list(components.image_paths) != list(split_meta.image_paths):
        raise RuntimeError("Gallery image-path order differs between cached/inferred scores and official dataloader metadata")


def checkpoint_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def cache_metadata(
    args: SimpleNamespace,
    cli_args: argparse.Namespace,
    config_path: Path,
    base_checkpoint: Path,
    gate_checkpoint: Path,
    split_meta: SplitMeta,
    candidate_filter: Optional[List[str]],
) -> Dict[str, Any]:
    relevant_gate_values = {key: getattr(args, key, None) for key in sorted(GATE_CONFIG_KEYS) if hasattr(args, key)}
    return {
        "script": "compare_gate_qualitative_r1_r10.py",
        "cache_version": 2,
        "base_checkpoint": checkpoint_signature(base_checkpoint),
        "gate_checkpoint": checkpoint_signature(gate_checkpoint),
        "gate_config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "dataset": str(getattr(args, "dataset_name", "")),
        "split": str(cli_args.split),
        "query_count": len(split_meta.captions),
        "gallery_count": len(split_meta.image_paths),
        "query_caption_pid_digest": split_meta.query_digest,
        "gallery_path_pid_digest": split_meta.gallery_digest,
        "top_m": int(getattr(args, "top_m", 0)),
        "img_size": list(parse_img_size(getattr(args, "img_size", (384, 128)))),
        "text_length": int(getattr(args, "text_length", 77)),
        "retrieval_mode": {
            "host_model": str(getattr(cli_args, "host_model", "itself")),
            "only_global": bool(getattr(args, "only_global", False)),
            "enrichment_space": str(getattr(args, "enrichment_space", "global")),
            "topm_rank_space": str(getattr(args, "topm_rank_space", "host_global")),
        },
        "gate_values": relevant_gate_values,
        "candidate_filter": candidate_filter or [],
        "gallery_chunk_size": int(cli_args.gallery_chunk_size),
        "query_batch_size": int(cli_args.query_batch_size),
    }


def stable_json_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(_json_safe(payload), sort_keys=True).encode("utf-8")).hexdigest()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, str(tmp_path))
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def score_cache_path(output_dir: Path, metadata: Mapping[str, Any]) -> Path:
    return output_dir / "cache" / "inference_{}.pt".format(stable_json_digest(metadata)[:24])


def try_load_score_cache(path: Path, metadata: Mapping[str, Any]) -> Optional[ScoreComponents]:
    if not path.is_file():
        return None
    payload = torch_load(path)
    if payload.get("metadata") != dict(metadata):
        return None
    return ScoreComponents(
        qids=payload["qids"].cpu().long(),
        gids=payload["gids"].cpu().long(),
        captions=[str(value) for value in payload["captions"]],
        image_paths=[str(value) for value in payload["image_paths"]],
        image_ids=[str(value) for value in payload["image_ids"]],
        global_scores=payload["global_scores"].cpu().float(),
        alt_scores=payload["alt_scores"].cpu().float() if payload.get("alt_scores") is not None else None,
        target_scores=payload["target_scores"].cpu().float(),
        metadata=dict(payload["metadata"]),
    )


def save_score_cache(path: Path, components: ScoreComponents) -> None:
    atomic_torch_save(
        {
            "metadata": components.metadata,
            "qids": components.qids.cpu(),
            "gids": components.gids.cpu(),
            "captions": list(components.captions),
            "image_paths": list(components.image_paths),
            "image_ids": list(components.image_ids),
            "global_scores": components.global_scores.cpu(),
            "alt_scores": components.alt_scores.cpu() if components.alt_scores is not None else None,
            "target_scores": components.target_scores.cpu(),
        },
        path,
    )


def _unwrap_state_dict_flexible(checkpoint: Any) -> Tuple[Mapping[str, Any], Optional[str], List[str]]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping or contain a state dictionary")
    top_level_keys = [str(key) for key in checkpoint.keys()]
    for key in ("state_dict", "model", "model_state_dict", "module", "network", "net"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value, key, top_level_keys
    return checkpoint, None, top_level_keys


def _strip_repeated_prefixes(key: str, prefixes: Sequence[str]) -> str:
    current = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if current.startswith(prefix):
                current = current[len(prefix):]
                changed = True
    return current


def _candidate_keys_for_role(raw_key: str, role: str) -> List[str]:
    prefixes = ("module.", "model.", "network.", "net.")
    normalized = _strip_repeated_prefixes(str(raw_key), prefixes)
    candidates: List[str] = []

    def add(key: str) -> None:
        if key and key not in candidates:
            candidates.append(key)

    add(str(raw_key))
    add(normalized)
    if role == "gate":
        if normalized.startswith("gate."):
            add("target_enricher." + normalized[len("gate."):])
        if normalized.startswith("target_enricher."):
            add(normalized)
        else:
            add("target_enricher." + normalized)
        if normalized.startswith("base_model.gate."):
            add("target_enricher." + normalized[len("base_model.gate."):])
    else:
        if normalized.startswith("base_model."):
            add(normalized[len("base_model."):])
        else:
            add("base_model." + normalized)
    return candidates


def _load_selected_state_flexible(
    model: Any,
    checkpoint_path: Path,
    role: str,
    include_model_key: Any,
    allow_unexpected_key: Any,
) -> Dict[str, Any]:
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    state_dict, state_dict_key, top_level_keys = _unwrap_state_dict_flexible(checkpoint)
    model_state = model.state_dict()
    loaded_keys: List[str] = []
    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    shape_mismatches: List[Dict[str, Any]] = []
    prefix_transformations: List[Dict[str, str]] = []
    mapped: Dict[str, Any] = {}
    loaded_tensor_count = 0
    loaded_element_count = 0

    for raw_key, value in state_dict.items():
        raw_key_s = str(raw_key)
        if not torch.is_tensor(value):
            continue
        target_key = None
        for candidate in _candidate_keys_for_role(raw_key_s, role):
            if candidate in model_state and include_model_key(candidate):
                target_key = candidate
                break
        if target_key is None:
            normalized = _strip_repeated_prefixes(raw_key_s, ("module.", "model.", "network.", "net."))
            suffix_matches = [
                model_key
                for model_key in model_state.keys()
                if include_model_key(model_key) and model_key.endswith(normalized)
            ]
            if len(suffix_matches) == 1:
                target_key = suffix_matches[0]
                prefix_transformations.append({"from": normalized, "to": target_key, "mode": "suffix_match"})
        if target_key is None:
            normalized = _strip_repeated_prefixes(raw_key_s, ("module.", "model.", "network.", "net."))
            if allow_unexpected_key(normalized):
                continue
            unexpected_keys.append(normalized)
            continue
        if tuple(model_state[target_key].shape) != tuple(value.shape):
            shape_mismatches.append(
                {
                    "checkpoint_key": raw_key_s,
                    "model_key": target_key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[target_key].shape),
                }
            )
            continue
        mapped[target_key] = value.detach().clone()
        loaded_keys.append(target_key)
        loaded_tensor_count += 1
        loaded_element_count += int(value.numel())
        if target_key != raw_key_s:
            prefix_transformations.append({"from": raw_key_s, "to": target_key})

    for key in model_state.keys():
        if include_model_key(key) and key not in mapped:
            missing_keys.append(key)

    if loaded_tensor_count <= 0 or loaded_element_count <= 0:
        raise RuntimeError("{} checkpoint loaded no meaningful compatible tensors: {}".format(role, checkpoint_path))

    next_state = dict(model_state)
    next_state.update(mapped)
    model.load_state_dict(next_state, strict=True)
    return {
        "role": role,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "top_level_keys": top_level_keys,
        "state_dict_key_used": state_dict_key,
        "loaded_key_count": len(loaded_keys),
        "loaded_tensor_count": loaded_tensor_count,
        "loaded_element_count": loaded_element_count,
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "shape_mismatch_count": len(shape_mismatches),
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "prefix_transformations": prefix_transformations,
        "strict": False,
        "loader": "script_flexible",
    }


def load_base_and_gate_with_fallback(model: Any, base_checkpoint: Path, gate_checkpoint: Path) -> List[Dict[str, Any]]:
    official_loader_error = None
    try:
        reports = _load_base_and_gate(model, base_checkpoint, gate_checkpoint)
        for report in reports:
            report.setdefault("loaded_tensor_count", report.get("loaded_key_count", 0))
            report.setdefault("missing_key_count", len(report.get("missing_keys", [])))
            report.setdefault("unexpected_key_count", len(report.get("unexpected_keys", [])))
            report.setdefault("shape_mismatch_count", len(report.get("shape_mismatches", [])))
            report["loader"] = "gallery_conditioned_query_common._load_base_and_gate"
        return reports
    except Exception as official_error:
        official_loader_error = str(official_error)
        _log("Official strict checkpoint loader failed; retrying flexible compatibility loader: {}".format(official_loader_error))

    def base_include(key: str) -> bool:
        return "target_enricher" not in key

    def base_allow_unexpected(key: str) -> bool:
        return "target_enricher" in key

    def gate_include(key: str) -> bool:
        return key.startswith("target_enricher.")

    def gate_allow_unexpected(key: str) -> bool:
        return not key.startswith("target_enricher.")

    reports = [
        _load_selected_state_flexible(model, base_checkpoint, "base", base_include, base_allow_unexpected),
        _load_selected_state_flexible(model, gate_checkpoint, "gate", gate_include, gate_allow_unexpected),
    ]
    reports[0]["official_loader_error"] = official_loader_error
    return reports


def build_score_components(
    spec: RepoSpec,
    model: Any,
    img_loader: Any,
    txt_loader: Any,
    args: SimpleNamespace,
    repo_root: Path,
    device: torch.device,
    split_meta: SplitMeta,
    metadata: Mapping[str, Any],
    query_batch_size: int,
    gallery_chunk_size: int,
    log_interval: int,
) -> ScoreComponents:
    core = _core_model(model)
    features = _extract_features(spec, model, img_loader, txt_loader, args, repo_root, log_interval=log_interval)
    active_queries = _active_query_features(spec, args, features.host_text, features.alt_text)

    all_gallery_indices = list(range(int(features.gids.numel())))
    _log("Building and finalizing complete-gallery GATE evidence cache")
    full_cache = _subset_and_finalize_cache(core, features.raw_target_cache, features.gids, all_gallery_indices, device)
    full_retrieval = _cache_retrieval_features(full_cache)

    _log("Computing complete-gallery frozen-base score components")
    host_queries = F.normalize(features.host_text.float(), p=2, dim=1)
    host_gallery = F.normalize(features.host_image.float(), p=2, dim=1)
    global_scores = _score_matrix(host_queries, host_gallery, gallery_chunk_size).cpu().float()

    alt_scores = None
    if features.alt_text is not None and features.alt_image is not None:
        alt_scores = _score_matrix(
            F.normalize(features.alt_text.float(), p=2, dim=1),
            F.normalize(features.alt_image.float(), p=2, dim=1),
            gallery_chunk_size,
        ).cpu().float()

    _log("Computing complete-gallery GATE-enriched target score component")
    full_enriched, _ = _enrich_all_queries_imported(
        spec,
        core,
        args,
        full_cache,
        active_queries,
        features.host_text,
        features.alt_text,
        query_batch_size,
        stage_name="selected/full-gallery GATE inference",
        log_interval=log_interval,
    )
    target_scores = _score_matrix(
        F.normalize(full_enriched.float(), p=2, dim=1),
        full_retrieval,
        gallery_chunk_size,
    ).cpu().float()
    del full_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    components = ScoreComponents(
        qids=features.qids.cpu().long(),
        gids=features.gids.cpu().long(),
        captions=list(split_meta.captions),
        image_paths=list(split_meta.image_paths),
        image_ids=list(features.image_ids),
        global_scores=global_scores,
        alt_scores=alt_scores,
        target_scores=target_scores,
        metadata=dict(metadata),
    )
    ensure_order_integrity(components, split_meta)
    return components


def _enrich_all_queries_imported(*args: Any, **kwargs: Any) -> Any:
    common = importlib.import_module("gallery_conditioned_query_common")
    return common._enrich_all_queries(*args, **kwargs)


def is_gate_candidate(task: Mapping[str, Any]) -> bool:
    name = str(task.get("task", ""))
    return task.get("proto_lambda") is not None or "+proto(" in name or name.startswith("target+proto")


def task_settings(task: Mapping[str, Any], args: SimpleNamespace) -> Dict[str, Any]:
    settings = {
        key: _json_safe(value)
        for key, value in task.items()
        if key not in {"scores", "base_scores"}
    }
    settings["gate_inference"] = {
        "top_m": int(getattr(args, "top_m", 0)),
        "extractor_mode": str(getattr(args, "extractor_mode", "")),
        "enrichment_space": str(getattr(args, "enrichment_space", "")),
        "topm_rank_space": str(getattr(args, "topm_rank_space", "")),
        "topm_rank_lambda": float(getattr(args, "topm_rank_lambda", 0.0)),
        "residual_gate": str(getattr(args, "residual_gate", "")),
        "enrich_gamma": _json_safe(getattr(args, "enrich_gamma", None)),
        "context_pooling": str(getattr(args, "context_pooling", "")),
        "mixer_depth": int(getattr(args, "mixer_depth", 0)),
        "mixer_dim": int(getattr(args, "mixer_dim", 0)),
    }
    return settings


def matrix_gib(rows: int, cols: int, dtype_bytes: int = 4) -> float:
    return float(rows) * float(cols) * float(dtype_bytes) / float(1024**3)


def lazy_base_task_specs(spec: RepoSpec, args: SimpleNamespace, has_alt_scores: bool) -> List[Dict[str, Any]]:
    if spec.repo_kind == "rde":
        if not has_alt_scores:
            raise ValueError("RDE official evaluation requires TSE/retrieval score components")
        return [
            {"task": "BGE", "base_task": "BGE", "kind": "global"},
            {"task": "TSE", "base_task": "TSE", "kind": "alt"},
            {"task": "BGE+TSE", "base_task": "BGE+TSE", "kind": "avg_global_alt"},
        ]
    if spec.repo_kind == "irra":
        if not has_alt_scores:
            raise ValueError("IRRA official evaluation requires retrieval score components")
        return [
            {"task": "global", "base_task": "global", "kind": "global"},
            {"task": "retrieval", "base_task": "retrieval", "kind": "alt"},
        ]
    if spec.repo_kind == "adapter":
        return [{"task": "global", "base_task": "global", "kind": "global"}]

    base_tasks: List[Dict[str, Any]] = [{"task": "global", "base_task": "global", "kind": "global"}]
    if bool(getattr(args, "only_global", False)):
        return base_tasks
    if not has_alt_scores:
        raise ValueError("Prototype ITSELF non-global evaluation requires GRAB/TSE score components")
    base_tasks.append({"task": "grab", "base_task": "grab", "kind": "grab"})
    for lambda_value in _global_grab_lambdas():
        task = "global+grab({})".format(_format_lambda(lambda_value))
        base_tasks.append(
            {
                "task": task,
                "base_task": task,
                "kind": "global_grab",
                "global_grab_lambda": float(lambda_value),
            }
        )
    return base_tasks


def lazy_gate_task_specs(
    spec: RepoSpec,
    args: SimpleNamespace,
    base_tasks: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    gate_tasks: List[Dict[str, Any]] = []
    if spec.repo_kind == "irra":
        base_reference = dict(base_tasks[0]) if base_tasks else {"task": "global", "base_task": "global", "kind": "global"}
        gate_tasks.append(
            {
                "task": "target+proto(1)",
                "base_task": str(base_reference["task"]),
                "base_spec": base_reference,
                "proto_lambda": 1.0,
                "kind": "target_only",
            }
        )
    for proto_lambda in _prototype_lambdas():
        proto_value = _format_lambda(proto_lambda)
        for base in base_tasks:
            task = "{}+proto({})".format(base["task"], proto_value)
            gate_tasks.append(
                {
                    "task": task,
                    "base_task": str(base["task"]),
                    "base_spec": dict(base),
                    "proto_lambda": float(proto_lambda),
                }
            )
    return gate_tasks


def memory_safe_official_task_specs(
    spec: RepoSpec,
    args: SimpleNamespace,
    components: ScoreComponents,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_tasks = lazy_base_task_specs(spec, args, components.alt_scores is not None)
    gate_tasks = lazy_gate_task_specs(spec, args, base_tasks)
    return gate_tasks, base_tasks


def get_scaled_alt_like_global(spec: RepoSpec, components: ScoreComponents, args: SimpleNamespace) -> Optional[torch.Tensor]:
    if spec.repo_kind != "prototype":
        return None
    if bool(getattr(args, "only_global", False)):
        return None
    if components.alt_scores is None:
        raise ValueError("Cannot build global+grab score tasks without alternate/GRAB scores")
    _log("Preparing one reusable rowwise-scaled GRAB score matrix for global+grab candidates")
    return _scale_scores_like(components.alt_scores, components.global_scores).detach().cpu().float()


def compute_base_task_scores(
    task: Mapping[str, Any],
    components: ScoreComponents,
    scaled_alt_like_global: Optional[torch.Tensor],
) -> torch.Tensor:
    kind = str(task.get("kind", "global"))
    if kind == "global":
        return components.global_scores
    if kind in {"grab", "alt"}:
        if components.alt_scores is None:
            raise ValueError("Base task {} requires alternate/retrieval scores".format(task.get("task", kind)))
        return components.alt_scores
    if kind == "avg_global_alt":
        if components.alt_scores is None:
            raise ValueError("Base task {} requires alternate/retrieval scores".format(task.get("task", kind)))
        return ((components.global_scores + components.alt_scores) / 2.0).detach().cpu().float()
    if kind == "global_grab":
        if scaled_alt_like_global is None:
            raise ValueError("Base task global+grab requires scaled alternate scores")
        lambda_value = float(task["global_grab_lambda"])
        return (lambda_value * components.global_scores + (1.0 - lambda_value) * scaled_alt_like_global).detach().cpu().float()
    raise ValueError("Unknown base score-task kind: {}".format(kind))


def compute_gate_task_scores(
    task: Mapping[str, Any],
    components: ScoreComponents,
    scaled_alt_like_global: Optional[torch.Tensor],
) -> torch.Tensor:
    if str(task.get("kind", "")) == "target_only":
        return components.target_scores
    proto_lambda = float(task.get("proto_lambda", 0.0))
    if abs(proto_lambda - 1.0) < 1e-12:
        return components.target_scores
    base_scores = compute_base_task_scores(task["base_spec"], components, scaled_alt_like_global)
    if abs(proto_lambda) < 1e-12:
        return base_scores
    scaled_base = _scale_scores_like(base_scores, components.target_scores).detach().cpu().float()
    release_transient_score(base_scores, components)
    scaled_base.mul_(1.0 - proto_lambda)
    scaled_base.add_(components.target_scores, alpha=proto_lambda)
    return scaled_base

def release_transient_score(score: Optional[torch.Tensor], components: ScoreComponents) -> None:
    if score is None:
        return
    borrowed = score is components.global_scores or score is components.alt_scores or score is components.target_scores
    if not borrowed:
        del score
        gc.collect()


def _extract_combination_names(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        if "name" in value:
            return [str(value["name"])]
        if "task" in value:
            return [str(value["task"])]
        return [str(key) for key in value.keys()]
    if isinstance(value, Sequence):
        names: List[str] = []
        for item in value:
            names.extend(_extract_combination_names(item))
        return names
    return [str(value)]


def configured_combination_filter(
    gate_config_values: Mapping[str, Any],
    respect_fixed_combination: bool,
) -> Tuple[Optional[List[str]], Dict[str, Any]]:
    metadata: Dict[str, Any] = {"source": None, "requested_names": []}
    if respect_fixed_combination:
        for key in FIXED_COMBINATION_KEYS:
            if key in gate_config_values and gate_config_values[key] not in (None, ""):
                names = _extract_combination_names(gate_config_values[key])
                metadata = {"source": key, "requested_names": names, "mode": "respect_fixed_combination"}
                return names, metadata
    for key in COMBINATION_LIST_KEYS:
        if key in gate_config_values and gate_config_values[key] not in (None, ""):
            names = _extract_combination_names(gate_config_values[key])
            metadata = {"source": key, "requested_names": names, "mode": "configured_sweep"}
            return names, metadata
    return None, metadata


def discover_combinations(
    gate_tasks: Sequence[Mapping[str, Any]],
    args: SimpleNamespace,
    gate_config_values: Mapping[str, Any],
    respect_fixed_combination: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates = [
        {
            "name": str(task["task"]),
            "score_task": str(task["task"]),
            "task": task,
            "settings": task_settings(task, args),
        }
        for task in gate_tasks
        if is_gate_candidate(task)
    ]
    if not candidates:
        raise ValueError("The official GATE evaluator exposed no valid GATE candidate score tasks")

    requested_names, filter_metadata = configured_combination_filter(gate_config_values, respect_fixed_combination)
    if not requested_names:
        filter_metadata["resolved_names"] = [candidate["name"] for candidate in candidates]
        return candidates, filter_metadata

    by_name = {candidate["name"]: candidate for candidate in candidates}
    resolved: List[Dict[str, Any]] = []
    missing: List[str] = []
    for requested in requested_names:
        requested = str(requested)
        if requested in by_name:
            resolved.append(by_name[requested])
            continue
        missing.append(requested)

    if missing == ["itself"]:
        pure_gate = [
            candidate
            for candidate in candidates
            if abs(float(candidate["settings"].get("proto_lambda", -1.0)) - 1.0) < 1e-12
        ]
        if pure_gate:
            preferred = next((candidate for candidate in pure_gate if candidate["score_task"] == "global+proto(1)"), None)
            if preferred is None:
                preferred = sorted(pure_gate, key=lambda candidate: str(candidate["score_task"]))[0]
            alias = dict(preferred)
            alias["name"] = "itself"
            alias["settings"] = dict(alias["settings"])
            alias["settings"]["score_task"] = alias["score_task"]
            alias["settings"]["combination_alias"] = "itself"
            filter_metadata["alias_policy"] = "literal itself mapped to the official pure GATE target score task with proto_lambda=1"
            resolved = [alias]
            missing = []

    if missing and len(requested_names) == 1 and len(candidates) == 1:
        alias = dict(candidates[0])
        alias["name"] = str(requested_names[0])
        alias["settings"] = dict(alias["settings"])
        alias["settings"]["score_task"] = alias["score_task"]
        alias["settings"]["combination_alias"] = str(requested_names[0])
        filter_metadata["alias_policy"] = "single configured name mapped to sole official candidate"
        resolved = [alias]
        missing = []

    if missing:
        raise ValueError(
            "Configured combination(s) were not official GATE score tasks: {}. "
            "Available tasks: {}".format(", ".join(missing), ", ".join(sorted(by_name.keys())))
        )
    filter_metadata["resolved_names"] = [candidate["name"] for candidate in resolved]
    return resolved, filter_metadata


def metrics_percent(scores: torch.Tensor, qids: torch.Tensor, gids: torch.Tensor, query_chunk_size: int = 256) -> Dict[str, float]:
    similarity = scores.detach().cpu().float()
    qids = qids.detach().cpu().long()
    gids = gids.detach().cpu().long()
    if similarity.shape[0] != qids.numel() or similarity.shape[1] != gids.numel():
        raise ValueError("Metric score matrix shape does not match query/gallery ids")
    max_rank = min(10, similarity.shape[1])
    cmc_sum = torch.zeros(max_rank, dtype=torch.float64)
    ap_sum = 0.0
    minp_sum = 0.0
    total_queries = int(qids.numel())
    step = max(1, int(query_chunk_size))
    rank_positions = torch.arange(1, similarity.shape[1] + 1, dtype=torch.float32).view(1, -1)

    for start in range(0, total_queries, step):
        end = min(total_queries, start + step)
        chunk_scores = similarity[start:end]
        chunk_qids = qids[start:end]
        indices = torch.argsort(chunk_scores, dim=1, descending=True)
        pred_labels = gids[indices.cpu()]
        matches = pred_labels.eq(chunk_qids.view(-1, 1))
        num_rel = matches.sum(1)
        if bool((num_rel <= 0).any().item()):
            raise ValueError("At least one query has no positive gallery image; official retrieval metrics are undefined")
        all_cmc = matches[:, :max_rank].cumsum(1)
        all_cmc[all_cmc > 1] = 1
        cmc_sum += all_cmc.float().sum(0).double()
        tmp_cmc = matches.cumsum(1)
        last_rel_rank = (tmp_cmc != num_rel.view(-1, 1)).sum(1) + 1
        minp_sum += float((num_rel.float() / last_rel_rank.float()).sum().item())
        ap_curve = tmp_cmc.float()
        ap_curve.div_(rank_positions)
        ap_curve.mul_(matches)
        ap_sum += float((ap_curve.sum(1) / num_rel).sum().item())

    all_cmc = cmc_sum / float(total_queries) * 100.0
    m_ap = ap_sum / float(total_queries) * 100.0
    m_inp = minp_sum / float(total_queries) * 100.0

    def recall_at(k: int) -> float:
        return float(all_cmc[min(k, max_rank) - 1].item())

    return {
        "R1": recall_at(1),
        "R5": recall_at(5),
        "R10": recall_at(10),
        "mAP": float(m_ap),
        "mINP": float(m_inp),
    }


def evaluate_combinations(
    combinations: Sequence[Mapping[str, Any]],
    qids: torch.Tensor,
    gids: torch.Tensor,
    continue_on_error: bool,
    components: Optional[ScoreComponents] = None,
    scaled_alt_like_global: Optional[torch.Tensor] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, combination in enumerate(combinations):
        row: Dict[str, Any] = {
            "combination_index": int(index),
            "combination_name": str(combination["name"]),
            "score_task": str(combination["score_task"]),
            "status": "ok",
            "selected": False,
        }
        row.update({key: value for key, value in combination["settings"].items() if key not in {"gate_inference"}})
        scores = None
        try:
            if components is None:
                scores = combination["task"]["scores"]
            else:
                scores = compute_gate_task_scores(combination["task"], components, scaled_alt_like_global)
            metrics = metrics_percent(scores, qids, gids)
            row.update(metrics)
            row["rSum"] = float(metrics["R1"] + metrics["R5"] + metrics["R10"])
        except Exception as error:
            if not continue_on_error:
                raise
            row["status"] = "error"
            row["error"] = str(error)
        finally:
            if components is not None:
                release_transient_score(scores, components)
        rows.append(row)
    return rows


def select_best_combination(rows: Sequence[Mapping[str, Any]], selection_metric: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    valid = [dict(row) for row in rows if row.get("status") == "ok"]
    if not valid:
        raise RuntimeError("No combination completed successfully")
    tie_metrics = [metric for metric in ("mAP", "R5", "R10") if metric != selection_metric]

    def sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        return tuple([-float(row[selection_metric])] + [-float(row[metric]) for metric in tie_metrics] + [str(row["combination_name"])])

    ordered = sorted(valid, key=sort_key)
    selected = dict(ordered[0])
    same_main = [
        row["combination_name"]
        for row in ordered
        if abs(float(row[selection_metric]) - float(selected[selection_metric])) <= 1e-12
    ]
    tie_break = {
        "selection_metric": selection_metric,
        "tie_break_order": tie_metrics + ["lexicographic_combination_name"],
        "candidates_tied_on_selection_metric": same_main,
        "selected_after_tie_break": selected["combination_name"],
    }
    return selected, tie_break


def select_baseline_task(
    base_tasks: Sequence[Mapping[str, Any]],
    qids: torch.Tensor,
    gids: torch.Tensor,
    requested_task: Optional[str],
    official_reproduction: Mapping[str, Any],
    components: Optional[ScoreComponents] = None,
    scaled_alt_like_global: Optional[torch.Tensor] = None,
) -> Tuple[str, Dict[str, float], torch.Tensor, Dict[str, Any]]:
    base_by_name = {str(task["task"]): task for task in base_tasks}
    if requested_task:
        if requested_task not in base_by_name:
            raise ValueError("--baseline_task {!r} is not one of the official base tasks: {}".format(requested_task, ", ".join(base_by_name)))
        selected_name = requested_task
        policy = {"source": "cli", "requested_task": requested_task}
    else:
        official_task = official_reproduction.get("base_best_task")
        if official_task in base_by_name:
            selected_name = str(official_task)
            policy = {"source": "official_evaluator_best_task", "official_task": official_task}
        else:
            base_rows = []
            for task in base_tasks:
                if components is None:
                    scores = task["scores"]
                else:
                    scores = compute_base_task_scores(task, components, scaled_alt_like_global)
                metrics = metrics_percent(scores, qids, gids)
                if components is not None:
                    release_transient_score(scores, components)
                base_rows.append({"task": str(task["task"]), **metrics})
            selected = sorted(base_rows, key=lambda row: (-row["R1"], -row["mAP"], -row["R5"], -row["R10"], row["task"]))[0]
            selected_name = selected["task"]
            policy = {"source": "reconstructed_best_base_task", "official_task": official_task}
    if components is None:
        selected_scores = base_by_name[selected_name]["scores"].detach().cpu().float()
    else:
        selected_scores = compute_base_task_scores(base_by_name[selected_name], components, scaled_alt_like_global).detach().cpu().float()
    metrics = metrics_percent(selected_scores, qids, gids)
    return selected_name, metrics, selected_scores, policy


def top_list_values(
    order: torch.Tensor,
    scores: torch.Tensor,
    gallery_pids: torch.Tensor,
    image_paths: Sequence[str],
    k: int,
) -> Tuple[List[int], List[int], List[float], List[str]]:
    top_indices = [int(index) for index in order[:k].tolist()]
    top_pids = [int(gallery_pids[index].item()) for index in top_indices]
    top_scores = [float(scores[index].item()) for index in top_indices]
    top_paths = [str(image_paths[index]) for index in top_indices]
    return top_indices, top_pids, top_scores, top_paths


def green_vector_and_scores(matches: torch.Tensor, top_k: int) -> Tuple[List[int], int, float]:
    top_matches = matches[: min(top_k, matches.numel())]
    z_topk = [int(value) for value in top_matches.to(dtype=torch.int64).tolist()]
    green_count = int(sum(z_topk))
    discounted = 0.0
    for rank, is_green in enumerate(z_topk, start=1):
        if is_green:
            discounted += 1.0 / math.log2(rank + 1.0)
    return z_topk, green_count, float(discounted)


def showcase_score_from_values(
    gate_r1_correct: bool,
    baseline_r1_correct: bool,
    gate_green_count: int,
    baseline_green_count: int,
    gate_first_correct_rank: int,
    rank_gain: int,
    early_green_gap: float,
) -> float:
    green_gap = gate_green_count - baseline_green_count
    return float(
        100.0 * int(gate_r1_correct)
        + 1000.0 * int(gate_r1_correct and not baseline_r1_correct)
        + 120.0 * green_gap
        + 35.0 * gate_green_count
        - 25.0 * baseline_green_count
        + 30.0 * early_green_gap
        + 2.0 * rank_gain
        + 10.0 / max(float(gate_first_correct_rank), 1.0)
    )


def row_stats_for_model(
    scores: torch.Tensor,
    order: torch.Tensor,
    query_pid: int,
    gallery_pids: torch.Tensor,
    image_paths: Sequence[str],
    display_k: int,
    recall_k: int = 10,
) -> Dict[str, Any]:
    ranked_pids = gallery_pids[order.cpu()]
    matches = ranked_pids.eq(int(query_pid))
    positive_positions = matches.nonzero(as_tuple=False).view(-1)
    if positive_positions.numel() == 0:
        raise RuntimeError("Query identity has no positive gallery image")

    first_correct_rank = int(positive_positions[0].item()) + 1
    top1_index = int(order[0].item())
    top_indices, top_pids, top_scores, top_paths = top_list_values(
        order.cpu(),
        scores.cpu(),
        gallery_pids.cpu(),
        image_paths,
        display_k,
    )
    top10 = matches[: min(recall_k, matches.numel())]
    z_topk, green_count, discounted_green_score = green_vector_and_scores(matches, display_k)
    return {
        "top1_identity": int(gallery_pids[top1_index].item()),
        "top1_index": top1_index,
        "top1_score": float(scores[top1_index].item()),
        "r1_correct": first_correct_rank == 1,
        "first_correct_rank": first_correct_rank,
        "has_correct_top10": bool(top10.any().item()),
        "top10_positive_count": int(top10.sum().item()),
        "z_topK": z_topk,
        "green_count_at_k": green_count,
        "discounted_green_score_at_k": discounted_green_score,
        "top_indices": top_indices,
        "top_pids": top_pids,
        "top_scores": top_scores,
        "top_image_paths": top_paths,
    }


def compute_query_rows(
    baseline_sim: torch.Tensor,
    gate_sim: torch.Tensor,
    captions: Sequence[str],
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
    image_paths: Sequence[str],
    display_k: int,
    selected_combination: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if baseline_sim.shape != gate_sim.shape:
        raise ValueError("Baseline and GATE similarity matrices have different shapes")
    if baseline_sim.shape[0] != len(captions) or baseline_sim.shape[0] != query_pids.numel():
        raise ValueError("Similarity row count does not match query metadata")
    if baseline_sim.shape[1] != gallery_pids.numel() or baseline_sim.shape[1] != len(image_paths):
        raise ValueError("Similarity column count does not match gallery metadata")

    display_k = min(display_k, baseline_sim.shape[1])
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for query_index in tqdm(range(baseline_sim.shape[0]), desc="Computing per-query qualitative rows"):
        query_pid = int(query_pids[query_index].item())
        if not bool(gallery_pids.eq(query_pid).any().item()):
            skipped.append(
                {
                    "query_index": int(query_index),
                    "query_pid": query_pid,
                    "caption": str(captions[query_index]),
                    "reason": "query identity not found in gallery",
                }
            )
            continue

        baseline_order = torch.argsort(baseline_sim[query_index], descending=True)
        gate_order = torch.argsort(gate_sim[query_index], descending=True)
        baseline_stats = row_stats_for_model(
            baseline_sim[query_index],
            baseline_order,
            query_pid,
            gallery_pids,
            image_paths,
            display_k,
        )
        gate_stats = row_stats_for_model(
            gate_sim[query_index],
            gate_order,
            query_pid,
            gallery_pids,
            image_paths,
            display_k,
        )

        baseline_rank = int(baseline_stats["first_correct_rank"])
        gate_rank = int(gate_stats["first_correct_rank"])
        rank_gain = baseline_rank - gate_rank
        rr_improvement = (1.0 / gate_rank) - (1.0 / baseline_rank)
        baseline_green_count = int(baseline_stats["green_count_at_k"])
        gate_green_count = int(gate_stats["green_count_at_k"])
        green_gap = gate_green_count - baseline_green_count
        baseline_discounted = float(baseline_stats["discounted_green_score_at_k"])
        gate_discounted = float(gate_stats["discounted_green_score_at_k"])
        early_green_gap = gate_discounted - baseline_discounted
        showcase_score = showcase_score_from_values(
            gate_r1_correct=bool(gate_stats["r1_correct"]),
            baseline_r1_correct=bool(baseline_stats["r1_correct"]),
            gate_green_count=gate_green_count,
            baseline_green_count=baseline_green_count,
            gate_first_correct_rank=gate_rank,
            rank_gain=rank_gain,
            early_green_gap=early_green_gap,
        )

        rows.append(
            {
                "query_index": int(query_index),
                "query_pid": query_pid,
                "caption": str(captions[query_index]),
                "selected_combination": selected_combination,
                "baseline_top1_identity": baseline_stats["top1_identity"],
                "gate_top1_identity": gate_stats["top1_identity"],
                "baseline_top1_index": baseline_stats["top1_index"],
                "gate_top1_index": gate_stats["top1_index"],
                "baseline_top1_score": baseline_stats["top1_score"],
                "gate_top1_score": gate_stats["top1_score"],
                "baseline_r1_correct": bool(baseline_stats["r1_correct"]),
                "gate_r1_correct": bool(gate_stats["r1_correct"]),
                "baseline_z_topK": baseline_stats["z_topK"],
                "gate_z_topK": gate_stats["z_topK"],
                "baseline_green_count@K": baseline_green_count,
                "gate_green_count@K": gate_green_count,
                "green_gap": int(green_gap),
                "baseline_first_correct_rank": baseline_rank,
                "gate_first_correct_rank": gate_rank,
                "rank_gain": int(rank_gain),
                "baseline_has_correct_top10": bool(baseline_stats["has_correct_top10"]),
                "gate_has_correct_top10": bool(gate_stats["has_correct_top10"]),
                "rank_improvement": int(rank_gain),
                "reciprocal_rank_improvement": float(rr_improvement),
                "baseline_discounted_green_score@K": baseline_discounted,
                "gate_discounted_green_score@K": gate_discounted,
                "early_green_gap": float(early_green_gap),
                "showcase_score": showcase_score,
                "baseline_top10_positive_count": int(baseline_stats["top10_positive_count"]),
                "gate_top10_positive_count": int(gate_stats["top10_positive_count"]),
                "baseline_top_indices": baseline_stats["top_indices"],
                "gate_top_indices": gate_stats["top_indices"],
                "baseline_top_pids": baseline_stats["top_pids"],
                "gate_top_pids": gate_stats["top_pids"],
                "baseline_top_scores": baseline_stats["top_scores"],
                "gate_top_scores": gate_stats["top_scores"],
                "baseline_top_image_paths": baseline_stats["top_image_paths"],
                "gate_top_image_paths": gate_stats["top_image_paths"],
            }
        )
    return rows, skipped


def showcase_sort_key(row: Mapping[str, Any]) -> Tuple[float, int, int, int, int, float, int, int, int]:
    return (
        float(row["showcase_score"]),
        int(bool(row["gate_r1_correct"] and not row["baseline_r1_correct"])),
        int(row["green_gap"]),
        int(row["gate_green_count@K"]),
        -int(row["baseline_green_count@K"]),
        float(row["early_green_gap"]),
        int(row["rank_gain"]),
        -int(row["gate_first_correct_rank"]),
        int(row["baseline_first_correct_rank"]),
    )


def sort_query_rows(
    rows: Sequence[Mapping[str, Any]],
    sort_mode: str,
    top_k: int,
    min_best_green: int,
    max_baseline_green: int,
) -> List[Mapping[str, Any]]:
    if sort_mode == "best_r1_green_gap":
        candidates = [
            row
            for row in rows
            if row["gate_r1_correct"] and int(row["gate_green_count@K"]) > int(row["baseline_green_count@K"])
        ]
        return sorted(candidates, key=showcase_sort_key, reverse=True)
    if sort_mode == "strict_showcase":
        candidates = [
            row
            for row in rows
            if row["gate_r1_correct"]
            and not row["baseline_r1_correct"]
            and int(row["gate_green_count@K"]) >= min_best_green
            and int(row["baseline_green_count@K"]) <= max_baseline_green
        ]
        return sorted(candidates, key=showcase_sort_key, reverse=True)
    if sort_mode == "best_r1_baseline_not_r1":
        candidates = [row for row in rows if row["gate_r1_correct"] and not row["baseline_r1_correct"]]
        return sorted(
            candidates,
            key=lambda row: (
                row["rank_improvement"],
                row["reciprocal_rank_improvement"],
                row["baseline_first_correct_rank"],
            ),
            reverse=True,
        )
    if sort_mode == "best_r1_baseline_not_r10":
        candidates = [row for row in rows if row["gate_r1_correct"] and not row["baseline_has_correct_top10"]]
        return sorted(candidates, key=lambda row: (row["rank_improvement"], row["reciprocal_rank_improvement"]), reverse=True)
    if sort_mode == "rank_improvement":
        return sorted(rows, key=lambda row: (row["rank_improvement"], row["reciprocal_rank_improvement"]), reverse=True)
    if sort_mode == "rr_improvement":
        return sorted(rows, key=lambda row: (row["reciprocal_rank_improvement"], row["rank_improvement"]), reverse=True)
    if sort_mode == "best_r1":
        candidates = [row for row in rows if row["gate_r1_correct"]]
        return sorted(candidates, key=lambda row: row["baseline_first_correct_rank"], reverse=True)
    if sort_mode == "baseline_fail_best_success":
        candidates = [
            row
            for row in rows
            if row["baseline_first_correct_rank"] > top_k and row["gate_first_correct_rank"] <= top_k
        ]
        return sorted(
            candidates,
            key=lambda row: (
                row["rank_improvement"],
                row["reciprocal_rank_improvement"],
                row["baseline_first_correct_rank"],
            ),
            reverse=True,
        )
    if sort_mode == "all":
        return sorted(rows, key=lambda row: (row["rank_improvement"], row["reciprocal_rank_improvement"]), reverse=True)
    raise ValueError("Unsupported sort mode: {!r}".format(sort_mode))


def recall_from_rows(rows: Sequence[Mapping[str, Any]], prefix: str, rank: int) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if int(row["{}_first_correct_rank".format(prefix)]) <= rank)
    return 100.0 * float(hits) / float(len(rows))


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    values = [float(value) for value in values]
    if not values:
        return None
    return float(sum(values) / len(values))


def verify_reconstructed_recall(
    rows: Sequence[Mapping[str, Any]],
    baseline_metrics: Mapping[str, float],
    gate_metrics: Mapping[str, float],
    skipped_count: int,
    tolerance: float = 0.02,
) -> Dict[str, Any]:
    if skipped_count:
        return {"status": "skipped", "reason": "queries_without_positive_gallery_were_filtered", "skipped_count": int(skipped_count)}
    checks = []
    for prefix, metrics in (("baseline", baseline_metrics), ("gate", gate_metrics)):
        for rank, key in ((1, "R1"), (5, "R5"), (10, "R10")):
            reconstructed = recall_from_rows(rows, prefix, rank)
            official = float(metrics[key])
            checks.append(
                {
                    "name": "{}_{}".format(prefix, key),
                    "reconstructed_percent": reconstructed,
                    "official_percent": official,
                    "abs_diff_pp": abs(reconstructed - official),
                }
            )
    failures = [check for check in checks if check["abs_diff_pp"] > tolerance]
    if failures:
        raise RuntimeError("Reconstructed recall disagrees with official metrics: {}".format(failures))
    return {"status": "ok", "tolerance_percentage_points": tolerance, "checks": checks}


def csv_value(value: Any) -> Any:
    value = _json_safe(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def save_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path, field_order: Sequence[str]) -> None:
    all_fields = list(field_order)
    for row in rows:
        for key in row.keys():
            if key not in all_fields:
                all_fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: csv_value(row.get(key, "")) for key in all_fields})
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    ensure_pillow()
    candidates = []
    if bold:
        candidates.extend(["arialbd.ttf", "Arial Bold.ttf"])
    candidates.extend(
        [
            "arial.ttf",
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def safe_text(text: Any) -> str:
    return str(text).encode("ascii", errors="replace").decode("ascii")


def text_width(draw: ImageDraw.ImageDraw, text: str, used_font: ImageFont.ImageFont) -> int:
    try:
        bbox = draw.textbbox((0, 0), text, font=used_font)
    except UnicodeEncodeError:
        bbox = draw.textbbox((0, 0), safe_text(text), font=used_font)
    return int(bbox[2] - bbox[0])


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    used_font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] | str,
    anchor: Optional[str] = None,
) -> None:
    try:
        draw.text(xy, text, font=used_font, fill=fill, anchor=anchor)
    except UnicodeEncodeError:
        draw.text(xy, safe_text(text), font=used_font, fill=fill, anchor=anchor)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, used_font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = "{} {}".format(current, word)
        if text_width(draw, candidate, used_font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def load_thumbnail(path: str, size: Tuple[int, int]) -> Image.Image:
    ensure_pillow()
    width, height = size
    canvas = Image.new("RGB", size, (245, 247, 250))
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = ImageOps.contain(image, size, method=RESAMPLE_LANCZOS)
            x = (width - image.width) // 2
            y = (height - image.height) // 2
            canvas.paste(image, (x, y))
    except Exception:
        draw = ImageDraw.Draw(canvas)
        small = font(13)
        draw_text(draw, (width // 2, height // 2 - 8), "image", small, (120, 126, 138), anchor="mm")
        draw_text(draw, (width // 2, height // 2 + 10), "missing", small, (120, 126, 138), anchor="mm")
    return canvas


def draw_retrieval_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    row_name: str,
    top_paths: Sequence[str],
    top_pids: Sequence[int],
    top_scores: Sequence[float],
    query_pid: int,
    y: int,
    left: int,
    grid_left: int,
    thumb_w: int,
    thumb_h: int,
    gap: int,
) -> None:
    label_font = font(20, bold=True)
    small_font = font(14)
    rank_font = font(16, bold=True)
    draw_text(draw, (left, y + 36), row_name, label_font, (17, 24, 39))
    for idx, path in enumerate(top_paths):
        x = grid_left + idx * (thumb_w + gap)
        draw_text(draw, (x + thumb_w // 2, y), "R{}".format(idx + 1), rank_font, (17, 24, 39), anchor="mt")
        image_y = y + 24
        thumb = load_thumbnail(path, (thumb_w, thumb_h))
        canvas.paste(thumb, (x, image_y))
        correct = int(top_pids[idx]) == int(query_pid)
        color = (22, 163, 74) if correct else (220, 38, 38)
        for offset in range(4):
            draw.rectangle([x - offset, image_y - offset, x + thumb_w + offset, image_y + thumb_h + offset], outline=color)
        draw_text(
            draw,
            (x + thumb_w // 2, image_y + thumb_h + 10),
            "sim {:.4f}".format(float(top_scores[idx])),
            small_font,
            (55, 65, 81),
            anchor="mt",
        )
        draw_text(draw, (x + thumb_w // 2, image_y + thumb_h + 30), "pid {}".format(int(top_pids[idx])), small_font, (55, 65, 81), anchor="mt")


def render_query_figure(
    row: Mapping[str, Any],
    out_path: Path,
    baseline_name: str,
    gate_name: str,
    sort_mode: str,
    selected_combination: str,
    dpi: int,
) -> None:
    ensure_pillow()
    display_k = len(row["baseline_top_image_paths"])
    thumb_w = 140
    thumb_h = 210
    gap = 14
    left = 28
    row_label_w = 170
    grid_left = left + row_label_w
    right = 28
    width = grid_left + display_k * thumb_w + max(display_k - 1, 0) * gap + right
    title_font = font(22, bold=True)
    caption_font = font(17)
    meta_font = font(16)
    temp = Image.new("RGB", (width, 200), "white")
    temp_draw = ImageDraw.Draw(temp)
    caption_lines = wrap_text(temp_draw, "Caption: {}".format(row["caption"]), caption_font, width - 2 * left)[:4]
    top_h = 140 + len(caption_lines) * 24
    row_h = thumb_h + 70
    height = top_h + 2 * row_h + 36
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title = (
        "Baseline rank {} | GATE rank {} | gain {}".format(
            row["baseline_first_correct_rank"],
            row["gate_first_correct_rank"],
            row["rank_gain"],
        )
    )
    score_line = (
        "green {}->{} (gap {}) | score {:.1f} | {}".format(
            row["baseline_green_count@K"],
            row["gate_green_count@K"],
            row["green_gap"],
            float(row.get("showcase_score", 0.0)),
            sort_mode,
        )
    )
    combo_line = "Combination: {} | query index {} | query pid {}".format(
        selected_combination,
        row["query_index"],
        row["query_pid"],
    )
    draw_text(draw, (left, 18), title, title_font, (17, 24, 39))
    draw_text(draw, (left, 50), score_line, meta_font, (55, 65, 81))
    draw_text(draw, (left, 74), combo_line, meta_font, (75, 85, 99))
    y_text = 106
    for line in caption_lines:
        draw_text(draw, (left, y_text), line, caption_font, (31, 41, 55))
        y_text += 24
    baseline_y = top_h
    gate_y = top_h + row_h
    draw.line((left, baseline_y - 14, width - right, baseline_y - 14), fill=(229, 231, 235), width=1)
    draw_retrieval_row(
        canvas,
        draw,
        baseline_name,
        row["baseline_top_image_paths"],
        row["baseline_top_pids"],
        row["baseline_top_scores"],
        int(row["query_pid"]),
        baseline_y,
        left,
        grid_left,
        thumb_w,
        thumb_h,
        gap,
    )
    draw_retrieval_row(
        canvas,
        draw,
        gate_name,
        row["gate_top_image_paths"],
        row["gate_top_pids"],
        row["gate_top_scores"],
        int(row["query_pid"]),
        gate_y,
        left,
        grid_left,
        thumb_w,
        thumb_h,
        gap,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path), "PNG", dpi=(dpi, dpi))


def save_selected_score_cache(
    output_dir: Path,
    base_scores: torch.Tensor,
    gate_scores: torch.Tensor,
    metadata: Mapping[str, Any],
    baseline_task: str,
    selected_combination: Mapping[str, Any],
) -> Dict[str, Any]:
    selected_metadata = dict(metadata)
    selected_metadata["baseline_task"] = baseline_task
    selected_metadata["selected_combination"] = _json_safe(selected_combination)
    path = output_dir / "cache" / "selected_{}.pt".format(stable_json_digest(selected_metadata)[:24])
    atomic_torch_save(
        {
            "metadata": selected_metadata,
            "baseline_similarity": base_scores.cpu(),
            "gate_similarity": gate_scores.cpu(),
        },
        path,
    )
    return {"path": str(path), "metadata_digest": stable_json_digest(selected_metadata), "status": "saved"}


def build_summary(
    cli_args: argparse.Namespace,
    args: SimpleNamespace,
    base_checkpoint: Path,
    gate_checkpoint: Path,
    config_path: Path,
    selected_combination: Mapping[str, Any],
    baseline_task: str,
    baseline_metrics: Mapping[str, float],
    gate_metrics: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    sorted_candidate_count: int,
    saved_figures: int,
    cache_status: Mapping[str, Any],
    checkpoint_reports: Sequence[Mapping[str, Any]],
    overrides: Sequence[Mapping[str, Any]],
    reproduction_check: Mapping[str, Any],
    recall_check: Mapping[str, Any],
    combination_filter: Mapping[str, Any],
    started_at: float,
) -> Dict[str, Any]:
    rank_gains = [float(row["rank_gain"]) for row in rows]
    return {
        "dataset": str(getattr(args, "dataset_name", "")),
        "split": cli_args.split,
        "base_checkpoint": str(base_checkpoint),
        "gate_checkpoint": str(gate_checkpoint),
        "gate_config": str(config_path),
        "selected_combination": selected_combination.get("combination_name"),
        "selected_score_task": selected_combination.get("score_task"),
        "selection_metric": cli_args.selection_metric,
        "baseline_task": baseline_task,
        "baseline_metrics_percent": dict(baseline_metrics),
        "selected_gate_metrics_percent": dict(gate_metrics),
        "baseline_R@1": float(baseline_metrics["R1"]),
        "baseline_R@5": float(baseline_metrics["R5"]),
        "baseline_R@10": float(baseline_metrics["R10"]),
        "baseline_mAP": float(baseline_metrics["mAP"]),
        "selected_GATE_R@1": float(gate_metrics["R1"]),
        "selected_GATE_R@5": float(gate_metrics["R5"]),
        "selected_GATE_R@10": float(gate_metrics["R10"]),
        "selected_GATE_mAP": float(gate_metrics["mAP"]),
        "num_queries": int(len(rows)),
        "num_gallery_images": int(getattr(args, "number_of_gallery_images", 0)),
        "num_skipped_queries": int(len(skipped)),
        "number_of_improved_queries": int(sum(1 for row in rows if row["rank_gain"] > 0)),
        "number_of_degraded_queries": int(sum(1 for row in rows if row["rank_gain"] < 0)),
        "number_of_GATE_R1_correct_Baseline_R1_wrong_queries": int(
            sum(1 for row in rows if row["gate_r1_correct"] and not row["baseline_r1_correct"])
        ),
        "number_of_Baseline_R1_correct_GATE_R1_wrong_queries": int(
            sum(1 for row in rows if row["baseline_r1_correct"] and not row["gate_r1_correct"])
        ),
        "mean_first_correct_rank_improvement": mean_or_none(rank_gains),
        "qualitative_sort_mode": cli_args.sort_mode,
        "number_of_candidate_cases": int(sorted_candidate_count),
        "number_of_saved_figures": int(saved_figures),
        "top_k": int(cli_args.top_k),
        "min_best_green": int(resolve_showcase_thresholds(cli_args)[0]),
        "max_baseline_green": int(resolve_showcase_thresholds(cli_args)[1]),
        "cache_status": dict(cache_status),
        "checkpoint_loading_statistics": list(checkpoint_reports),
        "cli_overrides": list(overrides),
        "combination_filter": dict(combination_filter),
        "official_reproduction_check": dict(reproduction_check),
        "reconstructed_recall_check": dict(recall_check),
        "resolved_device": str(_json_safe(getattr(args, "resolved_device", ""))),
        "img_size": list(parse_img_size(getattr(args, "img_size", (384, 128)))),
        "text_length": int(getattr(args, "text_length", 77)),
        "top_m": int(getattr(args, "top_m", 0)),
        "showcase_score_definition": (
            "100*gate_r1 + 1000*(gate_r1 and not baseline_r1) + 120*green_gap "
            "+ 35*gate_green_count@K - 25*baseline_green_count@K + 30*early_green_gap "
            "+ 2*rank_gain + 10/gate_first_correct_rank"
        ),
        "discounted_green_score_definition": "sum_{rank=1..K} z_rank / log2(rank + 1)",
        "runtime_seconds": float(time.time() - started_at),
    }


def main() -> None:
    started_at = time.time()
    cli_args = parse_args()
    validate_cli_args(cli_args)
    min_best_green, max_baseline_green = resolve_showcase_thresholds(cli_args)

    spec = RepoSpec(
        repository_name="enrichment-rde",
        repo_kind="rde",
        code_root=(REPO_ROOT / "2024-CVPR-RDE").resolve(),
        default_host_model="rde",
        metrics_module="utils.metrics",
        logger_name="enrichment_rde.gate_qualitative",
        supports_host_model=False,
    )
    if str(spec.code_root) not in sys.path:
        sys.path.insert(0, str(spec.code_root))
    repo_root = _repo_root_from_spec(spec).resolve()

    base_checkpoint = resolve_path(cli_args.base_checkpoint)
    gate_checkpoint = resolve_path(cli_args.gate_checkpoint)
    config_path = resolve_path(cli_args.gate_config)
    output_dir = resolve_path(cli_args.output_dir)
    for label, path in (("base checkpoint", base_checkpoint), ("GATE checkpoint", gate_checkpoint), ("GATE config", config_path)):
        if not path.is_file():
            raise FileNotFoundError("{} not found: {}".format(label, path))

    _log("Loading host checkpoint metadata and GATE config")
    host_args, _, host_settings_source = _load_host_args(base_checkpoint)
    gate_args, gate_config_values = _load_config(config_path)
    dataset, root_dir, overrides = resolve_dataset_and_root(cli_args, gate_config_values, config_path)
    host_model = spec.default_host_model
    host_model_override = {"name": "host_model", "source": "repo_spec", "value": host_model}
    if host_model_override is not None:
        overrides.append(host_model_override)
    _validate_gate_dataset_compatibility(gate_config_values, dataset)

    explicit_query_batch_size = cli_args.query_batch_size is not None
    explicit_num_workers = cli_args.num_workers is not None
    common_cli = SimpleNamespace(
        data=dataset,
        root_dir=root_dir,
        split=cli_args.split,
        host_model=host_model,
        num_workers=int(cli_args.num_workers if cli_args.num_workers is not None else getattr(gate_args, "num_workers", 4)),
        query_batch_size=int(cli_args.query_batch_size if cli_args.query_batch_size is not None else getattr(gate_args, "test_batch_size", 512)),
        gallery_chunk_size=int(cli_args.gallery_chunk_size),
    )
    cli_args.host_model = host_model
    cli_args.query_batch_size = common_cli.query_batch_size
    cli_args.num_workers = common_cli.num_workers
    maybe_record_override(overrides, "query_batch_size", cli_args.query_batch_size if explicit_query_batch_size else None, getattr(gate_args, "test_batch_size", None))
    maybe_record_override(overrides, "num_workers", cli_args.num_workers if explicit_num_workers else None, getattr(gate_args, "num_workers", None))
    maybe_record_override(overrides, "gallery_chunk_size", cli_args.gallery_chunk_size if cli_args.gallery_chunk_size else None, 0)

    args, applied_gate_keys = _apply_gate_config_overlay(host_args, gate_args, gate_config_values)
    args.gate_config_file = str(config_path)
    args.config_file = str(config_path)
    args.base_checkpoint = str(base_checkpoint)
    args.gate_checkpoint = str(gate_checkpoint)
    args.eval_output_dir = str(output_dir)
    _ensure_eval_defaults(args, spec, common_cli)
    normalize_rde_loss_names_for_eval(args, gate_config_values, overrides)

    if cli_args.seed is not None:
        maybe_record_override(overrides, "seed", cli_args.seed, getattr(args, "seed", None))
        args.seed = int(cli_args.seed)
    if cli_args.img_size is not None:
        maybe_record_override(overrides, "img_size", parse_img_size(cli_args.img_size), getattr(args, "img_size", None))
        args.img_size = parse_img_size(cli_args.img_size)
    if cli_args.text_length is not None:
        maybe_record_override(overrides, "text_length", cli_args.text_length, getattr(args, "text_length", None))
        args.text_length = int(cli_args.text_length)
    args.eval_log_interval = max(0.0, float(cli_args.log_interval))

    figs_dir = prepare_output_dir(output_dir, cli_args.overwrite, cli_args.save_json)
    _atomic_write_text(output_dir / "resolved_config.yaml", _resolved_config_yaml(args))
    _atomic_write_text(output_dir / "resolved_gate_config.yaml", _resolved_config_yaml(gate_args))

    _configure_reproducibility(
        int(getattr(args, "seed", 1)),
        deterministic=bool(getattr(args, "deterministic", True)),
        warn_only=bool(getattr(args, "deterministic_warn_only", False)),
    )
    device = _resolve_device(cli_args.device)
    args.resolved_device = str(device)
    _log("Using device: {}".format(device))

    _log("Building model and loading checkpoints")
    first_checkpoint = _torch_load_checkpoint(base_checkpoint)
    first_state, _, _ = _unwrap_state_dict_flexible(first_checkpoint)
    num_classes = _infer_num_classes_from_state_dict(first_state, getattr(args, "num_classes", None))
    build_model = importlib.import_module("model").build_model
    model = build_model(args, num_classes)
    model.to(device)
    checkpoint_reports = load_base_and_gate_with_fallback(model, base_checkpoint, gate_checkpoint)
    _freeze_and_eval(model)
    core = _core_model(model)
    if not hasattr(core, "target_enricher"):
        raise RuntimeError("Loaded model does not expose required GATE module target_enricher")
    for report in checkpoint_reports:
        _log(
            "Checkpoint load [{}]: loaded_tensors={} missing={} unexpected={} shape_mismatch={}".format(
                report.get("role"),
                report.get("loaded_tensor_count", report.get("loaded_key_count", 0)),
                report.get("missing_key_count", len(report.get("missing_keys", []))),
                report.get("unexpected_key_count", len(report.get("unexpected_keys", []))),
                report.get("shape_mismatch_count", len(report.get("shape_mismatches", []))),
            )
        )
    _atomic_write_json(output_dir / "checkpoint_load_report.json", checkpoint_reports)

    _log("Building official dataloaders")
    img_loader, txt_loader = _build_eval_loaders(args, cli_args.split)
    _validate_loaded_eval_loaders(img_loader, txt_loader, args)
    split_meta = split_meta_from_loaders(img_loader, txt_loader)
    _log("Dataloaders ready: queries={} gallery={}".format(len(split_meta.captions), len(split_meta.image_paths)))

    candidate_filter, filter_metadata = configured_combination_filter(gate_config_values, cli_args.respect_fixed_combination)
    metadata = cache_metadata(args, cli_args, config_path, base_checkpoint, gate_checkpoint, split_meta, candidate_filter)
    cache_path = score_cache_path(output_dir, metadata)
    cache_status: Dict[str, Any] = {
        "enabled": bool(cli_args.cache_inference),
        "path": str(cache_path),
        "status": "disabled",
    }
    components = None
    if cli_args.cache_inference:
        components = try_load_score_cache(cache_path, metadata)
        if components is not None:
            cache_status["status"] = "hit"
            _log("Loaded validated inference cache: {}".format(cache_path))
            ensure_order_integrity(components, split_meta)
        else:
            cache_status["status"] = "miss"

    if components is None:
        _log("Running complete test-set baseline and GATE inference")
        components = build_score_components(
            spec,
            model,
            img_loader,
            txt_loader,
            args,
            repo_root,
            device,
            split_meta,
            metadata,
            int(cli_args.query_batch_size),
            int(cli_args.gallery_chunk_size),
            max(1, int(cli_args.log_interval)),
        )
        if cli_args.cache_inference:
            save_score_cache(cache_path, components)
            cache_status["status"] = "saved"
            _log("Saved inference cache: {}".format(cache_path))

    q_count = int(components.qids.numel())
    g_count = int(components.gids.numel())
    _log(
        "One full similarity matrix is {:.2f} GiB at float32 (queries={} gallery={}); "
        "using memory-safe lazy score-task evaluation".format(
            matrix_gib(q_count, g_count),
            q_count,
            g_count,
        )
    )
    scaled_alt_like_global = get_scaled_alt_like_global(spec, components, args)
    _log("Building official score-task candidate descriptors")
    gate_tasks, base_tasks = memory_safe_official_task_specs(spec, args, components)
    combinations, filter_metadata = discover_combinations(
        gate_tasks,
        args,
        gate_config_values,
        cli_args.respect_fixed_combination,
    )

    _log("Evaluating {} GATE candidate combination(s)".format(len(combinations)))
    combination_rows = evaluate_combinations(
        combinations,
        components.qids,
        components.gids,
        cli_args.continue_on_combination_error,
        components,
        scaled_alt_like_global,
    )
    selected_row, tie_break = select_best_combination(combination_rows, cli_args.selection_metric)
    for row in combination_rows:
        row["selected"] = str(row["combination_name"]) == str(selected_row["combination_name"])
        row["selection_metric"] = cli_args.selection_metric
        if row.get("status") == "ok":
            row["selection_metric_value"] = row.get(cli_args.selection_metric)
    _atomic_write_csv(output_dir / "combination_results.csv", combination_rows)
    _atomic_write_json(output_dir / "combination_results.json", combination_rows)

    combination_by_name = {str(combination["name"]): combination for combination in combinations}
    selected_combination = combination_by_name[str(selected_row["combination_name"])]
    selected_gate_scores = compute_gate_task_scores(
        selected_combination["task"],
        components,
        scaled_alt_like_global,
    ).detach().cpu().float()
    selected_gate_metrics = {
        key: float(selected_row[key])
        for key in ("R1", "R5", "R10", "mAP", "mINP")
        if key in selected_row
    }

    official_reproduction: Dict[str, Any]
    reproduction_check: Dict[str, Any]
    if cli_args.skip_official_check:
        official_reproduction = {"status": "skipped", "reason": "--skip_official_check"}
    else:
        official_reproduction = _run_official_reproduction(spec, model, img_loader, txt_loader, args)

    baseline_task, baseline_metrics, baseline_scores, baseline_policy = select_baseline_task(
        base_tasks,
        components.qids,
        components.gids,
        cli_args.baseline_task,
        official_reproduction,
        components,
        scaled_alt_like_global,
    )
    if cli_args.skip_official_check:
        reproduction_check = {"status": "skipped", "reason": "--skip_official_check"}
    else:
        protocol_base = {baseline_task: {key: float(value) / 100.0 for key, value in baseline_metrics.items() if key in {"R1", "R5", "R10", "mAP", "mINP"}}}
        protocol_gate = {
            str(row["score_task"]): {
                key: float(row[key]) / 100.0
                for key in ("R1", "R5", "R10", "mAP", "mINP")
                if key in row and row.get("status") == "ok"
            }
            for row in combination_rows
            if row.get("status") == "ok"
        }
        reproduction_check = _compare_reproduction(spec, protocol_base, protocol_gate, official_reproduction)
        if official_reproduction.get("status") == "failed" or reproduction_check.get("status") == "failed":
            _atomic_write_json(
                output_dir / "summary.json",
                {
                    "status": "failed",
                    "official_reproduction": official_reproduction,
                    "reproduction_check": reproduction_check,
                },
            )
            raise RuntimeError("Official evaluator reproduction check failed; see summary.json")
    selected_cache_info = None
    if cli_args.cache_inference:
        selected_cache_info = save_selected_score_cache(
            output_dir,
            baseline_scores,
            selected_gate_scores,
            metadata,
            baseline_task,
            selected_combination["settings"],
        )
        cache_status["selected_similarity_cache"] = selected_cache_info

    selected_combination_payload = {
        "selected_combination_name": selected_row["combination_name"],
        "selected_score_task": selected_row["score_task"],
        "resolved_combination_settings": selected_combination["settings"],
        "selection_metric": cli_args.selection_metric,
        "retrieval_metrics_percent": selected_gate_metrics,
        "tie_break_result": tie_break,
        "config_path": str(config_path),
        "base_checkpoint": str(base_checkpoint),
        "gate_checkpoint": str(gate_checkpoint),
        "dataset": str(getattr(args, "dataset_name", "")),
        "split": cli_args.split,
        "number_of_queries": int(components.qids.numel()),
        "number_of_gallery_images": int(components.gids.numel()),
        "baseline_task": baseline_task,
        "baseline_task_policy": baseline_policy,
        "candidate_filter": filter_metadata,
    }
    _atomic_write_json(output_dir / "selected_combination.json", selected_combination_payload)

    _log("Computing per-query qualitative comparison rows")
    rows, skipped = compute_query_rows(
        baseline_scores,
        selected_gate_scores,
        components.captions,
        components.qids,
        components.gids,
        components.image_paths,
        cli_args.top_k,
        str(selected_row["combination_name"]),
    )
    _atomic_write_json(output_dir / "skipped_queries.json", skipped)
    recall_check = verify_reconstructed_recall(rows, baseline_metrics, selected_gate_metrics, len(skipped))
    sorted_rows = sort_query_rows(rows, cli_args.sort_mode, cli_args.top_k, min_best_green, max_baseline_green)
    selected_rows = list(sorted_rows[: cli_args.num_figs])

    save_rows_csv(rows, output_dir / "ranking_results.csv", CSV_FIELD_ORDER)
    save_rows_csv(selected_rows, output_dir / "selected_results.csv", CSV_FIELD_ORDER)
    if cli_args.save_json:
        _atomic_write_json(output_dir / "ranking_results.json", rows)
        _atomic_write_json(output_dir / "selected_results.json", selected_rows)

    _log("Rendering {} qualitative figure(s)".format(len(selected_rows)))
    saved_figures = 0
    for selected_rank, row in enumerate(tqdm(selected_rows, desc="Saving figures"), start=1):
        filename = (
            "{:04d}_q{:06d}_pid{}_improve{:+d}.png".format(
                selected_rank,
                int(row["query_index"]),
                int(row["query_pid"]),
                int(row["rank_gain"]),
            )
        )
        render_query_figure(
            row,
            figs_dir / filename,
            cli_args.baseline_name,
            cli_args.gate_name,
            cli_args.sort_mode,
            str(selected_row["combination_name"]),
            cli_args.dpi,
        )
        saved_figures += 1

    summary = build_summary(
        cli_args,
        args,
        base_checkpoint,
        gate_checkpoint,
        config_path,
        selected_row,
        baseline_task,
        baseline_metrics,
        selected_gate_metrics,
        rows,
        skipped,
        len(sorted_rows),
        saved_figures,
        cache_status,
        checkpoint_reports,
        overrides + [{"name": "applied_gate_config_keys", "source": "gate_config_overlay", "value": applied_gate_keys}, {"name": "host_settings_source", "value": host_settings_source}],
        reproduction_check,
        recall_check,
        filter_metadata,
        started_at,
    )
    _atomic_write_json(output_dir / "summary.json", summary)
    _log("Done. Outputs saved to {}".format(output_dir))


if __name__ == "__main__":
    main()
