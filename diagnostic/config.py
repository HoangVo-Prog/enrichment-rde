"""CLI parsing, RDE config loading, and deterministic setup."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch

from diagnostic.constants import BOOTSTRAP_UNITS, CUE_SCORERS, DATASET_NAMES, RETRIEVER_NAMES


def default_repo_args() -> Dict[str, Any]:
    return {
        "noisy_rate": 0.2,
        "noisy_file": "",
        "tau": 0.015,
        "select_ratio": 0.3,
        "margin": 0.1,
        "local_rank": 0,
        "name": "baseline",
        "output_dir": "logs",
        "log_period": 100,
        "eval_period": 1,
        "val_dataset": "test",
        "resume": False,
        "resume_ckpt_file": "",
        "pretrain_choice": "ViT-B/16",
        "temperature": 0.02,
        "img_aug": False,
        "txt_aug": False,
        "cmt_depth": 4,
        "masked_token_rate": 0.8,
        "masked_token_unchanged_rate": 0.1,
        "lr_factor": 5.0,
        "loss_names": "TAL",
        "img_size": (384, 128),
        "stride_size": 16,
        "text_length": 77,
        "vocab_size": 49408,
        "optimizer": "Adam",
        "lr": 1e-5,
        "bias_lr_factor": 2.0,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "weight_decay_bias": 0.0,
        "alpha": 0.9,
        "beta": 0.999,
        "num_epoch": 60,
        "milestones": (20, 50),
        "gamma": 0.1,
        "warmup_factor": 0.1,
        "warmup_epochs": 5,
        "warmup_method": "linear",
        "lrscheduler": "cosine",
        "target_lr": 0,
        "power": 0.9,
        "dataset_name": "CUHK-PEDES",
        "sampler": "random",
        "num_instance": 4,
        "root_dir": "data",
        "batch_size": 128,
        "test_batch_size": 512,
        "num_workers": 8,
        "training": False,
        "distributed": False,
    }


def default_cue_cases_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "prototype" / "cues" / "cue_cases.yaml"
        if candidate.exists():
            return candidate
    return Path("prototype") / "cues" / "cue_cases.yaml"


def repo_code_root() -> Path:
    return Path(__file__).resolve().parents[1] / "2024-CVPR-RDE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cue-swap diagnostic for the RDE retriever"
    )
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--retriever_name", choices=RETRIEVER_NAMES, default="rde")
    parser.add_argument("--config_file", "--retriever_config", dest="retriever_config", type=Path, required=True)
    parser.add_argument("--checkpoint", "--retriever_checkpoint", dest="retriever_checkpoint", type=Path, default=None)
    parser.add_argument("--cue_scorer", choices=CUE_SCORERS, default="off_the_shelf_clip")
    parser.add_argument("--clip_model_name", default="ViT-B/16")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--root_dir", type=Path, default=None)
    parser.add_argument(
        "--cue-cases",
        "--cue_cases",
        "--cases_file",
        dest="cases_file",
        type=Path,
        default=None,
    )
    parser.add_argument("--auto_cases", action="store_true")
    parser.add_argument("--cue_vocab_file", type=Path, default=None)
    parser.add_argument("--gallery_size", type=int, default=500)
    parser.add_argument("--dense_ratio", type=float, default=0.5)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score_mode", choices=["auto", "global", "grab", "fusion"], default="auto")
    parser.add_argument("--lambda_global", type=float, default=0.68)
    parser.add_argument("--lambda_contrast", type=float, default=0.5)
    parser.add_argument("--cue_threshold_quantile", type=float, default=0.75)
    parser.add_argument("--tau_density", type=float, default=0.02)
    parser.add_argument("--min_pair_cue_shift", type=float, default=0.0)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=123)
    parser.add_argument(
        "--bootstrap_unit",
        choices=BOOTSTRAP_UNITS,
        default="unique_query",
        help=(
            "Cluster unit for confidence intervals: case_query preserves the original "
            "case-query-instance bootstrap, unique_query resamples underlying query_id "
            "clusters, both writes both analyses and uses unique_query as primary."
        ),
    )
    parser.add_argument("--enable_random_control", action="store_true")
    parser.add_argument("--save_galleries", action="store_true")
    parser.add_argument("--save_image_paths", action="store_true")
    parser.add_argument("--max_queries_per_case", type=int, default=None)
    parser.add_argument("--min_queries_per_auto_case", type=int, default=10)
    parser.add_argument("--max_auto_cases", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--neutral_strategy", choices=["low_affinity", "random"], default="low_affinity")
    parser.add_argument("--neutral_pool_factor", type=int, default=5)
    parser.add_argument("--tight_hardness_z_tolerance", type=float, default=0.10)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.cases_file is not None and args.auto_cases:
        raise ValueError("Provide either --cases_file or --auto_cases, not both.")
    if args.cases_file is None:
        args.auto_cases = True
    for path_arg in ("retriever_config", "retriever_checkpoint", "cases_file", "cue_vocab_file"):
        path = getattr(args, path_arg)
        if path is not None and not path.exists():
            raise FileNotFoundError(f"--{path_arg} does not exist: {path}")
    if args.gallery_size <= 0:
        raise ValueError("--gallery_size must be positive")
    if not 0.0 <= args.dense_ratio <= 1.0:
        raise ValueError("--dense_ratio must be in [0, 1]")
    if args.num_trials <= 0:
        raise ValueError("--num_trials must be positive")
    if not 0.0 <= args.lambda_global <= 1.0:
        raise ValueError("--lambda_global must be in [0, 1]")
    if args.tau_density <= 0:
        raise ValueError("--tau_density must be positive")
    if not 0.0 <= args.cue_threshold_quantile <= 1.0:
        raise ValueError("--cue_threshold_quantile must be in [0, 1]")
    if args.min_pair_cue_shift < 0:
        raise ValueError("--min_pair_cue_shift must be non-negative")
    if args.bootstrap_iters <= 0:
        raise ValueError("--bootstrap_iters must be positive")
    if args.neutral_pool_factor <= 0:
        raise ValueError("--neutral_pool_factor must be positive")
    if args.tight_hardness_z_tolerance < 0:
        raise ValueError("--tight_hardness_z_tolerance must be non-negative")
    if args.enable_random_control:
        raise NotImplementedError(
            "--enable_random_control is reserved for a future random replacement control. "
            "The mandatory hardness-matched control is always enabled."
        )


def load_repo_args(args: argparse.Namespace) -> SimpleNamespace:
    cfg = default_repo_args()
    import yaml

    with args.retriever_config.open("r", encoding="utf-8") as handle:
        loaded = yaml.load(handle, Loader=yaml.FullLoader)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Retriever config must be a YAML mapping: {args.retriever_config}")
    cfg.update(dict(loaded))

    cfg["dataset_name"] = args.dataset
    if args.root_dir is not None:
        cfg["root_dir"] = str(args.root_dir)
    root_dir = Path(str(cfg["root_dir"]))
    if not root_dir.is_absolute():
        code_root_candidate = repo_code_root() / root_dir
        if code_root_candidate.exists():
            cfg["root_dir"] = str(code_root_candidate)
    cfg["training"] = False
    cfg["distributed"] = False
    if args.test_batch_size is not None:
        cfg["test_batch_size"] = args.test_batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.retriever_checkpoint is None:
        args.retriever_checkpoint = args.retriever_config.resolve().parent / "best.pth"

    img_size = cfg.get("img_size", (384, 128))
    if isinstance(img_size, list):
        img_size = tuple(img_size)
    if isinstance(img_size, str):
        numbers = [int(value) for value in re.findall(r"\d+", img_size)]
        if len(numbers) != 2:
            raise ValueError(f"Could not parse img_size from config value: {img_size}")
        img_size = tuple(numbers)
    cfg["img_size"] = tuple(img_size)
    cfg["milestones"] = tuple(cfg.get("milestones", (20, 50)))
    cfg["diagnostic_output_dir"] = str(args.output_dir)
    return SimpleNamespace(**cfg)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_seed(base_seed: int, *parts: Any) -> int:
    import hashlib

    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**32)


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return device


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, SimpleNamespace):
        return {key: jsonable(value) for key, value in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(key): jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
