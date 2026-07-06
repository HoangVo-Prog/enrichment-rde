import argparse
import os
import os.path as op
import random
import sys
from types import SimpleNamespace


DATASET_CHOICES = ("CUHK-PEDES", "ICFG-PEDES", "RSTPReid")
DEFAULT_CHECKPOINT_NAMES = ("best.pth", "last.pth")

_CHECKPOINT_CONFIG_KEYS = (
    "checkpoint",
    "checkpoint_file",
    "checkpoint_path",
    "model_checkpoint",
    "weights",
    "weights_file",
    "resume_ckpt_file",
    "finetune_clip",
    "finetune",
    "pretrain",
)

_CLI_OVERRIDE_NAMES = (
    "root_dir",
    "test_batch_size",
    "num_workers",
    "seed",
    "deterministic",
    "deterministic_warn_only",
    "target_enrichment",
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
    "residual_gate",
    "enrich_gamma",
    "residual_gate_hidden_dim",
    "lambda_ret",
    "eval_score_chunk_size",
    "strict_target_checkpoint",
)


def _add_boolean_override(parser, name, help_text, negative_help=None):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--{}".format(name),
        dest=name,
        action="store_true",
        default=None,
        help=help_text,
    )
    group.add_argument(
        "--no_{}".format(name),
        dest=name,
        action="store_false",
        default=None,
        help=negative_help or "disable {}".format(name.replace("_", " ")),
    )


def _resolve_user_path(path, base_dir=None):
    if path is None:
        return None
    path = str(path).strip()
    if not path:
        return None
    path = op.expanduser(path)
    if op.isabs(path):
        return op.abspath(path)
    if base_dir is not None:
        return op.abspath(op.join(base_dir, path))
    return op.abspath(path)


def _unique_paths(paths):
    seen = set()
    unique = []
    for path in paths:
        if path is None:
            continue
        normalized = op.normcase(op.abspath(path))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(op.abspath(path))
    return unique


def _candidate_paths(path, base_dirs):
    if isinstance(path, bool) or path is None:
        return []
    path = str(path).strip()
    if not path:
        return []
    path = op.expanduser(path)
    if op.isabs(path):
        return [op.abspath(path)]
    candidates = [op.join(base_dir, path) for base_dir in base_dirs if base_dir]
    candidates.append(path)
    return _unique_paths(candidates)


def _resolve_existing_or_first(path, base_dirs):
    candidates = _candidate_paths(path, base_dirs)
    for candidate in candidates:
        if op.isfile(candidate):
            return candidate
    return candidates[0] if candidates else None


def _resolve_existing_dir_or_first(path, base_dirs):
    candidates = _candidate_paths(path, base_dirs)
    for candidate in candidates:
        if op.isdir(candidate):
            return candidate
    return candidates[0] if candidates else None


def _resolve_config_file(cli_args):
    run_dir = _resolve_user_path(cli_args.run_dir)
    config_file = _resolve_user_path(cli_args.config_file, run_dir)
    if config_file is None:
        if run_dir is None:
            raise ValueError("provide --run_dir or --config_file")
        for filename in ("config.yaml", "configs.yaml"):
            candidate = op.join(run_dir, filename)
            if op.isfile(candidate):
                return op.abspath(candidate), run_dir
        config_file = op.join(run_dir, "configs.yaml")
    return config_file, run_dir or op.dirname(config_file)


def _load_train_defaults():
    from utils.options import get_args as get_train_args

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        return get_train_args()
    finally:
        sys.argv = original_argv


def _load_inference_config(config_file):
    from utils.iotools import load_train_configs

    defaults = vars(_load_train_defaults())
    config_args = load_train_configs(config_file)
    config_values = dict(config_args)
    merged = dict(defaults)
    merged.update(config_values)
    return SimpleNamespace(**merged), set(config_values)


def _configured_path_bases(args, config_file, run_dir, config_keys):
    config_dir = op.dirname(config_file)
    bases = []

    output_dir = getattr(args, "output_dir", None)
    if output_dir and "output_dir" in config_keys:
        bases.extend(_candidate_paths(output_dir, [run_dir, config_dir]))

    bases.extend([run_dir, config_dir])
    return [path for path in _unique_paths(bases) if op.isdir(path)]


def _resolve_checkpoints(cli_args, args, base_dirs, config_keys):
    if cli_args.checkpoint is not None:
        checkpoint = _resolve_existing_or_first(cli_args.checkpoint, base_dirs)
        if not op.isfile(checkpoint):
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
        return [checkpoint]

    configured_missing = []
    for key in _CHECKPOINT_CONFIG_KEYS:
        value = getattr(args, key, None)
        if not value or isinstance(value, bool):
            continue
        checkpoint = _resolve_existing_or_first(value, base_dirs)
        if op.isfile(checkpoint):
            return [checkpoint]
        if key in config_keys and key.startswith("checkpoint"):
            configured_missing.append((key, checkpoint))

    checkpoints = []
    checkpoint_names = cli_args.checkpoint_names or list(DEFAULT_CHECKPOINT_NAMES)
    for filename in checkpoint_names:
        for base_dir in base_dirs:
            candidate = op.join(base_dir, filename)
            if op.isfile(candidate):
                checkpoints.append(op.abspath(candidate))
    checkpoints = _unique_paths(checkpoints)
    if checkpoints:
        return checkpoints

    if configured_missing:
        key, checkpoint = configured_missing[0]
        raise FileNotFoundError(
            "Checkpoint configured by '{}' was not found: {}".format(
                key,
                checkpoint,
            )
        )
    raise FileNotFoundError(
        "Checkpoint not found. Pass --checkpoint, add a checkpoint path to the config, "
        "or place one of {} next to the config/output directory.".format(
            ", ".join(checkpoint_names)
        )
    )


def _resolve_paths(cli_args, args, config_file, run_dir, config_keys):
    base_dirs = _configured_path_bases(args, config_file, run_dir, config_keys)
    checkpoints = _resolve_checkpoints(cli_args, args, base_dirs, config_keys)

    eval_base_dir = op.dirname(checkpoints[0])
    if "output_dir" in config_keys and getattr(args, "output_dir", None):
        output_dir = _resolve_existing_dir_or_first(
            args.output_dir,
            [run_dir, op.dirname(config_file)],
        )
        if output_dir is not None:
            eval_base_dir = output_dir

    eval_dir = _resolve_user_path(cli_args.output_eval_dir, eval_base_dir)
    if eval_dir is None:
        eval_dir = op.join(eval_base_dir, "eval")

    results_file = _resolve_user_path(cli_args.results_file, eval_dir)
    if results_file is None:
        results_file = op.join(eval_dir, "eval_results.json")

    return checkpoints, eval_dir, results_file


def _set_if_not_present(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def _ensure_inference_defaults(args, config_file, config_keys=None):
    config_keys = config_keys or set()
    config_dir = op.dirname(config_file)
    if "output_dir" not in config_keys or not getattr(args, "output_dir", None):
        args.output_dir = config_dir
    _set_if_not_present(args, "root_dir", "data")
    _set_if_not_present(args, "num_workers", 4)
    _set_if_not_present(args, "test_batch_size", 512)
    _set_if_not_present(args, "batch_size", args.test_batch_size)
    _set_if_not_present(args, "seed", 1)
    _set_if_not_present(args, "deterministic", True)
    _set_if_not_present(args, "deterministic_warn_only", False)
    _set_if_not_present(args, "training", False)
    _set_if_not_present(args, "distributed", False)
    _set_if_not_present(args, "local_rank", 0)
    _set_if_not_present(args, "target_enrichment", False)
    _set_if_not_present(args, "enrichment_space", "global")
    _set_if_not_present(args, "topm_rank_space", "host_global")
    _set_if_not_present(args, "eval_score_chunk_size", 0)
    _set_if_not_present(args, "strict_target_checkpoint", False)
    _set_if_not_present(args, "pretrain_choice", "ViT-B/16")
    _set_if_not_present(args, "img_size", (384, 128))
    _set_if_not_present(args, "stride_size", 16)
    _set_if_not_present(args, "loss_names", "TAL")
    _set_if_not_present(args, "select_ratio", 0.3)
    _set_if_not_present(args, "temperature", 0.02)

    if isinstance(args.img_size, list):
        args.img_size = tuple(args.img_size)


def _apply_cli_overrides(args, cli_args, config_file, eval_dir):
    args.training = False
    args.distributed = False
    args.config_file = config_file
    args.eval_output_dir = eval_dir

    for name in _CLI_OVERRIDE_NAMES:
        value = getattr(cli_args, name, None)
        if value is not None:
            setattr(args, name, value)


def _configure_reproducibility(seed, deterministic=True, warn_only=False):
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
            torch.use_deterministic_algorithms(
                bool(deterministic),
                warn_only=bool(warn_only),
            )
        except TypeError:
            torch.use_deterministic_algorithms(bool(deterministic))


def _resolve_device(device_name):
    import torch

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def _unwrap_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise TypeError("Checkpoint must be a state dict or contain a model/state_dict entry")


def _checkpoint_has_target_weights(checkpoint_file):
    import torch

    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    state_dict = _unwrap_checkpoint_state_dict(checkpoint)
    return any("target_enricher" in key for key in state_dict.keys())


def _infer_num_classes_from_checkpoint(checkpoint_file, logger):
    import torch

    try:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
        state_dict = _unwrap_checkpoint_state_dict(checkpoint)
    except Exception as error:
        logger.warning("Could not infer source num_classes from checkpoint: {}".format(error))
        return None

    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key.endswith("classifier.weight") and getattr(value, "ndim", 0) == 2:
            num_classes = int(value.shape[0])
            logger.info(
                "Inferred source num_classes={} from checkpoint classifier".format(
                    num_classes
                )
            )
            return num_classes
    return None


def _num_classes_from_source_dataset(args, source_domain, logger):
    from datasets.build import __factory as dataset_factory

    if source_domain not in dataset_factory:
        raise ValueError("Unsupported source domain: {}".format(source_domain))

    dataset = dataset_factory[source_domain](root=args.root_dir, verbose=False)
    num_classes = len(dataset.train_id_container)
    logger.info(
        "Read source num_classes={} from {} train split".format(
            num_classes,
            source_domain,
        )
    )
    return num_classes


def _task_enabled(args, *task_names):
    loss_names = str(getattr(args, "loss_names", ""))
    enabled = {name.strip() for name in loss_names.split("+")}
    return any(task_name in enabled for task_name in task_names)


def _resolve_num_classes(args, source_domain, checkpoint_file, logger):
    num_classes = _infer_num_classes_from_checkpoint(checkpoint_file, logger)
    if num_classes is not None:
        return num_classes
    configured_num_classes = getattr(args, "num_classes", None)
    if configured_num_classes is not None:
        return int(configured_num_classes)
    if _task_enabled(args, "id", "imkt", "cid"):
        logger.info(
            "Checkpoint has no classifier weights; using num_classes=0 because "
            "classifier heads are not used during retrieval evaluation"
        )
    return 0


def _build_model(args, source_domain, checkpoint_file, device, logger):
    from model import build_model
    from utils.checkpoint import Checkpointer

    if getattr(args, "target_enrichment", False) and not _checkpoint_has_target_weights(checkpoint_file):
        message = (
            "Target enrichment is enabled, but checkpoint has no target_enricher "
            "weights: {}. The enrichment module will be randomly initialized."
        ).format(checkpoint_file)
        if getattr(args, "strict_target_checkpoint", False):
            raise ValueError(message)
        logger.warning(message)

    num_classes = _resolve_num_classes(args, source_domain, checkpoint_file, logger)
    model = build_model(args, num_classes)
    checkpointer = Checkpointer(model, logger=logger)
    checkpointer.load(f=checkpoint_file)
    model.to(device)
    model.eval()
    return model


def _target_domains(cli_args, source_domain):
    if cli_args.target_domains:
        return list(cli_args.target_domains)
    return [source_domain]


def _source_check_enabled(cli_args):
    if cli_args.source_check is None:
        return True
    return bool(cli_args.source_check)


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _build_eval_loaders(args, split):
    from datasets import build_dataloader

    old_training = getattr(args, "training", False)
    old_val_dataset = getattr(args, "val_dataset", None)
    old_batch_size = getattr(args, "batch_size", None)
    try:
        if split == "test":
            args.training = False
            test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
            return test_img_loader, test_txt_loader, num_classes

        args.training = True
        args.val_dataset = split
        if hasattr(args, "test_batch_size"):
            args.batch_size = args.test_batch_size
        _, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
        return val_img_loader, val_txt_loader, num_classes
    finally:
        args.training = False
        if old_val_dataset is not None:
            args.val_dataset = old_val_dataset
        if old_batch_size is not None:
            args.batch_size = old_batch_size
        args.training = False


def _evaluate_domain(model, args, target_domain, split, cli_args, logger):
    from utils.metrics import Evaluator

    logger.info("Evaluating target domain: {} split: {}".format(target_domain, split))
    args.dataset_name = target_domain
    args.training = False

    test_img_loader, test_txt_loader, _ = _build_eval_loaders(args, split)
    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    top1 = evaluator.eval(
        model.eval(),
        i2t_metric=cli_args.i2t_metric,
        use_target_enrichment=getattr(args, "target_enrichment", False),
    )

    metrics = {
        key: _json_safe(value)
        for key, value in getattr(evaluator, "last_metrics", {}).items()
    }
    metrics["best_R1"] = float(top1)
    metrics["best_task"] = getattr(evaluator, "last_best_task", None)
    logger.info(
        "Finished {} split on {}: best_R1={:.4f}, best_task={}".format(
            split,
            target_domain,
            float(top1),
            metrics["best_task"],
        )
    )
    return metrics


def _build_parser():
    parser = argparse.ArgumentParser(
        description="RDE checkpoint transfer evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="training run directory containing configs.yaml and checkpoints",
    )
    parser.add_argument(
        "--config_file",
        "--config",
        dest="config_file",
        default=None,
        help="training config file; defaults to <run_dir>/config(s).yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint path; if omitted, test.py searches config paths and checkpoint_names",
    )
    parser.add_argument(
        "--checkpoint_names",
        nargs="+",
        default=list(DEFAULT_CHECKPOINT_NAMES),
        help="checkpoint filenames to search under the config/output directory",
    )
    parser.add_argument(
        "--source_domain",
        default=None,
        choices=DATASET_CHOICES,
        help="dataset used to train the checkpoint; defaults to dataset_name in config",
    )
    parser.add_argument(
        "--target_domain",
        nargs="+",
        default=None,
        choices=DATASET_CHOICES,
        help="target domain alias for --target_domains; accepts one or more datasets",
    )
    parser.add_argument(
        "--target_domains",
        nargs="+",
        default=None,
        choices=DATASET_CHOICES,
        help="one or more target datasets to evaluate",
    )
    parser.add_argument(
        "--split",
        "--target_split",
        dest="split",
        choices=("test", "val"),
        default="test",
        help="target split to evaluate",
    )
    parser.add_argument(
        "--cross_domain",
        action="store_true",
        help="mark the run as cross-domain evaluation in logs/results",
    )
    _add_boolean_override(
        parser,
        "source_check",
        "run source-domain sanity inference before target evaluation",
        "skip source-domain sanity inference before target evaluation",
    )

    parser.add_argument("--root_dir", default=None, help="dataset root override")
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="sets CUDA_VISIBLE_DEVICES before torch/model imports",
    )

    parser.add_argument(
        "--output_eval_dir",
        default=None,
        help="directory for test_log.txt and optional JSON results",
    )
    parser.add_argument("--save_json", action="store_true", help="save collected metrics")
    parser.add_argument(
        "--results_file",
        default=None,
        help="explicit JSON output path; also enables saving",
    )
    parser.add_argument("--i2t_metric", action="store_true", help="also report i2t metrics")

    parser.add_argument("--seed", type=int, default=None)
    deterministic_group = parser.add_mutually_exclusive_group()
    deterministic_group.add_argument("--deterministic", dest="deterministic", action="store_true", default=None)
    deterministic_group.add_argument("--non_deterministic", dest="deterministic", action="store_false", default=None)
    parser.add_argument("--deterministic_warn_only", action="store_true", default=None)

    _add_boolean_override(parser, "target_enrichment", "enable target-aware text enrichment")
    parser.add_argument("--enrichment_space", choices=("global", "tse", "grab"), default=None)
    parser.add_argument("--top_m", type=int, default=None)
    parser.add_argument(
        "--topm_rank_space",
        choices=("host_global", "retrieval", "hybrid_global_grab", "hybrid_global_tse"),
        default=None,
    )
    parser.add_argument("--topm_rank_lambda", type=float, default=None)
    parser.add_argument("--extractor_mode", default=None)
    parser.add_argument("--num_parts", type=int, default=None)
    parser.add_argument("--target_relative_space", choices=("host_global", "retrieval"), default=None)
    parser.add_argument("--target_relative_num_clusters", type=int, default=None)
    parser.add_argument("--target_relative_cluster_method", choices=("kmeans",), default=None)
    parser.add_argument("--evidence_token_budget", type=int, default=None)
    parser.add_argument("--evidence_projection", choices=("auto", "linear", "none"), default=None)
    parser.add_argument("--context_module", choices=("mixer",), default=None)
    parser.add_argument("--mixer_dim", type=int, default=None)
    parser.add_argument("--mixer_depth", type=int, default=None)
    parser.add_argument("--mixer_hidden_part", type=int, default=None)
    parser.add_argument("--mixer_hidden_rank", type=int, default=None)
    parser.add_argument("--mixer_hidden_channel", type=int, default=None)
    parser.add_argument("--mixer_hidden_readout", type=int, default=None)
    parser.add_argument("--context_pooling", "--mixer_context_pooling", dest="context_pooling", choices=("mlp",), default=None)
    parser.add_argument("--residual_gate", "--gate_mode", dest="residual_gate", choices=("static", "residual"), default=None)
    parser.add_argument("--enrich_gamma", type=float, default=None)
    parser.add_argument("--residual_gate_hidden_dim", type=int, default=None)
    parser.add_argument("--lambda_ret", type=float, default=None)
    parser.add_argument("--eval_score_chunk_size", type=int, default=None)
    _add_boolean_override(
        parser,
        "strict_target_checkpoint",
        "raise if target enrichment is enabled but checkpoint lacks target_enricher weights",
    )

    args = parser.parse_args()
    if args.run_dir is None and args.config_file is None:
        parser.error("provide --run_dir or --config_file")
    if args.target_domain is not None and args.target_domains is not None:
        parser.error("use either --target_domain or --target_domains, not both")
    if args.target_domain is not None:
        args.target_domains = list(args.target_domain)
    return args


def main():
    cli_args = _build_parser()
    if cli_args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.cuda_visible_devices

    from utils.iotools import write_json
    from utils.logger import setup_logger

    config_file, run_dir = _resolve_config_file(cli_args)
    if not op.isfile(config_file):
        raise FileNotFoundError("Config file not found: {}".format(config_file))

    args, config_keys = _load_inference_config(config_file)
    _ensure_inference_defaults(args, config_file, config_keys)
    checkpoints, eval_dir, results_file = _resolve_paths(
        cli_args,
        args,
        config_file,
        run_dir,
        config_keys,
    )
    _apply_cli_overrides(args, cli_args, config_file, eval_dir)

    _configure_reproducibility(
        args.seed,
        deterministic=args.deterministic,
        warn_only=args.deterministic_warn_only,
    )

    logger = setup_logger("RDE", save_dir=eval_dir, if_train=False)
    source_domain = cli_args.source_domain or args.dataset_name
    targets = _target_domains(cli_args, source_domain)
    cross_domain = cli_args.cross_domain or any(domain != source_domain for domain in targets)
    run_source_check = _source_check_enabled(cli_args)

    logger.info("Config file: {}".format(config_file))
    logger.info("Checkpoints: {}".format(", ".join(checkpoints)))
    logger.info("Source domain: {}".format(source_domain))
    logger.info("Target domains: {}".format(", ".join(targets)))
    logger.info("Evaluation split: {}".format(cli_args.split))
    logger.info("Cross-domain evaluation: {}".format(cross_domain))
    logger.info("Source-domain sanity inference: {}".format(run_source_check))
    logger.info("Evaluation output: {}".format(eval_dir))
    logger.info("Target enrichment: {}".format(bool(getattr(args, "target_enrichment", False))))

    device = _resolve_device(cli_args.device)
    logger.info("Device: {}".format(device))

    results = {
        "config_file": config_file,
        "checkpoints": {},
        "source_domain": source_domain,
        "target_domains": targets,
        "split": cli_args.split,
        "cross_domain": cross_domain,
        "source_check_enabled": run_source_check,
        "device": str(device),
    }

    for checkpoint in checkpoints:
        logger.info("Evaluating checkpoint: {}".format(checkpoint))
        args.dataset_name = source_domain
        args.checkpoint = checkpoint
        model = _build_model(args, source_domain, checkpoint, device, logger)

        checkpoint_results = {
            "checkpoint": checkpoint,
            "source_check": None,
            "results": {},
        }
        source_check_metrics = None
        if run_source_check:
            logger.info(
                "Running source-domain sanity inference before target evaluation: {}".format(
                    source_domain
                )
            )
            source_check_metrics = _evaluate_domain(
                model,
                args,
                source_domain,
                cli_args.split,
                cli_args,
                logger,
            )
            checkpoint_results["source_check"] = {
                "domain": source_domain,
                "split": cli_args.split,
                "metrics": source_check_metrics,
            }
        else:
            logger.info("Skipping source-domain sanity inference")

        for target_domain in targets:
            if run_source_check and target_domain == source_domain:
                logger.info(
                    "Reusing source-domain sanity inference for target domain: {}".format(
                        target_domain
                    )
                )
                checkpoint_results["results"][target_domain] = source_check_metrics
                continue
            checkpoint_results["results"][target_domain] = _evaluate_domain(
                model,
                args,
                target_domain,
                cli_args.split,
                cli_args,
                logger,
            )

        results["checkpoints"][checkpoint] = checkpoint_results

    if cli_args.save_json or cli_args.results_file is not None:
        write_json(results, results_file)
        logger.info("Saved evaluation results to {}".format(results_file))


if __name__ == "__main__":
    main()
