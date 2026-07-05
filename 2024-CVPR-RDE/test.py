import argparse
import os
import os.path as op

import torch

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.logger import setup_logger
from model import build_model


def _set_default(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def _ensure_inference_defaults(args, config_file):
    config_dir = op.dirname(op.abspath(config_file))
    _set_default(args, "output_dir", config_dir)
    _set_default(args, "root_dir", "data")
    _set_default(args, "num_workers", 4)
    _set_default(args, "test_batch_size", 512)
    _set_default(args, "training", False)
    _set_default(args, "distributed", False)
    _set_default(args, "local_rank", 0)
    _set_default(args, "target_enrichment", False)
    _set_default(args, "enrichment_space", "global")
    _set_default(args, "topm_rank_space", "host_global")
    _set_default(args, "eval_score_chunk_size", 0)
    _set_default(args, "strict_target_checkpoint", False)
    _set_default(args, "img_size", (384, 128))
    if isinstance(args.img_size, list):
        args.img_size = tuple(args.img_size)


def _override_if_present(args, cli_args, name):
    value = getattr(cli_args, name, None)
    if value is not None:
        setattr(args, name, value)


def _apply_cli_overrides(args, cli_args):
    args.training = False
    args.distributed = False

    for name in (
        "root_dir",
        "test_batch_size",
        "num_workers",
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
    ):
        _override_if_present(args, cli_args, name)

    if cli_args.output_dir is not None:
        args.output_dir = cli_args.output_dir


def _resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def _checkpoint_has_target_weights(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("model", checkpoint)
    return any("target_enricher" in key for key in state_dict.keys())


def _resolve_checkpoints(args, cli_args):
    if cli_args.checkpoint:
        checkpoint = op.abspath(op.expanduser(cli_args.checkpoint))
        if not op.isfile(checkpoint):
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
        return [checkpoint]

    output_dir = op.abspath(op.expanduser(args.output_dir))
    candidates = [op.join(output_dir, name) for name in cli_args.checkpoint_names]
    checkpoints = [path for path in candidates if op.isfile(path)]
    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint or place one of {} in {}".format(
                ", ".join(cli_args.checkpoint_names),
                output_dir,
            )
        )
    return checkpoints


def _build_parser():
    parser = argparse.ArgumentParser(description="RDE inference with optional transductive enrichment")
    parser.add_argument("--config_file", required=True,
                        help="training configs.yaml produced by train.py")
    parser.add_argument("--checkpoint", default="",
                        help="checkpoint to evaluate; if omitted, test.py evaluates checkpoint_names in output_dir")
    parser.add_argument("--checkpoint_names", nargs="+", default=["best.pth", "last.pth"])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output_dir", default=None,
                        help="override output_dir from the config")
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--i2t_metric", action="store_true", default=False)

    enrichment_group = parser.add_mutually_exclusive_group()
    enrichment_group.add_argument("--target_enrichment", dest="target_enrichment", action="store_true", default=None)
    enrichment_group.add_argument("--no_target_enrichment", dest="target_enrichment", action="store_false")
    parser.add_argument("--enrichment_space", choices=["global", "tse", "grab"], default=None)
    parser.add_argument("--top_m", type=int, default=None)
    parser.add_argument("--topm_rank_space",
                        choices=["host_global", "retrieval", "hybrid_global_grab", "hybrid_global_tse"],
                        default=None)
    parser.add_argument("--topm_rank_lambda", type=float, default=None)
    parser.add_argument("--extractor_mode", default=None)
    parser.add_argument("--num_parts", type=int, default=None)
    parser.add_argument("--target_relative_space", choices=["host_global", "retrieval"], default=None)
    parser.add_argument("--target_relative_num_clusters", type=int, default=None)
    parser.add_argument("--target_relative_cluster_method", choices=["kmeans"], default=None)
    parser.add_argument("--evidence_token_budget", type=int, default=None)
    parser.add_argument("--evidence_projection", choices=["auto", "linear", "none"], default=None)
    parser.add_argument("--context_module", choices=["mixer"], default=None)
    parser.add_argument("--mixer_dim", type=int, default=None)
    parser.add_argument("--mixer_depth", type=int, default=None)
    parser.add_argument("--mixer_hidden_part", type=int, default=None)
    parser.add_argument("--mixer_hidden_rank", type=int, default=None)
    parser.add_argument("--mixer_hidden_channel", type=int, default=None)
    parser.add_argument("--mixer_hidden_readout", type=int, default=None)
    parser.add_argument("--context_pooling", "--mixer_context_pooling",
                        dest="context_pooling", choices=["mlp"], default=None)
    parser.add_argument("--residual_gate", "--gate_mode",
                        dest="residual_gate", choices=["static", "residual"], default=None)
    parser.add_argument("--enrich_gamma", type=float, default=None)
    parser.add_argument("--residual_gate_hidden_dim", type=int, default=None)
    parser.add_argument("--lambda_ret", type=float, default=None)
    parser.add_argument("--eval_score_chunk_size", type=int, default=None)
    parser.add_argument("--strict_target_checkpoint", action="store_true", default=None)
    return parser


def main():
    cli_args = _build_parser().parse_args()
    config_file = op.abspath(op.expanduser(cli_args.config_file))
    if not op.isfile(config_file):
        raise FileNotFoundError("Config file not found: {}".format(config_file))

    args = load_train_configs(config_file)
    _ensure_inference_defaults(args, config_file)
    _apply_cli_overrides(args, cli_args)
    checkpoints = _resolve_checkpoints(args, cli_args)

    logger = setup_logger('RDE', save_dir=args.output_dir, if_train=False)
    logger.info("Config file: {}".format(config_file))
    logger.info("Checkpoints: {}".format(", ".join(checkpoints)))
    logger.info("Target enrichment: {}".format(bool(getattr(args, "target_enrichment", False))))

    device = _resolve_device(cli_args.device)
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)

    for checkpoint in checkpoints:
        if getattr(args, "target_enrichment", False) and not _checkpoint_has_target_weights(checkpoint):
            message = (
                "Target enrichment is enabled, but checkpoint has no target_enricher "
                "weights: {}. The enrichment module will be randomly initialized."
            ).format(checkpoint)
            if getattr(args, "strict_target_checkpoint", False):
                raise ValueError(message)
            logger.warning(message)

        logger.info("Evaluating checkpoint: {}".format(checkpoint))
        model = build_model(args, num_classes)
        checkpointer = Checkpointer(model, logger=logger)
        checkpointer.load(f=checkpoint)
        model = model.to(device)
        do_inference(
            model,
            test_img_loader,
            test_txt_loader,
            args=args,
            use_target_enrichment=getattr(args, "target_enrichment", False),
        )


if __name__ == '__main__':
    main()
