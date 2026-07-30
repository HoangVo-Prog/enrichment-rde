#!/usr/bin/env python3
"""Qualitative R1-RK comparison for enrichment-rde GATE runs.

Example:
    python scripts/compare_gate_qualitative_r1_r10.py \
      --gate_config /path/to/gate_config.yaml \
      --base_checkpoint /path/to/base_retriever.pth \
      --gate_checkpoint /path/to/gate_best.pth \
      --output_dir outputs/qual_rde_gate \
      --dataset ICFG-PEDES \
      --root_dir /path/to/dataset_parent \
      --selection_metric R1 \
      --num_figs 30 \
      --top_k 10 \
      --sort_mode best_r1_green_gap \
      --device cuda \
      --cache_inference
"""

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
CODE_ROOT = REPO_ROOT / "2024-CVPR-RDE"
SHARED_PROTOTYPE_ROOT = WORKSPACE_ROOT / "prototype"
SHARED_SCRIPT_ROOT = SHARED_PROTOTYPE_ROOT / "scripts"

for path in (SHARED_SCRIPT_ROOT, SHARED_PROTOTYPE_ROOT):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

from compare_gate_qualitative_r1_r10 import RepoSpec, run_compare_gate_qualitative  # noqa: E402


def main() -> None:
    run_compare_gate_qualitative(
        RepoSpec(
            repository_name="enrichment-rde",
            repo_kind="rde",
            code_root=CODE_ROOT,
            default_host_model="rde",
            metrics_module="utils.metrics",
            logger_name="enrichment_rde.gate_qualitative",
            supports_host_model=False,
        )
    )


if __name__ == "__main__":
    main()
