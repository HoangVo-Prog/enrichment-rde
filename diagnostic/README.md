# RDE Cue-Swap Diagnostic

This package evaluates a frozen RDE text-based person search retriever under
matched cue-biased gallery perturbations. Gallery construction uses an
independent off-the-shelf CLIP cue scorer; RDE scores are used for retrieval,
hardness matching, and the score audit.

Cue-swap galleries share one neutral distractor set. For each query/trial, both
cue galleries contain the complete positive set and the same neutral
distractors; only the cue-dense distractor subset differs. The
hardness-matched control still replaces the full distractor set independently
for each cue direction.

Typical invocation:

```bash
python diagnostic/run_cue_swap_diagnostic.py \
  --dataset RSTPReid \
  --split test \
  --retriever_name rde \
  --retriever_config /path/to/config.yaml \
  --retriever_checkpoint /path/to/best.pth \
  --output_dir outputs/rde_diagnostic \
  --gallery_size 500 \
  --num_trials 3 \
  --bootstrap_iters 1000 \
  --tight_hardness_z_tolerance 0.10
```

Generated files use stable schemas. The run reports hardest-negative and
positive-negative margin audits in `per_gallery_results.csv`, paired Cue-vs-HM
hardness gaps in `paired_delta_results.csv`, and summary files:

```text
hardness_audit_with_ci.csv
tight_hardness_summary_with_ci.csv
```

The HM control approximately matches retriever-score difficulty, and residual
top-rank mismatch is audited using hardest-negative scores, positive-negative
margins, and a fixed tight-match robustness subset controlled by
`--tight_hardness_z_tolerance` (default `0.10`). The tight-hardness subset is a
robustness result, not a replacement for the primary Cue-minus-HM result.
