import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RETRIEVER_BY_ROOT = {
    "enrichment-irra": "irra",
    "enrichment-rde": "rde",
    "enrichment-rde-adapter": "dm_adapter",
}
RETRIEVER_NAME = RETRIEVER_BY_ROOT[REPO_ROOT.name]

from diagnostic.audit import build_hardness_audit_with_ci, build_tight_hardness_summary_with_ci
from diagnostic.constants import OUTPUT_FILES, TIGHT_HARDNESS_SUMMARY_COLUMNS
from diagnostic.gallery_construction import construct_cue_swap_galleries
from diagnostic.metrics import paired_hardness_gaps, query_negative_score_scale, retriever_hardness_stats
from diagnostic.outputs import write_run_outputs


def build_fixture(
    *,
    gallery_pids,
    psi_a,
    psi_b,
    gallery_size=8,
    dense_ratio=0.5,
    seed=13,
    case_id="case",
    query_id=101,
    trial_id=0,
):
    return construct_cue_swap_galleries(
        pid=1,
        gallery_pids=np.asarray(gallery_pids, dtype=np.int64),
        psi_a=np.asarray(psi_a, dtype=float),
        psi_b=np.asarray(psi_b, dtype=float),
        gallery_size=gallery_size,
        dense_ratio=dense_ratio,
        lambda_contrast=0.0,
        seed=seed,
        case_id=case_id,
        query_id=query_id,
        trial_id=trial_id,
        neutral_strategy="random",
        neutral_pool_factor=5,
    )


def make_stats(scores, is_positive):
    return retriever_hardness_stats(
        np.asarray(scores, dtype=float),
        np.asarray(is_positive, dtype=bool),
    )


def make_tight_rows():
    return pd.DataFrame(
        [
            {
                "dataset": "RSTPReid",
                "retriever_name": RETRIEVER_NAME,
                "case_id": "case_a",
                "query_id": 10,
                "trial_id": 0,
                "r1_flip": 1.0,
                "hm_r1_flip": 0.0,
                "delta_r1_flip": 1.0,
                "tight_hardness_match": True,
                "mean_signed_max_negative_gap": 0.10,
                "mean_signed_margin_gap": -0.10,
                "mean_abs_max_negative_gap": 0.10,
            },
            {
                "dataset": "RSTPReid",
                "retriever_name": RETRIEVER_NAME,
                "case_id": "case_a",
                "query_id": 10,
                "trial_id": 1,
                "r1_flip": 0.0,
                "hm_r1_flip": 0.0,
                "delta_r1_flip": 0.0,
                "tight_hardness_match": True,
                "mean_signed_max_negative_gap": 0.20,
                "mean_signed_margin_gap": -0.20,
                "mean_abs_max_negative_gap": 0.20,
            },
            {
                "dataset": "RSTPReid",
                "retriever_name": RETRIEVER_NAME,
                "case_id": "case_b",
                "query_id": 10,
                "trial_id": 0,
                "r1_flip": 1.0,
                "hm_r1_flip": 1.0,
                "delta_r1_flip": 0.0,
                "tight_hardness_match": True,
                "mean_signed_max_negative_gap": -0.10,
                "mean_signed_margin_gap": 0.10,
                "mean_abs_max_negative_gap": 0.10,
            },
            {
                "dataset": "RSTPReid",
                "retriever_name": RETRIEVER_NAME,
                "case_id": "case_c",
                "query_id": 20,
                "trial_id": 0,
                "r1_flip": 0.0,
                "hm_r1_flip": 1.0,
                "delta_r1_flip": -1.0,
                "tight_hardness_match": False,
                "mean_signed_max_negative_gap": 3.0,
                "mean_signed_margin_gap": -3.0,
                "mean_abs_max_negative_gap": 3.0,
            },
        ]
    )


class SharedNeutralGalleryConstructionTest(unittest.TestCase):
    def test_shared_neutral_invariants(self):
        gallery_pids = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        psi_a = [0, 0, 0.95, 0.90, 0.85, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
        psi_b = [0, 0, 0.10, 0.09, 0.08, 0.95, 0.90, 0.85, 0.07, 0.06, 0.05, 0.04]

        build, reason = build_fixture(gallery_pids=gallery_pids, psi_a=psi_a, psi_b=psi_b)

        self.assertIsNone(reason)
        self.assertIsNotNone(build)
        assert build is not None
        shared = build.shared_neutral
        positives = build.positive_indices

        np.testing.assert_array_equal(build.neutral["a_dense"], shared)
        np.testing.assert_array_equal(build.neutral["b_dense"], shared)
        self.assertTrue(set(shared.tolist()).isdisjoint(set(positives.tolist())))
        self.assertTrue(
            set(shared.tolist()).isdisjoint(
                set(build.dense["a_dense"].tolist()) | set(build.dense["b_dense"].tolist())
            )
        )
        self.assertEqual(set(build.galleries["a_dense"][: len(positives)].tolist()), set(positives.tolist()))
        self.assertEqual(set(build.galleries["b_dense"][: len(positives)].tolist()), set(positives.tolist()))
        self.assertEqual(len(build.galleries["a_dense"]), 8)
        self.assertEqual(len(build.galleries["b_dense"]), 8)
        self.assertEqual(len(np.unique(build.galleries["a_dense"])), 8)
        self.assertEqual(len(np.unique(build.galleries["b_dense"])), 8)

    def test_deterministic_for_identical_inputs(self):
        gallery_pids = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        psi_a = [0, 0, 0.95, 0.90, 0.85, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
        psi_b = [0, 0, 0.10, 0.09, 0.08, 0.95, 0.90, 0.85, 0.07, 0.06, 0.05, 0.04]

        first, _ = build_fixture(gallery_pids=gallery_pids, psi_a=psi_a, psi_b=psi_b)
        second, _ = build_fixture(gallery_pids=gallery_pids, psi_a=psi_a, psi_b=psi_b)

        assert first is not None and second is not None
        for key in ("a_dense", "b_dense"):
            np.testing.assert_array_equal(first.dense[key], second.dense[key])
            np.testing.assert_array_equal(first.galleries[key], second.galleries[key])
        np.testing.assert_array_equal(first.shared_neutral, second.shared_neutral)

    def test_insufficient_shared_neutral_pool_has_stable_reason(self):
        build, reason = build_fixture(
            gallery_pids=[1, 2, 3, 4, 5, 6],
            psi_a=[0, 0.95, 0.90, 0.10, 0.09, 0.08],
            psi_b=[0, 0.10, 0.09, 0.95, 0.90, 0.08],
            gallery_size=6,
            dense_ratio=0.4,
        )

        self.assertIsNone(build)
        self.assertEqual(reason, "not_enough_remaining_for_shared_neutral_fill")

    def test_dense_overlap_does_not_break_construction(self):
        build, reason = build_fixture(
            gallery_pids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            psi_a=[0, 0.99, 0.98, 0.97, 0.10, 0.09, 0.08, 0.07, 0.06],
            psi_b=[0, 0.99, 0.98, 0.10, 0.97, 0.09, 0.08, 0.07, 0.06],
            gallery_size=7,
            dense_ratio=0.5,
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(build)
        assert build is not None
        self.assertTrue(set(build.dense["a_dense"].tolist()) & set(build.dense["b_dense"].tolist()))
        self.assertTrue(
            set(build.shared_neutral.tolist()).isdisjoint(
                set(build.dense["a_dense"].tolist()) | set(build.dense["b_dense"].tolist())
            )
        )
        self.assertEqual(len(np.unique(build.galleries["a_dense"])), len(build.galleries["a_dense"]))
        self.assertEqual(len(np.unique(build.galleries["b_dense"])), len(build.galleries["b_dense"]))


class HardnessStatisticAndBootstrapTest(unittest.TestCase):
    def test_gallery_stats_and_gap_signs(self):
        cue_metrics = {
            "a_dense": make_stats([0.60, 0.30, 0.55, 0.20], [True, True, False, False]),
            "b_dense": make_stats([0.60, 0.30, 0.40, 0.20], [True, True, False, False]),
        }
        hm_metrics = {
            "hm_a": make_stats([0.60, 0.30, 0.45, 0.10], [True, True, False, False]),
            "hm_b": make_stats([0.60, 0.30, 0.50, 0.10], [True, True, False, False]),
        }

        self.assertAlmostEqual(cue_metrics["a_dense"]["best_positive_score"], 0.60)
        self.assertAlmostEqual(cue_metrics["a_dense"]["max_negative_score"], 0.55)
        self.assertAlmostEqual(cue_metrics["a_dense"]["positive_negative_margin"], 0.05)

        gaps = paired_hardness_gaps(
            cue_metrics,
            hm_metrics,
            safe_scale=0.20,
            tight_hardness_z_tolerance=0.60,
        )
        self.assertAlmostEqual(gaps["a_signed_max_negative_gap"], 0.10)
        self.assertAlmostEqual(gaps["a_signed_margin_gap"], -0.10)
        self.assertAlmostEqual(gaps["b_signed_max_negative_gap"], -0.10)
        self.assertAlmostEqual(gaps["b_signed_margin_gap"], 0.10)
        self.assertAlmostEqual(gaps["a_normalized_max_negative_gap"], 0.50)
        self.assertAlmostEqual(gaps["b_normalized_max_negative_gap"], -0.50)
        self.assertTrue(gaps["tight_hardness_match"])
        self.assertFalse(
            paired_hardness_gaps(
                cue_metrics,
                hm_metrics,
                safe_scale=0.20,
                tight_hardness_z_tolerance=0.40,
            )["tight_hardness_match"]
        )

    def test_query_negative_score_scale_excludes_positives_and_clamps(self):
        scale, safe_scale = query_negative_score_scale(
            np.asarray([0.60, 0.30, 0.55, 0.20, 0.45, 0.10], dtype=float),
            np.asarray([1, 1, 2, 3, 4, 5], dtype=np.int64),
            query_pid=1,
        )
        self.assertAlmostEqual(scale, float(np.std([0.55, 0.20, 0.45, 0.10], ddof=0)))
        self.assertAlmostEqual(safe_scale, max(scale, 1e-12))

        zero_scale, zero_safe = query_negative_score_scale(
            np.asarray([0.60, 0.60, 0.10, 0.10, 0.10], dtype=float),
            np.asarray([1, 1, 2, 3, 4], dtype=np.int64),
            query_pid=1,
        )
        self.assertAlmostEqual(zero_scale, 0.0)
        self.assertEqual(zero_safe, 1e-12)

    def test_tight_summary_filters_rows_and_clusters_by_unique_query(self):
        df = make_tight_rows()
        summary = build_tight_hardness_summary_with_ci(
            df,
            retriever_name=RETRIEVER_NAME,
            tight_hardness_z_tolerance=0.10,
            bootstrap_iters=25,
            bootstrap_seed=5,
        )

        r1_row = summary.loc[summary["metric"] == "r1_flip"].iloc[0]
        self.assertAlmostEqual(r1_row["mean"], 2.0 / 3.0)
        self.assertEqual(int(r1_row["cluster_count"]), 1)
        self.assertEqual(int(r1_row["trial_count"]), 3)
        self.assertEqual(int(r1_row["tight_trial_count"]), 3)
        self.assertEqual(int(r1_row["tight_unique_query_count"]), 1)
        self.assertEqual(int(r1_row["tight_case_query_count"]), 2)
        self.assertAlmostEqual(r1_row["tight_trial_rate"], 3.0 / 4.0)

    def test_tight_summary_empty_result_has_expected_rows(self):
        df = make_tight_rows()
        df["tight_hardness_match"] = False
        summary = build_tight_hardness_summary_with_ci(
            df,
            retriever_name=RETRIEVER_NAME,
            tight_hardness_z_tolerance=0.10,
            bootstrap_iters=25,
            bootstrap_seed=5,
        )

        self.assertEqual(summary.columns.tolist(), TIGHT_HARDNESS_SUMMARY_COLUMNS)
        self.assertEqual(summary["metric"].tolist(), ["r1_flip", "hm_r1_flip", "delta_r1_flip"])
        self.assertTrue(summary["mean"].map(math.isnan).all())
        self.assertEqual(summary["tight_trial_count"].unique().tolist(), [0])
        self.assertEqual(summary["tight_trial_rate"].unique().tolist(), [0.0])

    def test_hardness_audit_uses_all_valid_trials_and_outputs_write(self):
        df = make_tight_rows()
        audit = build_hardness_audit_with_ci(
            df,
            bootstrap_iters=25,
            bootstrap_seed=7,
            retriever_name=RETRIEVER_NAME,
        )
        signed_row = audit.loc[audit["metric"] == "mean_signed_max_negative_gap"].iloc[0]
        self.assertAlmostEqual(signed_row["mean"], df["mean_signed_max_negative_gap"].mean())
        self.assertEqual(int(signed_row["cluster_count"]), 2)
        self.assertEqual(int(signed_row["trial_count"]), 4)

        empty_tight = build_tight_hardness_summary_with_ci(
            df.assign(tight_hardness_match=False),
            retriever_name=RETRIEVER_NAME,
            tight_hardness_z_tolerance=0.10,
            bootstrap_iters=10,
            bootstrap_seed=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_run_outputs(
                output_dir,
                selected_queries=[],
                constructibility_rows=[],
                validity_counts=[],
                per_gallery_rows=[],
                paired_cue_rows=[],
                paired_hm_rows=[],
                paired_delta_rows=[],
                summary_by_case=pd.DataFrame(),
                summary_overall=pd.DataFrame(),
                summary_ci=pd.DataFrame(),
                summary_ci_by_unit={},
                hardness_audit_ci=audit,
                tight_hardness_summary_ci=empty_tight,
                skipped_rows=[],
                gallery_rows=[],
                save_galleries=False,
            )
            self.assertTrue((output_dir / OUTPUT_FILES["hardness_audit_ci"]).exists())
            self.assertTrue((output_dir / OUTPUT_FILES["tight_hardness_summary_ci"]).exists())
            written = pd.read_csv(output_dir / OUTPUT_FILES["tight_hardness_summary_ci"])
            self.assertEqual(written.columns.tolist(), TIGHT_HARDNESS_SUMMARY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
