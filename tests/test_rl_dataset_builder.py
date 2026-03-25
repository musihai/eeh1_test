import json
import unittest
from pathlib import Path

import pandas as pd

from recipe.time_series_forecast.build_etth1_rl_dataset import (
    build_ground_truth,
    build_prompt,
    build_train_stage_slices,
    build_split_configs,
    compute_teacher_metadata_coverage,
    compute_normalized_permutation_entropy,
    iter_split_samples,
)


class TestETTh1RLDatasetBuilder(unittest.TestCase):
    def test_build_prompt_uses_multivariate_history_protocol(self) -> None:
        history = pd.DataFrame(
            {
                "date": ["2016-01-01 00:00:00", "2016-01-01 01:00:00", "2016-01-01 02:00:00"],
                "HUFL": [10.0, 11.0, 12.0],
                "HULL": [20.0, 21.0, 22.0],
                "MUFL": [30.0, 31.0, 32.0],
                "MULL": [40.0, 41.0, 42.0],
                "LUFL": [50.0, 51.0, 52.0],
                "LULL": [60.0, 61.0, 62.0],
                "OT": [1.0, 2.5, 3.75],
            }
        )
        prompt = build_prompt(history, lookback_window=3, forecast_horizon=2, target_column="OT")
        self.assertIn("[Task] Multivariate time-series forecasting.", prompt)
        self.assertIn("Observed Covariates: HUFL, HULL, MUFL, MULL, LUFL, LULL", prompt)
        self.assertIn("2016-01-01 00:00:00 HUFL=10.0000 HULL=20.0000 MUFL=30.0000 MULL=40.0000 LUFL=50.0000 LULL=60.0000 OT=1.0000", prompt)
        self.assertIn("Target Column: OT", prompt)

    def test_build_ground_truth_preserves_timestamps(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["2016-01-01 00:00:00", "2016-01-01 01:00:00"],
                "OT": [1.0, 2.5],
            }
        )
        text = build_ground_truth(df, target_column="OT")
        self.assertEqual(text, "2016-01-01 00:00:00 1.0000\n2016-01-01 01:00:00 2.5000")

    def test_split_configs_match_historical_window_counts(self) -> None:
        splits = build_split_configs(total_rows=17420, train_rows=12251, val_rows=1913, test_rows=3256)
        counts = [split.num_rows - 96 - 96 + 1 for split in splits]
        self.assertEqual(counts, [12060, 1722, 3065])

    def test_normalized_permutation_entropy_stays_in_unit_interval(self) -> None:
        entropy = compute_normalized_permutation_entropy([1.0, 2.0, 3.0, 2.5, 2.0, 1.5])
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)

    def test_iter_split_samples_builds_expected_record_shape(self) -> None:
        source = pd.read_csv(Path("dataset/ETT-small/ETTh1.csv")).iloc[:210].copy()
        split = build_split_configs(total_rows=len(source), train_rows=210, val_rows=0, test_rows=0)[0]
        records = list(
            iter_split_samples(
                source,
                split,
                lookback_window=96,
                forecast_horizon=96,
                target_column="OT",
            )
        )
        self.assertEqual(len(records), 19)
        first = records[0]
        self.assertEqual(first["agent_name"], "time_series_forecast_agent")
        self.assertEqual(first["data_source"], "ETTh1")
        self.assertEqual(first["reward_model"]["style"], "rule")
        self.assertEqual(first["raw_prompt"][0]["role"], "user")

        prompt_lines = first["raw_prompt"][0]["content"].splitlines()
        hist_start = prompt_lines.index("Historical Data:") + 1
        history_lines = prompt_lines[hist_start:]
        self.assertEqual(len(history_lines), 96)
        self.assertTrue(all("OT=" in line for line in history_lines))
        self.assertTrue(all("HUFL=" in line for line in history_lines))

        gt_lines = first["reward_model"]["ground_truth"].splitlines()
        self.assertEqual(len(gt_lines), 96)
        self.assertIn("normalized_permutation_entropy", first)
        self.assertIn("normalized_permutation_entropy_band", first)
        self.assertIn("reference_teacher_error_band", first)
        self.assertIn("difficulty_stage", first)
        self.assertIn("quality_issue_flag", first)
        self.assertGreaterEqual(first["normalized_permutation_entropy"], 0.0)
        self.assertLessEqual(first["normalized_permutation_entropy"], 1.0)
        self.assertIn(first["normalized_permutation_entropy_band"], {"low", "medium", "high"})
        self.assertIn(first["reference_teacher_error_band"], {"unknown", "low", "medium", "high"})
        self.assertIn(first["difficulty_stage"], {"easy", "medium", "hard", "unknown"})

    def test_teacher_metadata_coverage_reports_fraction(self) -> None:
        coverage = compute_teacher_metadata_coverage(
            num_samples=5,
            teacher_metadata_by_index={
                0: {"best_model": "arima"},
                3: {"best_model": "patchtst"},
            },
        )
        self.assertAlmostEqual(coverage, 0.4)

    def test_iter_split_samples_prefers_reference_teacher_model_for_offline_best(self) -> None:
        source = pd.read_csv(Path("dataset/ETT-small/ETTh1.csv")).iloc[:210].copy()
        split = build_split_configs(total_rows=len(source), train_rows=210, val_rows=0, test_rows=0)[0]
        first = next(
            iter(
                iter_split_samples(
                    source,
                    split,
                    lookback_window=96,
                    forecast_horizon=96,
                    target_column="OT",
                    teacher_metadata_by_index={
                        0: {
                            "best_model": "arima",
                            "reference_teacher_model": "itransformer",
                            "reference_teacher_error": 1.0,
                        }
                    },
                )
            )
        )

        self.assertEqual(first["offline_best_model"], "itransformer")

    def test_build_train_stage_slices_emits_stage1_stage12_stage123(self) -> None:
        records = [
            {"uid": "a", "curriculum_stage": "easy"},
            {"uid": "b", "curriculum_stage": "medium"},
            {"uid": "c", "curriculum_stage": "hard"},
            {"uid": "d", "curriculum_stage": "unknown"},
        ]
        staged = build_train_stage_slices(records)
        self.assertEqual(list(staged.keys()), ["train_stage1", "train_stage12", "train_stage123"])
        self.assertEqual([record["uid"] for record in staged["train_stage1"]], ["a"])
        self.assertEqual([record["uid"] for record in staged["train_stage12"]], ["a", "b"])
        self.assertEqual([record["uid"] for record in staged["train_stage123"]], ["a", "b", "c", "d"])


if __name__ == "__main__":
    unittest.main()
