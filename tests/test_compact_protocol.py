import unittest

import pandas as pd

from recipe.time_series_forecast.prompts import build_runtime_user_prompt
from recipe.time_series_forecast.reward import (
    ENABLE_CHANGE_POINT_SCORE,
    ENABLE_SEASON_TREND_SCORE,
    compute_score,
)
from recipe.time_series_forecast.utils import compact_prediction_tool_output_from_string, format_prediction_tool_output


class CompactProtocolTests(unittest.TestCase):
    def test_turn_two_prompt_omits_duplicate_analysis_history(self) -> None:
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data="1.0000\n2.0000\n3.0000",
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            prediction_results=None,
            conversation_has_tool_history=True,
        )
        self.assertIn("### Analysis Summary", prompt)
        self.assertIn("Median: 2.0000", prompt)
        self.assertNotIn("already present earlier in this conversation", prompt)

    def test_turn_three_prompt_omits_duplicate_predictions(self) -> None:
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data="1.0000\n2.0000\n3.0000",
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            prediction_results="Start Timestamp: 2017-05-02 00:00:00\nFrequency Hours: 1\nForecast Values:\n4.0000",
            prediction_model_used="patchtst",
            conversation_has_tool_history=True,
        )
        self.assertIn("### Analysis Summary", prompt)
        self.assertIn("### Prediction Tool Output", prompt)
        self.assertIn("Forecast Values:", prompt)
        self.assertNotIn("already present earlier in this conversation", prompt)

    def test_compact_prediction_tool_output_keeps_single_timestamp_anchor(self) -> None:
        prediction_text = (
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010"
        )
        compact = compact_prediction_tool_output_from_string(prediction_text, model_name="chronos2")
        self.assertIn("Model: chronos2", compact)
        self.assertIn("Start Timestamp: 2017-05-02 00:00:00", compact)
        self.assertIn("Frequency Hours: 1", compact)
        self.assertIn("Forecast Values:", compact)
        self.assertNotIn("2017-05-02 01:00:00 12.6780", compact)
        self.assertTrue(compact.strip().endswith("13.0010"))

    def test_reward_matches_for_timestamped_and_value_only_answers(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010"
        )
        timestamped_solution = (
            "<answer>\n"
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010\n"
            "</answer>"
        )
        value_only_solution = "<answer>\n12.3450\n12.6780\n13.0010\n</answer>"
        timestamped_result = compute_score(
            data_source="time_series",
            solution_str=timestamped_solution,
            ground_truth=ground_truth,
        )
        value_only_result = compute_score(
            data_source="time_series",
            solution_str=value_only_solution,
            ground_truth=ground_truth,
        )
        timestamped_score = timestamped_result["score"] if isinstance(timestamped_result, dict) else timestamped_result
        value_only_score = value_only_result["score"] if isinstance(value_only_result, dict) else value_only_result
        self.assertAlmostEqual(timestamped_score, value_only_score, places=6)

    def test_strict_ablation_disables_auxiliary_rewards(self) -> None:
        self.assertFalse(ENABLE_CHANGE_POINT_SCORE)
        self.assertFalse(ENABLE_SEASON_TREND_SCORE)

        ground_truth = (
            "2017-05-02 00:00:00 10.0000\n"
            "2017-05-02 01:00:00 20.0000\n"
            "2017-05-02 02:00:00 30.0000"
        )
        solution = (
            "<answer>\n"
            "2017-05-02 00:00:00 10.0000\n"
            "2017-05-02 01:00:00 20.0000\n"
            "2017-05-02 02:00:00 30.0000\n"
            "</answer>"
        )
        result = compute_score(data_source="time_series", solution_str=solution, ground_truth=ground_truth)
        score_val = result["score"] if isinstance(result, dict) else result
        self.assertAlmostEqual(
            score_val,
            0.7,
            places=6,
        )

    def test_dataframe_prediction_tool_output_uses_compact_format(self) -> None:
        pred_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2017-05-02 00:00:00", "2017-05-02 01:00:00", "2017-05-02 02:00:00"]
                ),
                "target_0.5": [12.345, 12.678, 13.001],
            }
        )
        compact = format_prediction_tool_output(
            pred_df,
            last_timestamp="2017-05-01 23:00:00",
            model_name="patchtst",
        )
        self.assertIn("Start Timestamp: 2017-05-02 00:00:00", compact)
        self.assertIn("12.6780", compact)
        self.assertNotIn("2017-05-02 01:00:00 12.6780", compact)


if __name__ == "__main__":
    unittest.main()
