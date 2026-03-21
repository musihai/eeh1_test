import unittest

import pandas as pd

from recipe.time_series_forecast.prompts import build_runtime_user_prompt, build_timeseries_system_prompt
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
        self.assertIn("base forecast produced by the selected model", prompt)
        self.assertIn("Do NOT rewrite the forecast arbitrarily", prompt)
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
            "<think>x</think><answer>\n"
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010\n"
            "</answer>"
        )
        value_only_solution = "<think>x</think><answer>\n12.3450\n12.6780\n13.0010\n</answer>"
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

    def test_composite_reward_uses_paper_aligned_defaults(self) -> None:
        self.assertTrue(ENABLE_CHANGE_POINT_SCORE)
        self.assertTrue(ENABLE_SEASON_TREND_SCORE)

        ground_truth = (
            "2017-05-02 00:00:00 10.0000\n"
            "2017-05-02 01:00:00 20.0000\n"
            "2017-05-02 02:00:00 30.0000"
        )
        solution = (
            "<think>x</think><answer>\n"
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
        self.assertTrue(result["strict_length_match"])
        self.assertAlmostEqual(result["change_point_score"], 0.0, places=6)
        self.assertAlmostEqual(result["season_trend_score"], 0.0, places=6)
        self.assertAlmostEqual(result["orig_mse"], 0.0, places=6)
        self.assertAlmostEqual(result["norm_mse"], 0.0, places=6)

    def test_reward_length_mismatch_uses_soft_penalty(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 10.0000\n"
            "2017-05-02 01:00:00 20.0000\n"
            "2017-05-02 02:00:00 30.0000"
        )
        solution = "<think>x</think><answer>\n10.0000\n20.0000\n</answer>"
        result = compute_score(data_source="time_series", solution_str=solution, ground_truth=ground_truth)
        self.assertFalse(result["length_hard_fail"])
        self.assertFalse(result["strict_length_match"])
        self.assertEqual(result["format_failure_reason"], "length_mismatch:2!=3")
        self.assertAlmostEqual(result["mse_score"], 0.6, places=6)
        self.assertAlmostEqual(result["length_penalty"], 0.03, places=6)
        self.assertAlmostEqual(result["score"], 0.57, places=6)
        self.assertAlmostEqual(result["orig_mse"], 0.0, places=6)
        self.assertAlmostEqual(result["norm_mse"], 0.0, places=6)

    def test_reward_passthroughs_runtime_tool_debug_fields(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 10.0000\n"
            "2017-05-02 01:00:00 20.0000\n"
            "2017-05-02 02:00:00 30.0000"
        )
        solution = (
            "<think>x</think><answer>\n"
            "2017-05-02 00:00:00 10.0000\n"
            "2017-05-02 01:00:00 20.0000\n"
            "2017-05-02 02:00:00 30.0000\n"
            "</answer>"
        )
        result = compute_score(
            data_source="time_series",
            solution_str=solution,
            ground_truth=ground_truth,
            extra_info={
                "reward_extra_info": {
                    "generation_stop_reason": "stop",
                    "feature_tool_signature": "extract_basic_statistics->extract_event_summary",
                    "tool_call_sequence": "extract_basic_statistics->extract_event_summary->predict_time_series",
                    "analysis_state_signature": "basic_statistics|event_summary",
                    "prediction_requested_model": "itransformer",
                    "workflow_status": "accepted",
                    "turn_stage": "refinement",
                    "selected_forecast_orig_mse": 0.25,
                    "selected_forecast_len_match": True,
                    "selected_forecast_exact_copy": False,
                    "final_vs_selected_mse": 0.05,
                    "refinement_compare_len": 96,
                    "refinement_changed_value_count": 4,
                    "refinement_first_changed_index": 72,
                    "refinement_changed": True,
                    "prediction_model_defaulted": False,
                }
            },
        )
        self.assertEqual(result["generation_stop_reason"], "stop")
        self.assertEqual(result["feature_tool_signature"], "extract_basic_statistics->extract_event_summary")
        self.assertEqual(
            result["tool_call_sequence"],
            "extract_basic_statistics->extract_event_summary->predict_time_series",
        )
        self.assertEqual(result["analysis_state_signature"], "basic_statistics|event_summary")
        self.assertEqual(result["prediction_requested_model"], "itransformer")
        self.assertEqual(result["workflow_status"], "accepted")
        self.assertEqual(result["turn_stage"], "refinement")
        self.assertAlmostEqual(result["selected_forecast_orig_mse"], 0.25, places=6)
        self.assertTrue(result["selected_forecast_len_match"])
        self.assertFalse(result["selected_forecast_exact_copy"])
        self.assertAlmostEqual(result["final_vs_selected_mse"], 0.05, places=6)
        self.assertEqual(result["refinement_compare_len"], 96)
        self.assertEqual(result["refinement_changed_value_count"], 4)
        self.assertEqual(result["refinement_first_changed_index"], 72)
        self.assertTrue(result["refinement_changed"])
        self.assertFalse(result["prediction_model_defaulted"])

    def test_system_prompt_avoids_fixed_model_preferences(self) -> None:
        prompt = build_timeseries_system_prompt(data_source="ETTh1", target_column="OT")
        self.assertNotIn("Usually strong for ETTh1-like seasonal univariate patterns", prompt)
        self.assertNotIn("usually less preferred", prompt)
        self.assertIn("smooth and shows strong local periodicity", prompt)

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
