import unittest
from unittest.mock import patch

import pandas as pd

from recipe.time_series_forecast.prompts import build_runtime_user_prompt, build_timeseries_system_prompt
from recipe.time_series_forecast.reward import (
    ENABLE_CHANGE_POINT_SCORE,
    ENABLE_SEASON_TREND_SCORE,
    compute_score,
)
from recipe.time_series_forecast.reward_metrics import (
    CHANGE_POINT_COMPONENT_SCORE_WEIGHT,
    PREDICTION_ERROR_SCORE_WEIGHT,
    SEASON_COMPONENT_SCORE_WEIGHT,
    STRUCTURAL_TIE_BREAK_MAX_NORM_MSE,
    STRUCTURAL_TIE_BREAK_SCALE,
    TREND_COMPONENT_SCORE_WEIGHT,
    compute_structural_tie_break_gate,
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
            required_feature_tools=["extract_basic_statistics"],
            completed_feature_tools=["extract_basic_statistics"],
            turn_stage="routing",
        )
        self.assertIn("### Analysis Summary", prompt)
        self.assertIn("### Historical Data", prompt)
        self.assertIn("Median: 2.0000", prompt)
        self.assertNotIn("already present earlier in this conversation", prompt)

    def test_turn_two_prompt_uses_neutral_routing_instruction(self) -> None:
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data="1.0000\n2.0000\n3.0000",
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            prediction_results=None,
            required_feature_tools=["extract_basic_statistics"],
            completed_feature_tools=["extract_basic_statistics"],
            turn_stage="routing",
        )
        self.assertIn("inductive bias best matches the observed evidence", prompt)
        self.assertIn("not on a fixed model-to-pattern template", prompt)
        self.assertNotIn("patchtst` for local motifs", prompt)
        self.assertNotIn("arima` for stable autocorrelation structure", prompt)
        self.assertNotIn("chronos2` for irregular or quality-stressed windows", prompt)
        self.assertNotIn("itransformer` for broader structural drift", prompt)

    def test_turn_three_prompt_omits_duplicate_predictions(self) -> None:
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data="1.0000\n2.0000\n3.0000",
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            prediction_results="2017-05-02 00:00:00 4.0000",
            prediction_model_used="patchtst",
            required_feature_tools=["extract_basic_statistics"],
            completed_feature_tools=["extract_basic_statistics"],
            turn_stage="refinement",
        )
        self.assertIn("### Analysis Summary", prompt)
        self.assertIn("### Recent Historical Window", prompt)
        self.assertIn("### Prediction Tool Output", prompt)
        self.assertIn("2017-05-02 00:00:00 4.0000", prompt)
        self.assertIn("canonical base forecast produced by the selected model", prompt)
        self.assertIn("You must choose exactly one action", prompt)
        self.assertIn("KEEP: copy the 96 forecast rows", prompt)
        self.assertIn("LOCAL_REFINE", prompt)
        self.assertIn("If unsure, choose KEEP", prompt)
        self.assertIn("No tool schema is available in this turn", prompt)
        self.assertIn("Output ONLY one <think>...</think> block followed immediately by one <answer>...</answer> block", prompt)
        self.assertIn("must use `YYYY-MM-DD HH:MM:SS value`", prompt)
        self.assertIn("Never synthesize timestamps such as `24:00:00` or `25:00:00`", prompt)
        self.assertIn("exact row-for-row copy of \"Prediction Tool Output\"", prompt)
        self.assertIn("Your reply must start with <think>", prompt)
        self.assertNotIn("already present earlier in this conversation", prompt)
        self.assertNotIn("[Brief reflection", prompt)
        self.assertNotIn("[Final prediction", prompt)

    def test_turn_three_prompt_relaxed_keep_bias_is_opt_in(self) -> None:
        with patch.dict("os.environ", {"TS_RELAX_TURN3_KEEP_BIAS": "1"}):
            prompt = build_runtime_user_prompt(
                data_source="ETTh1",
                target_column="OT",
                lookback_window=96,
                forecast_horizon=96,
                time_series_data="1.0000\n2.0000\n3.0000",
                history_analysis=["Basic Statistics:\n  Median: 2.0000"],
                prediction_results="2017-05-02 00:00:00 4.0000",
                prediction_model_used="patchtst",
                required_feature_tools=["extract_basic_statistics"],
                completed_feature_tools=["extract_basic_statistics"],
                turn_stage="refinement",
            )
        self.assertIn("apply a limited local refinement", prompt)
        self.assertIn("multiple short separated spans", prompt)
        self.assertIn("constrained rewrite of more rows", prompt)
        self.assertIn("Choose KEEP only when the base forecast already looks internally consistent", prompt)
        self.assertNotIn("If unsure, choose KEEP", prompt)

    def test_turn_one_prompt_lists_available_diagnostic_tools(self) -> None:
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data="1.0000\n2.0000\n3.0000",
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            prediction_results=None,
            required_feature_tools=["extract_basic_statistics", "extract_event_summary"],
            completed_feature_tools=["extract_basic_statistics"],
            diagnostic_plan_reason="The window shows local oscillation, so I need a focused diagnostic pass.",
            diagnostic_primary_model="patchtst",
            diagnostic_runner_up_model="itransformer",
            turn_stage="diagnostic",
        )
        self.assertIn("### Diagnostic Plan", prompt)
        self.assertNotIn("patchtst", prompt)
        self.assertNotIn("itransformer", prompt)
        self.assertIn("### Diagnostic Tool Schemas Available This Turn", prompt)
        self.assertIn("extract_event_summary", prompt)
        self.assertIn("call one or more feature tools", prompt)
        self.assertIn("Follow the diagnostic plan", prompt)
        self.assertIn("Do NOT call predict_time_series", prompt)

    def test_turn_one_prompt_can_opt_in_diagnostic_model_hints(self) -> None:
        with patch.dict("os.environ", {"TS_INCLUDE_DIAGNOSTIC_MODEL_HINTS": "1"}):
            prompt = build_runtime_user_prompt(
                data_source="ETTh1",
                target_column="OT",
                lookback_window=96,
                forecast_horizon=96,
                time_series_data="1.0000\n2.0000\n3.0000",
                history_analysis=["Basic Statistics:\n  Median: 2.0000"],
                prediction_results=None,
                required_feature_tools=["extract_basic_statistics", "extract_event_summary"],
                completed_feature_tools=["extract_basic_statistics"],
                diagnostic_plan_reason="The window shows local oscillation, so I need a focused diagnostic pass.",
                diagnostic_primary_model="patchtst",
                diagnostic_runner_up_model="itransformer",
                turn_stage="diagnostic",
            )
        self.assertIn("patchtst", prompt)
        self.assertIn("itransformer", prompt)

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
            allow_recovery=True,
        )
        value_only_result = compute_score(
            data_source="time_series",
            solution_str=value_only_solution,
            ground_truth=ground_truth,
            allow_recovery=True,
        )
        timestamped_score = timestamped_result["score"] if isinstance(timestamped_result, dict) else timestamped_result
        value_only_score = value_only_result["score"] if isinstance(value_only_result, dict) else value_only_result
        self.assertAlmostEqual(timestamped_score, value_only_score, places=6)

    def test_reward_rejects_mixed_timestamped_and_value_only_answer_lines(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010"
        )
        mixed_solution = (
            "<think>x</think><answer>\n"
            "2017-05-02 00:00:00 12.3450\n"
            "12.6780\n"
            "2017-05-02 02:00:00 13.0010\n"
            "</answer>"
        )
        result = compute_score(
            data_source="time_series",
            solution_str=mixed_solution,
            ground_truth=ground_truth,
            allow_recovery=False,
        )
        self.assertEqual(result["format_failure_reason"], "invalid_answer_shape:mixed_line_formats")
        self.assertAlmostEqual(result["score"], -1.0, places=6)

    def test_composite_reward_uses_mse_first_structural_tie_break(self) -> None:
        self.assertTrue(ENABLE_CHANGE_POINT_SCORE)
        self.assertTrue(ENABLE_SEASON_TREND_SCORE)
        self.assertGreater(
            PREDICTION_ERROR_SCORE_WEIGHT,
            2 * CHANGE_POINT_COMPONENT_SCORE_WEIGHT + SEASON_COMPONENT_SCORE_WEIGHT + TREND_COMPONENT_SCORE_WEIGHT,
        )
        self.assertGreater(STRUCTURAL_TIE_BREAK_MAX_NORM_MSE, 0.0)
        self.assertGreater(STRUCTURAL_TIE_BREAK_SCALE, 0.0)
        self.assertLess(STRUCTURAL_TIE_BREAK_SCALE, 1.0)

        ground_truth = (
            "2017-05-02 00:00:00 10.0000\n"
            "2017-05-02 01:00:00 20.0000\n"
            "2017-05-02 02:00:00 30.0000"
        )
        solution = (
            "<think>x</think><answer>\n"
            "10.0000\n"
            "20.0000\n"
            "30.0000\n"
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
        self.assertAlmostEqual(result["structural_tie_break_gate"], STRUCTURAL_TIE_BREAK_SCALE, places=6)
        self.assertAlmostEqual(result["change_point_score"], 0.0, places=6)
        self.assertAlmostEqual(result["season_trend_score"], 0.0, places=6)
        self.assertAlmostEqual(result["orig_mse"], 0.0, places=6)
        self.assertAlmostEqual(result["norm_mse"], 0.0, places=6)

    def test_structural_tie_break_gate_turns_off_for_large_norm_mse(self) -> None:
        self.assertAlmostEqual(compute_structural_tie_break_gate(STRUCTURAL_TIE_BREAK_MAX_NORM_MSE), 0.0, places=6)
        self.assertAlmostEqual(compute_structural_tie_break_gate(STRUCTURAL_TIE_BREAK_MAX_NORM_MSE * 2), 0.0, places=6)

    def test_composite_reward_adds_only_small_structural_bonus_when_prediction_is_close(self) -> None:
        ground_truth_values = [10.0, 12.0, 15.0, 18.0, 16.0, 13.0, 11.0, 14.0]
        prediction_values = [10.1, 12.1, 15.1, 18.1, 16.1, 13.1, 11.1, 14.1]
        ground_truth = "\n".join(
            f"2017-05-02 {idx:02d}:00:00 {value:.4f}" for idx, value in enumerate(ground_truth_values)
        )
        solution = "<think>x</think><answer>\n" + "\n".join(f"{value:.4f}" for value in prediction_values) + "\n</answer>"

        result = compute_score(data_source="time_series", solution_str=solution, ground_truth=ground_truth)

        self.assertGreater(result["mse_score"], 0.0)
        self.assertGreater(result["structural_tie_break_gate"], 0.0)
        self.assertGreaterEqual(result["change_point_score"] + result["season_trend_score"], 0.0)
        self.assertLessEqual(
            result["change_point_score"] + result["season_trend_score"],
            (2 * CHANGE_POINT_COMPONENT_SCORE_WEIGHT + SEASON_COMPONENT_SCORE_WEIGHT + TREND_COMPONENT_SCORE_WEIGHT)
            * STRUCTURAL_TIE_BREAK_SCALE
            + 1e-6,
        )

    def test_reward_rejects_under_generation_against_ground_truth_length(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 10.0000\n"
            "2017-05-02 01:00:00 20.0000\n"
            "2017-05-02 02:00:00 30.0000"
        )
        solution = "<answer>\n10.0000\n20.0000\n</answer>"
        result = compute_score(data_source="time_series", solution_str=solution, ground_truth=ground_truth)
        self.assertFalse(result["length_hard_fail"])
        self.assertFalse(result["strict_length_match"])
        self.assertEqual(result["format_failure_reason"], "missing_think_block")
        self.assertAlmostEqual(result["score"], -1.0, places=6)

    def test_reward_accepts_answer_only_protocol(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010"
        )
        solution = "<answer>\n12.3450\n12.6780\n13.0010\n</answer>"
        result = compute_score(
            data_source="time_series",
            solution_str=solution,
            ground_truth=ground_truth,
            allow_recovery=True,
        )
        self.assertEqual(result["format_failure_reason"], "ok")
        self.assertEqual(result["format_parse_mode"], "recovered_missing_think_block_answer_block")
        self.assertTrue(result["was_recovered"])
        self.assertAlmostEqual(result["score"], 0.7, places=6)

    def test_reward_default_strict_mode_rejects_answer_only_protocol(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010"
        )
        solution = "<answer>\n12.3450\n12.6780\n13.0010\n</answer>"
        result = compute_score(data_source="time_series", solution_str=solution, ground_truth=ground_truth)
        self.assertEqual(result["format_failure_reason"], "missing_think_block")
        self.assertEqual(result["format_parse_mode"], "rejected_missing_think_block")
        self.assertFalse(result["was_recovered"])
        self.assertAlmostEqual(result["score"], -1.0, places=6)

    def test_reward_recovers_missing_answer_close_tag_and_keeps_clip_signal(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010"
        )
        solution = "<think>x</think><answer>\n12.3450\n12.6780\n13.0010\n"
        result = compute_score(
            data_source="time_series",
            solution_str=solution,
            ground_truth=ground_truth,
            allow_recovery=True,
        )
        self.assertEqual(result["format_failure_reason"], "ok")
        self.assertEqual(result["format_parse_mode"], "recovered_missing_answer_close_tag_answer_block")
        self.assertTrue(result["was_recovered"])
        self.assertTrue(result["missing_answer_close_tag"])
        self.assertTrue(result["was_clipped"])
        self.assertAlmostEqual(result["score"], 0.7, places=6)

    def test_reward_recovers_overlong_answer_by_canonicalizing_to_ground_truth_length(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010"
        )
        solution = "<answer>\n12.3450\n12.6780\n13.0010\n14.0000\n</answer>"
        result = compute_score(
            data_source="time_series",
            solution_str=solution,
            ground_truth=ground_truth,
            allow_recovery=True,
        )
        self.assertEqual(result["format_failure_reason"], "ok")
        self.assertEqual(result["format_parse_mode"], "recovered_missing_think_block_answer_block")
        self.assertTrue(result["was_recovered"])
        self.assertAlmostEqual(result["score"], 0.7, places=6)

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
        self.assertNotIn("12.3450", prompt)
        self.assertIn("Follow the CURRENT user turn instructions only.", prompt)
        self.assertIn("If no tool schema is available, NEVER emit `<tool_call>`.", prompt)
        self.assertIn("output ONLY the required `<think>...</think><answer>...</answer>` blocks", prompt)

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
