import unittest

import pandas as pd

from recipe.time_series_forecast.prompts import (
    build_refinement_evidence_card,
    build_runtime_user_prompt,
    build_timeseries_system_prompt,
)
from recipe.time_series_forecast.reward import (
    ENABLE_CHANGE_POINT_SCORE,
    ENABLE_SEASON_TREND_SCORE,
    compute_score,
)
from recipe.time_series_forecast.reward_metrics import (
    CHANGE_POINT_COMPONENT_SCORE_WEIGHT,
    PREDICTION_ERROR_SCORE_WEIGHT,
    SEASON_COMPONENT_SCORE_WEIGHT,
    TREND_COMPONENT_SCORE_WEIGHT,
    compute_length_penalty,
    compute_norm_mse_score,
)
from recipe.time_series_forecast.time_series_io import compact_historical_data_for_prompt
from recipe.time_series_forecast.utils import compact_prediction_tool_output_from_string, format_prediction_tool_output


class CompactProtocolTests(unittest.TestCase):
    def test_turn_two_prompt_keeps_analysis_summary_and_raw_statistics(self) -> None:
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data="1.0000\n2.0000\n3.0000",
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            prediction_results=None,
            available_feature_tools=["extract_basic_statistics"],
            completed_feature_tools=["extract_basic_statistics"],
            routing_feature_payload={
                "extract_basic_statistics": {
                    "acf1": 0.91,
                    "acf_seasonal": 0.12,
                    "cusum_max": 42.0,
                }
            },
            turn_stage="routing",
        )
        self.assertIn("### Routing Evidence Card", prompt)
        self.assertIn("observed_tools=[extract_basic_statistics]", prompt)
        self.assertIn("tool_fields:", prompt)
        self.assertIn("- extract_basic_statistics: acf1=0.9100, acf_seasonal=0.1200, cusum_max=42.0000", prompt)
        self.assertIn("### Analysis Summary", prompt)
        self.assertIn("Median: 2.0000", prompt)
        self.assertNotIn("### Historical Data", prompt)
        self.assertNotIn("expert_support_signals:", prompt)
        self.assertNotIn("already present earlier in this conversation", prompt)

    def test_turn_two_prompt_omits_routing_decision_guide(self) -> None:
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data="1.0000\n2.0000\n3.0000",
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            prediction_results=None,
            available_feature_tools=["extract_basic_statistics"],
            completed_feature_tools=["extract_basic_statistics"],
            routing_feature_payload={
                "extract_basic_statistics": {
                    "acf1": 0.91,
                    "acf_seasonal": 0.12,
                    "cusum_max": 42.0,
                }
            },
            turn_stage="routing",
        )
        self.assertNotIn("### Routing Decision Guide", prompt)
        self.assertNotIn("stable linear trend or seasonality", prompt)
        self.assertIn("Use the observed tool statistics and analysis summary to choose one expert.", prompt)
        self.assertIn("Do NOT rely on hidden heuristics or template rules", prompt)

    def test_routing_prompt_warns_against_placeholder_tool_names(self) -> None:
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data="1.0000\n2.0000\n3.0000",
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            prediction_results=None,
            available_feature_tools=["extract_basic_statistics"],
            completed_feature_tools=["extract_basic_statistics"],
            routing_feature_payload={
                "extract_basic_statistics": {
                    "acf1": 0.91,
                    "acf_seasonal": 0.12,
                    "cusum_max": 42.0,
                }
            },
            turn_stage="routing",
        )
        self.assertIn("Use the exact function name `predict_time_series`.", prompt)
        self.assertIn('{"name":"predict_time_series","arguments":{"model_name":"arima"}}', prompt)

    def test_routing_prompt_supports_default_override_route_schema(self) -> None:
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data="1.0000\n2.0000\n3.0000",
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            prediction_results=None,
            available_feature_tools=["extract_basic_statistics"],
            completed_feature_tools=["extract_basic_statistics"],
            routing_feature_payload={
                "extract_basic_statistics": {
                    "acf1": 0.91,
                    "acf_seasonal": 0.12,
                    "cusum_max": 42.0,
                }
            },
            turn_stage="routing",
            route_default_expert="itransformer",
        )
        self.assertIn("### Default Expert: itransformer", prompt)
        self.assertIn("route_time_series", prompt)
        self.assertIn('{"name":"route_time_series","arguments":{"decision":"keep_default"}}', prompt)
        self.assertIn('{"name":"route_time_series","arguments":{"decision":"override","model_name":"patchtst"}}', prompt)
        self.assertIn("Do NOT override back to the default expert.", prompt)

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
            available_feature_tools=["extract_basic_statistics"],
            completed_feature_tools=["extract_basic_statistics"],
            turn_stage="refinement",
        )
        self.assertIn("### Refinement Evidence Card", prompt)
        self.assertIn("keep_support=[none]", prompt)
        self.assertIn("support_signals=[evidence_consistent]", prompt)
        self.assertIn("candidate_adjustments=[none]", prompt)
        self.assertIn("### Analysis Summary", prompt)
        self.assertIn("### Recent Historical Window", prompt)
        self.assertIn("### Prediction Tool Output", prompt)
        self.assertIn("2017-05-02 00:00:00 4.0000", prompt)
        self.assertIn("base forecast from the selected model", prompt)
        self.assertIn("Make the refinement decision from the Refinement Evidence Card first", prompt)
        self.assertIn("decision_options=[keep_baseline]", prompt)
        self.assertIn("Choose `keep_baseline` only when no listed adjustment is directly supported", prompt)
        self.assertIn("Only choose a local edit when that exact edit's `support=[...]` line clearly justifies", prompt)
        self.assertIn("No tool schema is available in this turn", prompt)
        self.assertIn("exactly one `<think>...</think>` block followed immediately by one `<answer>...</answer>` block", prompt)
        self.assertIn("must contain exactly one non-empty line in the form `decision=<name>`", prompt)
        self.assertNotIn("Reuse the provided timestamps and row order exactly", prompt)
        self.assertIn("Stop immediately after `</answer>`", prompt)
        self.assertNotIn("already present earlier in this conversation", prompt)
        self.assertNotIn("[Brief reflection", prompt)
        self.assertNotIn("[Final prediction", prompt)

    def test_refinement_evidence_card_puts_keep_after_adjustments(self) -> None:
        card = build_refinement_evidence_card(
            refinement_feature_payload={
                "observed_tools": ["extract_forecast_residuals"],
                "support_signals": ["residual_mismatch"],
                "keep_support_signals": [],
                "candidate_adjustments": ["local_level_adjust", "local_slope_adjust"],
                "edit_support_signals": {
                    "local_level_adjust": ["level_shift_detected"],
                    "local_slope_adjust": ["slope_drift_detected"],
                },
                "keep_baseline_allowed": True,
            },
            prediction_model_used="patchtst",
        )

        self.assertIn("decision_options=[local_level_adjust, local_slope_adjust, keep_baseline]", card)

    def test_turn_one_prompt_lists_available_diagnostic_tools(self) -> None:
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data="1.0000\n2.0000\n3.0000",
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            prediction_results=None,
            available_feature_tools=["extract_basic_statistics", "extract_event_summary"],
            completed_feature_tools=["extract_basic_statistics"],
            turn_stage="diagnostic",
        )
        self.assertIn("### Diagnostic Tool Schemas Available This Turn", prompt)
        self.assertIn("extract_event_summary", prompt)
        self.assertIn("call one or more feature tools", prompt)
        self.assertIn("This is the planning and diagnostic stage", prompt)
        self.assertIn("First decide what evidence you need", prompt)
        self.assertIn("Do NOT call predict_time_series", prompt)

    def test_turn_one_prompt_compacts_multivariate_historical_window(self) -> None:
        historical_data = "\n".join(
            f"2016-08-29 {hour:02d}:00:00 HUFL={15.0 + hour:.4f} OT={25.8170 + hour:.4f}"
            for hour in range(16)
        )
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data=historical_data,
            history_analysis=[],
            prediction_results=None,
            available_feature_tools=["extract_basic_statistics"],
            completed_feature_tools=[],
            turn_stage="diagnostic",
        )
        self.assertIn("timestamp,HUFL,OT", prompt)
        self.assertIn("2016-08-29 00:00:00,15.0000,25.8170", prompt)
        self.assertNotIn("HUFL=15.0000 OT=25.8170", prompt)

    def test_compact_historical_data_keeps_short_value_only_series_verbatim(self) -> None:
        historical_data = "1.0000\n2.0000\n3.0000"
        compact = compact_historical_data_for_prompt(historical_data, target_column="OT")
        self.assertEqual(compact, historical_data)

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

    def test_composite_reward_uses_fixed_weight_multi_view_aggregation(self) -> None:
        self.assertTrue(ENABLE_CHANGE_POINT_SCORE)
        self.assertTrue(ENABLE_SEASON_TREND_SCORE)
        self.assertGreater(
            PREDICTION_ERROR_SCORE_WEIGHT,
            2 * CHANGE_POINT_COMPONENT_SCORE_WEIGHT + SEASON_COMPONENT_SCORE_WEIGHT + TREND_COMPONENT_SCORE_WEIGHT,
        )

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
        self.assertAlmostEqual(result["change_point_score"], 0.0, places=6)
        self.assertAlmostEqual(result["season_trend_score"], 0.0, places=6)
        self.assertAlmostEqual(result["orig_mse"], 0.0, places=6)
        self.assertAlmostEqual(result["norm_mse"], 0.0, places=6)

    def test_norm_mse_score_uses_stronger_bad_side_separation(self) -> None:
        self.assertAlmostEqual(compute_norm_mse_score(0.0), PREDICTION_ERROR_SCORE_WEIGHT, places=6)
        self.assertAlmostEqual(compute_norm_mse_score(1.0), 0.3, places=6)
        self.assertAlmostEqual(compute_norm_mse_score(100.0), 0.6 / 11.0, places=6)
        self.assertLess(compute_norm_mse_score(100.0), compute_norm_mse_score(10.0))

    def test_composite_reward_keeps_structural_components_active_beyond_low_mse_region(self) -> None:
        ground_truth_values = [10.0, 12.0, 15.0, 18.0, 16.0, 13.0, 11.0, 14.0]
        prediction_values = [15.0, 17.0, 20.0, 23.0, 21.0, 18.0, 16.0, 19.0]
        ground_truth = "\n".join(
            f"2017-05-02 {idx:02d}:00:00 {value:.4f}" for idx, value in enumerate(ground_truth_values)
        )
        solution = "<think>x</think><answer>\n" + "\n".join(f"{value:.4f}" for value in prediction_values) + "\n</answer>"

        result = compute_score(data_source="time_series", solution_str=solution, ground_truth=ground_truth)

        self.assertGreater(result["mse_score"], 0.0)
        self.assertGreater(result["norm_mse"], 0.5)
        self.assertGreater(result["change_point_score"] + result["season_trend_score"], 0.0)
        self.assertLessEqual(
            result["change_point_score"] + result["season_trend_score"],
            (2 * CHANGE_POINT_COMPONENT_SCORE_WEIGHT + SEASON_COMPONENT_SCORE_WEIGHT + TREND_COMPONENT_SCORE_WEIGHT)
            + 1e-6,
        )
        self.assertAlmostEqual(
            result["score"],
            result["format_score"]
            + result["length_score"]
            + result["mse_score"]
            + result["change_point_score"]
            + result["season_trend_score"]
            - result["length_penalty"],
            places=6,
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
        self.assertEqual(result["format_failure_reason"], "missing_think_block")
        self.assertEqual(result["format_parse_mode"], "recovered_missing_think_block_answer_block")
        self.assertTrue(result["was_recovered"])
        self.assertAlmostEqual(result["recovery_penalty"], 0.05, places=6)
        self.assertAlmostEqual(result["score"], 0.65, places=6)

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
        self.assertEqual(result["format_failure_reason"], "missing_answer_close_tag")
        self.assertEqual(result["format_parse_mode"], "recovered_missing_answer_close_tag_answer_block")
        self.assertTrue(result["was_recovered"])
        self.assertTrue(result["missing_answer_close_tag"])
        self.assertTrue(result["was_clipped"])
        self.assertAlmostEqual(result["recovery_penalty"], 0.03, places=6)
        self.assertAlmostEqual(result["score"], 0.67, places=6)

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
        self.assertEqual(result["format_failure_reason"], "missing_think_block")
        self.assertEqual(result["format_parse_mode"], "recovered_missing_think_block_answer_block")
        self.assertTrue(result["was_recovered"])
        self.assertAlmostEqual(result["recovery_penalty"], 0.05, places=6)
        self.assertAlmostEqual(result["score"], 0.65, places=6)

    def test_reward_strictly_scores_clamped_turn3_horizon_answers(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010"
        )
        solution = (
            "<think>x</think><answer>\n"
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010\n"
            "2017-05-02 03:00:00 14.0000\n"
            "2017-05-02 04:00:00 15.0000\n"
        )
        result = compute_score(
            data_source="time_series",
            solution_str=solution,
            ground_truth=ground_truth,
            allow_recovery=False,
            extra_info={"validate": True},
        )

        self.assertTrue(result["turn3_horizon_clamped"])
        self.assertEqual(result["turn3_horizon_clamp_reason"], "truncate_after_expected_timestamp_rows")
        self.assertEqual(result["turn3_horizon_clamp_discarded_lines"], 2)
        self.assertEqual(result["turn3_horizon_clamp_valid_prefix_lines"], 5)
        self.assertEqual(result["turn3_horizon_clamp_raw_answer_lines"], 5)
        self.assertEqual(result["answer_line_count"], 3)
        self.assertEqual(result["raw_response_answer_line_count"], 5)
        self.assertEqual(result["expected_answer_line_count"], 3)
        self.assertEqual(result["format_parse_mode"], "strict_protocol")
        self.assertAlmostEqual(result["raw_overrun_penalty"], compute_length_penalty(5, 3), places=6)
        self.assertAlmostEqual(result["score"], 0.7 - compute_length_penalty(5, 3), places=6)

    def test_reward_clamps_etth1_turn3_horizon_answers(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010"
        )
        solution = (
            "<think>x</think><answer>\n"
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010\n"
            "2017-05-02 03:00:00 14.0000\n"
            "2017-05-02 04:00:00 15.0000\n"
        )
        result = compute_score(
            data_source="ETTh1",
            solution_str=solution,
            ground_truth=ground_truth,
            allow_recovery=False,
            extra_info={"validate": True},
        )

        self.assertTrue(result["turn3_horizon_clamped"])
        self.assertEqual(result["turn3_horizon_clamp_reason"], "truncate_after_expected_timestamp_rows")
        self.assertEqual(result["turn3_horizon_clamp_discarded_lines"], 2)
        self.assertEqual(result["turn3_horizon_clamp_valid_prefix_lines"], 5)
        self.assertEqual(result["turn3_horizon_clamp_raw_answer_lines"], 5)
        self.assertEqual(result["answer_line_count"], 3)
        self.assertEqual(result["raw_response_answer_line_count"], 5)
        self.assertEqual(result["expected_answer_line_count"], 3)
        self.assertEqual(result["format_parse_mode"], "strict_protocol")
        self.assertAlmostEqual(result["raw_overrun_penalty"], compute_length_penalty(5, 3), places=6)
        self.assertAlmostEqual(result["score"], 0.7 - compute_length_penalty(5, 3), places=6)

    def test_reward_uses_materialized_solution_from_reward_extra_info(self) -> None:
        ground_truth = (
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010"
        )
        raw_solution = "<think>x</think><answer>\ndecision=keep_baseline\n</answer>"
        materialized_solution = (
            "<think>x</think><answer>\n"
            "2017-05-02 00:00:00 12.3450\n"
            "2017-05-02 01:00:00 12.6780\n"
            "2017-05-02 02:00:00 13.0010\n"
            "</answer>"
        )

        result = compute_score(
            data_source="time_series",
            solution_str=raw_solution,
            ground_truth=ground_truth,
            allow_recovery=False,
            extra_info={
                "validate": True,
                "reward_extra_info": {
                    "materialized_solution_str": materialized_solution,
                },
            },
        )

        self.assertTrue(result["used_materialized_solution"])
        self.assertEqual(result["format_parse_mode"], "strict_protocol")
        self.assertEqual(result["answer_line_count"], 3)
        self.assertEqual(result["raw_response_answer_line_count"], 1)
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
