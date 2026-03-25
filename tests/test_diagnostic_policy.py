import unittest

from recipe.time_series_forecast.diagnostic_policy import (
    FEATURE_TOOL_ORDER,
    build_diagnostic_plan,
    plan_diagnostic_tool_batches,
    select_feature_tool_names,
)


class TestDiagnosticPolicy(unittest.TestCase):
    def test_select_feature_tool_names_prefers_minimal_plan_for_stable_series(self) -> None:
        stable_values = [10.0 + 0.1 * idx for idx in range(96)]
        selected = select_feature_tool_names(stable_values)
        self.assertIn("extract_basic_statistics", selected)
        self.assertIn("extract_forecast_residuals", selected)
        self.assertNotIn("extract_data_quality", selected)
        self.assertLess(len(selected), len(FEATURE_TOOL_ORDER))

    def test_build_diagnostic_plan_surfaces_candidate_models_and_rationale(self) -> None:
        values = [0.0, 1.0, 0.2, 1.2] * 24
        plan = build_diagnostic_plan(values)
        self.assertTrue(plan.tool_names)
        self.assertIn(plan.primary_model, {"patchtst", "itransformer", "arima", "chronos2"})
        self.assertTrue(plan.runner_up_model)
        self.assertIsInstance(plan.score_gap, float)
        self.assertIn("I will inspect", plan.rationale)

    def test_batches_keep_paper_turn1_parallel_when_capacity_allows(self) -> None:
        batches = plan_diagnostic_tool_batches(
            [
                "extract_basic_statistics",
                "extract_within_channel_dynamics",
                "extract_forecast_residuals",
                "extract_data_quality",
            ],
            max_parallel_calls=5,
        )
        self.assertEqual(
            batches,
            [[
                "extract_basic_statistics",
                "extract_within_channel_dynamics",
                "extract_forecast_residuals",
                "extract_data_quality",
            ]],
        )

    def test_batches_chunk_in_order_only_when_parallel_cap_is_tight(self) -> None:
        batches = plan_diagnostic_tool_batches(
            [
                "extract_basic_statistics",
                "extract_within_channel_dynamics",
                "extract_forecast_residuals",
                "extract_data_quality",
            ],
            max_parallel_calls=2,
        )
        self.assertEqual(
            batches,
            [
                ["extract_basic_statistics", "extract_within_channel_dynamics"],
                ["extract_forecast_residuals", "extract_data_quality"],
            ],
        )

    def test_select_feature_tool_names_adds_quality_tool_for_quantized_window(self) -> None:
        quantized_values = [float(idx % 3) for idx in range(96)]
        selected = select_feature_tool_names(quantized_values)
        self.assertIn("extract_data_quality", selected)


if __name__ == "__main__":
    unittest.main()
