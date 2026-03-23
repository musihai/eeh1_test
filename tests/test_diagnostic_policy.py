import unittest

from recipe.time_series_forecast.diagnostic_policy import plan_diagnostic_tool_batches


class TestDiagnosticPolicy(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
