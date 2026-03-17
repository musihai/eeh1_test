import unittest

from recipe.time_series_forecast.task_protocol import (
    extract_historical_data_block,
    parse_task_prompt,
    parse_time_series_records,
)


SAMPLE_PROMPT = """[Task] Single-variable time-series forecasting.
Target Column: OT
Lookback Window: 96
Forecast Horizon: 96
Requirements:
1) Extract feature evidence before selecting a forecasting model.
2) Choose one model from the enabled experts and then predict.
Historical Data:
2016-08-29 11:00:00 OT=25.8170
2016-08-29 12:00:00 OT=27.7870
2016-08-29 13:00:00 OT=28.4200
"""


class TaskProtocolTest(unittest.TestCase):
    def test_extract_prompt_metadata(self) -> None:
        spec = parse_task_prompt(SAMPLE_PROMPT, data_source="ETTh1")
        self.assertEqual(spec.data_source, "ETTh1")
        self.assertEqual(spec.task_type, "Single-variable time-series forecasting.")
        self.assertEqual(spec.target_column, "OT")
        self.assertEqual(spec.lookback_window, 96)
        self.assertEqual(spec.forecast_horizon, 96)

    def test_extract_historical_data_block(self) -> None:
        historical_data = extract_historical_data_block(SAMPLE_PROMPT)
        self.assertTrue(historical_data.startswith("2016-08-29 11:00:00 OT=25.8170"))
        self.assertNotIn("Target Column", historical_data)

    def test_parse_labeled_target_series(self) -> None:
        timestamps, values = parse_time_series_records(SAMPLE_PROMPT, target_column="OT")
        self.assertEqual(
            timestamps,
            [
                "2016-08-29 11:00:00",
                "2016-08-29 12:00:00",
                "2016-08-29 13:00:00",
            ],
        )
        self.assertEqual(values, [25.8170, 27.7870, 28.4200])

    def test_parse_multicolumn_line_uses_requested_target(self) -> None:
        text = "Historical Data:\n2016-08-29 11:00:00 HUFL=15.0 OT=25.8170\n"
        _, values = parse_time_series_records(text, target_column="OT")
        self.assertEqual(values, [25.8170])


if __name__ == "__main__":
    unittest.main()
