import json
import unittest
from pathlib import Path

import pandas as pd

from recipe.time_series_forecast.model_server import load_config
from recipe.time_series_forecast.prompts import build_timeseries_system_prompt
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.time_series_forecast_agent_flow import TimeSeriesForecastAgentFlow
from recipe.time_series_forecast.utils import (
    extract_basic_statistics,
    extract_data_quality,
    extract_event_summary,
    extract_forecast_residuals,
    extract_within_channel_dynamics,
    format_predictions_to_string,
    parse_time_series_string,
    parse_time_series_to_dataframe,
)


class ETTh1FeatureSmokeTest(unittest.TestCase):
    def _build_timestamped_prompt_from_csv(self) -> str:
        df = pd.read_csv("dataset/ETT-small/ETTh1.csv").iloc[:96]
        self.timestamped_expected_first = str(df.iloc[0]["date"])
        self.timestamped_expected_last = str(df.iloc[-1]["date"])
        lines = [f"{row.date} OT={float(row.OT):.4f}" for row in df.itertuples(index=False)]
        return (
            "[Task] Single-variable time-series forecasting.\n"
            "Target Column: OT\n"
            "Lookback Window: 96\n"
            "Forecast Horizon: 96\n"
            "Requirements:\n"
            "1) Extract feature evidence before selecting a forecasting model.\n"
            "2) Choose one model from the enabled experts and then predict.\n"
            "3) Follow the required output protocol with <think>...</think><answer>...</answer>.\n"
            "Historical Data:\n"
            + "\n".join(lines)
        )

    def _run_dataset_case(self, dataset_path: str, protocol_kind: str) -> None:
        if protocol_kind == "timestamped":
            prompt_text = self._build_timestamped_prompt_from_csv()
            spec = parse_task_prompt(prompt_text, data_source="ETTh1")
        else:
            sample = json.loads(Path(dataset_path).read_text(encoding="utf-8").splitlines()[0])
            prompt_text = sample["raw_prompt"][0]["content"]
            spec = parse_task_prompt(prompt_text, data_source=sample.get("data_source"))

        self.assertEqual(spec.data_source, "ETTh1")
        self.assertEqual(spec.target_column, "OT")
        self.assertEqual(spec.lookback_window, 96)
        self.assertEqual(spec.forecast_horizon, 96)

        timestamps, values = parse_time_series_string(spec.historical_data, target_column=spec.target_column)
        self.assertEqual(len(values), 96)
        self.assertEqual(len(timestamps), 96)

        if protocol_kind == "value_only":
            self.assertTrue(all(ts is None for ts in timestamps))
        else:
            self.assertEqual(timestamps[0], self.timestamped_expected_first)
            self.assertEqual(timestamps[-1], self.timestamped_expected_last)

        df = parse_time_series_to_dataframe(
            spec.historical_data,
            series_id="ETTh1",
            target_column=spec.target_column,
        )
        self.assertEqual(len(df), 96)
        self.assertEqual(list(df.columns), ["id", "timestamp", "target"])
        if protocol_kind == "value_only":
            self.assertEqual(str(df.iloc[0]["timestamp"]), "2000-01-01 00:00:00")
            self.assertEqual(str(df.iloc[1]["timestamp"]), "2000-01-01 01:00:00")

        feature_sets = [
            extract_basic_statistics(values),
            extract_within_channel_dynamics(values),
            extract_forecast_residuals(values),
            extract_data_quality(values),
            extract_event_summary(values),
        ]
        for feature_dict in feature_sets:
            self.assertIsInstance(feature_dict, dict)
            self.assertTrue(feature_dict)

        prompt = build_timeseries_system_prompt(data_source=spec.data_source, target_column=spec.target_column)
        self.assertIn("ETTh1", prompt)
        self.assertIn("OT", prompt)

    def test_paper_aligned_value_only_protocol(self) -> None:
        self._run_dataset_case(
            "dataset/ett_rl_etth1_paper_same2/train.jsonl",
            protocol_kind="value_only",
        )

    def test_singlevar_timestamped_protocol(self) -> None:
        self._run_dataset_case(
            "",
            protocol_kind="timestamped",
        )

    def test_model_configs_and_agent_import(self) -> None:
        patchtst_cfg = load_config("patchtst")
        itransformer_cfg = load_config("itransformer")

        self.assertEqual(patchtst_cfg.get("target"), "OT")
        self.assertEqual(itransformer_cfg.get("target"), "OT")
        self.assertEqual(TimeSeriesForecastAgentFlow.__name__, "TimeSeriesForecastAgentFlow")

    def test_format_predictions_without_last_timestamp_uses_fixed_anchor(self) -> None:
        pred_df = pd.DataFrame({"target_0.5": [1.25, 2.5]})
        text = format_predictions_to_string(pred_df, last_timestamp=None)
        self.assertEqual(
            text,
            "2000-01-01 01:00:00 1.2500\n2000-01-01 02:00:00 2.5000",
        )


if __name__ == "__main__":
    unittest.main()
