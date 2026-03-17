import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recipe.time_series_forecast.build_etth1_sft_dataset import convert_jsonl_to_sft_parquet
from recipe.time_series_forecast.prompts import TIMESERIES_TOOL_SCHEMAS
from recipe.time_series_forecast.utils import compact_prediction_tool_output_from_string


class TestETTh1SFTDatasetBuilder(unittest.TestCase):
    def test_convert_small_rl_slice_to_multiturn_parquet(self):
        source_jsonl = Path("dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/train.jsonl")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "train.parquet"
            dataframe = convert_jsonl_to_sft_parquet(
                input_path=source_jsonl,
                output_path=output_path,
                prediction_mode="ground_truth",
                max_samples=2,
            )

            self.assertEqual(len(dataframe), 2)
            self.assertTrue(output_path.exists())

            loaded = pd.read_parquet(output_path)
            self.assertEqual(len(loaded), 2)
            self.assertIn("messages", loaded.columns)
            self.assertIn("tools", loaded.columns)
            self.assertIn("enable_thinking", loaded.columns)
            self.assertFalse(bool(loaded.iloc[0]["enable_thinking"]))

            messages = loaded.iloc[0]["messages"]
            roles = [message["role"] for message in messages]
            self.assertEqual(
                roles,
                [
                    "system",
                    "user",
                    "assistant",
                    "tool",
                    "tool",
                    "tool",
                    "tool",
                    "tool",
                    "user",
                    "assistant",
                    "tool",
                    "user",
                    "assistant",
                ],
            )

            first_tool_calls = messages[2]["tool_calls"]
            self.assertEqual(len(first_tool_calls), 5)
            self.assertEqual(first_tool_calls[0]["function"]["name"], "extract_basic_statistics")
            self.assertEqual(messages[9]["tool_calls"][0]["function"]["name"], "predict_time_series")
            self.assertEqual(messages[2]["content"], "")
            self.assertEqual(messages[9]["content"], "")
            self.assertIn("### Analysis Summary", messages[8]["content"])
            self.assertIn("### Prediction Tool Output", messages[11]["content"])
            self.assertIn("<think>", messages[-1]["content"])
            self.assertIn("<answer>", messages[-1]["content"])
            tools = list(loaded.iloc[0]["tools"])
            self.assertEqual(len(tools), len(TIMESERIES_TOOL_SCHEMAS))
            self.assertEqual(tools[0]["function"]["name"], TIMESERIES_TOOL_SCHEMAS[0]["function"]["name"])
            self.assertEqual(tools[-1]["function"]["name"], TIMESERIES_TOOL_SCHEMAS[-1]["function"]["name"])

    def test_convert_uses_cached_teacher_prediction_when_present(self):
        source_jsonl = Path("dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/train.jsonl")

        with source_jsonl.open("r", encoding="utf-8") as handle:
            sample = json.loads(next(handle))

        sample["reference_teacher_model"] = "chronos2"
        sample["teacher_prediction_text"] = "2016-01-01 00:00:00 1.0000\n2016-01-01 01:00:00 2.0000"
        sample["teacher_prediction_source"] = "reference_teacher"

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "cached.jsonl"
            parquet_path = Path(tmpdir) / "cached.parquet"
            jsonl_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataframe = convert_jsonl_to_sft_parquet(
                input_path=jsonl_path,
                output_path=parquet_path,
                prediction_mode="reference_teacher",
                max_samples=1,
            )

            messages = dataframe.iloc[0]["messages"]
            self.assertEqual(
                messages[10]["content"],
                compact_prediction_tool_output_from_string(
                    sample["teacher_prediction_text"],
                    model_name="chronos2",
                ),
            )
            self.assertIn(sample["teacher_prediction_text"].splitlines()[0], messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
