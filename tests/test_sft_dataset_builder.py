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
        source_jsonl = Path("dataset/ett_rl_etth1_paper_same/train.jsonl")

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
            self.assertEqual(roles[0:3], ["system", "user", "assistant"])
            self.assertEqual(roles[-5:], ["user", "assistant", "tool", "user", "assistant"])

            first_tool_calls = messages[2]["tool_calls"]
            self.assertGreaterEqual(len(first_tool_calls), 2)
            self.assertLessEqual(len(first_tool_calls), 5)
            self.assertEqual(first_tool_calls[0]["function"]["name"], "extract_basic_statistics")
            self.assertTrue(all(call["function"]["name"] != "predict_time_series" for call in first_tool_calls))

            prediction_assistant_index = next(
                idx
                for idx, message in enumerate(messages)
                if message["role"] == "assistant"
                and "tool_calls" in message
                and len(message["tool_calls"]) > 0
                and message["tool_calls"][0]["function"]["name"] == "predict_time_series"
            )
            self.assertEqual(messages[2]["content"], "")
            self.assertEqual(messages[prediction_assistant_index]["content"], "")
            self.assertIn("### Analysis Summary", messages[prediction_assistant_index - 1]["content"])
            self.assertIn("### Prediction Tool Output", messages[prediction_assistant_index + 2]["content"])
            self.assertIn("<think>", messages[-1]["content"])
            self.assertIn("<answer>", messages[-1]["content"])
            self.assertIn(loaded.iloc[0]["sft_trajectory_type"], {"route_only", "route_then_refine"})
            self.assertIsInstance(list(loaded.iloc[0]["selected_feature_tools"]), list)
            self.assertIn("selected_feature_tool_count", loaded.columns)
            self.assertIn("selected_feature_tool_signature", loaded.columns)
            self.assertIn("refinement_supervision_type", loaded.columns)
            self.assertIn("refinement_trigger_reason", loaded.columns)
            self.assertEqual(
                loaded.iloc[0]["selected_feature_tool_signature"],
                "->".join(list(loaded.iloc[0]["selected_feature_tools"])),
            )
            self.assertEqual(
                int(loaded.iloc[0]["selected_feature_tool_count"]),
                len(list(loaded.iloc[0]["selected_feature_tools"])),
            )
            if loaded.iloc[0]["sft_trajectory_type"] == "route_then_refine":
                self.assertEqual(loaded.iloc[0]["refinement_supervision_type"], "language_only_refinement_hint")
            else:
                self.assertEqual(loaded.iloc[0]["refinement_supervision_type"], "keep_selected_forecast")
            tools = list(loaded.iloc[0]["tools"])
            self.assertEqual(len(tools), len(TIMESERIES_TOOL_SCHEMAS))
            self.assertEqual(tools[0]["function"]["name"], TIMESERIES_TOOL_SCHEMAS[0]["function"]["name"])
            self.assertEqual(tools[-1]["function"]["name"], TIMESERIES_TOOL_SCHEMAS[-1]["function"]["name"])

    def test_convert_uses_cached_teacher_prediction_when_present(self):
        source_jsonl = Path("dataset/ett_rl_etth1_paper_same/train.jsonl")

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
            prediction_tool_message = next(
                message for message in messages if message["role"] == "tool" and "Forecast Values:" in message["content"]
            )
            self.assertEqual(
                prediction_tool_message["content"],
                compact_prediction_tool_output_from_string(
                    sample["teacher_prediction_text"],
                    model_name="chronos2",
                ),
            )
            self.assertIn("1.0000", messages[-1]["content"])
            self.assertIn("2.0000", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
