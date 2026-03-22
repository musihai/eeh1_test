import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recipe.time_series_forecast.build_etth1_sft_dataset import _build_turn3_target, convert_jsonl_to_sft_parquet
from recipe.time_series_forecast.utils import compact_prediction_tool_output_from_string


class TestETTh1SFTDatasetBuilder(unittest.TestCase):
    def _load_base_sample(self) -> dict:
        source_jsonl = Path("dataset/ett_rl_etth1_paper_same2/train.jsonl")
        with source_jsonl.open("r", encoding="utf-8") as handle:
            sample = json.loads(next(handle))

        reward_model = sample.get("reward_model", {})
        teacher_prediction_text = str(reward_model.get("ground_truth", "") or "").strip()
        sample["reference_teacher_model"] = "chronos2"
        sample["teacher_eval_second_best_model"] = "patchtst"
        sample["teacher_eval_score_margin"] = 0.20
        sample["teacher_prediction_text"] = teacher_prediction_text
        sample["teacher_prediction_source"] = "reference_teacher"
        return sample

    def _make_flat_tail_teacher_prediction(self, sample: dict, tail_length: int = 8) -> str:
        ground_truth = str(sample.get("reward_model", {}).get("ground_truth", "") or "").strip()
        lines = [line for line in ground_truth.splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), tail_length + 1)
        anchor_tokens = lines[-(tail_length + 1)].split()
        anchor_value = anchor_tokens[-1]

        rewritten: list[str] = []
        for idx, line in enumerate(lines):
            tokens = line.split()
            if idx >= len(lines) - tail_length:
                tokens[-1] = anchor_value
            rewritten.append(" ".join(tokens))
        return "\n".join(rewritten)

    def _make_spike_teacher_prediction(self, sample: dict, spike_index: int = 24, spike_delta: float = 50.0) -> str:
        ground_truth = str(sample.get("reward_model", {}).get("ground_truth", "") or "").strip()
        lines = [line for line in ground_truth.splitlines() if line.strip()]
        self.assertGreater(len(lines), spike_index)

        rewritten: list[str] = []
        for idx, line in enumerate(lines):
            tokens = line.split()
            if idx == spike_index:
                tokens[-1] = f"{float(tokens[-1]) + spike_delta:.4f}"
            rewritten.append(" ".join(tokens))
        return "\n".join(rewritten)

    def test_convert_small_rl_slice_to_multiturn_parquet(self):
        sample = self._load_base_sample()
        sample_b = copy.deepcopy(sample)
        sample_b["index"] = int(sample.get("index", 0)) + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "mini.jsonl"
            output_path = Path(tmpdir) / "train.parquet"
            with input_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                handle.write(json.dumps(sample_b, ensure_ascii=False) + "\n")

            dataframe = convert_jsonl_to_sft_parquet(
                input_path=input_path,
                output_path=output_path,
                max_samples=2,
            )

            self.assertEqual(len(dataframe), 2)
            self.assertTrue(output_path.exists())

            loaded = pd.read_parquet(output_path)
            self.assertEqual(len(loaded), 2)
            self.assertIn("messages", loaded.columns)
            self.assertNotIn("tools", loaded.columns)
            self.assertNotIn("enable_thinking", loaded.columns)

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
            self.assertIn("<answer>", messages[-1]["content"])
            self.assertIn(loaded.iloc[0]["turn3_target_type"], {"validated_keep", "local_refine"})
            self.assertIsInstance(list(loaded.iloc[0]["selected_feature_tools"]), list)
            self.assertIn("selected_feature_tool_count", loaded.columns)
            self.assertIn("selected_feature_tool_signature", loaded.columns)
            self.assertIn("turn3_trigger_reason", loaded.columns)
            self.assertIn("refine_ops_signature", loaded.columns)
            self.assertIn("refine_changed_value_count", loaded.columns)
            self.assertIn("refine_first_changed_index", loaded.columns)
            self.assertIn("refine_last_changed_index", loaded.columns)
            self.assertIn("refine_changed_span", loaded.columns)
            self.assertIn("refine_mean_abs_delta", loaded.columns)
            self.assertIn("refine_max_abs_delta", loaded.columns)
            self.assertIn("base_prediction_source", loaded.columns)
            self.assertIn("base_teacher_prediction_text", loaded.columns)
            self.assertIn("refined_prediction_text", loaded.columns)
            self.assertEqual(
                loaded.iloc[0]["selected_feature_tool_signature"],
                "->".join(list(loaded.iloc[0]["selected_feature_tools"])),
            )
            self.assertEqual(
                int(loaded.iloc[0]["selected_feature_tool_count"]),
                len(list(loaded.iloc[0]["selected_feature_tools"])),
            )
            self.assertEqual(loaded.iloc[0]["selected_prediction_model"], "chronos2")
            self.assertEqual(loaded.iloc[0]["base_prediction_source"], "reference_teacher_cached")
            self.assertEqual(loaded.iloc[0]["turn3_target_type"], "validated_keep")
            self.assertEqual(loaded.iloc[0]["refine_ops_signature"], "none")
            self.assertAlmostEqual(float(loaded.iloc[0]["refine_gain_mse"]), 0.0, places=6)
            self.assertEqual(int(loaded.iloc[0]["refine_changed_value_count"]), 0)
            self.assertEqual(int(loaded.iloc[0]["refine_first_changed_index"]), -1)

    def test_convert_uses_cached_teacher_prediction_when_present(self):
        sample = self._load_base_sample()
        sample["reference_teacher_model"] = "chronos2"
        sample["teacher_prediction_source"] = "reference_teacher"

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "cached.jsonl"
            parquet_path = Path(tmpdir) / "cached.parquet"
            jsonl_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataframe = convert_jsonl_to_sft_parquet(
                input_path=jsonl_path,
                output_path=parquet_path,
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
            self.assertEqual(
                messages[-1]["content"],
                "<answer>\n"
                + "\n".join(
                    line.split()[-1]
                    for line in str(sample["teacher_prediction_text"]).splitlines()
                    if line.strip()
                )
                + "\n</answer>",
            )
            self.assertEqual(dataframe.iloc[0]["selected_prediction_model"], "chronos2")

    def test_convert_builds_local_refine_target_for_spike_teacher_prediction(self):
        sample = self._load_base_sample()
        sample["teacher_eval_score_margin"] = 0.01
        sample["teacher_prediction_text"] = self._make_spike_teacher_prediction(sample)

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "spike.jsonl"
            parquet_path = Path(tmpdir) / "spike.parquet"
            jsonl_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataframe = convert_jsonl_to_sft_parquet(
                input_path=jsonl_path,
                output_path=parquet_path,
                max_samples=1,
            )

            self.assertEqual(dataframe.iloc[0]["turn3_target_type"], "local_refine")
            self.assertIn("isolated_spike_smoothing", dataframe.iloc[0]["refine_ops_signature"])
            self.assertGreater(int(dataframe.iloc[0]["refine_changed_value_count"]), 0)
            self.assertGreaterEqual(int(dataframe.iloc[0]["refine_first_changed_index"]), 0)
            self.assertGreater(float(dataframe.iloc[0]["refine_gain_mse"]), 0.0)

    def test_build_turn3_target_does_not_refine_without_evidence(self):
        sample = self._load_base_sample()
        sample["teacher_eval_score_margin"] = 0.20
        spike_prediction = self._make_spike_teacher_prediction(sample)

        turn3_target = _build_turn3_target(
            sample=sample,
            history_values=[float(idx) for idx in range(128)],
            base_prediction_text=spike_prediction,
            forecast_horizon=96,
            model_name="chronos2",
            selected_feature_tools=["extract_basic_statistics"],
        )

        self.assertEqual(turn3_target["turn3_target_type"], "validated_keep")
        self.assertEqual(turn3_target["refine_ops_signature"], "none")
        self.assertEqual(int(turn3_target["refine_changed_value_count"]), 0)


if __name__ == "__main__":
    unittest.main()
