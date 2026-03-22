import asyncio
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from arft.ray_agent_trainer import RayAgentTrainer, evaluate_validation_reward_manager
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class _AsyncRewardManager:
    async def run_single(self, data: DataProto):
        uid = str(data.non_tensor_batch["uid"][0])
        if uid.endswith("0"):
            return {"reward_score": 0.25, "reward_extra_info": {"acc": 0.25, "tag": "a"}}
        return {"reward_score": 0.5, "reward_extra_info": {"acc": 0.5}}


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "2016-01-01 00:00:00 1.0000"


class TestValidationRewardManager(unittest.TestCase):
    def _build_batch(self) -> DataProto:
        return DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[11, 12], [21, 22]], dtype=torch.long),
                "responses": torch.tensor([[31, 32, 33], [41, 42, 0]], dtype=torch.long),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0],
                    ],
                    dtype=torch.long,
                ),
            },
            non_tensors={
                "uid": np.array(["sample-0", "sample-1"], dtype=object),
                "reward_model": np.array(
                    [
                        {"ground_truth": "2016-01-01 00:00:00 1.0", "style": "rule"},
                        {"ground_truth": "2016-01-01 00:00:00 2.0", "style": "rule"},
                    ],
                    dtype=object,
                ),
                "data_source": np.array(["ETTh1", "ETTh1"], dtype=object),
            },
        )

    def test_evaluate_validation_reward_manager_supports_async_reward_loop_manager(self) -> None:
        batch = self._build_batch()
        result = evaluate_validation_reward_manager(_AsyncRewardManager(), batch)
        reward_tensor = result["reward_tensor"]
        reward_extra_info = result["reward_extra_info"]

        self.assertEqual(tuple(reward_tensor.shape), (2, 3))
        self.assertAlmostEqual(float(reward_tensor[0, 2].item()), 0.25, places=6)
        self.assertAlmostEqual(float(reward_tensor[1, 1].item()), 0.5, places=6)
        self.assertEqual(reward_extra_info["acc"], [0.25, 0.5])
        self.assertEqual(reward_extra_info["tag"], ["a", None])

    def test_evaluate_validation_reward_manager_handles_reward_manager_with_stale_loop(self) -> None:
        batch = self._build_batch()
        manager = NaiveRewardManager(
            config=OmegaConf.create({}),
            tokenizer=_Tokenizer(),
            compute_score=lambda **kwargs: 0.3,
        )
        stale_loop = asyncio.new_event_loop()
        manager.loop = stale_loop

        try:
            result = evaluate_validation_reward_manager(manager, batch)
            reward_tensor = result["reward_tensor"]

            self.assertAlmostEqual(float(reward_tensor[0, 2].item()), 0.3, places=6)
            self.assertAlmostEqual(float(reward_tensor[1, 1].item()), 0.3, places=6)
        finally:
            stale_loop.close()

    def test_min_eval_debug_writer_emits_extended_metrics(self) -> None:
        trainer = RayAgentTrainer.__new__(RayAgentTrainer)
        trainer.global_steps = 20

        gt_values = [float(i) for i in range(1, 97)]
        gt_text = "<answer>\n" + "\n".join(f"{v:.4f}" for v in gt_values) + "\n</answer>"
        success_output = gt_text
        failure_output = "\n".join(f"{v:.4f}" for v in gt_values[:20])
        near_miss_output = "<answer>\n" + "\n".join(f"{v:.4f}" for v in gt_values[:-1]) + "\n</answer>"

        reward_extra_infos = {
            "pred_len": [96, 585, 95],
            "expected_len": [96, 96, 96],
            "orig_mse": [0.0, float("nan"), 1.5],
            "orig_mae": [0.0, float("nan"), 0.9],
            "norm_mse": [0.0, float("nan"), 0.4],
            "norm_mae": [0.0, float("nan"), 0.3],
            "has_answer_tag": [True, False, True],
            "has_answer_close": [True, False, True],
            "was_clipped": [False, True, False],
            "format_failure_reason": ["", "missing_answer_close_tag", "length_mismatch:95!=96"],
            "final_answer_reject_reason": ["", "missing_answer_close_tag", "invalid_answer_shape:lines=95,expected=96"],
            "length_hard_fail": [False, False, True],
            "strict_length_match": [True, False, False],
            "trainer_seq_score": [0.7, -1.0, -0.55],
            "selected_model": ["itransformer", "itransformer", "chronos2"],
            "generation_stop_reason": ["stop", "length", "stop"],
            "selected_forecast_orig_mse": [0.2, 0.8, 1.7],
            "selected_forecast_len_match": [True, False, True],
            "selected_forecast_exact_copy": [True, False, False],
            "final_vs_selected_mse": [0.0, float("nan"), 0.15],
            "refinement_delta_orig_mse": [0.2, float("nan"), 0.2],
            "refinement_compare_len": [96, float("nan"), 95],
            "refinement_changed_value_count": [0, float("nan"), 3],
            "refinement_first_changed_index": [-1, float("nan"), 72],
            "refinement_change_mean_abs": [0.0, float("nan"), 0.05],
            "refinement_change_max_abs": [0.0, float("nan"), 0.2],
            "refinement_changed": [False, False, True],
            "refinement_improved": [True, False, True],
            "refinement_degraded": [False, False, False],
            "analysis_coverage_ratio": [1.0, 0.4, 0.8],
            "feature_tool_count": [3, 1, 2],
            "required_feature_tool_count": [3, 3, 2],
            "missing_required_feature_tool_count": [0, 2, 0],
            "prediction_call_count": [1, 1, 1],
            "tool_call_count": [4, 0, 3],
            "history_analysis_count": [3, 0, 2],
            "illegal_turn3_tool_call_count": [0, 0, 1],
            "prediction_requested_model": ["itransformer", "__missing__", "chronos2"],
            "prediction_model_defaulted": [False, True, False],
            "feature_tool_signature": [
                "extract_basic_statistics->extract_event_summary->extract_data_quality",
                "extract_data_quality",
                "extract_basic_statistics->extract_within_channel_dynamics",
            ],
            "required_feature_tool_signature": [
                "extract_basic_statistics->extract_event_summary->extract_data_quality",
                "extract_basic_statistics->extract_event_summary->extract_data_quality",
                "extract_basic_statistics->extract_within_channel_dynamics",
            ],
            "tool_call_sequence": [
                "extract_basic_statistics->extract_event_summary->extract_data_quality->predict_time_series",
                "",
                "extract_basic_statistics->extract_within_channel_dynamics->predict_time_series",
            ],
            "analysis_state_signature": [
                "basic_statistics|data_quality|event_summary",
                "data_quality",
                "basic_statistics|within_channel_dynamics",
            ],
            "workflow_status": ["accepted", "not_attempted", "rejected"],
            "turn_stage": ["refinement", "refinement", "refinement"],
            "prediction_tool_error": ["", "RuntimeError: timeout", ""],
            "selected_forecast_preview": ["1.0000, 2.0000 ... 95.0000, 96.0000", "", "1.0000, 2.0000 ... 94.0000, 95.0000"],
            "final_answer_preview": ["1.0000, 2.0000 ... 95.0000, 96.0000", "", "1.0000, 2.0000 ... 94.5000, 95.5000"],
        }

        previous_debug_dir = os.environ.get("TS_MIN_DEBUG_DIR")
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["TS_MIN_DEBUG_DIR"] = tmp_dir
            try:
                trainer._write_min_eval_debug_files(
                    sample_uids=["sample-0", "sample-1", "sample-2"],
                    sample_outputs=[success_output, failure_output, near_miss_output],
                    sample_gts=[gt_text, gt_text, gt_text],
                    sample_scores=[0.7, -1.0, -0.55],
                    reward_extra_infos_dict=reward_extra_infos,
                )
            finally:
                if previous_debug_dir is None:
                    os.environ.pop("TS_MIN_DEBUG_DIR", None)
                else:
                    os.environ["TS_MIN_DEBUG_DIR"] = previous_debug_dir

            with open(os.path.join(tmp_dir, "eval_step_aggregate.jsonl"), "r", encoding="utf-8") as handle:
                agg_row = json.loads(handle.readline())

            self.assertAlmostEqual(agg_row["exact_96_ratio"], 1.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["final_answer_accept_ratio"], 1.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["orig_mse_mean"], 0.0, places=6)
            self.assertAlmostEqual(agg_row["norm_mse_mean"], 0.0, places=6)
            self.assertAlmostEqual(agg_row["length_hard_fail_ratio"], 1.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["selected_forecast_orig_mse_mean"], 0.9, places=6)
            self.assertAlmostEqual(agg_row["selected_forecast_len_match_ratio"], 2.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["selected_forecast_exact_copy_ratio"], 1.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["final_vs_selected_mse_mean"], 0.075, places=6)
            self.assertAlmostEqual(agg_row["refinement_delta_orig_mse_mean"], 0.2, places=6)
            self.assertAlmostEqual(agg_row["refinement_compare_len_mean"], 95.5, places=6)
            self.assertAlmostEqual(agg_row["refinement_changed_value_count_mean"], 1.5, places=6)
            self.assertAlmostEqual(agg_row["refinement_first_changed_index_mean"], 72.0, places=6)
            self.assertAlmostEqual(agg_row["refinement_changed_ratio"], 1.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["refinement_improved_ratio"], 2.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["prediction_model_defaulted_ratio"], 1.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["analysis_coverage_ratio_mean"], (1.0 + 0.4 + 0.8) / 3.0, places=6)
            self.assertAlmostEqual(agg_row["feature_tool_count_mean"], 2.0, places=6)
            self.assertAlmostEqual(agg_row["required_feature_tool_count_mean"], 8.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["missing_required_feature_tool_count_mean"], 2.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["tool_call_count_mean"], 7.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["history_analysis_count_mean"], 5.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["no_tool_call_ratio"], 1.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["no_history_analysis_ratio"], 1.0 / 3.0, places=6)
            self.assertAlmostEqual(agg_row["illegal_turn3_tool_call_ratio"], 1.0 / 3.0, places=6)
            self.assertEqual(agg_row["prediction_tool_error_count"], 1)
            self.assertEqual(agg_row["selected_model_distribution"], {"chronos2": 1, "itransformer": 2})
            self.assertEqual(agg_row["prediction_requested_model_distribution"], {"__missing__": 1, "chronos2": 1, "itransformer": 1})
            self.assertEqual(
                agg_row["required_feature_tool_signature_distribution"],
                {
                    "extract_basic_statistics->extract_event_summary->extract_data_quality": 2,
                    "extract_basic_statistics->extract_within_channel_dynamics": 1,
                },
            )
            self.assertEqual(
                agg_row["tool_call_sequence_distribution"],
                {
                    "extract_basic_statistics->extract_event_summary->extract_data_quality->predict_time_series": 1,
                    "extract_basic_statistics->extract_within_channel_dynamics->predict_time_series": 1,
                    "none": 1,
                },
            )
            self.assertEqual(agg_row["workflow_status_distribution"], {"accepted": 1, "not_attempted": 1, "rejected": 1})
            self.assertEqual(agg_row["format_failure_reason_distribution"]["missing_answer_close_tag"], 1)
            self.assertEqual(agg_row["generation_stop_reason_distribution"], {"length": 1, "stop": 2})
            self.assertEqual(
                agg_row["final_answer_reject_reason_distribution"]["invalid_answer_shape:lines=95,expected=96"],
                1,
            )

            with open(os.path.join(tmp_dir, "eval_step_samples.jsonl"), "r", encoding="utf-8") as handle:
                sample_rows = [json.loads(line) for line in handle if line.strip()]

            near_miss_row = next(
                row for row in sample_rows if row["category"] == "near_miss_94_95" and row["sample_id"] == "sample-2"
            )
            self.assertEqual(near_miss_row["generation_stop_reason"], "stop")
            self.assertEqual(near_miss_row["prediction_requested_model"], "chronos2")
            self.assertTrue(near_miss_row["refinement_changed"])
            self.assertTrue(near_miss_row["selected_forecast_len_match"])
            self.assertFalse(near_miss_row["selected_forecast_exact_copy"])
            self.assertEqual(near_miss_row["refinement_changed_value_count"], 3)
            self.assertEqual(near_miss_row["refinement_first_changed_index"], 72)
            self.assertEqual(near_miss_row["tool_call_count"], 3)
            self.assertEqual(near_miss_row["history_analysis_count"], 2)
            self.assertEqual(near_miss_row["required_feature_tool_count"], 2)
            self.assertEqual(near_miss_row["missing_required_feature_tool_count"], 0)
            self.assertEqual(
                near_miss_row["required_feature_tool_signature"],
                "extract_basic_statistics->extract_within_channel_dynamics",
            )
            self.assertEqual(
                near_miss_row["tool_call_sequence"],
                "extract_basic_statistics->extract_within_channel_dynamics->predict_time_series",
            )
            self.assertEqual(near_miss_row["analysis_state_signature"], "basic_statistics|within_channel_dynamics")
            self.assertIn("94.5000", near_miss_row["final_answer_preview"])
            self.assertIn("filled_orig_mse", near_miss_row)
            self.assertIn("filled_norm_mse", near_miss_row)
            self.assertAlmostEqual(near_miss_row["filled_raw_mse"], near_miss_row["filled_orig_mse"], places=6)


if __name__ == "__main__":
    unittest.main()
