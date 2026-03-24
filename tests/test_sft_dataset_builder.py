import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from recipe.time_series_forecast.build_etth1_sft_dataset import (
    DEFAULT_BALANCE_TRAIN_ROUTING_MODELS,
    FEATURE_TOOL_BUILDERS,
    SUPPORTED_PREDICTION_MODELS,
    TURN3_TARGET_MODE_ENGINEERING_REFINE,
    TURN3_TARGET_MODE_PAPER_STRICT,
    _build_turn3_target,
    _resolve_prediction_text,
    _resolve_reference_teacher_model,
    _select_model_from_scores,
    _select_prediction_model_by_heuristic,
    convert_jsonl_to_sft_parquet,
    parse_args,
    rebalance_train_routing_model_records,
    rebalance_train_stage_records,
    rebalance_train_turn3_targets,
)
from recipe.time_series_forecast.diagnostic_policy import (
    FEATURE_TOOL_ORDER,
    plan_diagnostic_tool_batches,
    select_feature_tool_names,
)
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.utils import compact_prediction_tool_output_from_string
from recipe.time_series_forecast.utils import parse_time_series_string


class TestETTh1SFTDatasetBuilder(unittest.TestCase):
    def _heuristic_model_for_sample(self, sample: dict) -> str:
        raw_prompt = sample["raw_prompt"][0]["content"]
        task_spec = parse_task_prompt(raw_prompt, data_source=sample.get("data_source"))
        historical_data = task_spec.historical_data or raw_prompt
        _, history_values = parse_time_series_string(historical_data, target_column=task_spec.target_column or "OT")
        model_name, _feature_snapshot, _reason = _select_prediction_model_by_heuristic(history_values)
        return model_name

    def _selected_feature_tools_for_sample(self, sample: dict) -> list[str]:
        raw_prompt = sample["raw_prompt"][0]["content"]
        task_spec = parse_task_prompt(raw_prompt, data_source=sample.get("data_source"))
        historical_data = task_spec.historical_data or raw_prompt
        _, history_values = parse_time_series_string(historical_data, target_column=task_spec.target_column or "OT")
        return select_feature_tool_names(history_values)

    def test_build_feature_tool_results_uses_state_aware_diagnostic_subset(self):
        sample = self._load_base_sample()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "single.jsonl"
            output_path = Path(tmpdir) / "single.parquet"
            input_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataframe = convert_jsonl_to_sft_parquet(
                input_path=input_path,
                output_path=output_path,
                max_samples=1,
            )

        expected_tool_names = list(FEATURE_TOOL_ORDER)
        first_row = dataframe.sort_values("turn_stage_order").iloc[0]
        self.assertEqual(list(first_row["selected_feature_tools"]), expected_tool_names)
        self.assertEqual(int(first_row["selected_feature_tool_count"]), len(expected_tool_names))
        self.assertEqual(first_row["selected_feature_tool_signature"], "->".join(expected_tool_names))

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

    def test_convert_small_rl_slice_to_stepwise_parquet(self):
        sample = self._load_base_sample()
        sample_b = copy.deepcopy(sample)
        sample_b["index"] = int(sample.get("index", 0)) + 1
        expected_model = self._heuristic_model_for_sample(sample)

        async def fake_predict_with_runtime_tools(**kwargs):
            self.assertIn(kwargs["model_name"], SUPPORTED_PREDICTION_MODELS)
            return str(sample["teacher_prediction_text"])

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "mini.jsonl"
            output_path = Path(tmpdir) / "train.parquet"
            with input_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                handle.write(json.dumps(sample_b, ensure_ascii=False) + "\n")

            with mock.patch(
                "recipe.time_series_forecast.build_etth1_sft_dataset._predict_with_runtime_tools",
                new=fake_predict_with_runtime_tools,
            ):
                dataframe = convert_jsonl_to_sft_parquet(
                    input_path=input_path,
                    output_path=output_path,
                    max_samples=2,
                )

            self.assertTrue(output_path.exists())

            loaded = pd.read_parquet(output_path)
            self.assertIn("messages", loaded.columns)
            self.assertIn("tools", loaded.columns)
            self.assertNotIn("enable_thinking", loaded.columns)
            self.assertIn("source_sample_index", loaded.columns)
            self.assertIn("turn_stage", loaded.columns)
            self.assertIn("turn_stage_order", loaded.columns)
            self.assertIn("paper_turn3_required", loaded.columns)
            self.assertIn("trajectory_turn_count", loaded.columns)
            self.assertIn("turn3_target_mode", loaded.columns)
            self.assertEqual(list(loaded["sample_index"]), list(range(len(loaded))))

            grouped = {
                int(source_idx): frame.sort_values("turn_stage_order").reset_index(drop=True)
                for source_idx, frame in loaded.groupby("source_sample_index")
            }
            self.assertEqual(sorted(grouped.keys()), [int(sample["index"]), int(sample_b["index"])])

            for source_idx, frame in grouped.items():
                selected_tools = list(frame.iloc[0]["selected_feature_tools"])
                expected_turn_count = len(plan_diagnostic_tool_batches(selected_tools, max_parallel_calls=5)) + 2
                self.assertEqual(len(frame), expected_turn_count)
                self.assertEqual(list(frame["turn_stage"][:-2]), ["diagnostic"] * (expected_turn_count - 2))
                self.assertEqual(list(frame["turn_stage"][-2:]), ["routing", "refinement"])
                self.assertEqual(int(frame.iloc[0]["trajectory_turn_count"]), expected_turn_count)
                self.assertTrue(bool(frame.iloc[-1]["paper_turn3_required"]))
                self.assertTrue(all(not bool(value) for value in frame.iloc[:-1]["paper_turn3_required"].tolist()))

                first_diag_messages = frame.iloc[0]["messages"]
                self.assertEqual([msg["role"] for msg in first_diag_messages], ["system", "user", "assistant"])
                first_tool_calls = first_diag_messages[-1]["tool_calls"]
                self.assertGreaterEqual(len(first_tool_calls), 1)
                self.assertLessEqual(len(first_tool_calls), 5)
                self.assertEqual(first_tool_calls[0]["function"]["name"], "extract_basic_statistics")
                self.assertTrue(all(call["function"]["name"] != "predict_time_series" for call in first_tool_calls))
                first_diag_tools = list(frame.iloc[0]["tools"])
                self.assertEqual(
                    [tool["function"]["name"] for tool in first_diag_tools],
                    list(frame.iloc[0]["current_required_feature_tools"]),
                )

                routing_messages = frame.iloc[-2]["messages"]
                self.assertEqual([msg["role"] for msg in routing_messages], ["system", "user", "assistant"])
                self.assertEqual(str(routing_messages[-1]["content"]), "")
                self.assertTrue(str(routing_messages[-1]["reasoning_content"]).strip())
                self.assertIn("Observed diagnostics:", routing_messages[-1]["reasoning_content"])
                self.assertIn("ACF(1)=", routing_messages[-1]["reasoning_content"])
                self.assertIn("Decision:", routing_messages[-1]["reasoning_content"])
                self.assertEqual(routing_messages[-1]["tool_calls"][0]["function"]["name"], "predict_time_series")
                self.assertIn(frame.iloc[-2]["selected_prediction_model"], routing_messages[-1]["reasoning_content"])
                self.assertIn("predict_time_series", routing_messages[-1]["reasoning_content"])
                self.assertIn("### Analysis Summary", routing_messages[1]["content"])
                routing_tools = list(frame.iloc[-2]["tools"])
                self.assertEqual([tool["function"]["name"] for tool in routing_tools], ["predict_time_series"])

                refinement_messages = frame.iloc[-1]["messages"]
                self.assertEqual([msg["role"] for msg in refinement_messages], ["system", "user", "assistant"])
                self.assertIn("### Prediction Tool Output", refinement_messages[1]["content"])
                self.assertIn("<think>", refinement_messages[-1]["content"])
                self.assertIn("<answer>", refinement_messages[-1]["content"])
                self.assertIsNone(frame.iloc[-1]["tools"])
                self.assertEqual(frame.iloc[-1]["turn3_target_type"], "validated_keep")
                self.assertEqual(frame.iloc[-1]["turn3_target_mode"], TURN3_TARGET_MODE_PAPER_STRICT)
                self.assertEqual(int(frame.iloc[-1]["source_sample_index"]), source_idx)
                self.assertIsInstance(list(frame.iloc[-1]["selected_feature_tools"]), list)

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
                grouped[int(sample["index"])].iloc[-1]["selected_feature_tool_signature"],
                "->".join(list(grouped[int(sample["index"])].iloc[-1]["selected_feature_tools"])),
            )
            self.assertEqual(
                int(grouped[int(sample["index"])].iloc[-1]["selected_feature_tool_count"]),
                len(list(grouped[int(sample["index"])].iloc[-1]["selected_feature_tools"])),
            )
            refinement_row = grouped[int(sample["index"])].iloc[-1]
            self.assertEqual(refinement_row["selected_prediction_model"], expected_model)
            self.assertEqual(refinement_row["routing_policy_source"], "heuristic_rule_based")
            expected_prediction_source = (
                "reference_teacher_cached"
                if expected_model == str(sample["reference_teacher_model"])
                else "reference_teacher_runtime"
            )
            self.assertEqual(refinement_row["base_prediction_source"], expected_prediction_source)
            self.assertEqual(refinement_row["turn3_target_type"], "validated_keep")
            self.assertEqual(refinement_row["refine_ops_signature"], "none")
            self.assertAlmostEqual(float(refinement_row["refine_gain_mse"]), 0.0, places=6)
            self.assertEqual(int(refinement_row["refine_changed_value_count"]), 0)
            self.assertEqual(int(refinement_row["refine_first_changed_index"]), -1)

    def test_convert_uses_cached_teacher_prediction_when_present(self):
        sample = self._load_base_sample()
        sample["reference_teacher_model"] = self._heuristic_model_for_sample(sample)
        sample["teacher_prediction_source"] = "reference_teacher"

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "cached.jsonl"
            parquet_path = Path(tmpdir) / "cached.parquet"
            jsonl_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataframe = convert_jsonl_to_sft_parquet(
                input_path=jsonl_path,
                output_path=parquet_path,
                max_samples=1,
                turn3_target_mode=TURN3_TARGET_MODE_ENGINEERING_REFINE,
            )

            refinement_row = dataframe.loc[dataframe["turn_stage"] == "refinement"].iloc[0]
            messages = refinement_row["messages"]
            expected_tool_output = compact_prediction_tool_output_from_string(
                sample["teacher_prediction_text"],
                model_name=str(sample["reference_teacher_model"]),
            )
            self.assertEqual(refinement_row["base_prediction_source"], "reference_teacher_cached")
            self.assertIn(
                expected_tool_output,
                messages[1]["content"],
            )

    def test_rebalance_train_stage_records_repeats_routing_only(self):
        sample = self._load_base_sample()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "single.jsonl"
            output_path = Path(tmpdir) / "single.parquet"
            input_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataframe = convert_jsonl_to_sft_parquet(
                input_path=input_path,
                output_path=output_path,
                max_samples=1,
            )

        balanced = rebalance_train_stage_records(
            dataframe,
            stage_repeat_factors={"diagnostic": 1, "routing": 4, "refinement": 1},
        )
        self.assertEqual(
            balanced["turn_stage"].value_counts().to_dict(),
            {"routing": 4, "diagnostic": 1, "refinement": 1},
        )
        self.assertEqual(list(balanced["sample_index"]), list(range(len(balanced))))

    def test_rebalance_train_routing_model_records_balances_model_classes(self):
        dataframe = pd.DataFrame(
            [
                {"sample_index": 0, "source_sample_index": 0, "turn_stage": "diagnostic", "turn_stage_order": 0},
                {
                    "sample_index": 1,
                    "source_sample_index": 0,
                    "turn_stage": "routing",
                    "turn_stage_order": 1,
                    "selected_prediction_model": "patchtst",
                },
                {
                    "sample_index": 2,
                    "source_sample_index": 1,
                    "turn_stage": "routing",
                    "turn_stage_order": 1,
                    "selected_prediction_model": "patchtst",
                },
                {
                    "sample_index": 3,
                    "source_sample_index": 2,
                    "turn_stage": "routing",
                    "turn_stage_order": 1,
                    "selected_prediction_model": "arima",
                },
                {"sample_index": 4, "source_sample_index": 0, "turn_stage": "refinement", "turn_stage_order": 2},
            ]
        )

        balanced = rebalance_train_routing_model_records(dataframe, enabled=True)
        routing_counts = balanced.loc[balanced["turn_stage"] == "routing", "selected_prediction_model"].value_counts().to_dict()
        self.assertEqual(routing_counts, {"patchtst": 2, "arima": 2})
        self.assertEqual(list(balanced["sample_index"]), list(range(len(balanced))))

    def test_parse_args_enables_routing_model_balance_by_default(self):
        with mock.patch("sys.argv", ["build_etth1_sft_dataset.py"]):
            args = parse_args()
        self.assertEqual(bool(args.balance_train_routing_models), DEFAULT_BALANCE_TRAIN_ROUTING_MODELS)

    def test_resolve_reference_teacher_model_uses_offline_best_model_for_rl_source(self):
        sample = self._load_base_sample()
        sample.pop("reference_teacher_model", None)
        sample["offline_best_model"] = "itransformer"

        self.assertEqual(_resolve_reference_teacher_model(sample), "itransformer")

    def test_resolve_prediction_text_falls_back_when_cached_teacher_output_is_invalid(self):
        sample = self._load_base_sample()
        sample["teacher_prediction_text"] = "1.0000"

        async def fake_predict_with_runtime_tools(**kwargs):
            self.assertEqual(kwargs["model_name"], "chronos2")
            self.assertEqual(kwargs["forecast_horizon"], 3)
            return "1.0000\n2.0000\n3.0000"

        with mock.patch(
            "recipe.time_series_forecast.build_etth1_sft_dataset._predict_with_runtime_tools",
            new=fake_predict_with_runtime_tools,
        ):
            prediction_text, prediction_source = _resolve_prediction_text(
                sample=sample,
                historical_data="1.0000\n2.0000\n3.0000",
                data_source="ETTh1",
                target_column="OT",
                forecast_horizon=3,
                model_name="chronos2",
                allow_cached_reference=True,
            )

        self.assertEqual(prediction_text, "1.0000\n2.0000\n3.0000")
        self.assertEqual(prediction_source, "reference_teacher_runtime")

    def test_convert_builds_local_refine_target_for_spike_teacher_prediction(self):
        sample = self._load_base_sample()
        sample["teacher_eval_score_margin"] = 0.01
        sample["teacher_prediction_text"] = self._make_spike_teacher_prediction(sample)

        async def fake_predict_with_runtime_tools(**kwargs):
            self.assertIn(kwargs["model_name"], SUPPORTED_PREDICTION_MODELS)
            return str(sample["teacher_prediction_text"])

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "spike.jsonl"
            parquet_path = Path(tmpdir) / "spike.parquet"
            jsonl_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            with mock.patch(
                "recipe.time_series_forecast.build_etth1_sft_dataset._predict_with_runtime_tools",
                new=fake_predict_with_runtime_tools,
            ):
                dataframe = convert_jsonl_to_sft_parquet(
                    input_path=jsonl_path,
                    output_path=parquet_path,
                    max_samples=1,
                    turn3_target_mode=TURN3_TARGET_MODE_ENGINEERING_REFINE,
                )

            refinement_row = dataframe.loc[dataframe["turn_stage"] == "refinement"].iloc[0]
            self.assertEqual(refinement_row["turn3_target_type"], "local_refine")
            self.assertIn("isolated_spike_smoothing", refinement_row["refine_ops_signature"])
            self.assertGreater(int(refinement_row["refine_changed_value_count"]), 0)
            self.assertGreaterEqual(int(refinement_row["refine_first_changed_index"]), 0)
            self.assertGreater(float(refinement_row["refine_gain_mse"]), 0.0)

    def test_convert_rl_sample_uses_offline_best_model_as_reference_teacher_model(self):
        sample = self._load_base_sample()
        sample.pop("reference_teacher_model", None)
        sample["offline_best_model"] = "itransformer"

        async def fake_predict_with_runtime_tools(**kwargs):
            self.assertIn(kwargs["model_name"], SUPPORTED_PREDICTION_MODELS)
            return str(sample["teacher_prediction_text"])

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "single.jsonl"
            output_path = Path(tmpdir) / "single.parquet"
            input_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            with mock.patch(
                "recipe.time_series_forecast.build_etth1_sft_dataset._predict_with_runtime_tools",
                new=fake_predict_with_runtime_tools,
            ):
                dataframe = convert_jsonl_to_sft_parquet(
                    input_path=input_path,
                    output_path=output_path,
                    max_samples=1,
                )

        self.assertEqual(set(dataframe["reference_teacher_model"].astype(str)), {"itransformer"})

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
            turn3_target_mode=TURN3_TARGET_MODE_ENGINEERING_REFINE,
        )

        self.assertEqual(turn3_target["turn3_target_type"], "validated_keep")
        self.assertEqual(turn3_target["refine_ops_signature"], "none")
        self.assertEqual(int(turn3_target["refine_changed_value_count"]), 0)

    def test_build_turn3_target_paper_strict_keeps_selected_forecast_even_with_refine_signal(self):
        sample = self._load_base_sample()
        sample["teacher_eval_score_margin"] = 0.01
        spike_prediction = self._make_spike_teacher_prediction(sample)

        turn3_target = _build_turn3_target(
            sample=sample,
            history_values=[float(idx) for idx in range(128)],
            base_prediction_text=spike_prediction,
            forecast_horizon=96,
            model_name="chronos2",
            selected_feature_tools=[name for name, _builder in FEATURE_TOOL_BUILDERS],
            turn3_target_mode=TURN3_TARGET_MODE_PAPER_STRICT,
        )

        self.assertEqual(turn3_target["turn3_target_type"], "validated_keep")
        self.assertEqual(turn3_target["refine_ops_signature"], "none")
        self.assertEqual(float(turn3_target["refine_gain_mse"]), 0.0)
        self.assertEqual(turn3_target["turn3_trigger_reason"], "evidence_consistent")

    def test_select_prediction_model_by_heuristic_returns_supported_model_and_reason(self):
        sample = self._load_base_sample()
        raw_prompt = sample["raw_prompt"][0]["content"]
        task_spec = parse_task_prompt(raw_prompt, data_source=sample.get("data_source"))
        historical_data = task_spec.historical_data or raw_prompt
        _, history_values = parse_time_series_string(historical_data, target_column=task_spec.target_column or "OT")

        model_name, feature_snapshot, routing_reason = _select_prediction_model_by_heuristic(history_values)

        self.assertIn(model_name, SUPPORTED_PREDICTION_MODELS)
        self.assertTrue(str(routing_reason).strip())
        self.assertIn("acf1", feature_snapshot)
        self.assertIn("changepoint_count", feature_snapshot)
        self.assertIn("dominant_pattern", feature_snapshot)

    def test_select_model_from_scores_prefers_simpler_model_on_score_tie(self):
        scores = {
            "arima": 1.0,
            "patchtst": 1.0,
            "itransformer": 1.0,
            "chronos2": 1.0,
        }

        self.assertEqual(_select_model_from_scores(scores), "arima")

    def test_heuristic_does_not_force_chronos2_on_mild_quality_or_oscillation(self):
        feature_snapshot = {
            "acf1": 0.90,
            "acf_seasonal": 0.08,
            "cusum_max": 48.0,
            "changepoint_count": 1.0,
            "peak_count": 3.0,
            "peak_spacing_cv": 0.20,
            "monotone_duration": 0.08,
            "residual_exceed_ratio": 0.055,
            "quality_quantization_score": 0.17,
            "quality_saturation_ratio": 0.03,
            "dominant_pattern": "oscillation",
        }

        with mock.patch(
            "recipe.time_series_forecast.build_etth1_sft_dataset._compute_routing_feature_snapshot",
            return_value=feature_snapshot,
        ):
            model_name, _snapshot, routing_reason = _select_prediction_model_by_heuristic([0.0, 1.0, 2.0])

        self.assertEqual(model_name, "patchtst")
        self.assertIn("patch", routing_reason.lower())

    def test_heuristic_keeps_chronos2_for_strong_irregularity(self):
        feature_snapshot = {
            "acf1": 0.82,
            "acf_seasonal": 0.01,
            "cusum_max": 55.0,
            "changepoint_count": 2.0,
            "peak_count": 6.0,
            "peak_spacing_cv": 0.42,
            "monotone_duration": 0.07,
            "residual_exceed_ratio": 0.09,
            "quality_quantization_score": 0.26,
            "quality_saturation_ratio": 0.09,
            "dominant_pattern": "flat",
        }

        with mock.patch(
            "recipe.time_series_forecast.build_etth1_sft_dataset._compute_routing_feature_snapshot",
            return_value=feature_snapshot,
        ):
            model_name, _snapshot, routing_reason = _select_prediction_model_by_heuristic([0.0, 1.0, 2.0])

        self.assertEqual(model_name, "chronos2")
        self.assertIn("robust", routing_reason.lower())

    def test_rebalance_keeps_complete_source_trajectories(self):
        dataframe = pd.DataFrame(
            [
                {"sample_index": 0, "source_sample_index": 10, "turn_stage": "diagnostic", "turn_stage_order": 0, "turn3_target_type": "local_refine"},
                {"sample_index": 1, "source_sample_index": 10, "turn_stage": "refinement", "turn_stage_order": 1, "turn3_target_type": "local_refine"},
                {"sample_index": 2, "source_sample_index": 11, "turn_stage": "diagnostic", "turn_stage_order": 0, "turn3_target_type": "validated_keep"},
                {"sample_index": 3, "source_sample_index": 11, "turn_stage": "refinement", "turn_stage_order": 1, "turn3_target_type": "validated_keep"},
                {"sample_index": 4, "source_sample_index": 12, "turn_stage": "diagnostic", "turn_stage_order": 0, "turn3_target_type": "validated_keep"},
                {"sample_index": 5, "source_sample_index": 12, "turn_stage": "refinement", "turn_stage_order": 1, "turn3_target_type": "validated_keep"},
            ]
        )

        balanced = rebalance_train_turn3_targets(dataframe, min_local_refine_ratio=0.5)

        self.assertEqual(sorted(balanced["source_sample_index"].unique().tolist()), [10, 11])
        self.assertEqual(len(balanced), 4)
        grouped_counts = balanced.groupby("source_sample_index").size().to_dict()
        self.assertEqual(grouped_counts, {10: 2, 11: 2})


if __name__ == "__main__":
    unittest.main()
