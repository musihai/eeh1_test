import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from recipe.time_series_forecast.build_etth1_sft_dataset import (
    DEFAULT_BALANCE_TRAIN_ROUTING_MODELS,
    DEFAULT_ROUTING_LABEL_SOURCE,
    DEFAULT_SFT_STAGE_MODE,
    DEFAULT_TRAIN_LOCAL_REFINE_REFINEMENT_REPEAT_FACTOR,
    DEFAULT_TRAIN_TURN3_REBALANCE_MODE,
    DEFAULT_TURN3_TARGET_MODE,
    FEATURE_TOOL_BUILDERS,
    SFT_STAGE_MODE_REFINEMENT_ONLY,
    SFT_STAGE_MODE_ROUTING_ONLY,
    SUPPORTED_PREDICTION_MODELS,
    TURN3_TARGET_MODE_ENGINEERING_REFINE,
    TURN3_TARGET_MODE_PAPER_STRICT,
    _build_turn3_target,
    _summarize_paper_turn3_protocol,
    _resolve_prediction_text,
    _resolve_reference_teacher_model,
    _select_model_from_scores,
    _select_prediction_model_by_heuristic,
    convert_jsonl_to_sft_parquet,
    filter_train_routing_records_by_confidence,
    parse_args,
    rebalance_refinement_stage_targets,
    rebalance_train_routing_model_records,
    rebalance_train_stage_records,
    repeat_high_confidence_routing_rows,
    rebalance_train_turn3_targets,
    repeat_local_refine_refinement_rows,
    repeat_priority_validated_keep_refinement_rows,
)
from recipe.time_series_forecast.config_utils import ETTH1_COVARIATE_COLUMNS, ETTH1_FEATURE_COLUMNS, ETTH1_TARGET_COLUMN
from recipe.time_series_forecast.dataset_identity import HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS
from recipe.time_series_forecast.diagnostic_policy import (
    FEATURE_TOOL_ORDER,
    plan_diagnostic_tool_batches,
    select_feature_tool_names,
)
from recipe.time_series_forecast.refinement_support import (
    build_refinement_candidate_prediction_text_map,
    build_refinement_support_payload,
    filter_refinement_candidates_for_model,
)
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.utils import parse_time_series_string


class TestETTh1SFTDatasetBuilder(unittest.TestCase):
    def test_default_turn3_target_mode_uses_engineering_refine(self):
        self.assertEqual(DEFAULT_TURN3_TARGET_MODE, TURN3_TARGET_MODE_ENGINEERING_REFINE)

    def test_default_routing_label_source_uses_reference_teacher(self):
        self.assertEqual(DEFAULT_ROUTING_LABEL_SOURCE, "reference_teacher")

    def test_refinement_candidate_filter_applies_model_aware_gating(self):
        base_prediction_text = "\n".join(
            f"2026-03-01 {hour:02d}:00:00 {30.0 + hour / 1000:.4f}" for hour in range(24)
        )
        history_values = [30.0 + hour / 1000 for hour in range(24)]
        base_values = [30.0 + hour / 1000 for hour in range(24)]
        candidate_refinements = [
            ([value + 0.2 for value in base_values], ["local_level_adjust"]),
            ([value + 0.3 for value in base_values], ["isolated_spike_smoothing"]),
            ([value + 0.4 for value in base_values], ["local_slope_adjust"]),
        ]

        arima_filtered = filter_refinement_candidates_for_model(candidate_refinements, prediction_model_used="arima")
        self.assertNotIn("local_level_adjust", [ops[0] for _vals, ops in arima_filtered])
        itransformer_filtered = filter_refinement_candidates_for_model(
            candidate_refinements,
            prediction_model_used="itransformer",
        )
        self.assertNotIn("local_level_adjust", [ops[0] for _vals, ops in itransformer_filtered])

        arima_payload = build_refinement_support_payload(
            base_values=base_values,
            history_values=history_values,
            selected_feature_tools=["extract_within_channel_dynamics"],
            candidate_refinements=candidate_refinements,
            prediction_model_used="arima",
        )
        self.assertNotIn("local_level_adjust", arima_payload["candidate_adjustments"])

        arima_map = build_refinement_candidate_prediction_text_map(
            base_prediction_text=base_prediction_text,
            candidate_refinements=candidate_refinements,
            prediction_model_used="arima",
        )
        self.assertNotIn("local_level_adjust", arima_map)
        self.assertIn("isolated_spike_smoothing", arima_map)
        self.assertIn("local_slope_adjust", arima_map)

    def test_build_turn3_target_uses_filtered_refinement_candidates(self):
        base_prediction_text = "\n".join(
            f"2026-03-01 {hour:02d}:00:00 {30.0 + hour / 1000:.4f}" for hour in range(24)
        )
        history_values = [30.0 + hour / 1000 for hour in range(24)]
        base_values = [30.0 + hour / 1000 for hour in range(24)]
        sample = {
            "reward_model": {
                "ground_truth": "\n".join(
                    f"2026-03-01 {hour:02d}:00:00 {30.2 + hour / 1000:.4f}" for hour in range(24)
                )
            }
        }
        candidate_refinements = [
            ([value + 0.2 for value in base_values], ["local_level_adjust"]),
            ([value + 0.4 for value in base_values], ["local_slope_adjust"]),
        ]

        with mock.patch(
            "recipe.time_series_forecast.build_etth1_sft_dataset.generate_local_refinement_candidates",
            return_value=candidate_refinements,
        ):
            turn3_target = _build_turn3_target(
                sample=sample,
                historical_data=base_prediction_text,
                history_values=history_values,
                base_prediction_text=base_prediction_text,
                forecast_horizon=24,
                model_name="arima",
                selected_feature_tools=["extract_forecast_residuals"],
                turn3_target_mode=TURN3_TARGET_MODE_ENGINEERING_REFINE,
            )

        self.assertNotEqual(turn3_target["refinement_decision_name"], "local_level_adjust")
        self.assertNotIn("local_level_adjust", turn3_target["refinement_candidate_adjustments"])

    def _heuristic_model_for_sample(self, sample: dict) -> str:
        raw_prompt = sample["raw_prompt"][0]["content"]
        task_spec = parse_task_prompt(raw_prompt, data_source=sample.get("data_source"))
        historical_data = task_spec.historical_data or raw_prompt
        _, history_values = parse_time_series_string(historical_data, target_column=task_spec.target_column or "OT")
        selected_feature_tools = self._selected_feature_tools_for_sample(sample)
        model_name, _feature_snapshot, _reason = _select_prediction_model_by_heuristic(
            history_values,
            selected_feature_tools=selected_feature_tools,
        )
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

        expected_tool_names = self._selected_feature_tools_for_sample(sample)
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
        expected_model = str(sample["reference_teacher_model"])

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
                self.assertTrue(str(first_diag_messages[-1]["reasoning_content"]).strip())
                self.assertIn("I will", first_diag_messages[-1]["reasoning_content"])
                first_diag_tools = list(frame.iloc[0]["tools"])
                self.assertEqual(
                    [tool["function"]["name"] for tool in first_diag_tools],
                    list(frame.iloc[0]["current_required_feature_tools"]),
                )

                routing_messages = frame.iloc[-2]["messages"]
                self.assertEqual([msg["role"] for msg in routing_messages], ["system", "user", "assistant"])
                self.assertEqual(str(routing_messages[-1]["content"]), "")
                self.assertTrue(str(routing_messages[-1]["reasoning_content"]).strip())
                self.assertEqual(
                    str(routing_messages[-1]["reasoning_content"]).strip(),
                    "Use the diagnostic evidence to choose one forecasting model.",
                )
                self.assertNotIn("RouteDecision(", routing_messages[-1]["reasoning_content"])
                self.assertNotIn("reason_codes=[", routing_messages[-1]["reasoning_content"])
                self.assertNotIn("RouteSummary:", routing_messages[-1]["reasoning_content"])
                self.assertEqual(routing_messages[-1]["tool_calls"][0]["function"]["name"], "predict_time_series")
                self.assertIn("### Routing Evidence Card", routing_messages[1]["content"])
                self.assertIn("tool_fields:", routing_messages[1]["content"])
                self.assertIn("extract_basic_statistics", routing_messages[1]["content"])
                self.assertIn("missing_tool_groups=[", routing_messages[1]["content"])
                self.assertIn("### Analysis Summary", routing_messages[1]["content"])
                self.assertNotIn("expert_support_signals:", routing_messages[1]["content"])
                self.assertNotIn("### Routing Decision Guide", routing_messages[1]["content"])
                routing_tools = list(frame.iloc[-2]["tools"])
                self.assertEqual([tool["function"]["name"] for tool in routing_tools], ["predict_time_series"])

                refinement_messages = frame.iloc[-1]["messages"]
                self.assertEqual([msg["role"] for msg in refinement_messages], ["system", "user", "assistant"])
                self.assertIn("### Refinement Evidence Card", refinement_messages[1]["content"])
                self.assertIn("keep_support=[", refinement_messages[1]["content"])
                self.assertIn("support_signals=[", refinement_messages[1]["content"])
                self.assertIn("edit_support:", refinement_messages[1]["content"])
                self.assertIn("### Prediction Tool Output", refinement_messages[1]["content"])
                self.assertRegex(
                    refinement_messages[1]["content"],
                    r"### Prediction Tool Output\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} -?\d+(?:\.\d+)?",
                )
                self.assertIn("<think>", refinement_messages[-1]["content"])
                self.assertIn("<answer>", refinement_messages[-1]["content"])
                self.assertIn("<answer>\ndecision=", refinement_messages[-1]["content"])
                self.assertIsNone(frame.iloc[-1]["tools"])
                self.assertEqual(frame.iloc[-1]["turn3_target_type"], "validated_keep")
                self.assertEqual(frame.iloc[-1]["turn3_target_mode"], TURN3_TARGET_MODE_ENGINEERING_REFINE)
                self.assertEqual(int(frame.iloc[-1]["source_sample_index"]), source_idx)
                self.assertIsInstance(list(frame.iloc[-1]["selected_feature_tools"]), list)

            self.assertIn("selected_feature_tool_count", loaded.columns)
            self.assertIn("selected_feature_tool_signature", loaded.columns)
            self.assertIn("turn3_trigger_reason", loaded.columns)
            self.assertIn("refinement_support_signals", loaded.columns)
            self.assertIn("refinement_support_signal_signature", loaded.columns)
            self.assertIn("refinement_keep_support_signals", loaded.columns)
            self.assertIn("refinement_keep_support_signal_signature", loaded.columns)
            self.assertIn("refinement_edit_support_signals", loaded.columns)
            self.assertIn("refinement_candidate_adjustments", loaded.columns)
            self.assertIn("refinement_candidate_adjustment_signature", loaded.columns)
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
            self.assertEqual(refinement_row["routing_policy_source"], "reference_teacher_offline_best")
            expected_prediction_source = "reference_teacher_cached"
            self.assertEqual(refinement_row["base_prediction_source"], expected_prediction_source)
            self.assertEqual(refinement_row["turn3_target_type"], "validated_keep")
            self.assertEqual(refinement_row["refine_ops_signature"], "none")
            self.assertAlmostEqual(float(refinement_row["refine_gain_mse"]), 0.0, places=6)
            self.assertEqual(int(refinement_row["refine_changed_value_count"]), 0)
            self.assertEqual(int(refinement_row["refine_first_changed_index"]), -1)
            self.assertRegex(
                str(refinement_row["base_teacher_prediction_text"]).splitlines()[0],
                r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} -?\d+(?:\.\d+)?$",
            )
            self.assertRegex(
                str(refinement_row["refined_prediction_text"]).splitlines()[0],
                r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} -?\d+(?:\.\d+)?$",
            )

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
            self.assertEqual(refinement_row["base_prediction_source"], "reference_teacher_cached")
            self.assertIn(
                str(sample["teacher_prediction_text"]),
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
        self.assertEqual(str(args.sft_stage_mode), DEFAULT_SFT_STAGE_MODE)
        self.assertEqual(str(args.turn3_target_mode), DEFAULT_TURN3_TARGET_MODE)
        self.assertEqual(str(args.train_turn3_rebalance_mode), DEFAULT_TRAIN_TURN3_REBALANCE_MODE)
        self.assertEqual(
            int(args.train_local_refine_refinement_repeat_factor),
            DEFAULT_TRAIN_LOCAL_REFINE_REFINEMENT_REPEAT_FACTOR,
        )

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

    def test_convert_can_build_routing_only_records_with_heuristic_labels(self):
        sample = self._load_base_sample()
        expected_model = self._heuristic_model_for_sample(sample)

        async def fake_predict_with_runtime_tools(**kwargs):
            self.assertIn(kwargs["model_name"], SUPPORTED_PREDICTION_MODELS)
            return str(sample["teacher_prediction_text"])

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "single.jsonl"
            output_path = Path(tmpdir) / "routing_only.parquet"
            input_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            with mock.patch(
                "recipe.time_series_forecast.build_etth1_sft_dataset._predict_with_runtime_tools",
                new=fake_predict_with_runtime_tools,
            ):
                dataframe = convert_jsonl_to_sft_parquet(
                    input_path=input_path,
                    output_path=output_path,
                    max_samples=1,
                    routing_label_source="heuristic",
                    sft_stage_mode=SFT_STAGE_MODE_ROUTING_ONLY,
                )

        self.assertEqual(set(dataframe["turn_stage"].astype(str)), {"routing"})
        self.assertEqual(set(dataframe["sft_stage_mode"].astype(str)), {SFT_STAGE_MODE_ROUTING_ONLY})
        self.assertEqual(set(dataframe["routing_label_source"].astype(str)), {"heuristic"})
        self.assertEqual(
            set(dataframe["selected_prediction_model"].astype(str)),
            {expected_model},
        )
        self.assertEqual(set(dataframe["routing_policy_source"].astype(str)), {"heuristic_rule_based"})
        reasoning_content = str(dataframe.iloc[0]["messages"][-1]["reasoning_content"])
        self.assertEqual(reasoning_content.strip(), "Use the diagnostic evidence to choose one forecasting model.")
        self.assertNotIn("RouteDecision(", reasoning_content)
        self.assertNotIn(expected_model, reasoning_content)

    def test_filter_train_routing_records_by_confidence_keeps_only_mid_and_high(self):
        dataframe = pd.DataFrame(
            [
                {"sample_index": 0, "source_sample_index": 0, "turn_stage": "diagnostic", "turn_stage_order": 0},
                {
                    "sample_index": 1,
                    "source_sample_index": 0,
                    "turn_stage": "routing",
                    "turn_stage_order": 1,
                    "routing_label_source": "heuristic",
                    "routing_confidence_tier": "low",
                },
                {
                    "sample_index": 2,
                    "source_sample_index": 1,
                    "turn_stage": "routing",
                    "turn_stage_order": 1,
                    "routing_label_source": "heuristic",
                    "routing_confidence_tier": "mid",
                },
                {
                    "sample_index": 3,
                    "source_sample_index": 2,
                    "turn_stage": "routing",
                    "turn_stage_order": 1,
                    "routing_label_source": "heuristic",
                    "routing_confidence_tier": "high",
                },
                {
                    "sample_index": 4,
                    "source_sample_index": 3,
                    "turn_stage": "routing",
                    "turn_stage_order": 1,
                    "routing_label_source": "reference_teacher",
                    "routing_confidence_tier": "low",
                },
            ]
        )

        filtered = filter_train_routing_records_by_confidence(dataframe, min_tier="mid")
        self.assertEqual(
            list(filtered.loc[filtered["turn_stage"] == "routing", "routing_confidence_tier"]),
            ["mid", "high", "low"],
        )
        self.assertEqual(list(filtered["sample_index"]), list(range(len(filtered))))

    def test_repeat_high_confidence_routing_rows_duplicates_only_high_conf_heuristic(self):
        dataframe = pd.DataFrame(
            [
                {
                    "sample_index": 0,
                    "source_sample_index": 0,
                    "turn_stage": "routing",
                    "turn_stage_order": 1,
                    "routing_label_source": "heuristic",
                    "routing_confidence_tier": "high",
                },
                {
                    "sample_index": 1,
                    "source_sample_index": 1,
                    "turn_stage": "routing",
                    "turn_stage_order": 1,
                    "routing_label_source": "heuristic",
                    "routing_confidence_tier": "mid",
                },
                {
                    "sample_index": 2,
                    "source_sample_index": 2,
                    "turn_stage": "routing",
                    "turn_stage_order": 1,
                    "routing_label_source": "reference_teacher",
                    "routing_confidence_tier": "high",
                },
            ]
        )

        repeated = repeat_high_confidence_routing_rows(dataframe, repeat_factor=3)
        self.assertEqual(len(repeated), 5)
        high_conf_rows = repeated.loc[
            (repeated["routing_label_source"] == "heuristic")
            & (repeated["routing_confidence_tier"] == "high")
        ]
        self.assertEqual(len(high_conf_rows), 3)
        self.assertEqual(list(repeated["sample_index"]), list(range(len(repeated))))

    def test_convert_can_build_refinement_only_records(self):
        sample = self._load_base_sample()

        async def fake_predict_with_runtime_tools(**kwargs):
            self.assertIn(kwargs["model_name"], SUPPORTED_PREDICTION_MODELS)
            return str(sample["teacher_prediction_text"])

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "single.jsonl"
            output_path = Path(tmpdir) / "refinement_only.parquet"
            input_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            with mock.patch(
                "recipe.time_series_forecast.build_etth1_sft_dataset._predict_with_runtime_tools",
                new=fake_predict_with_runtime_tools,
            ):
                dataframe = convert_jsonl_to_sft_parquet(
                    input_path=input_path,
                    output_path=output_path,
                    max_samples=1,
                    sft_stage_mode=SFT_STAGE_MODE_REFINEMENT_ONLY,
                )

        self.assertEqual(set(dataframe["turn_stage"].astype(str)), {"refinement"})
        self.assertEqual(set(dataframe["sft_stage_mode"].astype(str)), {SFT_STAGE_MODE_REFINEMENT_ONLY})
        self.assertTrue(all(bool(flag) for flag in dataframe["paper_turn3_required"].tolist()))

    def test_build_turn3_target_does_not_refine_without_evidence(self):
        sample = self._load_base_sample()
        sample["teacher_eval_score_margin"] = 0.20
        spike_prediction = self._make_spike_teacher_prediction(sample)

        turn3_target = _build_turn3_target(
            sample=sample,
            historical_data=str(sample["raw_prompt"][0]["content"]),
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
        self.assertRegex(
            str(turn3_target["base_teacher_prediction_text"]).splitlines()[0],
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} -?\d+(?:\.\d+)?$",
        )
        self.assertRegex(
            str(turn3_target["refined_prediction_text"]).splitlines()[0],
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} -?\d+(?:\.\d+)?$",
        )

    def test_build_turn3_target_paper_strict_keeps_selected_forecast_even_with_refine_signal(self):
        sample = self._load_base_sample()
        sample["teacher_eval_score_margin"] = 0.01
        spike_prediction = self._make_spike_teacher_prediction(sample)

        turn3_target = _build_turn3_target(
            sample=sample,
            historical_data=str(sample["raw_prompt"][0]["content"]),
            history_values=[float(idx) for idx in range(128)],
            base_prediction_text=spike_prediction,
            forecast_horizon=96,
            model_name="chronos2",
            selected_feature_tools=[name for name, _builder in FEATURE_TOOL_BUILDERS],
            turn3_target_mode=TURN3_TARGET_MODE_PAPER_STRICT,
        )

        self.assertEqual(turn3_target["turn3_target_type"], "local_refine")
        self.assertIn("isolated_spike_smoothing", turn3_target["refine_ops_signature"])
        self.assertGreater(float(turn3_target["refine_gain_mse"]), 0.0)
        self.assertGreater(int(turn3_target["refine_changed_value_count"]), 0)
        self.assertRegex(
            str(turn3_target["refined_prediction_text"]).splitlines()[0],
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} -?\d+(?:\.\d+)?$",
        )

    def test_convert_builds_local_refine_target_for_spike_teacher_prediction_in_paper_strict_mode(self):
        sample = self._load_base_sample()
        sample["teacher_eval_score_margin"] = 0.01
        sample["teacher_prediction_text"] = self._make_spike_teacher_prediction(sample)

        async def fake_predict_with_runtime_tools(**kwargs):
            self.assertIn(kwargs["model_name"], SUPPORTED_PREDICTION_MODELS)
            return str(sample["teacher_prediction_text"])

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "spike_paper.jsonl"
            parquet_path = Path(tmpdir) / "spike_paper.parquet"
            jsonl_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            with mock.patch(
                "recipe.time_series_forecast.build_etth1_sft_dataset._predict_with_runtime_tools",
                new=fake_predict_with_runtime_tools,
            ):
                dataframe = convert_jsonl_to_sft_parquet(
                    input_path=jsonl_path,
                    output_path=parquet_path,
                    max_samples=1,
                    turn3_target_mode=TURN3_TARGET_MODE_PAPER_STRICT,
                )

        refinement_row = dataframe.loc[dataframe["turn_stage"] == "refinement"].iloc[0]
        self.assertEqual(refinement_row["turn3_target_mode"], TURN3_TARGET_MODE_PAPER_STRICT)
        self.assertEqual(refinement_row["turn3_target_type"], "local_refine")
        self.assertIn("isolated_spike_smoothing", refinement_row["refine_ops_signature"])
        self.assertRegex(
            str(refinement_row["messages"][-1]["content"]),
            r"<answer>\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} -?\d+(?:\.\d+)?",
        )

    def test_main_keeps_train_local_refine_ratio_enabled_in_paper_strict_mode(self):
        def _protocol_messages() -> list[dict]:
            return [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "<think>\nkeep\n</think>\n<answer>\n1.0000\n</answer>"},
            ]

        def _make_df(turn3_target_type: str) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "sample_index": 0,
                        "source_sample_index": 0,
                        "turn_stage": "refinement",
                        "turn_stage_order": 2,
                        "forecast_horizon": 1,
                        "messages": _protocol_messages(),
                        "turn3_target_type": turn3_target_type,
                        "reference_teacher_model": "patchtst",
                        "selected_prediction_model": "patchtst",
                        "turn3_trigger_reason": "evidence_consistent",
                        "refine_ops_signature": "none",
                        "selected_feature_tool_signature": "extract_basic_statistics",
                        "routing_policy_source": "heuristic_rule_based",
                        "base_prediction_source": "reference_teacher_cached",
                    }
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_jsonl = tmp_path / "train.jsonl"
            val_jsonl = tmp_path / "val.jsonl"
            train_jsonl.write_text("{}\n", encoding="utf-8")
            val_jsonl.write_text("{}\n", encoding="utf-8")
            output_dir = tmp_path / "out"

            args = Namespace(
                train_jsonl=str(train_jsonl),
                val_jsonl=str(val_jsonl),
                test_jsonl=str(tmp_path / "missing_test.jsonl"),
                output_dir=str(output_dir),
                max_train_samples=1,
                max_val_samples=1,
                max_test_samples=0,
                sft_stage_mode="full",
                turn3_target_mode=TURN3_TARGET_MODE_PAPER_STRICT,
                routing_label_source="heuristic",
                train_min_local_refine_ratio=0.35,
                train_turn3_rebalance_mode="downsample_keep",
                train_stage_repeat_factors='{"diagnostic":1,"routing":1,"refinement":1}',
                balance_train_routing_models=True,
                train_priority_validated_keep_repeat_factor=1,
                train_local_refine_refinement_repeat_factor=1,
            )

            captured_ratios: list[float] = []
            source_metadata = {
                "dataset_kind": "etth1_teacher_curated_sft",
                "pipeline_stage": "teacher200_curated",
                "task_type": "multivariate time-series forecasting",
                "historical_data_protocol": HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
                "target_column": ETTH1_TARGET_COLUMN,
                "observed_feature_columns": list(ETTH1_FEATURE_COLUMNS),
                "observed_covariates": list(ETTH1_COVARIATE_COLUMNS),
                "model_input_width": len(ETTH1_FEATURE_COLUMNS),
            }

            def _fake_convert_jsonl_to_sft_parquet(
                *,
                input_path,
                output_path,
                max_samples=-1,
                turn3_target_mode=None,
                routing_label_source=None,
                sft_stage_mode=None,
            ):
                if Path(input_path).name == "train.jsonl":
                    return _make_df("local_refine")
                return _make_df("validated_keep")

            def _fake_rebalance_train_turn3_targets(dataframe, min_local_refine_ratio, rebalance_mode):
                captured_ratios.append(float(min_local_refine_ratio))
                self.assertEqual(str(rebalance_mode), "downsample_keep")
                return dataframe

            with mock.patch("recipe.time_series_forecast.build_etth1_sft_dataset.parse_args", return_value=args):
                with mock.patch(
                    "recipe.time_series_forecast.build_etth1_sft_dataset.validate_sibling_metadata",
                    return_value=(source_metadata, tmp_path / "metadata.json"),
                ):
                    with mock.patch(
                        "recipe.time_series_forecast.build_etth1_sft_dataset.convert_jsonl_to_sft_parquet",
                        side_effect=_fake_convert_jsonl_to_sft_parquet,
                    ):
                        with mock.patch(
                            "recipe.time_series_forecast.build_etth1_sft_dataset.rebalance_train_turn3_targets",
                            side_effect=_fake_rebalance_train_turn3_targets,
                        ):
                            with mock.patch(
                                "recipe.time_series_forecast.build_etth1_sft_dataset.rebalance_train_routing_model_records",
                                side_effect=lambda dataframe, enabled: dataframe,
                            ):
                                with mock.patch(
                                    "recipe.time_series_forecast.build_etth1_sft_dataset.rebalance_train_stage_records",
                                    side_effect=lambda dataframe, stage_repeat_factors: dataframe,
                                ):
                                    with mock.patch(
                                        "recipe.time_series_forecast.build_etth1_sft_dataset._validate_paper_turn3_protocol",
                                        return_value={"turn3_protocol_checked_count": 1, "turn3_protocol_invalid_count": 0},
                                    ):
                                        with mock.patch(
                                            "recipe.time_series_forecast.build_etth1_sft_dataset.write_metadata_file",
                                        ) as mock_write_metadata:
                                            from recipe.time_series_forecast.build_etth1_sft_dataset import main

                                            main()

            self.assertEqual(captured_ratios, [0.35])
            metadata_kwargs = mock_write_metadata.call_args.args[1]
            self.assertEqual(metadata_kwargs["turn3_target_mode"], TURN3_TARGET_MODE_PAPER_STRICT)
            self.assertEqual(metadata_kwargs["target_column"], ETTH1_TARGET_COLUMN)
            self.assertAlmostEqual(metadata_kwargs["train_min_local_refine_ratio"], 0.35, places=6)
            self.assertAlmostEqual(metadata_kwargs["requested_train_min_local_refine_ratio"], 0.35, places=6)

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
        self.assertIn("repeatable local peaks", routing_reason.lower())

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
        self.assertIn("irregular", routing_reason.lower())

    def test_heuristic_does_not_use_hidden_quality_or_residual_signals(self):
        feature_snapshot = {
            "acf1": 0.83,
            "acf_seasonal": 0.07,
            "cusum_max": 44.0,
            "changepoint_count": 2.0,
            "peak_count": 3.0,
            "peak_spacing_cv": 0.22,
            "monotone_duration": 0.06,
            "residual_exceed_ratio": 0.12,
            "quality_quantization_score": 0.35,
            "quality_saturation_ratio": 0.14,
            "dominant_pattern": "oscillation",
        }

        with mock.patch(
            "recipe.time_series_forecast.build_etth1_sft_dataset._compute_routing_feature_snapshot",
            return_value=feature_snapshot,
        ):
            hidden_model, _snapshot, _reason = _select_prediction_model_by_heuristic([0.0, 1.0, 2.0])
            visible_model, _snapshot, _reason = _select_prediction_model_by_heuristic(
                [0.0, 1.0, 2.0],
                selected_feature_tools=[
                    "extract_basic_statistics",
                    "extract_within_channel_dynamics",
                    "extract_event_summary",
                ],
            )

        self.assertEqual(hidden_model, "chronos2")
        self.assertNotEqual(visible_model, "chronos2")

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

    def test_rebalance_prioritizes_arima_validated_keep_flat_tail_examples(self):
        def _timestamp_line(idx: int, value: float) -> str:
            day = 20 + idx // 24
            hour = idx % 24
            return f"2026-03-{day:02d} {hour:02d}:00:00 {value:.4f}"

        plateau_prediction = "\n".join(
            _timestamp_line(idx, 2.5) for idx in range(96)
        )
        varied_prediction = "\n".join(
            _timestamp_line(idx, 2.0 + idx / 100.0) for idx in range(96)
        )
        dataframe = pd.DataFrame(
            [
                {"sample_index": 0, "source_sample_index": 1, "turn_stage": "diagnostic", "turn_stage_order": 0, "turn3_target_type": "local_refine"},
                {"sample_index": 1, "source_sample_index": 1, "turn_stage": "refinement", "turn_stage_order": 1, "turn3_target_type": "local_refine", "selected_prediction_model": "patchtst", "base_teacher_prediction_text": varied_prediction},
                {"sample_index": 2, "source_sample_index": 2, "turn_stage": "diagnostic", "turn_stage_order": 0, "turn3_target_type": "validated_keep"},
                {"sample_index": 3, "source_sample_index": 2, "turn_stage": "refinement", "turn_stage_order": 1, "turn3_target_type": "validated_keep", "selected_prediction_model": "arima", "base_teacher_prediction_text": plateau_prediction},
                {"sample_index": 4, "source_sample_index": 3, "turn_stage": "diagnostic", "turn_stage_order": 0, "turn3_target_type": "validated_keep"},
                {"sample_index": 5, "source_sample_index": 3, "turn_stage": "refinement", "turn_stage_order": 1, "turn3_target_type": "validated_keep", "selected_prediction_model": "patchtst", "base_teacher_prediction_text": varied_prediction},
                {"sample_index": 6, "source_sample_index": 4, "turn_stage": "diagnostic", "turn_stage_order": 0, "turn3_target_type": "validated_keep"},
                {"sample_index": 7, "source_sample_index": 4, "turn_stage": "refinement", "turn_stage_order": 1, "turn3_target_type": "validated_keep", "selected_prediction_model": "patchtst", "base_teacher_prediction_text": varied_prediction},
            ]
        )

        balanced = rebalance_train_turn3_targets(dataframe, min_local_refine_ratio=0.5)

        self.assertEqual(sorted(balanced["source_sample_index"].unique().tolist()), [1, 2])
        kept_refine = balanced.loc[balanced["turn_stage"] == "refinement"].sort_values("source_sample_index")
        self.assertEqual(kept_refine["selected_prediction_model"].tolist(), ["patchtst", "arima"])

    def test_rebalance_can_oversample_local_refine_without_dropping_sources(self):
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

        balanced = rebalance_train_turn3_targets(
            dataframe,
            min_local_refine_ratio=0.5,
            rebalance_mode="oversample_local_refine",
        )

        self.assertEqual(sorted(balanced["source_sample_index"].unique().tolist()), [10, 11, 12])
        refine_rows = balanced.loc[balanced["turn_stage"] == "refinement"]
        self.assertEqual(refine_rows["turn3_target_type"].value_counts().to_dict(), {"local_refine": 2, "validated_keep": 2})
        self.assertEqual(list(balanced["sample_index"]), list(range(len(balanced))))

    def test_rebalance_refinement_stage_targets_keeps_changes_local_to_refinement_rows(self):
        dataframe = pd.DataFrame(
            [
                {"sample_index": 0, "source_sample_index": 10, "turn_stage": "diagnostic", "turn_stage_order": 0, "turn3_target_type": "local_refine"},
                {"sample_index": 1, "source_sample_index": 10, "turn_stage": "refinement", "turn_stage_order": 1, "turn3_target_type": "local_refine"},
                {"sample_index": 2, "source_sample_index": 11, "turn_stage": "routing", "turn_stage_order": 1, "turn3_target_type": "validated_keep"},
                {"sample_index": 3, "source_sample_index": 11, "turn_stage": "refinement", "turn_stage_order": 2, "turn3_target_type": "validated_keep"},
                {"sample_index": 4, "source_sample_index": 12, "turn_stage": "refinement", "turn_stage_order": 2, "turn3_target_type": "validated_keep"},
            ]
        )

        balanced = rebalance_refinement_stage_targets(
            dataframe,
            min_local_refine_ratio=0.5,
            rebalance_mode="oversample_local_refine",
        )

        self.assertEqual(
            balanced.loc[balanced["turn_stage"] == "diagnostic", "source_sample_index"].tolist(),
            [10],
        )
        self.assertEqual(
            balanced.loc[balanced["turn_stage"] == "routing", "source_sample_index"].tolist(),
            [11],
        )
        self.assertEqual(
            balanced.loc[balanced["turn_stage"] == "refinement", "turn3_target_type"].value_counts().to_dict(),
            {"local_refine": 2, "validated_keep": 2},
        )
        self.assertEqual(list(balanced["sample_index"]), list(range(len(balanced))))

    def test_repeat_local_refine_refinement_rows_duplicates_only_local_refine_rows(self):
        dataframe = pd.DataFrame(
            [
                {"sample_index": 0, "source_sample_index": 1, "turn_stage": "diagnostic", "turn_stage_order": 0, "turn3_target_type": "local_refine"},
                {"sample_index": 1, "source_sample_index": 1, "turn_stage": "refinement", "turn_stage_order": 1, "turn3_target_type": "local_refine"},
                {"sample_index": 2, "source_sample_index": 2, "turn_stage": "refinement", "turn_stage_order": 1, "turn3_target_type": "validated_keep"},
            ]
        )

        repeated = repeat_local_refine_refinement_rows(
            dataframe,
            repeat_factor=3,
        )

        repeated_refine = repeated.loc[repeated["turn_stage"] == "refinement"]
        self.assertEqual(
            repeated_refine["turn3_target_type"].astype(str).value_counts().to_dict(),
            {"local_refine": 3, "validated_keep": 1},
        )

    def test_repeat_priority_validated_keep_refinement_rows_duplicates_only_priority_refinement(self):
        def _timestamp_line(idx: int, value: float) -> str:
            day = 20 + idx // 24
            hour = idx % 24
            return f"2026-03-{day:02d} {hour:02d}:00:00 {value:.4f}"

        plateau_prediction = "\n".join(_timestamp_line(idx, 2.5) for idx in range(96))
        varied_prediction = "\n".join(_timestamp_line(idx, 2.0 + idx / 100.0) for idx in range(96))
        dataframe = pd.DataFrame(
            [
                {"sample_index": 0, "source_sample_index": 2, "turn_stage": "diagnostic", "turn_stage_order": 0, "turn3_target_type": "validated_keep"},
                {"sample_index": 1, "source_sample_index": 2, "turn_stage": "routing", "turn_stage_order": 1, "turn3_target_type": "validated_keep", "selected_prediction_model": "arima"},
                {"sample_index": 2, "source_sample_index": 2, "turn_stage": "refinement", "turn_stage_order": 2, "turn3_target_type": "validated_keep", "selected_prediction_model": "arima", "base_teacher_prediction_text": plateau_prediction},
                {"sample_index": 3, "source_sample_index": 3, "turn_stage": "refinement", "turn_stage_order": 2, "turn3_target_type": "validated_keep", "selected_prediction_model": "patchtst", "base_teacher_prediction_text": varied_prediction},
            ]
        )

        repeated = repeat_priority_validated_keep_refinement_rows(
            dataframe,
            repeat_factor=3,
        )

        repeated_refine = repeated.loc[repeated["turn_stage"] == "refinement"]
        self.assertEqual(
            repeated_refine["selected_prediction_model"].astype(str).value_counts().to_dict(),
            {"arima": 3, "patchtst": 1},
        )

    def test_summarize_paper_turn3_protocol_accepts_numpy_messages(self):
        frame = pd.DataFrame(
            [
                {
                    "turn_stage": "refinement",
                    "forecast_horizon": 96,
                    "sample_index": 7,
                    "messages": np.array(
                        [
                            {"role": "system", "content": "s"},
                            {"role": "user", "content": "u"},
                            {
                                "role": "assistant",
                                "content": "<think>reason</think>\n<answer>\ndecision=keep_baseline\n</answer>",
                            },
                        ],
                        dtype=object,
                    ),
                }
            ]
        )

        summary = _summarize_paper_turn3_protocol(frame)

        self.assertEqual(summary["turn3_protocol_valid_count"], 1)
        self.assertEqual(summary["turn3_protocol_invalid_count"], 0)


if __name__ == "__main__":
    unittest.main()
