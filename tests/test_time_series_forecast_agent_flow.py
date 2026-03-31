import unittest
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from arft.agent_flow.agent_flow import (
    AgentFlowMetrics,
    AgentFlowOutput,
    AgentFlowWorkerBase,
    _InternalAgentFlowStep,
)
from recipe.time_series_forecast.reward import compute_score
from recipe.time_series_forecast.time_series_forecast_agent_flow import TimeSeriesForecastAgentFlow


class TestTimeSeriesForecastAgentFlow(unittest.IsolatedAsyncioTestCase):
    def _make_flow(self) -> TimeSeriesForecastAgentFlow:
        flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
        flow.route_default_expert = "itransformer"
        flow.route_decision = None
        flow.route_override_model = None
        flow.route_resolved_model = None
        flow.prediction_results = None
        flow.prediction_requested_model = None
        flow.prediction_model_defaulted = False
        flow.prediction_tool_error = None
        flow.prediction_step_index = None
        flow.prediction_turn_stage = None
        flow.prediction_attempt_count = 0
        flow.prediction_call_count = 0
        flow.max_prediction_attempts = 2
        flow.max_steps = 4
        flow.max_parallel_calls = 5
        flow.illegal_turn3_tool_call_count = 0
        flow.forecast_horizon = 96
        flow.response_length = 4096
        flow.steps = []
        flow.feature_tool_sequence = []
        flow.history_analysis = []
        flow.required_feature_tools = []
        flow.prediction_requested_model = ""
        flow.prediction_model_used = ""
        flow.final_answer_step_index = None
        flow.final_answer_reject_reason = ""
        flow.final_answer_parse_mode = ""
        flow.basic_statistics = None
        flow.within_channel_dynamics = None
        flow.forecast_residuals = None
        flow.data_quality = None
        flow.event_summary = None
        return flow

    def test_required_step_budget_is_fixed_episode_upper_bound(self) -> None:
        flow = self._make_flow()
        self.assertEqual(flow._required_step_budget(), 4)

        flow.steps = [object()] * 450
        self.assertEqual(flow._required_step_budget(), 4)

        flow.prediction_results = "1.0000\n2.0000\n3.0000"
        self.assertEqual(flow._required_step_budget(), 4)

    def test_required_step_budget_stays_fixed_even_if_parallel_cap_changes(self) -> None:
        flow = self._make_flow()
        flow.max_parallel_calls = 2
        self.assertEqual(flow._required_step_budget(), 4)

    def test_required_step_budget_respects_configured_floor(self) -> None:
        flow = self._make_flow()
        flow.max_steps = 6
        self.assertEqual(flow._required_step_budget(), 6)

    def test_final_answer_budget_leaves_headroom_for_closing_tag(self) -> None:
        flow = self._make_flow()
        self.assertEqual(flow._final_answer_max_tokens(), 2944)

        flow.response_length = 1200
        self.assertEqual(flow._final_answer_max_tokens(), 1200)

    async def test_failed_prediction_attempt_does_not_count_as_success(self) -> None:
        flow = self._make_flow()
        flow.basic_statistics = {}
        tool_call = SimpleNamespace(name="predict_time_series", arguments={"model_name": "chronos2"})

        async def _prediction_side_effect(*, model_name: str):
            self.assertEqual(model_name, "chronos2")
            if flow.prediction_attempt_count == 1:
                return None
            flow.prediction_results = "1.0000\n2.0000\n3.0000"
            return "Forecast Values:\n1.0000\n2.0000\n3.0000"

        flow._run_prediction_tool = AsyncMock(side_effect=_prediction_side_effect)

        first_output = await flow._execute_tool_call(tool_call, turn_stage="routing")
        self.assertIsNone(first_output)
        self.assertEqual(flow.prediction_attempt_count, 1)
        self.assertEqual(flow.prediction_call_count, 0)
        self.assertIsNone(flow.prediction_step_index)
        self.assertIsNone(flow.prediction_turn_stage)

        second_output = await flow._execute_tool_call(tool_call, turn_stage="routing")
        self.assertIsNotNone(second_output)
        self.assertEqual(flow.prediction_attempt_count, 2)
        self.assertEqual(flow.prediction_call_count, 1)
        self.assertEqual(flow.prediction_step_index, 1)
        self.assertEqual(flow.prediction_turn_stage, "routing")

    async def test_prediction_retry_budget_rejects_third_attempt(self) -> None:
        flow = self._make_flow()
        flow.basic_statistics = {}
        tool_call = SimpleNamespace(name="predict_time_series", arguments={"model_name": "chronos2"})
        flow._run_prediction_tool = AsyncMock(return_value=None)

        first_output = await flow._execute_tool_call(tool_call, turn_stage="routing")
        second_output = await flow._execute_tool_call(tool_call, turn_stage="routing")
        third_output = await flow._execute_tool_call(tool_call, turn_stage="routing")

        self.assertIsNone(first_output)
        self.assertIsNone(second_output)
        self.assertIsNone(third_output)
        self.assertEqual(flow.prediction_attempt_count, 2)
        self.assertEqual(flow.prediction_call_count, 0)
        self.assertEqual(flow._run_prediction_tool.await_count, 2)

    async def test_invalid_prediction_model_is_rejected_without_defaulting(self) -> None:
        flow = self._make_flow()
        flow.basic_statistics = {}
        tool_call = SimpleNamespace(name="predict_time_series", arguments={"model_name": "unknown_model"})
        flow._run_prediction_tool = AsyncMock()

        output = await flow._execute_tool_call(tool_call, turn_stage="routing")

        self.assertIsNone(output)
        self.assertEqual(flow.prediction_requested_model, "unknown_model")
        self.assertFalse(flow.prediction_model_defaulted)
        self.assertIn("Invalid model_name", flow.prediction_tool_error)
        self.assertEqual(flow.prediction_attempt_count, 0)
        self.assertEqual(flow.prediction_call_count, 0)
        self.assertEqual(flow._run_prediction_tool.await_count, 0)

    def test_shared_reward_tracking_fields_include_required_feature_metrics_and_uid(self) -> None:
        flow = self._make_flow()
        flow.required_feature_tools = [
            "extract_basic_statistics",
            "extract_within_channel_dynamics",
            "extract_event_summary",
        ]
        flow.feature_tool_sequence = [
            "extract_basic_statistics",
            "extract_within_channel_dynamics",
            "extract_event_summary",
        ]
        flow.history_analysis = ["basic", "dynamics", "event"]
        flow.prediction_requested_model = "patchtst"
        flow.prediction_model_used = "patchtst"
        flow.prediction_step_index = 2
        flow.prediction_turn_stage = "routing"
        flow.final_answer_step_index = 3
        flow.basic_statistics = {}
        flow.within_channel_dynamics = {}
        flow.event_summary = {}

        info = flow._shared_reward_tracking_fields(sample_uid="etth1-val-00001")

        self.assertEqual(info["sample_uid"], "etth1-val-00001")
        self.assertEqual(info["required_feature_tool_count"], 3)
        self.assertEqual(info["missing_required_feature_tool_count"], 0)
        self.assertEqual(
            info["required_feature_tool_signature"],
            "extract_basic_statistics->extract_within_channel_dynamics->extract_event_summary",
        )

    def test_tool_schemas_are_stage_specific(self) -> None:
        flow = self._make_flow()
        diagnostic_names = [tool["function"]["name"] for tool in flow._tool_schemas_for_turn("diagnostic")]
        routing_names = [tool["function"]["name"] for tool in flow._tool_schemas_for_turn("routing")]
        refinement_names = [tool["function"]["name"] for tool in flow._tool_schemas_for_turn("refinement")]

        self.assertIn("extract_basic_statistics", diagnostic_names)
        self.assertIn("extract_event_summary", diagnostic_names)
        self.assertNotIn("predict_time_series", diagnostic_names)
        self.assertEqual(routing_names, ["route_time_series"])
        self.assertEqual(refinement_names, [])

    async def test_route_time_series_keep_default_resolves_default_expert(self) -> None:
        flow = self._make_flow()
        flow.basic_statistics = {}
        tool_call = SimpleNamespace(name="route_time_series", arguments={"decision": "keep_default"})
        flow._run_prediction_tool = AsyncMock(return_value="Forecast Values:\n1.0000\n2.0000\n3.0000")

        output = await flow._execute_tool_call(tool_call, turn_stage="routing")

        self.assertIsNotNone(output)
        self.assertEqual(flow.route_decision, "keep_default")
        self.assertEqual(flow.prediction_requested_model, "itransformer")
        self.assertEqual(flow.route_resolved_model, "itransformer")
        flow._run_prediction_tool.assert_awaited_once_with(model_name="itransformer")

    def test_build_user_prompt_uses_policy_owned_diagnostic_stage(self) -> None:
        flow = self._make_flow()
        flow.data_source = "ETTh1"
        flow.target_column = "OT"
        flow.lookback_window = 96
        flow.time_series_data = "1.0000\n2.0000\n3.0000"

        prompt = flow._build_user_prompt()

        self.assertNotIn("### Diagnostic Plan", prompt)
        self.assertIn("planning and diagnostic stage", prompt)
        self.assertIn("extract_event_summary", prompt)
        self.assertIn("First decide what evidence you need", prompt)

    def test_build_user_prompt_uses_canonical_timestamped_prediction_values_in_refinement(self) -> None:
        flow = self._make_flow()
        flow.data_source = "ETTh1"
        flow.target_column = "OT"
        flow.lookback_window = 96
        flow.time_series_data = "2024-01-01 00:00:00 1.0000\n2024-01-01 01:00:00 2.0000"
        flow.history_analysis = ["diagnostic complete"]
        flow.prediction_results = "2024-01-02 00:00:00 3.1000\n2024-01-02 01:00:00 3.2000"
        flow.prediction_tool_output = "Model: arima\nForecast Values:\n3.1000\n3.2000"
        flow.prediction_model_used = "arima"
        flow.forecast_horizon = 2

        prompt = flow._build_user_prompt()

        self.assertIn("### Refinement Evidence Card", prompt)
        self.assertIn("support_signals=[", prompt)
        self.assertIn("### Prediction Tool Output", prompt)
        self.assertIn("2024-01-02 00:00:00 3.1000", prompt)
        self.assertIn("2024-01-02 01:00:00 3.2000", prompt)
        self.assertIn("### Selected Forecast Model: arima", prompt)
        self.assertNotIn("Forecast Values:", prompt)
        self.assertIn("decision_options=[keep_baseline]", prompt)
        self.assertIn("must contain exactly one non-empty line in the form `decision=<name>`", prompt)
        self.assertIn("Stop immediately after `</answer>`", prompt)

    def test_current_turn_stage_moves_to_routing_after_any_diagnostic_analysis(self) -> None:
        flow = self._make_flow()
        self.assertEqual(flow._current_turn_stage(), "diagnostic")
        flow.basic_statistics = {"median": 1.0}
        self.assertEqual(flow._current_turn_stage(), "routing")

    def test_refinement_sampling_params_keep_training_temperature(self) -> None:
        flow = self._make_flow()
        params = flow._prepare_sampling_params(
            {"temperature": 1.0, "top_p": 0.95, "max_tokens": 4096},
            turn_stage="refinement",
        )
        self.assertEqual(params["temperature"], 1.0)
        self.assertEqual(params["top_p"], 0.95)
        self.assertIn("</answer>", params["stop"])

    def test_apply_turn3_horizon_clamp_tracks_discarded_rows(self) -> None:
        flow = self._make_flow()
        timestamp_lines = "\n".join(
            f"2026-03-{1 + (idx // 24):02d} {idx % 24:02d}:00:00 {30.0 + idx / 1000:.4f}"
            for idx in range(100)
        )
        response_text = f"<think>x</think><answer>\n{timestamp_lines}"

        clamped = flow._apply_turn3_horizon_clamp(response_text, turn_stage="refinement")

        self.assertTrue(flow.turn3_horizon_clamped)
        self.assertEqual(flow.turn3_horizon_clamp_discarded_lines, 4)
        self.assertEqual(flow.turn3_horizon_clamp_valid_prefix_lines, 100)
        self.assertEqual(flow.turn3_horizon_clamp_raw_answer_lines, 100)
        self.assertIn("</answer>", clamped)

    def test_apply_turn3_horizon_clamp_is_noop_for_short_answers(self) -> None:
        flow = self._make_flow()
        timestamp_lines = "\n".join(
            f"2026-03-{1 + (idx // 24):02d} {idx % 24:02d}:00:00 {30.0 + idx / 1000:.4f}"
            for idx in range(95)
        )
        response_text = f"<think>x</think><answer>\n{timestamp_lines}"

        clamped = flow._apply_turn3_horizon_clamp(response_text, turn_stage="refinement")

        self.assertFalse(flow.turn3_horizon_clamped)
        self.assertEqual(flow.turn3_horizon_clamp_discarded_lines, 0)
        self.assertEqual(clamped, response_text)

    def test_append_turn_debug_records_sample_uid(self) -> None:
        flow = self._make_flow()
        flow.required_feature_tools = ["extract_basic_statistics"]
        flow.basic_statistics = {"mean": 1.0}
        flow.final_answer_step_index = 3
        reward_extra_info = flow._shared_reward_tracking_fields(sample_uid="etth1-val-00077")
        reward_extra_info.update({"required_step_budget": 4})

        with patch("recipe.time_series_forecast.time_series_forecast_agent_flow.append_chain_debug") as mock_append:
            flow._append_turn_debug(
                step_index=3,
                request_id="req-1",
                sample_index=0,
                sample_uid="etth1-val-00077",
                turn_stage="refinement",
                prompt_text="prompt",
                response_text="response",
                generation_stop_reason="completed",
                generation_finish_reason="stop",
                tool_call_names=[],
                workflow_status="accepted",
                workflow_message="",
                reward_extra_info=reward_extra_info,
            )

        payload = mock_append.call_args.args[1]
        self.assertEqual(payload["sample_uid"], "etth1-val-00077")
        self.assertEqual(payload["debug_bucket"], "ok")
        self.assertEqual(payload["debug_reason"], "ok")
        self.assertEqual(payload["required_feature_tool_count"], 1)
        self.assertEqual(payload["required_feature_tool_signature"], "extract_basic_statistics")
        self.assertEqual(payload["final_answer_step_index"], 3)

    def test_final_reward_uses_same_strict_protocol_as_validation(self) -> None:
        flow = self._make_flow()
        ground_truth = "2024-01-02 00:00:00 3.1000\n2024-01-02 01:00:00 3.2000"
        malformed_final = (
            "<think>\n"
            "I keep the selected forecast because it already matches the diagnostics.\n"
            "</think>\n\n"
            "<answer>\n"
            "2024-01-02 00:00:00 3.1000\n"
            "2024-01-02 01:00:00 3.2000"
        )

        recovered = compute_score(
            data_source="time_series",
            solution_str=malformed_final,
            ground_truth=ground_truth,
            allow_recovery=True,
        )
        strict = compute_score(
            data_source="time_series",
            solution_str=malformed_final,
            ground_truth=ground_truth,
            allow_recovery=False,
        )

        self.assertGreater(float(recovered["score"]), -1.0)
        self.assertGreater(float(strict["score"]), -1.0)
        self.assertTrue(strict["turn3_horizon_clamped"])
        self.assertEqual(strict["turn3_horizon_clamp_discarded_lines"], 0)
        self.assertEqual(flow._compute_final_reward(malformed_final, ground_truth), float(strict["score"]))

    def test_agent_flow_worker_postprocess_preserves_explicit_group_uid(self) -> None:
        worker = AgentFlowWorkerBase.__new__(AgentFlowWorkerBase)

        def make_step(token_id: int, *, group_uid: str) -> _InternalAgentFlowStep:
            return _InternalAgentFlowStep(
                prompt_ids=torch.tensor([[11, 12]], dtype=torch.long),
                response_ids=torch.tensor([[token_id, 0]], dtype=torch.long),
                input_ids=torch.tensor([[11, 12, token_id, 0]], dtype=torch.long),
                position_ids=torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
                response_mask=torch.tensor([[1, 0]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
                response_logprobs=None,
                routed_experts=None,
                multi_modal_inputs=None,
                multi_modal_data=None,
                reward_score=None,
                num_turns=2,
                extra_fields={"group_uid": group_uid},
            )

        outputs = [
            AgentFlowOutput(
                steps=[make_step(21, group_uid="uid-a"), make_step(22, group_uid="uid-a")],
                metrics=AgentFlowMetrics(),
            ),
            AgentFlowOutput(
                steps=[make_step(31, group_uid="uid-b")],
                metrics=AgentFlowMetrics(),
            ),
        ]

        result = worker._postprocess(outputs)

        self.assertIn("group_uid", result.non_tensor_batch)
        np.testing.assert_array_equal(
            result.non_tensor_batch["group_uid"],
            np.array(["uid-a", "uid-a", "uid-b"], dtype=object),
        )

    def test_agent_flow_worker_postprocess_keeps_union_of_reward_extra_info_keys(self) -> None:
        worker = AgentFlowWorkerBase.__new__(AgentFlowWorkerBase)

        def make_step(token_id: int, reward_extra_info: dict[str, object]) -> _InternalAgentFlowStep:
            return _InternalAgentFlowStep(
                prompt_ids=torch.tensor([[11, 12]], dtype=torch.long),
                response_ids=torch.tensor([[token_id, 0]], dtype=torch.long),
                input_ids=torch.tensor([[11, 12, token_id, 0]], dtype=torch.long),
                position_ids=torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
                response_mask=torch.tensor([[1, 0]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
                response_logprobs=None,
                routed_experts=None,
                multi_modal_inputs=None,
                multi_modal_data=None,
                reward_score=None,
                num_turns=2,
                extra_fields={"reward_extra_info": reward_extra_info},
            )

        outputs = [
            AgentFlowOutput(
                steps=[
                    make_step(21, {"a": 1, "shared": "x"}),
                    make_step(22, {"shared": "y", "b": 2}),
                ],
                metrics=AgentFlowMetrics(),
            )
        ]

        result = worker._postprocess(outputs)

        self.assertIn("a", result.non_tensor_batch)
        self.assertIn("b", result.non_tensor_batch)
        self.assertIn("shared", result.non_tensor_batch)
        self.assertEqual(result.non_tensor_batch["a"].tolist(), [1, None])
        self.assertEqual(result.non_tensor_batch["b"].tolist(), [None, 2])
        self.assertEqual(result.non_tensor_batch["shared"].tolist(), ["x", "y"])


if __name__ == "__main__":
    unittest.main()
