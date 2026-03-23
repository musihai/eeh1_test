import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from recipe.time_series_forecast.time_series_forecast_agent_flow import TimeSeriesForecastAgentFlow


class TestTimeSeriesForecastAgentFlow(unittest.IsolatedAsyncioTestCase):
    def _make_flow(self) -> TimeSeriesForecastAgentFlow:
        flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
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
        flow.diagnostic_tool_batches = []
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
        self.assertGreater(flow._final_answer_max_tokens(), 1024)

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
        self.assertNotIn("predict_time_series", diagnostic_names)
        self.assertEqual(routing_names, ["predict_time_series"])
        self.assertEqual(refinement_names, [])

    def test_append_turn_debug_records_sample_uid(self) -> None:
        flow = self._make_flow()
        flow.required_feature_tools = ["extract_basic_statistics"]
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
        self.assertEqual(payload["required_feature_tool_count"], 1)
        self.assertEqual(payload["required_feature_tool_signature"], "extract_basic_statistics")
        self.assertEqual(payload["final_answer_step_index"], 3)


if __name__ == "__main__":
    unittest.main()
