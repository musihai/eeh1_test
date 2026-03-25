import asyncio
import unittest
from types import SimpleNamespace

from recipe.time_series_forecast.reward_protocol import extract_values_from_time_series_string
from recipe.time_series_forecast.time_series_forecast_agent_flow import TimeSeriesForecastAgentFlow


class FinalAnswerParsingTest(unittest.TestCase):
    def _make_agent(self) -> TimeSeriesForecastAgentFlow:
        agent = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
        agent.forecast_horizon = 96
        agent.response_length = 3072
        agent.max_parallel_calls = 5
        agent.prediction_results = "available"
        agent.prediction_requested_model = None
        agent.prediction_model_defaulted = False
        agent.prediction_step_index = None
        agent.prediction_turn_stage = None
        agent.prediction_call_count = 1
        agent.illegal_turn3_tool_call_count = 0
        agent.final_answer_reject_reason = None
        agent.final_answer_parse_mode = None
        agent.required_feature_tools = ["extract_basic_statistics"]
        agent.feature_tool_sequence = []
        agent.basic_statistics = {"median": 1.0}
        agent.within_channel_dynamics = None
        agent.forecast_residuals = None
        agent.data_quality = None
        agent.event_summary = None
        agent.timestamps = []
        agent.values = [1.0, 2.0, 3.0]
        return agent

    def _numeric_answer_lines(self, count: int) -> str:
        return "\n".join(f"{30.0 + idx / 1000:.4f}" for idx in range(count))

    def test_extracts_well_formed_answer_block(self) -> None:
        agent = self._make_agent()
        text = f"<think>x</think><answer>\n{self._numeric_answer_lines(96)}\n</answer>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNotNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertIsNone(agent.final_answer_reject_reason)
        self.assertEqual(agent.final_answer_parse_mode, "strict_protocol")
        self.assertTrue(answer.startswith("30.0000"))

    def test_accepts_empty_think_when_answer_is_valid(self) -> None:
        agent = self._make_agent()
        text = f"<think></think><answer>\n{self._numeric_answer_lines(96)}\n</answer>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNotNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertIsNone(agent.final_answer_reject_reason)
        self.assertEqual(agent.final_answer_parse_mode, "strict_protocol")

    def test_rejects_truncated_answer_block(self) -> None:
        agent = self._make_agent()
        text = f"<think>x</think><answer>\n{self._numeric_answer_lines(95)}\n</answer>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "invalid_answer_shape:lines=95,expected=96")

    def test_rejects_timestamped_answer_block(self) -> None:
        agent = self._make_agent()
        timestamped_lines = "\n".join(
            f"2026-03-20 {idx:02d}:00:00 {30.0 + idx / 1000:.4f}" for idx in range(24)
        )
        text = f"<think>x</think><answer>\n{timestamped_lines}\n</answer>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "invalid_answer_shape:lines=24,expected=96")

    def test_rejects_invalid_timestamp_hour_in_answer_block(self) -> None:
        agent = self._make_agent()
        timestamped_lines = []
        for idx in range(96):
            hour = idx % 24
            if idx == 95:
                hour = 24
            timestamped_lines.append(f"2026-03-20 {hour:02d}:00:00 {30.0 + idx / 1000:.4f}")
        text = f"<think>x</think><answer>\n{chr(10).join(timestamped_lines)}\n</answer>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "invalid_answer_shape:values=95,expected=96")
        self.assertEqual(len(extract_values_from_time_series_string(text)), 95)

    def test_rejects_bare_forecast_block(self) -> None:
        agent = self._make_agent()
        agent.prediction_results = None
        text = f"I will output the final forecast candidate.\n{self._numeric_answer_lines(96)}\n<|im_end|>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "missing_answer_block")

    def test_rejects_missing_close_tag_without_runtime_recovery(self) -> None:
        agent = self._make_agent()
        text = f"<think>x</think><answer>\n{self._numeric_answer_lines(96)}"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "missing_answer_close_tag")
        self.assertEqual(agent.final_answer_parse_mode, "rejected_missing_answer_close_tag")

    def test_rejects_overlong_answer_block_instead_of_canonicalizing(self) -> None:
        agent = self._make_agent()
        text = f"<think>x</think><answer>\n{self._numeric_answer_lines(97)}\n</answer>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "invalid_answer_shape:lines=97,expected=96")
        self.assertEqual(agent.final_answer_parse_mode, "rejected_invalid_answer_shape:lines=97,expected=96")

    def test_rejects_answer_only_protocol_during_refinement(self) -> None:
        agent = self._make_agent()
        text = f"<answer>\n{self._numeric_answer_lines(96)}\n</answer>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "missing_think_block")
        self.assertEqual(agent.final_answer_parse_mode, "rejected_missing_think_block")

    def test_rejects_plain_numeric_forecast_block_during_refinement(self) -> None:
        agent = self._make_agent()
        text = f"{self._numeric_answer_lines(96)}\n<|im_end|>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "missing_answer_block")
        self.assertEqual(agent.final_answer_parse_mode, "rejected_missing_answer_block")

    def test_tool_call_response_is_not_treated_as_final_answer_failure(self) -> None:
        agent = self._make_agent()
        text = '<tool_call>\n{"name":"predict_time_series","arguments":{"model_name":"chronos2"}}\n</tool_call>'
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "missing_answer_block")
        self.assertEqual(agent.final_answer_parse_mode, "rejected_missing_answer_block")

    def test_rejects_non_forecast_text(self) -> None:
        agent = self._make_agent()
        answer, penalty = agent._extract_final_answer("I think chronos2 is a good fit for this window.")
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "missing_answer_block")

    def test_accepts_answer_block_before_prediction_but_workflow_rejects_later(self) -> None:
        agent = self._make_agent()
        agent.prediction_results = None
        text = f"<think>x</think><answer>\n{self._numeric_answer_lines(96)}\n</answer>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNotNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertIsNone(agent.final_answer_reject_reason)
        self.assertEqual(agent.final_answer_parse_mode, "strict_protocol")

    def test_rejects_extra_text_outside_tags(self) -> None:
        agent = self._make_agent()
        text = f"<think>x</think><answer>\n{self._numeric_answer_lines(96)}\n</answer>\nextra"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "extra_text_outside_tags")
        self.assertEqual(agent.final_answer_parse_mode, "rejected_extra_text_outside_tags")

    def test_final_turn_sampling_params_add_stop_and_cap_tokens(self) -> None:
        agent = self._make_agent()
        params = agent._prepare_sampling_params({"temperature": 0.3, "top_p": 0.95})
        self.assertEqual(params["stop"], ["</answer>"])
        self.assertTrue(params["include_stop_str_in_output"])
        self.assertEqual(params["max_tokens"], 2944)
        self.assertAlmostEqual(params["temperature"], 0.0, places=6)
        self.assertAlmostEqual(params["top_p"], 1.0, places=6)
        self.assertAlmostEqual(params["repetition_penalty"], 1.05, places=6)

    def test_non_final_turn_sampling_params_add_stop_and_cap_tokens(self) -> None:
        agent = self._make_agent()
        agent.prediction_results = None
        params = agent._prepare_sampling_params({"temperature": 0.3, "top_p": 0.95})
        self.assertEqual(params["stop"], ["<answer>"])
        self.assertNotIn("include_stop_str_in_output", params)
        self.assertEqual(params["max_tokens"], 640)
        self.assertAlmostEqual(params["temperature"], 0.3, places=6)
        self.assertAlmostEqual(params["top_p"], 0.95, places=6)

    def test_validate_workflow_rejects_copying_historical_tail(self) -> None:
        agent = self._make_agent()
        agent.prediction_step_index = 2
        agent.prediction_turn_stage = "routing"
        agent.values = [float(idx) for idx in range(200)]
        copied_tail = "\n".join(f"{float(value):.4f}" for value in agent.values[-96:])
        valid, penalty, message = agent._validate_workflow_completion(copied_tail)
        self.assertFalse(valid)
        self.assertLess(penalty, 0.0)
        self.assertIn("copy input data", message)

    def test_validate_workflow_rejects_tool_calls_after_prediction(self) -> None:
        agent = self._make_agent()
        agent.prediction_step_index = 2
        agent.prediction_turn_stage = "routing"
        agent.illegal_turn3_tool_call_count = 1
        valid, penalty, message = agent._validate_workflow_completion(self._numeric_answer_lines(96))
        self.assertFalse(valid)
        self.assertLess(penalty, 0.0)
        self.assertIn("may not call tools", message)

    def test_validate_workflow_rejects_prediction_called_outside_routing_stage(self) -> None:
        agent = self._make_agent()
        agent.prediction_step_index = 3
        agent.prediction_turn_stage = "diagnostic"
        valid, penalty, message = agent._validate_workflow_completion(self._numeric_answer_lines(96))
        self.assertFalse(valid)
        self.assertLess(penalty, 0.0)
        self.assertIn("routing stage", message)

    def test_validate_workflow_rejects_missing_diagnostic_analysis(self) -> None:
        agent = self._make_agent()
        agent.basic_statistics = None
        agent.prediction_step_index = 2
        agent.prediction_turn_stage = "routing"
        valid, penalty, message = agent._validate_workflow_completion(self._numeric_answer_lines(96))
        self.assertFalse(valid)
        self.assertLess(penalty, 0.0)
        self.assertIn("Complete the required diagnostic feature tools", message)

    def test_execute_tool_call_blocks_prediction_during_diagnostic_turn(self) -> None:
        agent = self._make_agent()
        agent.prediction_results = None
        agent.prediction_call_count = 0

        async def _fake_run_prediction_tool(model_name: str = "chronos2") -> str:
            self.fail("prediction tool should not run during the diagnostic turn")

        agent._run_prediction_tool = _fake_run_prediction_tool
        tool_call = SimpleNamespace(name="predict_time_series", arguments={"model_name": "patchtst"})

        result = asyncio.run(agent._execute_tool_call(tool_call, turn_stage="diagnostic"))
        self.assertIsNone(result)
        self.assertEqual(agent.prediction_requested_model, "patchtst")
        self.assertEqual(agent.prediction_call_count, 0)

    def test_execute_tool_call_blocks_prediction_until_some_diagnostic_tool_completed(self) -> None:
        agent = self._make_agent()
        agent.prediction_results = None
        agent.prediction_call_count = 0
        agent.basic_statistics = None

        async def _fake_run_prediction_tool(model_name: str = "chronos2") -> str:
            self.fail("prediction tool should not run before any diagnostic feature tool is complete")

        agent._run_prediction_tool = _fake_run_prediction_tool
        tool_call = SimpleNamespace(name="predict_time_series", arguments={"model_name": "patchtst"})

        result = asyncio.run(agent._execute_tool_call(tool_call, turn_stage="routing"))
        self.assertIsNone(result)
        self.assertEqual(agent.prediction_requested_model, "patchtst")
        self.assertEqual(agent.prediction_call_count, 0)

    def test_execute_tool_call_blocks_feature_tools_during_routing_turn(self) -> None:
        agent = self._make_agent()
        agent.prediction_results = None
        tool_call = SimpleNamespace(name="extract_basic_statistics", arguments={})

        result = asyncio.run(agent._execute_tool_call(tool_call, turn_stage="routing"))
        self.assertIsNone(result)

    def test_execute_tool_call_blocks_duplicate_feature_tool(self) -> None:
        agent = self._make_agent()
        agent.prediction_results = None
        agent.basic_statistics = {"median": 1.0}
        tool_call = SimpleNamespace(name="extract_basic_statistics", arguments={})

        result = asyncio.run(agent._execute_tool_call(tool_call, turn_stage="diagnostic"))
        self.assertIsNone(result)

    def test_current_turn_stage_waits_until_required_diagnostics_finish(self) -> None:
        agent = self._make_agent()
        agent.prediction_results = None
        agent.basic_statistics = None
        agent.event_summary = None
        agent.required_feature_tools = ["extract_basic_statistics", "extract_event_summary"]
        self.assertEqual(agent._current_turn_stage(), "diagnostic")
        agent.event_summary = {"event_segment_count": 4.0}
        self.assertEqual(agent._current_turn_stage(), "diagnostic")
        agent.basic_statistics = {"median": 1.0}
        self.assertEqual(agent._current_turn_stage(), "routing")


if __name__ == "__main__":
    unittest.main()
