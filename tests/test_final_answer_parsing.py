import unittest

from recipe.time_series_forecast.time_series_forecast_agent_flow import TimeSeriesForecastAgentFlow


class FinalAnswerParsingTest(unittest.TestCase):
    def _make_agent(self) -> TimeSeriesForecastAgentFlow:
        agent = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
        agent.forecast_horizon = 96
        agent.response_length = 3072
        agent.prediction_results = "available"
        agent.prediction_call_count = 1
        agent.illegal_turn3_tool_call_count = 0
        agent.final_answer_reject_reason = None
        agent.basic_statistics = {"median": 1.0}
        agent.within_channel_dynamics = None
        agent.forecast_residuals = None
        agent.data_quality = None
        agent.event_summary = None
        agent.timestamps = []
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
        self.assertTrue(answer.startswith("30.0000"))

    def test_rejects_truncated_answer_block(self) -> None:
        agent = self._make_agent()
        text = f"<answer>\n{self._numeric_answer_lines(95)}\n</answer>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "invalid_answer_shape:lines=95,expected=96")

    def test_rejects_timestamped_answer_block(self) -> None:
        agent = self._make_agent()
        timestamped_lines = "\n".join(
            f"2026-03-20 {idx:02d}:00:00 {30.0 + idx / 1000:.4f}" for idx in range(24)
        )
        text = f"<answer>\n{timestamped_lines}\n</answer>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "invalid_answer_shape:lines=24,expected=96")

    def test_rejects_bare_forecast_block(self) -> None:
        agent = self._make_agent()
        text = f"I will output the final forecast candidate.\n{self._numeric_answer_lines(96)}\n<|im_end|>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "missing_answer_block")

    def test_rejects_missing_close_tag(self) -> None:
        agent = self._make_agent()
        text = f"<answer>\n{self._numeric_answer_lines(96)}"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "missing_answer_close_tag")

    def test_rejects_non_forecast_text(self) -> None:
        agent = self._make_agent()
        answer, penalty = agent._extract_final_answer("I think chronos2 is a good fit for this window.")
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(agent.final_answer_reject_reason, "missing_answer_block")

    def test_final_turn_sampling_params_add_stop_and_cap_tokens(self) -> None:
        agent = self._make_agent()
        params = agent._prepare_sampling_params({"temperature": 0.3, "top_p": 0.95})
        self.assertEqual(params["stop"], ["</answer>"])
        self.assertTrue(params["include_stop_str_in_output"])
        self.assertEqual(params["max_tokens"], 1024)

    def test_non_final_turn_sampling_params_are_unchanged(self) -> None:
        agent = self._make_agent()
        agent.prediction_results = None
        params = agent._prepare_sampling_params({"temperature": 0.3, "top_p": 0.95})
        self.assertEqual(params, {"temperature": 0.3, "top_p": 0.95})

    def test_validate_workflow_rejects_tool_calls_after_prediction(self) -> None:
        agent = self._make_agent()
        agent.illegal_turn3_tool_call_count = 1
        valid, penalty, message = agent._validate_workflow_completion(self._numeric_answer_lines(96))
        self.assertFalse(valid)
        self.assertLess(penalty, 0.0)
        self.assertIn("may not call tools", message)


if __name__ == "__main__":
    unittest.main()
