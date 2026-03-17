import unittest

from recipe.time_series_forecast.time_series_forecast_agent_flow import TimeSeriesForecastAgentFlow


class FinalAnswerParsingTest(unittest.TestCase):
    def _make_agent(self) -> TimeSeriesForecastAgentFlow:
        agent = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
        agent.forecast_horizon = 96
        agent.prediction_results = "available"
        return agent

    def test_extracts_well_formed_answer_block(self) -> None:
        agent = self._make_agent()
        text = "<think>x</think><answer>\n2026-03-20 20:00:00 31.0027\n2026-03-20 21:00:00 30.7912\n2026-03-20 22:00:00 30.6538\n2026-03-20 23:00:00 30.6209\n2026-03-21 00:00:00 30.5656\n2026-03-21 01:00:00 30.6607\n2026-03-21 02:00:00 30.6778\n2026-03-21 03:00:00 30.7042\n2026-03-21 04:00:00 30.7848\n2026-03-21 05:00:00 30.9305\n2026-03-21 06:00:00 30.8620\n2026-03-21 07:00:00 30.9294\n2026-03-21 08:00:00 30.9196\n2026-03-21 09:00:00 30.9809\n2026-03-21 10:00:00 31.1007\n2026-03-21 11:00:00 31.0463\n2026-03-21 12:00:00 31.2336\n2026-03-21 13:00:00 31.2834\n2026-03-21 14:00:00 31.2961\n2026-03-21 15:00:00 31.3414\n2026-03-21 16:00:00 31.2914\n2026-03-21 17:00:00 31.3630\n2026-03-21 18:00:00 31.3297\n2026-03-21 19:00:00 31.2643\n</answer>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNotNone(answer)
        self.assertEqual(penalty, 0.0)
        self.assertTrue(answer.startswith("2026-03-20 20:00:00 31.0027"))

    def test_rejects_truncated_answer_block(self) -> None:
        agent = self._make_agent()
        text = "<answer>\n2026-03-20 20:00:00 31.0027\n2026-03-20 21:00:00 30.7912\n2026-03-20 22:00:00 30.6538\n2026-03-20 23:00:00 30.6209\n2026-03-21 00:00:00 30.5656\n2026-03-21 01:00:00 30.6607\n2026-03-21 02:00:00 30.6778\n2026-03-21 03:00:00 30.7042\n2026-03-21 04:00:00 30.7848\n2026-03-21 05:00:00 30.9305\n2026-03-21 06:00:00 30.8620\n2026-03-21 07:00:00 30.9294\n2026-03-21 08:00:00 30.9196\n2026-03-21 09:00:00 30.9809\n2026-03-21 10:00:00 31.1007\n2026-03-21 11:00:00 31.0463\n2026-03-21 12:00:00 31.2336\n2026-03-21 13:00:00 31.2834\n2026-03-21 14:00:00 31.2961\n2026-03-21 15:00:00 31.3414\n2026-03-21 16:00:00 31.2914\n2026-03-21 17:00:00 31.3630\n2026-03-21 18:00:00 31.3297\n2026-03-21 19:00:00 31.2643"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)

    def test_rejects_bare_forecast_block(self) -> None:
        agent = self._make_agent()
        text = "I will output the final forecast candidate based on the selected model.\n2026-03-20 20:00:00 31.0027\n2026-03-20 21:00:00 30.7912\n2026-03-20 22:00:00 30.6538\n2026-03-20 23:00:00 30.6209\n2026-03-21 00:00:00 30.5656\n2026-03-21 01:00:00 30.6607\n2026-03-21 02:00:00 30.6778\n2026-03-21 03:00:00 30.7042\n2026-03-21 04:00:00 30.7848\n2026-03-21 05:00:00 30.9305\n2026-03-21 06:00:00 30.8620\n2026-03-21 07:00:00 30.9294\n2026-03-21 08:00:00 30.9196\n2026-03-21 09:00:00 30.9809\n2026-03-21 10:00:00 31.1007\n2026-03-21 11:00:00 31.0463\n2026-03-21 12:00:00 31.2336\n2026-03-21 13:00:00 31.2834\n2026-03-21 14:00:00 31.2961\n2026-03-21 15:00:00 31.3414\n2026-03-21 16:00:00 31.2914\n2026-03-21 17:00:00 31.3630\n2026-03-21 18:00:00 31.3297\n2026-03-21 19:00:00 31.2643\n<|im_end|>"
        answer, penalty = agent._extract_final_answer(text)
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)

    def test_rejects_non_forecast_text(self) -> None:
        agent = self._make_agent()
        answer, penalty = agent._extract_final_answer("I think chronos2 is a good fit for this window.")
        self.assertIsNone(answer)
        self.assertEqual(penalty, 0.0)


if __name__ == "__main__":
    unittest.main()
