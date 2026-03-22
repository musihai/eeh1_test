import unittest

from recipe.time_series_forecast.validate_turn3_format import check_answer_format


class ValidateTurn3FormatTests(unittest.TestCase):
    def _numeric_lines(self, count: int) -> str:
        return "\n".join(f"{1.0 + idx / 10:.4f}" for idx in range(count))

    def test_accepts_answer_only_protocol(self) -> None:
        ok, reason, pred_len = check_answer_format(f"<answer>\n{self._numeric_lines(3)}\n</answer>", expected_len=3)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        self.assertEqual(pred_len, 3)

    def test_accepts_recovered_missing_answer_close_tag(self) -> None:
        ok, reason, pred_len = check_answer_format(
            f"<think>x</think><answer>\n{self._numeric_lines(3)}\n",
            expected_len=3,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        self.assertEqual(pred_len, 3)

    def test_rejects_tool_call_payloads(self) -> None:
        ok, reason, pred_len = check_answer_format(
            '<tool_call>\n{"name":"predict_time_series","arguments":{"model_name":"chronos2"}}\n</tool_call>',
            expected_len=3,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_answer_block")
        self.assertEqual(pred_len, 0)


if __name__ == "__main__":
    unittest.main()
