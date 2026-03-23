import unittest

from recipe.time_series_forecast.validate_turn3_format import (
    check_answer_format,
    check_paper_turn3_protocol,
    check_record_format,
    record_requires_paper_turn3_protocol,
)


class ValidateTurn3FormatTests(unittest.TestCase):
    def _numeric_lines(self, count: int) -> str:
        return "\n".join(f"{1.0 + idx / 10:.4f}" for idx in range(count))

    def test_accepts_answer_only_protocol(self) -> None:
        ok, reason, pred_len = check_answer_format(
            f"<answer>\n{self._numeric_lines(3)}\n</answer>",
            expected_len=3,
            allow_recovery=True,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        self.assertEqual(pred_len, 3)

    def test_accepts_recovered_missing_answer_close_tag(self) -> None:
        ok, reason, pred_len = check_answer_format(
            f"<think>x</think><answer>\n{self._numeric_lines(3)}\n",
            expected_len=3,
            allow_recovery=True,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        self.assertEqual(pred_len, 3)

    def test_default_check_answer_format_is_strict(self) -> None:
        ok, reason, pred_len = check_answer_format(f"<answer>\n{self._numeric_lines(3)}\n</answer>", expected_len=3)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_think_block")
        self.assertEqual(pred_len, 3)

    def test_rejects_tool_call_payloads(self) -> None:
        ok, reason, pred_len = check_answer_format(
            '<tool_call>\n{"name":"predict_time_series","arguments":{"model_name":"chronos2"}}\n</tool_call>',
            expected_len=3,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_answer_block")
        self.assertEqual(pred_len, 0)

    def test_paper_protocol_accepts_think_answer(self) -> None:
        ok, reason, pred_len = check_paper_turn3_protocol(
            f"<think>x</think><answer>\n{self._numeric_lines(3)}\n</answer>",
            expected_len=3,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        self.assertEqual(pred_len, 3)

    def test_paper_protocol_rejects_answer_only(self) -> None:
        ok, reason, pred_len = check_paper_turn3_protocol(
            f"<answer>\n{self._numeric_lines(3)}\n</answer>",
            expected_len=3,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_think_block")
        self.assertEqual(pred_len, 3)

    def test_record_format_uses_strict_protocol_for_refinement_records(self) -> None:
        ok, source, reason, pred_len, _content = check_record_format(
            {
                "turn_stage": "refinement",
                "paper_turn3_required": True,
                "messages": [
                    {"role": "assistant", "content": f"<think>x</think><answer>\n{self._numeric_lines(3)}\n"}
                ],
            },
            expected_len=3,
        )
        self.assertFalse(ok)
        self.assertEqual(source, "messages.last_assistant")
        self.assertEqual(reason, "missing_answer_close_tag")
        self.assertEqual(pred_len, 3)

    def test_record_format_keeps_lenient_answer_only_check_for_non_refinement_records(self) -> None:
        ok, source, reason, pred_len, _content = check_record_format(
            {
                "turn_stage": "diagnostic",
                "paper_turn3_required": False,
                "messages": [{"role": "assistant", "content": f"<answer>\n{self._numeric_lines(3)}\n</answer>"}],
            },
            expected_len=3,
        )
        self.assertTrue(ok)
        self.assertEqual(source, "messages.last_assistant")
        self.assertEqual(reason, "ok")
        self.assertEqual(pred_len, 3)

    def test_non_refinement_record_is_skipped(self) -> None:
        self.assertFalse(
            record_requires_paper_turn3_protocol(
                {
                    "turn_stage": "diagnostic",
                    "paper_turn3_required": False,
                }
            )
        )

    def test_refinement_record_requires_paper_protocol(self) -> None:
        self.assertTrue(
            record_requires_paper_turn3_protocol(
                {
                    "turn_stage": "refinement",
                    "paper_turn3_required": True,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
