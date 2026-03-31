import unittest

from recipe.time_series_forecast.tool_call_protocol import (
    extract_tool_calls,
    extract_tool_calls_with_debug,
    load_time_series_chat_template,
)


class ToolCallProtocolTests(unittest.TestCase):
    def test_extract_tool_calls_accepts_strict_json_block(self) -> None:
        response_text = (
            "<tool_call>\n"
            '{"name":"extract_basic_statistics","arguments":{}}\n'
            "</tool_call>"
        )

        assistant_content, tool_calls = extract_tool_calls(
            response_text,
            allowed_tool_names=["extract_basic_statistics"],
            max_calls=5,
        )

        self.assertEqual(assistant_content, "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "extract_basic_statistics")
        self.assertEqual(tool_calls[0].arguments, {})

    def test_extract_tool_calls_rejects_placeholder_payload_without_error(self) -> None:
        response_text = (
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

        assistant_content, tool_calls = extract_tool_calls(
            response_text,
            allowed_tool_names=["extract_basic_statistics"],
            max_calls=5,
        )

        self.assertEqual(assistant_content, "")
        self.assertEqual(tool_calls, [])

    def test_custom_chat_template_uses_valid_json_example(self) -> None:
        template = load_time_series_chat_template()

        self.assertIn('{\\"name\\": \\"extract_basic_statistics\\", \\"arguments\\": {}}', template)
        self.assertNotIn('{\\"name\\": \\"tool_name\\", \\"arguments\\": {}}', template)
        self.assertNotIn('{"name": <function-name>, "arguments": <args-json-object>}', template)

    def test_extract_tool_calls_with_debug_reports_invalid_tool_name(self) -> None:
        response_text = (
            "<tool_call>\n"
            '{"name":"tool_name","arguments":{"model_name":"arima"}}\n'
            "</tool_call>"
        )

        assistant_content, tool_calls, diagnostics = extract_tool_calls_with_debug(
            response_text,
            allowed_tool_names=["predict_time_series"],
            max_calls=5,
        )

        self.assertEqual(assistant_content, "")
        self.assertEqual(tool_calls, [])
        self.assertEqual(diagnostics.raw_tool_call_block_count, 1)
        self.assertEqual(diagnostics.invalid_tool_call_name_count, 1)
        self.assertEqual(diagnostics.invalid_tool_call_name_sequence, "tool_name")


if __name__ == "__main__":
    unittest.main()
