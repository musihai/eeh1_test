from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class TimeSeriesToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallParseDiagnostics:
    raw_tool_call_block_count: int
    tool_call_json_decode_error_count: int
    tool_call_missing_name_count: int
    invalid_tool_call_name_count: int
    invalid_tool_call_arguments_count: int
    raw_tool_call_name_sequence: str
    invalid_tool_call_name_sequence: str


@lru_cache(maxsize=1)
def load_time_series_chat_template() -> str:
    template_path = Path(__file__).with_name("time_series_chat_template.jinja")
    return template_path.read_text(encoding="utf-8")


def extract_tool_calls_with_debug(
    response_text: str,
    *,
    allowed_tool_names: Sequence[str] | None = None,
    max_calls: Optional[int] = None,
) -> tuple[str, list[TimeSeriesToolCall], ToolCallParseDiagnostics]:
    allowed_names = {str(name).strip() for name in (allowed_tool_names or []) if str(name).strip()}
    collected: list[TimeSeriesToolCall] = []
    call_limit = None if max_calls is None else max(0, int(max_calls))
    raw_tool_names: list[str] = []
    invalid_tool_names: list[str] = []
    raw_tool_call_block_count = 0
    tool_call_json_decode_error_count = 0
    tool_call_missing_name_count = 0
    invalid_tool_call_name_count = 0
    invalid_tool_call_arguments_count = 0

    for match in _TOOL_CALL_BLOCK_RE.finditer(str(response_text or "")):
        raw_tool_call_block_count += 1
        payload_text = match.group(1).strip()
        if not payload_text:
            tool_call_json_decode_error_count += 1
            continue

        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            tool_call_json_decode_error_count += 1
            continue

        if not isinstance(payload, dict):
            tool_call_json_decode_error_count += 1
            continue

        tool_name = str(payload.get("name") or "").strip()
        if not tool_name:
            tool_call_missing_name_count += 1
            continue
        raw_tool_names.append(tool_name)

        if allowed_names and tool_name not in allowed_names:
            invalid_tool_call_name_count += 1
            invalid_tool_names.append(tool_name)
            continue

        arguments = payload.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            invalid_tool_call_arguments_count += 1
            invalid_tool_names.append(tool_name)
            continue

        collected.append(TimeSeriesToolCall(name=tool_name, arguments=dict(arguments)))
        if call_limit is not None and len(collected) >= call_limit:
            break

    assistant_content = _TOOL_CALL_BLOCK_RE.sub("", str(response_text or "")).strip()
    diagnostics = ToolCallParseDiagnostics(
        raw_tool_call_block_count=int(raw_tool_call_block_count),
        tool_call_json_decode_error_count=int(tool_call_json_decode_error_count),
        tool_call_missing_name_count=int(tool_call_missing_name_count),
        invalid_tool_call_name_count=int(invalid_tool_call_name_count),
        invalid_tool_call_arguments_count=int(invalid_tool_call_arguments_count),
        raw_tool_call_name_sequence="->".join(raw_tool_names),
        invalid_tool_call_name_sequence="->".join(invalid_tool_names),
    )
    return assistant_content, collected, diagnostics


def extract_tool_calls(
    response_text: str,
    *,
    allowed_tool_names: Sequence[str] | None = None,
    max_calls: Optional[int] = None,
) -> tuple[str, list[TimeSeriesToolCall]]:
    assistant_content, collected, _ = extract_tool_calls_with_debug(
        response_text,
        allowed_tool_names=allowed_tool_names,
        max_calls=max_calls,
    )
    return assistant_content, collected


__all__ = [
    "TimeSeriesToolCall",
    "ToolCallParseDiagnostics",
    "extract_tool_calls",
    "extract_tool_calls_with_debug",
    "load_time_series_chat_template",
]
