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


@lru_cache(maxsize=1)
def load_time_series_chat_template() -> str:
    template_path = Path(__file__).with_name("time_series_chat_template.jinja")
    return template_path.read_text(encoding="utf-8")


def extract_tool_calls(
    response_text: str,
    *,
    allowed_tool_names: Sequence[str] | None = None,
    max_calls: Optional[int] = None,
) -> tuple[str, list[TimeSeriesToolCall]]:
    allowed_names = {str(name).strip() for name in (allowed_tool_names or []) if str(name).strip()}
    collected: list[TimeSeriesToolCall] = []
    call_limit = None if max_calls is None else max(0, int(max_calls))

    for match in _TOOL_CALL_BLOCK_RE.finditer(str(response_text or "")):
        payload_text = match.group(1).strip()
        if not payload_text:
            continue

        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        tool_name = str(payload.get("name") or "").strip()
        if not tool_name:
            continue
        if allowed_names and tool_name not in allowed_names:
            continue

        arguments = payload.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            continue

        collected.append(TimeSeriesToolCall(name=tool_name, arguments=dict(arguments)))
        if call_limit is not None and len(collected) >= call_limit:
            break

    assistant_content = _TOOL_CALL_BLOCK_RE.sub("", str(response_text or "")).strip()
    return assistant_content, collected
