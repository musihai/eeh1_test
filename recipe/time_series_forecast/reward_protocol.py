from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from verl.utils.chain_debug import append_chain_debug, short_text


def extract_answer(text: str) -> str:
    """Extract content within <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text


def normalized_nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def extract_forecast_block(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = (
        str(text)
        .replace("<|im_end|>", "\n")
        .replace("<answer>", "\n")
        .replace("</answer>", "\n")
        .strip()
    )
    lines = cleaned.splitlines()
    collected: List[str] = []
    started = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        is_timestamp_value = re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+-?\d+\.?\d*$", line) is not None
        is_numeric_value = re.match(r"^-?\d+\.?\d*$", line) is not None
        if not started:
            if is_timestamp_value or is_numeric_value:
                started = True
                collected.append(line)
            continue

        if is_timestamp_value or is_numeric_value:
            collected.append(line)
            continue
        break

    if not collected:
        return None
    return "\n".join(collected)


def canonicalize_forecast_values(values: List[float], expected_len: int) -> str:
    return "\n".join(f"{float(value):.4f}" for value in values[:expected_len])


def looks_like_forecast_answer(answer_text: Optional[str], expected_len: int) -> bool:
    if not answer_text:
        return False
    lines = normalized_nonempty_lines(answer_text)
    values = extract_values_from_time_series_string(answer_text)
    if len(lines) != expected_len or len(values) != expected_len:
        return False
    for line in lines:
        if re.fullmatch(r"-?\d+(?:\.\d+)?", line) is None:
            return False
    return True


def infer_answer_shape_failure(answer_text: str, expected_len: int) -> str:
    lines = normalized_nonempty_lines(answer_text)
    values = extract_values_from_time_series_string(answer_text)
    if not lines:
        return "empty_answer_block"
    if len(lines) != expected_len:
        return f"invalid_answer_shape:lines={len(lines)},expected={expected_len}"
    if len(values) != expected_len:
        return f"invalid_answer_shape:values={len(values)},expected={expected_len}"
    for line in lines:
        if re.fullmatch(r"-?\d+(?:\.\d+)?", line) is None:
            return "invalid_answer_shape:non_numeric_line"
    return "invalid_answer_shape:unknown"


def is_plain_forecast_block_response(response_text: Optional[str]) -> bool:
    forecast_block = extract_forecast_block(response_text)
    if not forecast_block or response_text is None:
        return False

    cleaned = (
        str(response_text)
        .replace("<|im_end|>", "\n")
        .replace("<think>", "\n")
        .replace("</think>", "\n")
        .replace("<answer>", "\n")
        .replace("</answer>", "\n")
        .strip()
    )
    return normalized_nonempty_lines(cleaned) == normalized_nonempty_lines(forecast_block)


def extract_strict_protocol_answer(solution_str: Optional[str], expected_len: int) -> Tuple[Optional[str], Optional[str]]:
    if solution_str is None:
        return None, "empty_solution"

    if "<think>" in solution_str and "</think>" not in solution_str:
        return None, "missing_think_close_tag"
    if "</think>" in solution_str and "<think>" not in solution_str:
        return None, "missing_think_open_tag"
    if "<answer>" in solution_str and "</answer>" not in solution_str:
        return None, "missing_answer_close_tag"
    if "</answer>" in solution_str and "<answer>" not in solution_str:
        return None, "missing_answer_open_tag"
    if "<answer>" not in solution_str and "</answer>" not in solution_str:
        return None, "missing_answer_block"
    if "<think>" not in solution_str and "</think>" not in solution_str:
        return None, "missing_think_block"

    protocol_match = re.fullmatch(
        r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*",
        solution_str,
        re.DOTALL,
    )
    if not protocol_match:
        return None, "extra_text_outside_tags"

    candidate = protocol_match.group(2).strip()
    if looks_like_forecast_answer(candidate, expected_len):
        return candidate, None
    return None, infer_answer_shape_failure(candidate, expected_len)


def recover_protocol_answer(
    solution_str: Optional[str],
    reject_reason: str,
    expected_len: int,
) -> Tuple[Optional[str], Optional[str]]:
    if not solution_str:
        return None, None
    if "<tool_call>" in solution_str or "</tool_call>" in solution_str:
        return None, None

    candidate_sources: List[Tuple[str, str]] = []
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if answer_match:
        candidate_sources.append(("answer_block", answer_match.group(1)))
    elif "<answer>" in solution_str:
        candidate_sources.append(("answer_block", solution_str.split("<answer>", 1)[1]))

    if is_plain_forecast_block_response(solution_str):
        forecast_block = extract_forecast_block(solution_str)
        if forecast_block:
            candidate_sources.append(("plain_forecast_block", forecast_block))

    seen_sources: set[str] = set()
    for source_name, source_text in candidate_sources:
        normalized_source = source_text.strip()
        if not normalized_source or normalized_source in seen_sources:
            continue
        seen_sources.add(normalized_source)

        values = extract_values_from_time_series_string(normalized_source)
        if len(values) < expected_len:
            continue

        canonical_answer = canonicalize_forecast_values(values, expected_len)
        if not looks_like_forecast_answer(canonical_answer, expected_len):
            continue

        return canonical_answer, f"recovered_{reject_reason}_{source_name}"

    return None, None


def parse_final_answer_protocol(
    solution_str: Optional[str],
    expected_len: int,
    *,
    allow_recovery: bool = False,
) -> Tuple[Optional[str], str, Optional[str]]:
    strict_answer, reject_reason = extract_strict_protocol_answer(solution_str, expected_len)
    if strict_answer is not None:
        return strict_answer, "strict_protocol", None

    reject_reason = reject_reason or "unknown_format_failure"
    if allow_recovery:
        recovered_answer, parse_mode = recover_protocol_answer(solution_str, reject_reason, expected_len)
        if recovered_answer is not None and parse_mode is not None:
            return recovered_answer, parse_mode, reject_reason

    return None, f"rejected_{reject_reason}", reject_reason


def extract_tail_lines(text: Optional[str], max_lines: int = 10) -> List[str]:
    if text is None:
        return []
    lines = str(text).splitlines()
    if max_lines <= 0:
        return []
    return lines[-max_lines:]


def extract_answer_region(text: Optional[str]) -> str:
    if text is None:
        return ""
    raw_text = str(text)
    open_tag = "<answer>"
    close_tag = "</answer>"
    start = raw_text.rfind(open_tag)
    if start < 0:
        return ""
    start += len(open_tag)
    end = raw_text.find(close_tag, start)
    if end < 0:
        return raw_text[start:].strip()
    return raw_text[start:end].strip()


def count_numeric_only_lines(text: str) -> Tuple[int, int]:
    numeric_count = 0
    non_numeric_count = 0
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"-?\d+(?:\.\d+)?", line):
            numeric_count += 1
        else:
            non_numeric_count += 1
    return numeric_count, non_numeric_count


def trailing_text_after_close(text: Optional[str]) -> str:
    if text is None:
        return ""
    raw_text = str(text)
    close_tag = "</answer>"
    if close_tag not in raw_text:
        return ""
    return raw_text.split(close_tag, 1)[1].strip()


def detect_suffix_repetition(
    values: List[float],
    *,
    max_period: int = 8,
    min_repeats: int = 2,
    atol: float = 1e-8,
) -> Tuple[bool, int, int]:
    n = len(values)
    if n < 2:
        return False, 0, 0

    def _segments_equal(a: List[float], b: List[float]) -> bool:
        if len(a) != len(b):
            return False
        return all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))

    best_period = 0
    best_repeats = 0
    upper_period = min(max_period, n // min_repeats)
    for period in range(1, upper_period + 1):
        base = values[-period:]
        repeats = 1
        cursor = n - 2 * period
        while cursor >= 0:
            segment = values[cursor : cursor + period]
            if not _segments_equal(segment, base):
                break
            repeats += 1
            cursor -= period
        if repeats >= min_repeats and repeats * period > best_repeats * max(best_period, 1):
            best_period = period
            best_repeats = repeats
    if best_repeats >= min_repeats:
        return True, best_period, best_repeats
    return False, 0, 0


def extract_values_from_time_series_string(text: str) -> List[float]:
    """Extract numeric values from forecast text in multiple accepted formats."""
    raw_text = text
    text = extract_answer(text)
    values = []

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(-?\d+\.?\d*)", line)
        if match:
            try:
                values.append(float(match.group(1)))
                continue
            except ValueError:
                pass

        match = re.search(r"\|\s*\d+\s*\|\s*(-?\d+\.?\d*)\s*\|", line)
        if match:
            try:
                values.append(float(match.group(1)))
                continue
            except ValueError:
                pass

        match = re.search(r"(-?\d+\.?\d*)$", line)
        if match:
            try:
                values.append(float(match.group(1)))
                continue
            except ValueError:
                pass

    if os.getenv("TS_REWARD_PARSE_DEBUG", "0").lower() in {"1", "true", "yes", "on"}:
        append_chain_debug(
            "reward_parse_extract",
            {
                "has_answer_tag": bool(re.search(r"<answer>(.*?)</answer>", raw_text or "", re.DOTALL)),
                "raw_text_head": short_text(raw_text, 300),
                "parsed_answer_head": short_text(text, 300),
                "num_values": len(values),
                "values_head": values[:10],
            },
        )
    return values


def extract_ground_truth_values(text: str) -> List[float]:
    return extract_values_from_time_series_string(text)


__all__ = [
    "canonicalize_forecast_values",
    "count_numeric_only_lines",
    "detect_suffix_repetition",
    "extract_answer",
    "extract_answer_region",
    "extract_forecast_block",
    "extract_ground_truth_values",
    "extract_strict_protocol_answer",
    "extract_tail_lines",
    "extract_values_from_time_series_string",
    "infer_answer_shape_failure",
    "is_plain_forecast_block_response",
    "looks_like_forecast_answer",
    "normalized_nonempty_lines",
    "parse_final_answer_protocol",
    "recover_protocol_answer",
    "trailing_text_after_close",
]
