from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from recipe.time_series_forecast.dataset_file_utils import load_jsonl_records, write_jsonl_records
from recipe.time_series_forecast.reward import extract_values_from_time_series_string, parse_final_answer_protocol


def get_last_assistant_content(record: dict[str, Any]) -> str:
    messages = record.get("messages", [])
    if isinstance(messages, tuple):
        messages = list(messages)
    elif hasattr(messages, "tolist") and not isinstance(messages, list):
        try:
            messages = messages.tolist()
        except Exception:
            messages = []
    if not isinstance(messages, list):
        messages = []
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            content = msg["content"].strip()
            if content:
                return content
    return ""


def record_requires_paper_turn3_protocol(record: dict[str, Any]) -> bool:
    explicit = record.get("paper_turn3_required")
    if explicit is not None:
        if isinstance(explicit, bool):
            return explicit
        if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
            return bool(int(explicit))
        text = str(explicit).strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False

    turn_stage = str(record.get("turn_stage", "") or "").strip().lower()
    if turn_stage:
        return turn_stage == "refinement"
    return True


def extract_candidate_answer_text(record: dict[str, Any]) -> tuple[str, str]:
    assistant_content = get_last_assistant_content(record)
    if assistant_content:
        return assistant_content, "messages.last_assistant"

    teacher_prediction = str(record.get("teacher_prediction_text", "") or "").strip()
    if teacher_prediction:
        return f"<answer>\n{teacher_prediction}\n</answer>", "teacher_prediction_text"

    return "", "none"


def check_answer_format(
    text: str,
    expected_len: int = 96,
    *,
    allow_recovery: bool = False,
) -> tuple[bool, str, int]:
    if not text:
        return False, "empty_assistant_content", 0
    answer_text, _, reject_reason = parse_final_answer_protocol(text, expected_len, allow_recovery=allow_recovery)
    if answer_text is None:
        values = extract_values_from_time_series_string(text)
        return False, reject_reason or "unknown_format_failure", len(values)

    values = extract_values_from_time_series_string(answer_text)
    pred_len = len(values)
    if pred_len != expected_len:
        return False, f"length_mismatch:{pred_len}!={expected_len}", pred_len

    return True, "ok", pred_len


def check_paper_turn3_protocol(text: str, expected_len: int = 96) -> tuple[bool, str, int]:
    """Validate the strict paper-aligned Turn-3 protocol.

    The paper protocol requires exactly one `<think>...</think>` block followed by
    exactly one `<answer>...</answer>` block, with the answer containing the
    expected number of numeric forecast lines.
    """
    if not text:
        return False, "empty_assistant_content", 0

    answer_text, _, reject_reason = parse_final_answer_protocol(text, expected_len, allow_recovery=False)
    if answer_text is None:
        values = extract_values_from_time_series_string(text)
        return False, reject_reason or "unknown_format_failure", len(values)

    values = extract_values_from_time_series_string(answer_text)
    pred_len = len(values)
    if pred_len != expected_len:
        return False, f"length_mismatch:{pred_len}!={expected_len}", pred_len

    return True, "ok", pred_len


def check_record_format(record: dict[str, Any], expected_len: int = 96) -> tuple[bool, str, str, int, str]:
    content, source = extract_candidate_answer_text(record)
    if record_requires_paper_turn3_protocol(record):
        ok, reason, pred_len = check_paper_turn3_protocol(content, expected_len=expected_len)
    else:
        ok, reason, pred_len = check_answer_format(content, expected_len=expected_len, allow_recovery=True)
    return ok, source, reason, pred_len, content


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Turn-3 final answer format in SFT JSONL records")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL path")
    parser.add_argument("--expected-len", type=int, default=96, help="Expected number of forecast values")
    parser.add_argument("--top-k-invalid", type=int, default=10, help="Print top-k invalid examples")
    parser.add_argument("--write-clean-jsonl", default="", help="Optional output path for valid-only records")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_jsonl_records(input_path)
    invalid: list[tuple[int, str, int, str, str]] = []
    valid_records: list[dict[str, Any]] = []
    source_count: dict[str, int] = {}
    checked_count = 0
    skipped_count = 0

    for idx, rec in enumerate(records):
        if not record_requires_paper_turn3_protocol(rec):
            skipped_count += 1
            valid_records.append(rec)
            continue
        checked_count += 1
        ok, source, reason, pred_len, content = check_record_format(rec, expected_len=args.expected_len)
        source_count[source] = source_count.get(source, 0) + 1
        if ok:
            valid_records.append(rec)
        else:
            invalid.append((idx, source, reason, pred_len, content[:500]))

    total = len(records)
    valid = checked_count - len(invalid)
    invalid_n = len(invalid)
    valid_ratio = (valid / checked_count * 100.0) if checked_count > 0 else 0.0

    print(f"input={input_path}")
    print(
        f"total={total} checked={checked_count} skipped={skipped_count} "
        f"valid={valid} invalid={invalid_n} valid_ratio={valid_ratio:.2f}%"
    )
    print(f"source_count={source_count}")

    if invalid:
        print("\nTop invalid samples:")
        for n, (idx, source, reason, pred_len, head) in enumerate(invalid[: args.top_k_invalid], start=1):
            print(f"\n--- invalid {n}/{min(args.top_k_invalid, invalid_n)} ---")
            print(f"index={idx} source={source} reason={reason} pred_len={pred_len}")
            print(f"assistant_head={head}")

    if args.write_clean_jsonl:
        out_path = Path(args.write_clean_jsonl)
        write_jsonl_records(out_path, valid_records)
        print(f"\nWrote valid-only records: {out_path} ({len(valid_records)} samples)")


if __name__ == "__main__":
    main()
