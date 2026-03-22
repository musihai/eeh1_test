from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recipe.time_series_forecast.reward import extract_values_from_time_series_string, parse_final_answer_protocol


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def get_last_assistant_content(record: dict[str, Any]) -> str:
    messages = record.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            content = msg["content"].strip()
            if content:
                return content
    return ""


def extract_candidate_answer_text(record: dict[str, Any]) -> tuple[str, str]:
    # 1) SFT-style messages
    assistant_content = get_last_assistant_content(record)
    if assistant_content:
        return assistant_content, "messages.last_assistant"

    # 2) Teacher curated jsonl often has prediction text without XML tags
    teacher_prediction = str(record.get("teacher_prediction_text", "") or "").strip()
    if teacher_prediction:
        # Wrap to reuse the same parser/checker path.
        return f"<answer>\n{teacher_prediction}\n</answer>", "teacher_prediction_text"

    # 3) Other fallback fields
    for key in ("response", "answer", "target", "final_answer"):
        value = str(record.get(key, "") or "").strip()
        if value:
            return value, key

    return "", "none"


def check_answer_format(text: str, expected_len: int = 96) -> tuple[bool, str, int]:
    if not text:
        return False, "empty_assistant_content", 0
    answer_text, _, reject_reason = parse_final_answer_protocol(text, expected_len, allow_recovery=True)
    if answer_text is None:
        values = extract_values_from_time_series_string(text)
        return False, reject_reason or "unknown_format_failure", len(values)

    values = extract_values_from_time_series_string(answer_text)
    pred_len = len(values)
    if pred_len != expected_len:
        return False, f"length_mismatch:{pred_len}!={expected_len}", pred_len

    return True, "ok", pred_len


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

    records = load_jsonl(input_path)
    invalid: list[tuple[int, str, int, str, str]] = []
    valid_records: list[dict[str, Any]] = []
    source_count: dict[str, int] = {}

    for idx, rec in enumerate(records):
        content, source = extract_candidate_answer_text(rec)
        source_count[source] = source_count.get(source, 0) + 1
        ok, reason, pred_len = check_answer_format(content, expected_len=args.expected_len)
        if ok:
            valid_records.append(rec)
        else:
            invalid.append((idx, source, reason, pred_len, content[:500]))

    total = len(records)
    valid = len(valid_records)
    invalid_n = len(invalid)
    valid_ratio = (valid / total * 100.0) if total > 0 else 0.0

    print(f"input={input_path}")
    print(f"total={total} valid={valid} invalid={invalid_n} valid_ratio={valid_ratio:.2f}%")
    print(f"source_count={source_count}")

    if invalid:
        print("\nTop invalid samples:")
        for n, (idx, source, reason, pred_len, head) in enumerate(invalid[: args.top_k_invalid], start=1):
            print(f"\n--- invalid {n}/{min(args.top_k_invalid, invalid_n)} ---")
            print(f"index={idx} source={source} reason={reason} pred_len={pred_len}")
            print(f"assistant_head={head}")

    if args.write_clean_jsonl:
        out_path = Path(args.write_clean_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec in valid_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nWrote valid-only records: {out_path} ({len(valid_records)} samples)")


if __name__ == "__main__":
    main()
