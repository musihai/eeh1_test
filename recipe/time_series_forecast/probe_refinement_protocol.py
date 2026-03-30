from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.time_series_forecast.build_etth1_sft_dataset import build_sft_records
from recipe.time_series_forecast.dataset_file_utils import load_jsonl_records
from recipe.time_series_forecast.reward_protocol import (
    clamp_turn3_answer_horizon,
    normalized_nonempty_lines,
    parse_final_answer_protocol,
)


@dataclass(frozen=True)
class ProbeExample:
    gate_name: str
    source_sample_index: int
    prompt_messages: list[dict[str, Any]]
    forecast_horizon: int
    prompt_hash: str


def _stable_prompt_hash(messages: Iterable[dict[str, Any]]) -> str:
    payload = []
    for message in messages:
        payload.append(
            {
                "role": str(message.get("role") or ""),
                "content": str(message.get("content") or ""),
            }
        )
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()


def _iter_refinement_rows(stepwise_parquet: Path) -> list[dict[str, Any]]:
    dataframe = pd.read_parquet(stepwise_parquet)
    refinement_rows = dataframe.loc[dataframe["turn_stage"].astype(str).str.lower() == "refinement"].copy()
    if refinement_rows.empty:
        raise ValueError(f"No refinement rows found in {stepwise_parquet}")
    return refinement_rows.to_dict("records")


def _gate_a_examples(stepwise_parquet: Path) -> list[ProbeExample]:
    examples: list[ProbeExample] = []
    for row in _iter_refinement_rows(stepwise_parquet):
        messages = row.get("messages")
        if not hasattr(messages, "__len__") or len(messages) < 2:
            continue
        prompt_messages = [
            {"role": str(message.get("role") or ""), "content": str(message.get("content") or "")}
            for message in list(messages[:-1])
        ]
        examples.append(
            ProbeExample(
                gate_name="gate_a_stored_prompt",
                source_sample_index=int(row.get("source_sample_index", -1)),
                prompt_messages=prompt_messages,
                forecast_horizon=int(row.get("forecast_horizon") or 96),
                prompt_hash=_stable_prompt_hash(prompt_messages),
            )
        )
    return examples


def _load_jsonl_record_map(source_jsonl: Path) -> dict[int, dict[str, Any]]:
    record_map: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(load_jsonl_records(source_jsonl)):
        payload = dict(record)
        record_key = payload.get("index")
        if record_key is None:
            record_key = index
        record_map[int(record_key)] = payload
    return record_map


def _take_evenly_spaced(items: list[Any], max_items: int) -> list[Any]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    if max_items == 1:
        return [items[0]]
    positions = {
        min(len(items) - 1, max(0, round(index * (len(items) - 1) / (max_items - 1))))
        for index in range(max_items)
    }
    return [items[index] for index in sorted(positions)]


def _gate_b_examples(stepwise_parquet: Path, source_jsonl: Path) -> list[ProbeExample]:
    record_map = _load_jsonl_record_map(source_jsonl)
    seen_source_indexes: set[int] = set()
    examples: list[ProbeExample] = []
    for row in _iter_refinement_rows(stepwise_parquet):
        source_sample_index = int(row.get("source_sample_index", -1))
        if source_sample_index < 0 or source_sample_index in seen_source_indexes:
            continue
        source_record = record_map.get(source_sample_index)
        if source_record is None:
            continue
        records = build_sft_records(source_record)
        refinement_record = next(
            (record for record in records if str(record.get("turn_stage") or "").strip().lower() == "refinement"),
            None,
        )
        if refinement_record is None:
            continue
        messages = refinement_record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        prompt_messages = [
            {"role": str(message.get("role") or ""), "content": str(message.get("content") or "")}
            for message in messages[:-1]
            if isinstance(message, dict)
        ]
        examples.append(
            ProbeExample(
                gate_name="gate_b_rebuilt_prompt",
                source_sample_index=source_sample_index,
                prompt_messages=prompt_messages,
                forecast_horizon=int(refinement_record.get("forecast_horizon") or 96),
                prompt_hash=_stable_prompt_hash(prompt_messages),
            )
        )
        seen_source_indexes.add(source_sample_index)
    return examples


def _parse_generated_answer(text: str, forecast_horizon: int) -> dict[str, Any]:
    parsed_answer, parse_mode, reject_reason = parse_final_answer_protocol(
        text,
        forecast_horizon,
        allow_recovery=False,
    )
    line_count = len(normalized_nonempty_lines(parsed_answer or ""))
    has_close = "</answer>" in text
    return {
        "strict_ok": parsed_answer is not None,
        "parse_mode": parse_mode,
        "reject_reason": reject_reason,
        "answer_line_count": int(line_count),
        "has_close_tag": bool(has_close),
        "exact_96_lines": bool(line_count == forecast_horizon),
    }


def _generate(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt_messages: list[dict[str, Any]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": bool(do_sample),
    }
    if do_sample:
        generate_kwargs.update({"temperature": float(temperature), "top_p": 1.0})
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=False)


def _summarize_gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    if count <= 0:
        return {"count": 0}
    strict_ok = sum(1 for item in results if item["strict_ok"])
    close_ok = sum(1 for item in results if item["has_close_tag"])
    line_ok = sum(1 for item in results if item["exact_96_lines"])
    parse_mode_counts = Counter(item["parse_mode"] for item in results)
    reject_reason_counts = Counter(item["reject_reason"] for item in results if item["reject_reason"])
    answer_line_counts = Counter(int(item["answer_line_count"]) for item in results)
    return {
        "count": count,
        "strict_protocol_rate": strict_ok / count,
        "close_tag_rate": close_ok / count,
        "exact_line_rate": line_ok / count,
        "parse_mode_counts": dict(parse_mode_counts),
        "reject_reason_counts": dict(reject_reason_counts),
        "answer_line_count_distribution": dict(answer_line_counts),
    }


def _gate_passed(summary: dict[str, Any], *, strict_threshold: float, sample_threshold: float, mode: str) -> bool:
    threshold = strict_threshold if mode == "greedy" else sample_threshold
    if summary.get("count", 0) <= 0:
        return False
    return (
        float(summary.get("strict_protocol_rate", 0.0)) >= threshold
        and float(summary.get("close_tag_rate", 0.0)) >= threshold
        and float(summary.get("exact_line_rate", 0.0)) >= threshold
    )


def _can_still_pass(
    *,
    seen_count: int,
    total_count: int,
    strict_success_count: int,
    close_success_count: int,
    line_success_count: int,
    threshold: float,
) -> bool:
    remaining = max(total_count - seen_count, 0)
    max_strict_rate = (strict_success_count + remaining) / max(total_count, 1)
    max_close_rate = (close_success_count + remaining) / max(total_count, 1)
    max_line_rate = (line_success_count + remaining) / max(total_count, 1)
    return (
        max_strict_rate >= threshold
        and max_close_rate >= threshold
        and max_line_rate >= threshold
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Turn-3 refinement protocol stability for an SFT checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--stepwise-parquet", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit per-gate sample count; -1 uses all.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["greedy", "sample_t1"],
        choices=["greedy", "sample_t1"],
    )
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=3200)
    parser.add_argument("--strict-threshold", type=float, default=1.0)
    parser.add_argument("--sample-threshold", type=float, default=0.90)
    parser.add_argument("--fail-fast", action="store_true", help="Stop a gate/mode early once it can no longer pass.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gate_examples = {
        "gate_a_stored_prompt": _gate_a_examples(args.stepwise_parquet),
        "gate_b_rebuilt_prompt": _gate_b_examples(args.stepwise_parquet, args.source_jsonl),
    }
    if args.max_samples > 0:
        for gate_name, examples in gate_examples.items():
            gate_examples[gate_name] = _take_evenly_spaced(examples, int(args.max_samples))

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    summary_payload: dict[str, Any] = {
        "model_path": str(args.model_path),
        "stepwise_parquet": str(args.stepwise_parquet),
        "source_jsonl": str(args.source_jsonl),
        "modes": list(args.modes),
        "max_samples": int(args.max_samples),
        "strict_threshold": float(args.strict_threshold),
        "sample_threshold": float(args.sample_threshold),
        "gates": {},
    }
    per_sample_path = args.output_dir / "probe_refinement_protocol_samples.jsonl"
    per_sample_path.unlink(missing_ok=True)

    for gate_name, examples in gate_examples.items():
        gate_output: dict[str, Any] = {
            "example_count": len(examples),
            "prompt_hash_count": len({example.prompt_hash for example in examples}),
            "modes": {},
        }
        for mode in args.modes:
            mode_results: list[dict[str, Any]] = []
            do_sample = mode == "sample_t1"
            temperature = float(args.sample_temperature) if do_sample else 0.0
            threshold = float(args.strict_threshold) if mode == "greedy" else float(args.sample_threshold)
            strict_success_count = 0
            close_success_count = 0
            line_success_count = 0
            for sample_idx, example in enumerate(examples, start=1):
                raw_generated_text = _generate(
                    tokenizer=tokenizer,
                    model=model,
                    prompt_messages=example.prompt_messages,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=do_sample,
                    temperature=temperature,
                )
                raw_parsed = _parse_generated_answer(raw_generated_text, example.forecast_horizon)
                generated_text, clamp_info = clamp_turn3_answer_horizon(
                    raw_generated_text,
                    example.forecast_horizon,
                )
                parsed = _parse_generated_answer(str(generated_text or ""), example.forecast_horizon)
                result = {
                    "gate_name": gate_name,
                    "mode": mode,
                    "source_sample_index": int(example.source_sample_index),
                    "forecast_horizon": int(example.forecast_horizon),
                    "prompt_hash": example.prompt_hash,
                    "raw_generated_char_len": len(raw_generated_text),
                    "raw_generated_tail": raw_generated_text[-160:],
                    "raw_strict_ok": bool(raw_parsed["strict_ok"]),
                    "raw_parse_mode": raw_parsed["parse_mode"],
                    "raw_reject_reason": raw_parsed["reject_reason"],
                    "raw_answer_line_count": int(raw_parsed["answer_line_count"]),
                    "raw_has_close_tag": bool(raw_parsed["has_close_tag"]),
                    "raw_exact_96_lines": bool(raw_parsed["exact_96_lines"]),
                    "turn3_horizon_clamped": bool(clamp_info.get("applied", False)),
                    "turn3_horizon_clamp_reason": str(clamp_info.get("reason") or ""),
                    "turn3_horizon_clamp_discarded_lines": int(clamp_info.get("discarded_line_count") or 0),
                    "turn3_horizon_clamp_valid_prefix_lines": int(clamp_info.get("valid_prefix_line_count") or 0),
                    "turn3_horizon_clamp_raw_answer_lines": int(clamp_info.get("raw_answer_line_count") or 0),
                    "generated_char_len": len(str(generated_text or "")),
                    "generated_tail": str(generated_text or "")[-160:],
                    **parsed,
                }
                mode_results.append(result)
                strict_success_count += int(result["strict_ok"])
                close_success_count += int(result["has_close_tag"])
                line_success_count += int(result["exact_96_lines"])
                with per_sample_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                if sample_idx == 1 or sample_idx == len(examples) or sample_idx % 4 == 0:
                    print(
                        f"[{gate_name}][{mode}] {sample_idx}/{len(examples)} "
                        f"strict_ok={int(result['strict_ok'])} lines={result['answer_line_count']} "
                        f"close={int(result['has_close_tag'])}",
                        flush=True,
                    )
                if args.fail_fast and not _can_still_pass(
                    seen_count=sample_idx,
                    total_count=len(examples),
                    strict_success_count=strict_success_count,
                    close_success_count=close_success_count,
                    line_success_count=line_success_count,
                    threshold=threshold,
                ):
                    print(
                        f"[{gate_name}][{mode}] fail-fast at {sample_idx}/{len(examples)}",
                        flush=True,
                    )
                    break
            mode_summary = _summarize_gate(mode_results)
            mode_summary["passed"] = _gate_passed(
                mode_summary,
                strict_threshold=float(args.strict_threshold),
                sample_threshold=float(args.sample_threshold),
                mode=mode,
            )
            gate_output["modes"][mode] = mode_summary
        summary_payload["gates"][gate_name] = gate_output

    summary_path = args.output_dir / "probe_refinement_protocol_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
