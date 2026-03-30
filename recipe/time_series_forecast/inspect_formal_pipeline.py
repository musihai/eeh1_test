from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recipe.time_series_forecast.build_etth1_sft_dataset import build_sft_records
from recipe.time_series_forecast.dataset_file_utils import load_jsonl_records
from recipe.time_series_forecast.diagnostic_policy import build_diagnostic_plan
from recipe.time_series_forecast.prompts import build_runtime_user_prompt
from recipe.time_series_forecast.reward_protocol import normalized_nonempty_lines, parse_final_answer_protocol
from recipe.time_series_forecast.task_protocol import parse_task_prompt, parse_time_series_records


def _load_jsonl_record(path: Path, sample_index: int) -> dict[str, Any]:
    for index, record in enumerate(load_jsonl_records(path)):
        if index == sample_index:
            return dict(record)
    raise IndexError(f"Sample index {sample_index} out of range for {path}")


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _tool_names_from_record(record: dict[str, Any]) -> list[str]:
    tools = record.get("tools")
    if not isinstance(tools, list):
        return []
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
            continue
        if tool.get("name"):
            names.append(str(tool["name"]))
    return names


def _assistant_message(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return {}
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message
    return {}


def _user_message(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return {}
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    return {}


def _summarize_sft_record(record: dict[str, Any]) -> dict[str, Any]:
    turn_stage = str(record.get("turn_stage") or "")
    assistant = _assistant_message(record)
    user = _user_message(record)
    tool_calls = assistant.get("tool_calls") if isinstance(assistant.get("tool_calls"), list) else []
    tool_call_names = [
        str(call.get("function", {}).get("name") or call.get("name") or "")
        for call in tool_calls
        if isinstance(call, dict)
    ]
    issues: list[str] = []

    turn_stage_order = record.get("turn_stage_order")
    if turn_stage_order is None:
        turn_stage_order = -1

    if turn_stage == "diagnostic":
        exposed_tools = _tool_names_from_record(record)
        if not tool_call_names:
            issues.append("diagnostic_missing_tool_call")
        if len(tool_call_names) != len(set(tool_call_names)):
            issues.append("diagnostic_duplicate_tool_call")
        unexpected_tools = [name for name in tool_call_names if name not in exposed_tools]
        if unexpected_tools:
            issues.append(f"diagnostic_unexpected_tool_call:{','.join(unexpected_tools)}")

    elif turn_stage == "routing":
        if tool_call_names != ["predict_time_series"]:
            issues.append(f"routing_tool_calls={tool_call_names}")

    elif turn_stage == "refinement":
        forecast_horizon = int(record.get("forecast_horizon") or 96)
        assistant_content = str(assistant.get("content") or "")
        parsed_answer, parse_mode, reject_reason = parse_final_answer_protocol(
            assistant_content,
            forecast_horizon,
            allow_recovery=False,
        )
        answer_line_count = len(normalized_nonempty_lines(parsed_answer or ""))
        if parsed_answer is None:
            issues.append(f"refinement_protocol_reject:{reject_reason}")
        if answer_line_count != forecast_horizon:
            issues.append(f"refinement_answer_lines={answer_line_count}")
        return {
            "turn_stage": turn_stage,
            "turn_stage_order": int(turn_stage_order),
            "user_prompt_head": str(user.get("content") or "")[:400],
            "assistant_content_head": assistant_content[:400],
            "tool_call_names": tool_call_names,
            "parse_mode": parse_mode,
            "reject_reason": reject_reason,
            "answer_line_count": answer_line_count,
            "issues": issues,
        }

    return {
        "turn_stage": turn_stage,
        "turn_stage_order": int(turn_stage_order),
        "user_prompt_head": str(user.get("content") or "")[:400],
        "assistant_reasoning_head": str(assistant.get("reasoning_content") or "")[:400],
        "tool_call_names": tool_call_names,
        "issues": issues,
    }


def _render_sft_record(record: dict[str, Any]) -> str:
    assistant = _assistant_message(record)
    user = _user_message(record)
    parts = [
        f"# SFT Turn: {record.get('turn_stage')} (order={record.get('turn_stage_order')})",
        "",
        "## Summary",
        _safe_json(_summarize_sft_record(record)),
        "",
        "## User Prompt",
        str(user.get("content") or ""),
    ]
    if assistant.get("reasoning_content"):
        parts.extend(["", "## Assistant Reasoning", str(assistant.get("reasoning_content") or "")])
    if assistant.get("tool_calls"):
        parts.extend(["", "## Assistant Tool Calls", _safe_json(assistant.get("tool_calls"))])
    if assistant.get("content"):
        parts.extend(["", "## Assistant Content", str(assistant.get("content") or "")])
    return "\n".join(parts).strip() + "\n"


def inspect_sft_sample(sample: dict[str, Any]) -> dict[str, Any]:
    records = build_sft_records(sample)
    turn_stages = [str(record.get("turn_stage") or "") for record in records]
    issues: list[str] = []
    if not turn_stages:
        issues.append("no_sft_records")
    if "routing" not in turn_stages:
        issues.append("missing_routing_turn")
    if "refinement" not in turn_stages:
        issues.append("missing_refinement_turn")
    summaries = [_summarize_sft_record(record) for record in records]
    for summary in summaries:
        issues.extend(str(item) for item in summary.get("issues", []))
    return {
        "source_uid": str(sample.get("uid") or ""),
        "source_index": int(sample.get("index", -1)),
        "turn_count": len(records),
        "turn_stages": turn_stages,
        "selected_prediction_model": str(records[-1].get("selected_prediction_model") or "") if records else "",
        "turn3_target_type": str(records[-1].get("turn3_target_type") or "") if records else "",
        "issues": issues,
        "records": records,
        "record_summaries": summaries,
    }


def inspect_rl_sample(sample: dict[str, Any]) -> dict[str, Any]:
    raw_prompt = str(sample["raw_prompt"][0]["content"])
    task_spec = parse_task_prompt(raw_prompt, data_source=str(sample.get("data_source") or "ETTh1"))
    timestamps, values = parse_time_series_records(task_spec.historical_data, target_column=task_spec.target_column)
    ground_truth = str(sample.get("reward_model", {}).get("ground_truth") or "")
    gt_timestamps, gt_values = parse_time_series_records(ground_truth, target_column=task_spec.target_column)
    diagnostic_plan = build_diagnostic_plan(values)
    initial_diagnostic_prompt = build_runtime_user_prompt(
        data_source=str(sample.get("data_source") or "ETTh1"),
        target_column=str(task_spec.target_column or "OT"),
        lookback_window=int(task_spec.lookback_window or len(values)),
        forecast_horizon=int(task_spec.forecast_horizon or len(gt_values)),
        time_series_data=task_spec.historical_data,
        history_analysis=[],
        prediction_results=None,
        available_feature_tools=list(diagnostic_plan.tool_names),
        completed_feature_tools=[],
        turn_stage="diagnostic",
    )
    issues: list[str] = []
    if len(values) != int(task_spec.lookback_window or len(values)):
        issues.append(f"rl_history_len={len(values)}")
    if len(gt_values) != int(task_spec.forecast_horizon or len(gt_values)):
        issues.append(f"rl_ground_truth_len={len(gt_values)}")
    return {
        "uid": str(sample.get("uid") or ""),
        "index": int(sample.get("index", -1)),
        "curriculum_stage": str(sample.get("curriculum_stage") or ""),
        "difficulty_stage": str(sample.get("difficulty_stage") or ""),
        "lookback_window": int(task_spec.lookback_window or len(values)),
        "forecast_horizon": int(task_spec.forecast_horizon or len(gt_values)),
        "history_rows": len(values),
        "ground_truth_rows": len(gt_values),
        "diagnostic_tool_plan": list(diagnostic_plan.tool_names),
        "issues": issues,
        "raw_prompt": raw_prompt,
        "initial_diagnostic_prompt": initial_diagnostic_prompt,
        "ground_truth": ground_truth,
        "history_first_timestamp": timestamps[0] if timestamps else None,
        "history_last_timestamp": timestamps[-1] if timestamps else None,
        "ground_truth_first_timestamp": gt_timestamps[0] if gt_timestamps else None,
        "ground_truth_last_timestamp": gt_timestamps[-1] if gt_timestamps else None,
    }


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _render_rl_report(report: dict[str, Any]) -> str:
    return (
        "# RL Formal Sample\n\n"
        "## Summary\n"
        f"{_safe_json({k: v for k, v in report.items() if k not in {'raw_prompt', 'initial_diagnostic_prompt', 'ground_truth'}})}\n\n"
        "## Raw Prompt\n"
        f"{report['raw_prompt']}\n\n"
        "## Initial Diagnostic Prompt\n"
        f"{report['initial_diagnostic_prompt']}\n\n"
        "## Ground Truth\n"
        f"{report['ground_truth']}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect formal SFT/RL pipeline samples on real data.")
    parser.add_argument(
        "--sft-jsonl",
        type=Path,
        default=Path("dataset/ett_sft_etth1_runtime_teacher200_paper_same2/train_curated.jsonl"),
    )
    parser.add_argument(
        "--rl-jsonl",
        type=Path,
        default=Path("dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/val.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/formal_pipeline_inspection"),
    )
    parser.add_argument("--sft-sample-indexes", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--rl-sample-indexes", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "sft_jsonl": str(args.sft_jsonl),
        "rl_jsonl": str(args.rl_jsonl),
        "sft_samples": [],
        "rl_samples": [],
    }

    for sample_index in args.sft_sample_indexes:
        sample = _load_jsonl_record(args.sft_jsonl, sample_index)
        inspected = inspect_sft_sample(sample)
        sample_dir = args.output_dir / "sft" / f"sample_{sample_index:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for record_index, record in enumerate(inspected["records"]):
            _write_text(sample_dir / f"{record_index:02d}_{record['turn_stage']}.md", _render_sft_record(record))
        _write_text(
            sample_dir / "summary.json",
            _safe_json({k: v for k, v in inspected.items() if k not in {"records"}}) + "\n",
        )
        summary["sft_samples"].append({k: v for k, v in inspected.items() if k not in {"records"}})

    for sample_index in args.rl_sample_indexes:
        sample = _load_jsonl_record(args.rl_jsonl, sample_index)
        inspected = inspect_rl_sample(sample)
        sample_dir = args.output_dir / "rl" / f"sample_{sample_index:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        _write_text(sample_dir / "report.md", _render_rl_report(inspected))
        _write_text(
            sample_dir / "summary.json",
            _safe_json({k: v for k, v in inspected.items() if k not in {"raw_prompt", "initial_diagnostic_prompt", "ground_truth"}})
            + "\n",
        )
        summary["rl_samples"].append({k: v for k, v in inspected.items() if k not in {"raw_prompt", "initial_diagnostic_prompt", "ground_truth"}})

    _write_text(args.output_dir / "summary.json", _safe_json(summary) + "\n")
    print(f"Wrote inspection outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
