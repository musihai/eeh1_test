from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from recipe.time_series_forecast.prompts import (
    TIMESERIES_TOOL_SCHEMAS,
    build_runtime_user_prompt,
    build_timeseries_system_prompt,
)
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.utils import (
    compact_prediction_tool_output_from_string,
    extract_basic_statistics,
    extract_data_quality,
    extract_event_summary,
    extract_forecast_residuals,
    extract_within_channel_dynamics,
    format_basic_statistics,
    format_data_quality,
    format_event_summary,
    format_forecast_residuals,
    format_predictions_to_string,
    format_within_channel_dynamics,
    get_last_timestamp,
    parse_time_series_string,
    parse_time_series_to_dataframe,
    predict_time_series_async,
)


SUPPORTED_PREDICTION_MODELS = {"patchtst", "itransformer", "arima", "chronos2"}
DEFAULT_OUTPUT_DIR = Path("dataset/ett_sft_etth1_runtime_ot_teacher200_paper_v2")


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    tool_output: str


FEATURE_TOOL_BUILDERS = (
    ("extract_basic_statistics", lambda values: format_basic_statistics(extract_basic_statistics(values))),
    (
        "extract_within_channel_dynamics",
        lambda values: format_within_channel_dynamics(extract_within_channel_dynamics(values)),
    ),
    ("extract_forecast_residuals", lambda values: format_forecast_residuals(extract_forecast_residuals(values))),
    ("extract_data_quality", lambda values: format_data_quality(extract_data_quality(values))),
    ("extract_event_summary", lambda values: format_event_summary(extract_event_summary(values))),
)


def _normalize_teacher_model(model_name: Any) -> str:
    model = str(model_name or "patchtst").strip().lower()
    return model if model in SUPPORTED_PREDICTION_MODELS else "patchtst"


def _make_tool_call(tool_name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


def _slice_nonempty_lines(text: str, limit: int | None = None) -> list[str]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if limit is not None:
        return lines[:limit]
    return lines


def _infer_last_timestamp_from_ground_truth(ground_truth: str) -> str | None:
    lines = _slice_nonempty_lines(ground_truth, limit=2)
    if not lines:
        return None

    first_parts = lines[0].split()
    if len(first_parts) < 2:
        return None
    first_timestamp = pd.to_datetime(f"{first_parts[0]} {first_parts[1]}")

    if len(lines) >= 2:
        second_parts = lines[1].split()
        if len(second_parts) >= 2:
            second_timestamp = pd.to_datetime(f"{second_parts[0]} {second_parts[1]}")
            freq = second_timestamp - first_timestamp
            if freq > pd.Timedelta(0):
                return (first_timestamp - freq).strftime("%Y-%m-%d %H:%M:%S")

    return (first_timestamp - pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")


def _ground_truth_prediction_text(ground_truth: str, forecast_horizon: int) -> str:
    return "\n".join(_slice_nonempty_lines(ground_truth, limit=forecast_horizon))


async def _predict_with_runtime_tools(
    *,
    historical_data: str,
    data_source: str,
    target_column: str,
    forecast_horizon: int,
    model_name: str,
    fallback_ground_truth: str,
) -> tuple[str, str]:
    context_df = parse_time_series_to_dataframe(
        historical_data,
        series_id=data_source or "ETTh1",
        target_column=target_column,
    )
    pred_df = await predict_time_series_async(
        context_df,
        prediction_length=forecast_horizon,
        model_name=model_name,
    )
    last_timestamp = get_last_timestamp(historical_data) or _infer_last_timestamp_from_ground_truth(fallback_ground_truth)
    prediction_text = format_predictions_to_string(pred_df, last_timestamp)
    return prediction_text, "reference_teacher"


def build_feature_tool_results(values: list[float]) -> list[ToolResult]:
    basic_features = extract_basic_statistics(values)
    dynamics_features = extract_within_channel_dynamics(values)
    residual_features = extract_forecast_residuals(values)
    quality_features = extract_data_quality(values)
    event_features = extract_event_summary(values)

    tool_outputs = {
        "extract_basic_statistics": format_basic_statistics(basic_features),
        "extract_within_channel_dynamics": format_within_channel_dynamics(dynamics_features),
        "extract_forecast_residuals": format_forecast_residuals(residual_features),
        "extract_data_quality": format_data_quality(quality_features),
        "extract_event_summary": format_event_summary(event_features),
    }

    selected_tool_names = {"extract_basic_statistics"}

    strong_dynamics = (
        float(dynamics_features.get("changepoint_count", 0.0)) >= 1.0
        or float(dynamics_features.get("peak_count", 0.0)) >= 3.0
        or float(dynamics_features.get("slope_second_diff_max", 0.0)) >= max(float(basic_features.get("mad", 0.0)), 1e-3)
        or float(dynamics_features.get("monotone_duration", 0.0)) >= 0.35
    )
    residual_difficulty = (
        abs(float(residual_features.get("residual_acf1", 0.0))) >= 0.2
        or float(residual_features.get("residual_exceed_ratio", 0.0)) >= 0.08
        or float(residual_features.get("residual_concentration", 0.0)) >= 0.35
    )
    quality_issue = (
        float(quality_features.get("quality_quantization_score", 0.0)) >= 0.25
        or float(quality_features.get("quality_saturation_ratio", 0.0)) >= 0.08
        or float(quality_features.get("quality_constant_channel_ratio", 0.0)) > 0.0
        or float(quality_features.get("quality_dropout_ratio", 0.0)) > 0.0
    )
    eventful_window = (
        float(event_features.get("event_segment_count", 0.0)) >= 4.0
        or int(float(event_features.get("event_dominant_pattern", 0.0))) == 3
        or abs(float(basic_features.get("acf_seasonal", 0.0))) >= 0.25
    )

    if strong_dynamics:
        selected_tool_names.add("extract_within_channel_dynamics")
    if residual_difficulty:
        selected_tool_names.add("extract_forecast_residuals")
    if quality_issue:
        selected_tool_names.add("extract_data_quality")
    if eventful_window:
        selected_tool_names.add("extract_event_summary")

    if len(selected_tool_names) == 1:
        fallback_tool = "extract_event_summary" if abs(float(basic_features.get("acf_seasonal", 0.0))) >= 0.2 else "extract_within_channel_dynamics"
        selected_tool_names.add(fallback_tool)

    return [
        ToolResult(tool_name=name, tool_output=tool_outputs[name])
        for name, _builder in FEATURE_TOOL_BUILDERS
        if name in selected_tool_names
    ]


def _to_value_only_prediction_text(prediction_text: str, forecast_horizon: int) -> str:
    values: list[float] = []
    for line in str(prediction_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # Try last numeric token first (works for timestamp + value).
        parts = line.split()
        parsed = None
        for token in reversed(parts):
            try:
                parsed = float(token)
                break
            except Exception:
                continue
        # Fallback: regex extract the last float-like number.
        if parsed is None:
            matches = re.findall(r"-?\d+(?:\.\d+)?", line)
            if matches:
                try:
                    parsed = float(matches[-1])
                except Exception:
                    parsed = None
        if parsed is not None:
            values.append(float(parsed))

    if not values:
        return ""

    trimmed = values[:forecast_horizon]
    if len(trimmed) < forecast_horizon:
        trimmed = trimmed + [trimmed[-1]] * (forecast_horizon - len(trimmed))
    return "\n".join(f"{v:.4f}" for v in trimmed)


def _infer_trajectory_type(sample: dict[str, Any], selected_feature_tools: list[str]) -> str:
    score_margin = float(sample.get("teacher_eval_score_margin", 0.0) or 0.0)
    if "extract_data_quality" in selected_feature_tools or "extract_forecast_residuals" in selected_feature_tools:
        return "route_then_refine"
    if "extract_within_channel_dynamics" in selected_feature_tools and score_margin <= 0.05:
        return "route_then_refine"
    return "route_only"


def _feature_tool_signature(selected_feature_tools: list[str]) -> str:
    if not selected_feature_tools:
        return "none"
    return "->".join(selected_feature_tools)


def _build_refinement_supervision(
    sample: dict[str, Any],
    selected_feature_tools: list[str],
    *,
    model_name: str,
    prediction_source: str,
) -> dict[str, str]:
    trajectory_type = _infer_trajectory_type(sample, selected_feature_tools)
    score_margin = float(sample.get("teacher_eval_score_margin", 0.0) or 0.0)

    if trajectory_type == "route_then_refine":
        if "extract_data_quality" in selected_feature_tools or "extract_forecast_residuals" in selected_feature_tools:
            return {
                "trajectory_type": trajectory_type,
                "refinement_supervision_type": "language_only_refinement_hint",
                "refinement_trigger_reason": "quality_or_residual_signal",
                "reflection_text": (
                    f"I start from {model_name.upper()} as the selected forecast and make only a small "
                    "evidence-based refinement using the quality/residual diagnostics. I keep the selected "
                    "forecast as the backbone and only adjust local level or shape when the evidence supports it."
                ),
            }
        return {
            "trajectory_type": trajectory_type,
            "refinement_supervision_type": "language_only_refinement_hint",
            "refinement_trigger_reason": f"dynamics_signal_with_small_teacher_margin<= {score_margin:.4f}",
            "reflection_text": (
                f"I start from {model_name.upper()} as the selected forecast and make only a small "
                "evidence-based refinement because the dynamics diagnostics indicate local uncertainty. "
                "I preserve the selected forecast and only make constrained local adjustments."
            ),
        }

    if prediction_source == "reference_teacher":
        reflection_text = (
            f"The extracted evidence strongly supports {model_name.upper()} for this window, so I keep the "
            "selected forecast as the final answer without additional numerical adjustment."
        )
    else:
        reflection_text = "The selected forecast is consistent with the evidence, so I keep it as the final answer."
    return {
        "trajectory_type": trajectory_type,
        "refinement_supervision_type": "keep_selected_forecast",
        "refinement_trigger_reason": "none",
        "reflection_text": reflection_text,
    }


def build_final_answer(
    prediction_text: str,
    forecast_horizon: int,
    reflection_text: str,
) -> str:
    value_only_prediction_text = _to_value_only_prediction_text(prediction_text, forecast_horizon=forecast_horizon)
    return f"<think>{reflection_text}</think>\n<answer>\n{value_only_prediction_text}\n</answer>"


def build_sft_record(sample: dict[str, Any], prediction_mode: str = "preferred") -> dict[str, Any]:
    raw_prompt = sample["raw_prompt"][0]["content"]
    reward_model = sample.get("reward_model", {}) if isinstance(sample.get("reward_model"), dict) else {}
    ground_truth = str(reward_model.get("ground_truth", "") or "")
    task_spec = parse_task_prompt(raw_prompt, data_source=sample.get("data_source"))
    data_source = task_spec.data_source or str(sample.get("data_source") or "ETTh1")
    target_column = task_spec.target_column or "OT"
    lookback_window = int(task_spec.lookback_window or 96)
    forecast_horizon = int(task_spec.forecast_horizon or 96)
    historical_data = task_spec.historical_data or raw_prompt
    _, values = parse_time_series_string(historical_data, target_column=target_column)
    if not values:
        raise ValueError("No valid ETTh1 values found in raw_prompt")

    feature_results = build_feature_tool_results(values)
    selected_feature_tools = [result.tool_name for result in feature_results]
    history_analysis = [result.tool_output for result in feature_results]
    teacher_model = _normalize_teacher_model(sample.get("reference_teacher_model"))
    cached_teacher_prediction = str(sample.get("teacher_prediction_text", "") or "").strip()
    cached_prediction_source = str(sample.get("teacher_prediction_source", "reference_teacher") or "reference_teacher")

    prediction_text = ""
    prediction_source = "ground_truth"
    if prediction_mode not in {"ground_truth", "reference_teacher", "preferred"}:
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")
    if prediction_mode in {"reference_teacher", "preferred"} and cached_teacher_prediction:
        prediction_text = cached_teacher_prediction
        prediction_source = cached_prediction_source
    if prediction_mode in {"reference_teacher", "preferred"}:
        if not prediction_text:
            try:
                prediction_text, prediction_source = asyncio.run(
                    _predict_with_runtime_tools(
                        historical_data=historical_data,
                        data_source=data_source,
                        target_column=target_column,
                        forecast_horizon=forecast_horizon,
                        model_name=teacher_model,
                        fallback_ground_truth=ground_truth,
                    )
                )
            except Exception:
                if prediction_mode == "reference_teacher":
                    raise
    if not prediction_text:
        prediction_text = _ground_truth_prediction_text(ground_truth, forecast_horizon)
        prediction_source = "ground_truth"

    refinement_supervision = _build_refinement_supervision(
        sample,
        selected_feature_tools,
        model_name=teacher_model,
        prediction_source=prediction_source,
    )
    tool_prediction_text = compact_prediction_tool_output_from_string(
        prediction_text,
        model_name=teacher_model,
    )

    system_prompt = build_timeseries_system_prompt(data_source=data_source, target_column=target_column)
    turn_1_prompt = build_runtime_user_prompt(
        data_source=data_source,
        target_column=target_column,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        time_series_data=historical_data,
        history_analysis=[],
        prediction_results=None,
        conversation_has_tool_history=False,
    )
    turn_2_prompt = build_runtime_user_prompt(
        data_source=data_source,
        target_column=target_column,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        time_series_data=historical_data,
        history_analysis=history_analysis,
        prediction_results=None,
        conversation_has_tool_history=True,
    )
    turn_3_prompt = build_runtime_user_prompt(
        data_source=data_source,
        target_column=target_column,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        time_series_data=historical_data,
        history_analysis=history_analysis,
        prediction_results=tool_prediction_text,
        prediction_model_used=teacher_model,
        conversation_has_tool_history=True,
    )

    feature_tool_calls = [
        _make_tool_call(tool_name=result.tool_name, arguments={}, call_id=f"call_{idx}_{result.tool_name}")
        for idx, result in enumerate(feature_results, start=1)
    ]
    feature_tool_messages = [
        {
            "role": "tool",
            "content": result.tool_output,
            "tool_call_id": feature_tool_calls[idx]["id"],
        }
        for idx, result in enumerate(feature_results)
    ]
    prediction_tool_call = _make_tool_call(
        tool_name="predict_time_series",
        arguments={"model_name": teacher_model},
        call_id="call_predict_time_series",
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": turn_1_prompt},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": feature_tool_calls,
        },
        *feature_tool_messages,
        {"role": "user", "content": turn_2_prompt},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [prediction_tool_call],
        },
        {
            "role": "tool",
            "content": tool_prediction_text,
            "tool_call_id": prediction_tool_call["id"],
        },
        {"role": "user", "content": turn_3_prompt},
        {
            "role": "assistant",
            "content": build_final_answer(
                prediction_text,
                forecast_horizon=forecast_horizon,
                reflection_text=refinement_supervision["reflection_text"],
            ),
        },
    ]

    return {
        "messages": messages,
        "tools": TIMESERIES_TOOL_SCHEMAS,
        "enable_thinking": False,
        "sample_index": int(sample.get("index", -1)),
        "data_source": data_source,
        "target_column": target_column,
        "forecast_horizon": forecast_horizon,
        "lookback_window": lookback_window,
        "reference_teacher_model": teacher_model,
        "prediction_source": prediction_source,
        "selected_feature_tools": selected_feature_tools,
        "selected_feature_tool_count": len(selected_feature_tools),
        "selected_feature_tool_signature": _feature_tool_signature(selected_feature_tools),
        "sft_trajectory_type": refinement_supervision["trajectory_type"],
        "refinement_supervision_type": refinement_supervision["refinement_supervision_type"],
        "refinement_trigger_reason": refinement_supervision["refinement_trigger_reason"],
        "teacher_eval_best_score": sample.get("teacher_eval_best_score"),
        "teacher_eval_second_best_score": sample.get("teacher_eval_second_best_score"),
        "teacher_eval_score_margin": sample.get("teacher_eval_score_margin"),
        "teacher_eval_scores": sample.get("teacher_eval_scores"),
    }


def convert_jsonl_to_sft_parquet(
    *,
    input_path: str | Path,
    output_path: str | Path,
    prediction_mode: str = "preferred",
    max_samples: int = -1,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)
    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if max_samples > 0 and len(records) >= max_samples:
                break
            if not line.strip():
                continue
            sample = json.loads(line)
            records.append(build_sft_record(sample, prediction_mode=prediction_mode))
            if (line_idx + 1) % 500 == 0:
                print(f"Processed {line_idx + 1} samples from {input_path}")

    dataframe = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    print(f"Wrote {len(dataframe)} SFT samples to {output_path}")
    return dataframe


def _write_metadata(output_dir: Path, **kwargs: Any) -> None:
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(kwargs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _distribution_from_series(values: Iterable[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        if value is None:
            key = "__missing__"
        elif isinstance(value, float) and pd.isna(value):
            key = "__missing__"
        else:
            key = str(value)
        counter[key] += 1
    return {str(k): int(v) for k, v in sorted(counter.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ETTh1 OT multi-turn SFT parquet from RL jsonl samples.")
    parser.add_argument(
        "--train-jsonl",
        default="dataset/ett_rl_etth1_paper_same/train.jsonl",
        help="RL train jsonl path.",
    )
    parser.add_argument(
        "--val-jsonl",
        default="dataset/ett_rl_etth1_paper_same/val.jsonl",
        help="RL val jsonl path.",
    )
    parser.add_argument(
        "--test-jsonl",
        default="dataset/ett_rl_etth1_paper_same/test.jsonl",
        help="Optional RL test jsonl path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for parquet files.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["ground_truth", "reference_teacher", "preferred"],
        default="preferred",
        help="How to populate Turn 2 prediction tool outputs.",
    )
    parser.add_argument("--max-train-samples", type=int, default=-1, help="Limit train sample count for debugging.")
    parser.add_argument("--max-val-samples", type=int, default=-1, help="Limit val sample count for debugging.")
    parser.add_argument("--max-test-samples", type=int, default=-1, help="Limit test sample count for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = convert_jsonl_to_sft_parquet(
        input_path=args.train_jsonl,
        output_path=output_dir / "train.parquet",
        prediction_mode=args.prediction_mode,
        max_samples=args.max_train_samples,
    )
    val_df = convert_jsonl_to_sft_parquet(
        input_path=args.val_jsonl,
        output_path=output_dir / "val.parquet",
        prediction_mode=args.prediction_mode,
        max_samples=args.max_val_samples,
    )

    test_count = 0
    test_df: pd.DataFrame | None = None
    test_path = Path(args.test_jsonl)
    if test_path.exists():
        test_df = convert_jsonl_to_sft_parquet(
            input_path=test_path,
            output_path=output_dir / "test.parquet",
            prediction_mode=args.prediction_mode,
            max_samples=args.max_test_samples,
        )
        test_count = len(test_df)

    metadata_kwargs = dict(
        train_samples=len(train_df),
        val_samples=len(val_df),
        test_samples=test_count,
        prediction_mode=args.prediction_mode,
        source_train_jsonl=str(Path(args.train_jsonl)),
        source_val_jsonl=str(Path(args.val_jsonl)),
        source_test_jsonl=str(test_path),
        train_reference_teacher_model_distribution=_distribution_from_series(train_df["reference_teacher_model"]),
        train_sft_trajectory_type_distribution=_distribution_from_series(train_df["sft_trajectory_type"]),
        train_refinement_supervision_type_distribution=_distribution_from_series(train_df["refinement_supervision_type"]),
        train_refinement_trigger_reason_distribution=_distribution_from_series(train_df["refinement_trigger_reason"]),
        train_selected_feature_tool_signature_distribution=_distribution_from_series(
            train_df["selected_feature_tool_signature"]
        ),
        train_prediction_source_distribution=_distribution_from_series(train_df["prediction_source"]),
        val_reference_teacher_model_distribution=_distribution_from_series(val_df["reference_teacher_model"]),
        val_sft_trajectory_type_distribution=_distribution_from_series(val_df["sft_trajectory_type"]),
        val_refinement_supervision_type_distribution=_distribution_from_series(val_df["refinement_supervision_type"]),
        val_refinement_trigger_reason_distribution=_distribution_from_series(val_df["refinement_trigger_reason"]),
        val_selected_feature_tool_signature_distribution=_distribution_from_series(
            val_df["selected_feature_tool_signature"]
        ),
        val_prediction_source_distribution=_distribution_from_series(val_df["prediction_source"]),
    )
    if test_df is not None and len(test_df) > 0:
        metadata_kwargs.update(
            test_reference_teacher_model_distribution=_distribution_from_series(test_df["reference_teacher_model"]),
            test_sft_trajectory_type_distribution=_distribution_from_series(test_df["sft_trajectory_type"]),
            test_refinement_supervision_type_distribution=_distribution_from_series(
                test_df["refinement_supervision_type"]
            ),
            test_refinement_trigger_reason_distribution=_distribution_from_series(test_df["refinement_trigger_reason"]),
            test_selected_feature_tool_signature_distribution=_distribution_from_series(
                test_df["selected_feature_tool_signature"]
            ),
            test_prediction_source_distribution=_distribution_from_series(test_df["prediction_source"]),
        )

    _write_metadata(
        output_dir,
        **metadata_kwargs,
    )


if __name__ == "__main__":
    main()
