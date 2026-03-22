from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from recipe.time_series_forecast.prompts import (
    build_runtime_user_prompt,
    build_timeseries_system_prompt,
)
from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RUNTIME_SFT_PARQUET,
    DATASET_KIND_TEACHER_CURATED_SFT,
    validate_sibling_metadata,
)
from recipe.time_series_forecast.diagnostic_policy import select_feature_tool_names
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
DEFAULT_OUTPUT_DIR = Path("dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same2")


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


async def _predict_with_runtime_tools(
    *,
    historical_data: str,
    data_source: str,
    target_column: str,
    forecast_horizon: int,
    model_name: str,
) -> str:
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
    last_timestamp = get_last_timestamp(historical_data)
    if not last_timestamp:
        raise ValueError("Unable to infer the last historical timestamp for teacher prediction formatting.")
    return format_predictions_to_string(pred_df, last_timestamp)


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

    selected_tool_names = set(select_feature_tool_names(values))

    return [
        ToolResult(tool_name=name, tool_output=tool_outputs[name])
        for name, _builder in FEATURE_TOOL_BUILDERS
        if name in selected_tool_names
    ]


def _extract_prediction_values(prediction_text: str) -> list[float]:
    values: list[float] = []
    for line in str(prediction_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        parsed = None
        for token in reversed(parts):
            try:
                parsed = float(token)
                break
            except Exception:
                continue
        if parsed is None:
            matches = re.findall(r"-?\d+(?:\.\d+)?", line)
            if matches:
                try:
                    parsed = float(matches[-1])
                except Exception:
                    parsed = None
        if parsed is not None and np.isfinite(parsed):
            values.append(float(parsed))
    return values


def _require_prediction_values(prediction_text: str, forecast_horizon: int, *, source_name: str) -> list[float]:
    values = _extract_prediction_values(prediction_text)
    if len(values) != forecast_horizon:
        raise ValueError(
            f"{source_name} prediction length must equal forecast_horizon={forecast_horizon}, got {len(values)}"
        )
    return values


def _prediction_text_from_values(values: list[float]) -> str:
    return "\n".join(f"{float(v):.4f}" for v in values)


def _feature_tool_signature(selected_feature_tools: list[str]) -> str:
    return "->".join(selected_feature_tools) if selected_feature_tools else "none"


def _median_abs_deviation(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return mad


def _compute_diff_mad(values: list[float]) -> float:
    if len(values) < 2:
        return 1e-6
    diffs = np.diff(np.asarray(values, dtype=float))
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))
    return max(mad, 1e-6)


def _compute_error_metrics(candidate_values: list[float], ground_truth_values: list[float]) -> tuple[float, float]:
    min_len = min(len(candidate_values), len(ground_truth_values))
    if min_len <= 0:
        return float("inf"), float("inf")
    candidate = np.asarray(candidate_values[:min_len], dtype=float)
    reference = np.asarray(ground_truth_values[:min_len], dtype=float)
    mse = float(np.mean((candidate - reference) ** 2))
    mae = float(np.mean(np.abs(candidate - reference)))
    return mse, mae


def _detect_isolated_spikes(values: list[float], diff_mad: float) -> list[int]:
    if len(values) < 3:
        return []
    threshold = max(3.0 * diff_mad, 1e-4)
    spike_indices: list[int] = []
    for idx in range(1, len(values) - 1):
        left_delta = values[idx] - values[idx - 1]
        right_delta = values[idx + 1] - values[idx]
        if abs(left_delta) <= threshold or abs(right_delta) <= threshold:
            continue
        if left_delta * right_delta < 0:
            spike_indices.append(idx)
    return spike_indices


def _smooth_isolated_spikes(values: list[float]) -> tuple[list[float], list[str]]:
    smoothed = list(values)
    diff_mad = _compute_diff_mad(smoothed)
    spike_indices = _detect_isolated_spikes(smoothed, diff_mad)
    if not spike_indices:
        return smoothed, []
    for idx in spike_indices:
        smoothed[idx] = 0.5 * (smoothed[idx - 1] + smoothed[idx + 1])
    return smoothed, ["isolated_spike_smoothing"]


def _repair_flat_tail(values: list[float], history_values: list[float]) -> tuple[list[float], list[str]]:
    if len(values) < 8 or len(history_values) < 8:
        return list(values), []
    repaired = list(values)
    tail_value = repaired[-1]
    flat_run = 1
    for idx in range(len(repaired) - 2, -1, -1):
        if abs(repaired[idx] - tail_value) <= 1e-8:
            flat_run += 1
        else:
            break
    if flat_run < 6:
        return repaired, []
    history_tail = history_values[-flat_run:]
    if len(history_tail) != flat_run:
        return repaired, []
    history_center = float(np.mean(history_tail))
    tail_center = float(np.mean(repaired[-flat_run:]))
    adjusted_tail = [tail_center + (float(v) - history_center) for v in history_tail]
    if np.allclose(np.asarray(adjusted_tail, dtype=float), np.asarray(repaired[-flat_run:], dtype=float), atol=1e-6):
        return repaired, []
    repaired[-flat_run:] = adjusted_tail
    return repaired, ["flat_tail_repair"]


def _clip_implausible_amplitude(values: list[float], history_values: list[float]) -> tuple[list[float], list[str]]:
    if not history_values:
        return list(values), []
    history_median = float(np.median(np.asarray(history_values, dtype=float)))
    history_mad = max(_median_abs_deviation(history_values), 1e-6)
    lower = history_median - 6.0 * history_mad
    upper = history_median + 6.0 * history_mad
    clipped = [float(np.clip(v, lower, upper)) for v in values]
    if np.allclose(np.asarray(clipped, dtype=float), np.asarray(values, dtype=float), atol=1e-8):
        return list(values), []
    return clipped, ["amplitude_clip"]


def _adjust_local_level(values: list[float], history_values: list[float]) -> tuple[list[float], list[str]]:
    window = min(12, len(values), len(history_values))
    if window < 4:
        return list(values), []

    adjusted = list(values)
    pred_tail = np.asarray(adjusted[-window:], dtype=float)
    history_tail = np.asarray(history_values[-window:], dtype=float)
    level_gap = float(np.mean(history_tail) - np.mean(pred_tail))
    history_scale = max(_median_abs_deviation(history_values), 1e-4)
    if abs(level_gap) <= max(1.5 * history_scale, 1e-4):
        return adjusted, []

    correction = 0.5 * level_gap
    adjusted[-window:] = [float(v + correction) for v in adjusted[-window:]]
    return adjusted, ["local_level_adjust"]


def _adjust_local_slope(values: list[float], history_values: list[float]) -> tuple[list[float], list[str]]:
    window = min(12, len(values), len(history_values))
    if window < 4:
        return list(values), []

    adjusted = list(values)
    pred_tail = np.asarray(adjusted[-window:], dtype=float)
    history_tail = np.asarray(history_values[-window:], dtype=float)
    pred_slope = float((pred_tail[-1] - pred_tail[0]) / max(window - 1, 1))
    history_slope = float((history_tail[-1] - history_tail[0]) / max(window - 1, 1))
    slope_gap = history_slope - pred_slope
    slope_scale = max(_compute_diff_mad(history_values), 1e-4)
    if abs(slope_gap) <= max(2.0 * slope_scale, 1e-4):
        return adjusted, []

    slope_correction = 0.5 * slope_gap
    start_index = len(adjusted) - window
    for offset in range(window):
        adjusted[start_index + offset] = float(adjusted[start_index + offset] + slope_correction * offset)
    return adjusted, ["local_slope_adjust"]


def _should_attempt_refinement(
    selected_feature_tools: list[str],
    score_margin: float,
    candidate_refinements: Sequence[tuple[list[float], list[str]]],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if "extract_data_quality" in selected_feature_tools:
        reasons.append("quality_signal")
    if "extract_forecast_residuals" in selected_feature_tools:
        reasons.append("residual_signal")
    if "extract_within_channel_dynamics" in selected_feature_tools and score_margin <= 0.05:
        reasons.append("dynamics_small_margin")
    candidate_ops = {op for _values, ops in candidate_refinements for op in ops}
    if score_margin <= 0.05:
        if "isolated_spike_smoothing" in candidate_ops:
            reasons.append("prediction_spike_signal")
        if "flat_tail_repair" in candidate_ops:
            reasons.append("prediction_flat_tail_signal")
        if "local_level_adjust" in candidate_ops:
            reasons.append("prediction_level_signal")
        if "local_slope_adjust" in candidate_ops:
            reasons.append("prediction_slope_signal")
        if "amplitude_clip" in candidate_ops:
            reasons.append("prediction_amplitude_signal")
    return bool(reasons), reasons or ["evidence_consistent"]


def _generate_local_refinement_candidates(
    values: list[float],
    history_values: list[float],
) -> list[tuple[list[float], list[str]]]:
    candidate_builders = [
        lambda current: _smooth_isolated_spikes(current),
        lambda current: _repair_flat_tail(current, history_values),
        lambda current: _adjust_local_level(current, history_values),
        lambda current: _adjust_local_slope(current, history_values),
        lambda current: _clip_implausible_amplitude(current, history_values),
    ]
    candidates: list[tuple[list[float], list[str]]] = []
    seen_signatures: set[tuple[str, tuple[float, ...]]] = set()
    for builder in candidate_builders:
        candidate_values, candidate_ops = builder(list(values))
        deduped_ops = list(dict.fromkeys(candidate_ops))
        if not deduped_ops:
            continue
        signature = (
            "->".join(deduped_ops),
            tuple(round(float(value), 6) for value in candidate_values),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        candidates.append((candidate_values, deduped_ops))
    return candidates


def _summarize_refine_delta(base_values: list[float], refined_values: list[float]) -> dict[str, float]:
    if len(base_values) != len(refined_values):
        raise ValueError("base_values and refined_values must have the same length")
    deltas = np.abs(np.asarray(refined_values, dtype=float) - np.asarray(base_values, dtype=float))
    changed_indices = np.where(deltas > 1e-6)[0]
    if changed_indices.size == 0:
        return {
            "refine_changed_value_count": 0,
            "refine_first_changed_index": -1,
            "refine_last_changed_index": -1,
            "refine_changed_span": 0,
            "refine_mean_abs_delta": 0.0,
            "refine_max_abs_delta": 0.0,
        }
    first_idx = int(changed_indices[0])
    last_idx = int(changed_indices[-1])
    return {
        "refine_changed_value_count": int(changed_indices.size),
        "refine_first_changed_index": first_idx,
        "refine_last_changed_index": last_idx,
        "refine_changed_span": int(last_idx - first_idx + 1),
        "refine_mean_abs_delta": float(np.mean(deltas[changed_indices])),
        "refine_max_abs_delta": float(np.max(deltas[changed_indices])),
    }


def _is_meaningful_local_refine(
    *,
    base_values: list[float],
    refined_values: list[float],
    history_values: list[float],
    base_mse: float,
    base_mae: float,
    refined_mse: float,
    refined_mae: float,
) -> tuple[bool, dict[str, float]]:
    delta_summary = _summarize_refine_delta(base_values, refined_values)
    changed_count = int(delta_summary["refine_changed_value_count"])
    if changed_count <= 0:
        return False, delta_summary

    max_changed = min(24, max(4, len(base_values) // 4))
    max_span = min(32, max(8, len(base_values) // 3))
    local_enough = (
        changed_count <= max_changed
        and int(delta_summary["refine_changed_span"]) <= max_span
    )

    mse_gain = float(base_mse - refined_mse)
    mae_gain = float(base_mae - refined_mae)
    meaningful_gain = (
        mse_gain > max(1e-6, 0.005 * max(base_mse, 1e-6))
        or mae_gain > max(1e-6, 0.005 * max(base_mae, 1e-6))
    )
    return bool(local_enough and meaningful_gain), delta_summary


def _build_turn3_target(
    *,
    sample: dict[str, Any],
    history_values: list[float],
    base_prediction_text: str,
    forecast_horizon: int,
    model_name: str,
    selected_feature_tools: list[str],
) -> dict[str, Any]:
    base_values = _require_prediction_values(base_prediction_text, forecast_horizon, source_name=f"{model_name}_base")
    reward_model = sample.get("reward_model", {}) if isinstance(sample.get("reward_model"), dict) else {}
    ground_truth_values = _extract_prediction_values(str(reward_model.get("ground_truth", "") or ""))
    if len(ground_truth_values) < forecast_horizon:
        raise ValueError(
            f"ground_truth length must be at least forecast_horizon={forecast_horizon}, got {len(ground_truth_values)}"
        )
    ground_truth_values = ground_truth_values[:forecast_horizon]
    score_margin = float(sample.get("teacher_eval_score_margin", 0.0) or 0.0)
    candidate_refinements = _generate_local_refinement_candidates(base_values, history_values)
    attempt_refine, trigger_reasons = _should_attempt_refinement(
        selected_feature_tools,
        score_margin,
        candidate_refinements,
    )

    base_mse, base_mae = _compute_error_metrics(base_values, ground_truth_values)
    best_target_type = "validated_keep"
    best_values = list(base_values)
    best_ops: list[str] = []
    best_mse = base_mse
    best_mae = base_mae
    best_delta_summary = _summarize_refine_delta(base_values, base_values)
    best_trigger_reasons = ["evidence_consistent"]

    if attempt_refine:
        candidate_trigger_reasons = trigger_reasons
        for refined_values, refine_ops in candidate_refinements:
            refined_mse, refined_mae = _compute_error_metrics(refined_values, ground_truth_values)
            is_meaningful_refine, delta_summary = _is_meaningful_local_refine(
                base_values=base_values,
                refined_values=refined_values,
                history_values=history_values,
                base_mse=base_mse,
                base_mae=base_mae,
                refined_mse=refined_mse,
                refined_mae=refined_mae,
            )
            if not is_meaningful_refine or not np.isfinite(refined_mse) or refined_mse >= best_mse - 1e-8:
                continue
            best_target_type = "local_refine"
            best_values = refined_values
            best_ops = refine_ops
            best_mse = refined_mse
            best_mae = refined_mae
            best_delta_summary = delta_summary
            best_trigger_reasons = candidate_trigger_reasons
    trigger_reasons = best_trigger_reasons

    refine_gain_mse = float(base_mse - best_mse)
    refine_gain_mae = float(base_mae - best_mae)
    refine_ops_signature = "none" if not best_ops else "->".join(best_ops)
    trigger_reason = "none" if not trigger_reasons else "->".join(dict.fromkeys(trigger_reasons))

    return {
        "turn3_target_type": best_target_type,
        "refine_ops": list(dict.fromkeys(best_ops)),
        "refine_ops_signature": refine_ops_signature,
        "refine_gain_mse": refine_gain_mse,
        "refine_gain_mae": refine_gain_mae,
        "turn3_trigger_reason": trigger_reason,
        "base_teacher_prediction_text": _prediction_text_from_values(base_values),
        "refined_prediction_text": _prediction_text_from_values(best_values),
        **best_delta_summary,
    }


def build_final_answer(prediction_values: list[float]) -> str:
    return f"<answer>\n{_prediction_text_from_values(prediction_values)}\n</answer>"


def _resolve_prediction_text(
    *,
    sample: dict[str, Any],
    historical_data: str,
    data_source: str,
    target_column: str,
    forecast_horizon: int,
    model_name: str,
    allow_cached_reference: bool,
) -> tuple[str, str]:
    if allow_cached_reference:
        cached_teacher_prediction = str(sample.get("teacher_prediction_text", "") or "").strip()
        if cached_teacher_prediction:
            return cached_teacher_prediction, "reference_teacher_cached"
    prediction_text = asyncio.run(
        _predict_with_runtime_tools(
            historical_data=historical_data,
            data_source=data_source,
            target_column=target_column,
            forecast_horizon=forecast_horizon,
            model_name=model_name,
        )
    )
    return prediction_text, "reference_teacher_runtime"


def build_sft_record(sample: dict[str, Any]) -> dict[str, Any]:
    raw_prompt = sample["raw_prompt"][0]["content"]
    reward_model = sample.get("reward_model", {}) if isinstance(sample.get("reward_model"), dict) else {}
    task_spec = parse_task_prompt(raw_prompt, data_source=sample.get("data_source"))
    data_source = task_spec.data_source or str(sample.get("data_source") or "ETTh1")
    target_column = task_spec.target_column or "OT"
    lookback_window = int(task_spec.lookback_window or 96)
    forecast_horizon = int(task_spec.forecast_horizon or 96)
    historical_data = task_spec.historical_data or raw_prompt
    _, history_values = parse_time_series_string(historical_data, target_column=target_column)
    if not history_values:
        raise ValueError("No valid ETTh1 values found in raw_prompt")

    feature_results = build_feature_tool_results(history_values)
    selected_feature_tools = [result.tool_name for result in feature_results]
    history_analysis = [result.tool_output for result in feature_results]

    reference_teacher_model = _normalize_teacher_model(sample.get("reference_teacher_model"))
    second_teacher_model = _normalize_teacher_model(sample.get("teacher_eval_second_best_model"))

    try:
        base_prediction_text, base_prediction_source = _resolve_prediction_text(
            sample=sample,
            historical_data=historical_data,
            data_source=data_source,
            target_column=target_column,
            forecast_horizon=forecast_horizon,
            model_name=reference_teacher_model,
            allow_cached_reference=True,
        )
        selected_prediction_model = reference_teacher_model
    except Exception:
        if second_teacher_model == reference_teacher_model:
            raise
        base_prediction_text, base_prediction_source = _resolve_prediction_text(
            sample=sample,
            historical_data=historical_data,
            data_source=data_source,
            target_column=target_column,
            forecast_horizon=forecast_horizon,
            model_name=second_teacher_model,
            allow_cached_reference=False,
        )
        selected_prediction_model = second_teacher_model
        base_prediction_source = "second_teacher_runtime"

    turn3_target = _build_turn3_target(
        sample=sample,
        history_values=history_values,
        base_prediction_text=base_prediction_text,
        forecast_horizon=forecast_horizon,
        model_name=selected_prediction_model,
        selected_feature_tools=selected_feature_tools,
    )
    final_prediction_values = _require_prediction_values(
        turn3_target["refined_prediction_text"],
        forecast_horizon,
        source_name="turn3_target",
    )

    tool_prediction_text = compact_prediction_tool_output_from_string(
        base_prediction_text,
        model_name=selected_prediction_model,
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
        required_feature_tools=selected_feature_tools,
        completed_feature_tools=[],
    )
    turn_2_prompt = build_runtime_user_prompt(
        data_source=data_source,
        target_column=target_column,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        time_series_data=historical_data,
        history_analysis=history_analysis,
        prediction_results=None,
        required_feature_tools=selected_feature_tools,
        completed_feature_tools=selected_feature_tools,
    )
    turn_3_prompt = build_runtime_user_prompt(
        data_source=data_source,
        target_column=target_column,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        time_series_data=historical_data,
        history_analysis=history_analysis,
        prediction_results=tool_prediction_text,
        prediction_model_used=selected_prediction_model,
        required_feature_tools=selected_feature_tools,
        completed_feature_tools=selected_feature_tools,
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
        arguments={"model_name": selected_prediction_model},
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
            "content": build_final_answer(final_prediction_values),
        },
    ]

    return {
        "messages": messages,
        "sample_index": int(sample.get("index", -1)),
        "data_source": data_source,
        "target_column": target_column,
        "forecast_horizon": forecast_horizon,
        "lookback_window": lookback_window,
        "reference_teacher_model": reference_teacher_model,
        "selected_prediction_model": selected_prediction_model,
        "base_prediction_source": base_prediction_source,
        "selected_feature_tools": selected_feature_tools,
        "selected_feature_tool_count": len(selected_feature_tools),
        "selected_feature_tool_signature": _feature_tool_signature(selected_feature_tools),
        "turn3_target_type": turn3_target["turn3_target_type"],
        "turn3_trigger_reason": turn3_target["turn3_trigger_reason"],
        "refine_ops": turn3_target["refine_ops"],
        "refine_ops_signature": turn3_target["refine_ops_signature"],
        "refine_gain_mse": turn3_target["refine_gain_mse"],
        "refine_gain_mae": turn3_target["refine_gain_mae"],
        "refine_changed_value_count": turn3_target["refine_changed_value_count"],
        "refine_first_changed_index": turn3_target["refine_first_changed_index"],
        "refine_last_changed_index": turn3_target["refine_last_changed_index"],
        "refine_changed_span": turn3_target["refine_changed_span"],
        "refine_mean_abs_delta": turn3_target["refine_mean_abs_delta"],
        "refine_max_abs_delta": turn3_target["refine_max_abs_delta"],
        "base_teacher_prediction_text": turn3_target["base_teacher_prediction_text"],
        "refined_prediction_text": turn3_target["refined_prediction_text"],
        "teacher_eval_best_score": sample.get("teacher_eval_best_score"),
        "teacher_eval_second_best_model": sample.get("teacher_eval_second_best_model"),
        "teacher_eval_second_best_score": sample.get("teacher_eval_second_best_score"),
        "teacher_eval_score_margin": sample.get("teacher_eval_score_margin"),
        "teacher_eval_scores": sample.get("teacher_eval_scores"),
    }


def convert_jsonl_to_sft_parquet(
    *,
    input_path: str | Path,
    output_path: str | Path,
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
            records.append(build_sft_record(sample))
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


def _rebalance_train_turn3_targets(
    dataframe: pd.DataFrame,
    *,
    min_local_refine_ratio: float,
) -> pd.DataFrame:
    if dataframe.empty or min_local_refine_ratio <= 0:
        return dataframe
    if "turn3_target_type" not in dataframe.columns:
        return dataframe

    local_refine_mask = dataframe["turn3_target_type"] == "local_refine"
    validated_keep_mask = dataframe["turn3_target_type"] == "validated_keep"
    local_refine_df = dataframe.loc[local_refine_mask].copy()
    validated_keep_df = dataframe.loc[validated_keep_mask].copy()
    other_df = dataframe.loc[~(local_refine_mask | validated_keep_mask)].copy()

    local_count = len(local_refine_df)
    keep_count = len(validated_keep_df)
    if local_count <= 0 or keep_count <= 0:
        return dataframe

    max_keep_count = int(np.floor(local_count * (1.0 - min_local_refine_ratio) / min_local_refine_ratio))
    max_keep_count = max(max_keep_count, 0)
    if keep_count <= max_keep_count:
        return dataframe

    validated_keep_df = validated_keep_df.sort_values("sample_index").reset_index(drop=True)
    if max_keep_count <= 0:
        rebalanced_keep_df = validated_keep_df.iloc[0:0].copy()
    else:
        take_positions = np.linspace(0, keep_count - 1, num=max_keep_count, dtype=int)
        rebalanced_keep_df = validated_keep_df.iloc[take_positions].copy()

    balanced = pd.concat([local_refine_df, rebalanced_keep_df, other_df], ignore_index=True)
    if "sample_index" in balanced.columns:
        balanced = balanced.sort_values("sample_index").reset_index(drop=True)
    return balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ETTh1 OT multi-turn SFT parquet from RL jsonl samples.")
    parser.add_argument(
        "--train-jsonl",
        default="dataset/ett_rl_etth1_paper_same2/train.jsonl",
        help="RL train jsonl path.",
    )
    parser.add_argument(
        "--val-jsonl",
        default="dataset/ett_rl_etth1_paper_same2/val.jsonl",
        help="RL val jsonl path.",
    )
    parser.add_argument(
        "--test-jsonl",
        default="dataset/ett_rl_etth1_paper_same2/test.jsonl",
        help="Optional RL test jsonl path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for parquet files.",
    )
    parser.add_argument("--max-train-samples", type=int, default=-1, help="Limit train sample count for debugging.")
    parser.add_argument("--max-val-samples", type=int, default=-1, help="Limit val sample count for debugging.")
    parser.add_argument("--max-test-samples", type=int, default=-1, help="Limit test sample count for debugging.")
    parser.add_argument(
        "--train-min-local-refine-ratio",
        type=float,
        default=0.30,
        help="Minimum desired local_refine ratio in train parquet. Set <=0 to disable train rebalancing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_metadata_paths: list[Path] = []
    for split_path in (Path(args.train_jsonl), Path(args.val_jsonl), Path(args.test_jsonl)):
        if not split_path.exists():
            continue
        _, source_metadata_path = validate_sibling_metadata(
            split_path,
            expected_kind=DATASET_KIND_TEACHER_CURATED_SFT,
        )
        source_metadata_paths.append(source_metadata_path)
    if source_metadata_paths:
        unique_source_metadata_paths = {str(path) for path in source_metadata_paths}
        if len(unique_source_metadata_paths) != 1:
            raise ValueError(
                "All source curated jsonl splits must come from the same teacher-curated dataset directory. "
                f"Got metadata files: {sorted(unique_source_metadata_paths)}"
            )

    train_df_raw = convert_jsonl_to_sft_parquet(
        input_path=args.train_jsonl,
        output_path=output_dir / "train.parquet",
        max_samples=args.max_train_samples,
    )
    train_df = _rebalance_train_turn3_targets(
        train_df_raw,
        min_local_refine_ratio=float(args.train_min_local_refine_ratio),
    )
    if len(train_df) != len(train_df_raw) or not train_df.equals(train_df_raw):
        train_df.to_parquet(output_dir / "train.parquet", index=False)
        print(
            f"Rebalanced train.parquet turn3_target_type distribution: "
            f"{_distribution_from_series(train_df['turn3_target_type'])}"
        )
    val_df = convert_jsonl_to_sft_parquet(
        input_path=args.val_jsonl,
        output_path=output_dir / "val.parquet",
        max_samples=args.max_val_samples,
    )

    test_count = 0
    test_df: pd.DataFrame | None = None
    test_path = Path(args.test_jsonl)
    if test_path.exists():
        test_df = convert_jsonl_to_sft_parquet(
            input_path=test_path,
            output_path=output_dir / "test.parquet",
            max_samples=args.max_test_samples,
        )
        test_count = len(test_df)

    metadata_kwargs = dict(
        dataset_kind=DATASET_KIND_RUNTIME_SFT_PARQUET,
        pipeline_stage="runtime_multiturn_sft",
        train_samples_before_balance=len(train_df_raw),
        train_samples=len(train_df),
        val_samples=len(val_df),
        test_samples=test_count,
        train_min_local_refine_ratio=float(args.train_min_local_refine_ratio),
        source_train_jsonl=str(Path(args.train_jsonl)),
        source_val_jsonl=str(Path(args.val_jsonl)),
        source_test_jsonl=str(test_path),
        source_curated_metadata_path=str(source_metadata_paths[0]) if source_metadata_paths else "",
        train_turn3_target_type_distribution_before_balance=_distribution_from_series(train_df_raw["turn3_target_type"]),
        train_reference_teacher_model_distribution=_distribution_from_series(train_df["reference_teacher_model"]),
        train_selected_prediction_model_distribution=_distribution_from_series(train_df["selected_prediction_model"]),
        train_turn3_target_type_distribution=_distribution_from_series(train_df["turn3_target_type"]),
        train_turn3_trigger_reason_distribution=_distribution_from_series(train_df["turn3_trigger_reason"]),
        train_refine_ops_signature_distribution=_distribution_from_series(train_df["refine_ops_signature"]),
        train_selected_feature_tool_signature_distribution=_distribution_from_series(
            train_df["selected_feature_tool_signature"]
        ),
        train_base_prediction_source_distribution=_distribution_from_series(train_df["base_prediction_source"]),
        val_reference_teacher_model_distribution=_distribution_from_series(val_df["reference_teacher_model"]),
        val_selected_prediction_model_distribution=_distribution_from_series(val_df["selected_prediction_model"]),
        val_turn3_target_type_distribution=_distribution_from_series(val_df["turn3_target_type"]),
        val_turn3_trigger_reason_distribution=_distribution_from_series(val_df["turn3_trigger_reason"]),
        val_refine_ops_signature_distribution=_distribution_from_series(val_df["refine_ops_signature"]),
        val_selected_feature_tool_signature_distribution=_distribution_from_series(
            val_df["selected_feature_tool_signature"]
        ),
        val_base_prediction_source_distribution=_distribution_from_series(val_df["base_prediction_source"]),
    )
    if test_df is not None and len(test_df) > 0:
        metadata_kwargs.update(
            test_reference_teacher_model_distribution=_distribution_from_series(test_df["reference_teacher_model"]),
            test_selected_prediction_model_distribution=_distribution_from_series(test_df["selected_prediction_model"]),
            test_turn3_target_type_distribution=_distribution_from_series(test_df["turn3_target_type"]),
            test_turn3_trigger_reason_distribution=_distribution_from_series(test_df["turn3_trigger_reason"]),
            test_refine_ops_signature_distribution=_distribution_from_series(test_df["refine_ops_signature"]),
            test_selected_feature_tool_signature_distribution=_distribution_from_series(
                test_df["selected_feature_tool_signature"]
            ),
            test_base_prediction_source_distribution=_distribution_from_series(test_df["base_prediction_source"]),
        )

    _write_metadata(
        output_dir,
        **metadata_kwargs,
    )


if __name__ == "__main__":
    main()
