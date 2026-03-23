from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from recipe.time_series_forecast.prompts import (
    FEATURE_TOOL_SCHEMAS,
    PREDICT_TIMESERIES_TOOL_SCHEMA,
    build_runtime_user_prompt,
    build_timeseries_system_prompt,
)
from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RUNTIME_SFT_PARQUET,
    DATASET_KIND_TEACHER_CURATED_SFT,
    validate_sibling_metadata,
)
from recipe.time_series_forecast.diagnostic_policy import plan_diagnostic_tool_batches
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
TURN3_TARGET_MODE_PAPER_STRICT = "paper_strict"
TURN3_TARGET_MODE_ENGINEERING_REFINE = "engineering_refine"
SUPPORTED_TURN3_TARGET_MODES = {
    TURN3_TARGET_MODE_PAPER_STRICT,
    TURN3_TARGET_MODE_ENGINEERING_REFINE,
}
DEFAULT_TURN3_TARGET_MODE = TURN3_TARGET_MODE_PAPER_STRICT
DEFAULT_OUTPUT_DIR = Path("dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise_heuristicroute")
DEFAULT_CURATED_INPUT_DIR = Path("dataset/ett_sft_etth1_runtime_teacher200_paper_same2")
DEFAULT_TRAIN_JSONL = DEFAULT_CURATED_INPUT_DIR / "train_curated.jsonl"
DEFAULT_VAL_JSONL = DEFAULT_CURATED_INPUT_DIR / "val_curated.jsonl"
DEFAULT_TEST_JSONL = DEFAULT_CURATED_INPUT_DIR / "test_curated.jsonl"
DEFAULT_TRAIN_STAGE_REPEAT_FACTORS = {"diagnostic": 1, "routing": 1, "refinement": 1}


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
FEATURE_TOOL_SCHEMA_BY_NAME = {
    str(schema["function"]["name"]): schema for schema in FEATURE_TOOL_SCHEMAS
}
FEATURE_TOOL_LABELS = {
    "extract_basic_statistics": "basic statistics",
    "extract_within_channel_dynamics": "within-channel dynamics",
    "extract_forecast_residuals": "forecast residual patterns",
    "extract_data_quality": "data quality",
    "extract_event_summary": "event-level changes",
}
EVENT_PATTERN_NAMES = ["rise", "fall", "flat", "oscillation"]
ROUTING_MODEL_RATIONALES = {
    "patchtst": "patch-level local patterns and seasonality",
    "itransformer": "global temporal dependencies and trend interactions",
    "arima": "stable linear autocorrelation structure",
    "chronos2": "irregular dynamics that benefit from a strong general forecasting prior",
}


def _last_assistant_content(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            content = msg["content"].strip()
            if content:
                return content
    return ""


def _coerce_bool_flag(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, float) and pd.isna(value):
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _row_requires_paper_turn3_protocol(row: Any) -> bool:
    explicit_flag = row.get("paper_turn3_required") if hasattr(row, "get") else None
    if explicit_flag is not None and not (isinstance(explicit_flag, float) and pd.isna(explicit_flag)):
        return _coerce_bool_flag(explicit_flag, default=False)

    turn_stage = str((row.get("turn_stage") if hasattr(row, "get") else "") or "").strip().lower()
    if turn_stage:
        return turn_stage == "refinement"
    return True


def _source_level_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe

    if "turn_stage" in dataframe.columns:
        refinement_df = dataframe.loc[dataframe["turn_stage"] == "refinement"].copy()
        if not refinement_df.empty:
            sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in refinement_df.columns]
            if sort_columns:
                refinement_df = refinement_df.sort_values(sort_columns, kind="stable")
            return refinement_df.reset_index(drop=True)

    if "source_sample_index" in dataframe.columns:
        sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in dataframe.columns]
        ordered = dataframe.sort_values(sort_columns or ["source_sample_index"], kind="stable")
        return ordered.drop_duplicates(subset=["source_sample_index"], keep="last").reset_index(drop=True)
    return dataframe.reset_index(drop=True)


def _paper_turn3_protocol_reason(content: str, expected_len: int) -> str:
    if not content:
        return "empty_last_assistant"
    match = re.fullmatch(r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*", content, re.DOTALL)
    if match is None:
        if "<think>" not in content and "</think>" not in content:
            return "missing_think_block"
        if "<think>" in content and "</think>" not in content:
            return "missing_think_close_tag"
        if "</think>" in content and "<think>" not in content:
            return "missing_think_open_tag"
        if "<answer>" not in content and "</answer>" not in content:
            return "missing_answer_block"
        if "<answer>" in content and "</answer>" not in content:
            return "missing_answer_close_tag"
        if "</answer>" in content and "<answer>" not in content:
            return "missing_answer_open_tag"
        return "extra_text_outside_tags"

    answer_text = match.group(2).strip()
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    values = _extract_prediction_values(answer_text)
    if len(lines) != expected_len:
        return f"invalid_answer_shape:lines={len(lines)},expected={expected_len}"
    if len(values) != expected_len:
        return f"invalid_answer_shape:values={len(values)},expected={expected_len}"
    return "ok"


def _summarize_paper_turn3_protocol(dataframe: pd.DataFrame) -> dict[str, Any]:
    total = int(len(dataframe))
    reason_counter: Counter[str] = Counter()
    invalid_examples: list[dict[str, Any]] = []
    checked_count = 0

    for row_idx, row in dataframe.iterrows():
        if not _row_requires_paper_turn3_protocol(row):
            continue
        checked_count += 1
        expected_len = int(row.get("forecast_horizon", 96) or 96)
        content = _last_assistant_content(row.get("messages"))
        reason = _paper_turn3_protocol_reason(content, expected_len)
        reason_counter[reason] += 1
        if reason != "ok" and len(invalid_examples) < 5:
            invalid_examples.append(
                {
                    "row_index": int(row_idx),
                    "sample_index": int(row.get("sample_index", -1) or -1),
                    "reason": reason,
                    "assistant_head": content[:200],
                }
            )

    valid_count = int(reason_counter.get("ok", 0))
    invalid_count = checked_count - valid_count
    return {
        "turn3_protocol": "paper_think_answer_xml",
        "turn3_protocol_checked_count": int(checked_count),
        "turn3_protocol_skipped_count": int(max(total - checked_count, 0)),
        "turn3_protocol_valid_count": valid_count,
        "turn3_protocol_invalid_count": int(invalid_count),
        "turn3_protocol_valid_ratio": float(valid_count / checked_count) if checked_count > 0 else 0.0,
        "turn3_protocol_reason_distribution": {str(k): int(v) for k, v in sorted(reason_counter.items())},
        "turn3_protocol_invalid_examples": invalid_examples,
    }


def _validate_paper_turn3_protocol(dataframe: pd.DataFrame, *, split_name: str, output_path: Path) -> dict[str, Any]:
    summary = _summarize_paper_turn3_protocol(dataframe)
    if int(summary["turn3_protocol_checked_count"]) <= 0:
        raise ValueError(
            f"{split_name} SFT parquet at {output_path} does not contain any refinement records to validate."
        )
    if int(summary["turn3_protocol_invalid_count"]) > 0:
        raise ValueError(
            f"{split_name} SFT parquet is not paper-aligned at {output_path}. "
            f"reason_distribution={summary['turn3_protocol_reason_distribution']} "
            f"examples={summary['turn3_protocol_invalid_examples']}"
        )
    return summary


def _normalize_teacher_model(model_name: Any) -> str:
    model = str(model_name or "patchtst").strip().lower()
    return model if model in SUPPORTED_PREDICTION_MODELS else "patchtst"


def _compute_routing_feature_snapshot(history_values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in history_values]
    basic = extract_basic_statistics(values)
    dynamics = extract_within_channel_dynamics(values)
    residuals = extract_forecast_residuals(values)
    quality = extract_data_quality(values)
    events = extract_event_summary(values)
    dominant_pattern_idx = int(float(events.get("event_dominant_pattern", 0.0) or 0.0))
    dominant_pattern_idx = min(max(dominant_pattern_idx, 0), len(EVENT_PATTERN_NAMES) - 1)
    return {
        "acf1": float(basic.get("acf1", 0.0)),
        "acf_seasonal": float(basic.get("acf_seasonal", 0.0)),
        "cusum_max": float(basic.get("cusum_max", 0.0)),
        "changepoint_count": float(dynamics.get("changepoint_count", 0.0)),
        "peak_count": float(dynamics.get("peak_count", 0.0)),
        "peak_spacing_cv": float(dynamics.get("peak_spacing_cv", 0.0)),
        "monotone_duration": float(dynamics.get("monotone_duration", 0.0)),
        "residual_exceed_ratio": float(residuals.get("residual_exceed_ratio", 0.0)),
        "quality_quantization_score": float(quality.get("quality_quantization_score", 0.0)),
        "quality_saturation_ratio": float(quality.get("quality_saturation_ratio", 0.0)),
        "dominant_pattern": EVENT_PATTERN_NAMES[dominant_pattern_idx],
    }


def _heuristic_routing_scores(feature_snapshot: dict[str, Any]) -> dict[str, float]:
    acf1 = float(feature_snapshot.get("acf1", 0.0))
    acf_seasonal = float(feature_snapshot.get("acf_seasonal", 0.0))
    cusum_max = float(feature_snapshot.get("cusum_max", 0.0))
    changepoint_count = float(feature_snapshot.get("changepoint_count", 0.0))
    peak_count = float(feature_snapshot.get("peak_count", 0.0))
    peak_spacing_cv = float(feature_snapshot.get("peak_spacing_cv", 0.0))
    monotone_duration = float(feature_snapshot.get("monotone_duration", 0.0))
    residual_exceed_ratio = float(feature_snapshot.get("residual_exceed_ratio", 0.0))
    quality_quantization_score = float(feature_snapshot.get("quality_quantization_score", 0.0))
    quality_saturation_ratio = float(feature_snapshot.get("quality_saturation_ratio", 0.0))

    scores = {model_name: 0.0 for model_name in sorted(SUPPORTED_PREDICTION_MODELS)}

    scores["arima"] += 2.0 if acf1 >= 0.93 else (1.0 if acf1 >= 0.88 else 0.0)
    scores["arima"] += 1.5 if abs(acf_seasonal) <= 0.15 else 0.0
    scores["arima"] += 1.5 if changepoint_count <= 1.0 else (0.5 if changepoint_count <= 2.0 else -1.0)
    scores["arima"] += 1.0 if residual_exceed_ratio <= 0.04 else (-0.5 if residual_exceed_ratio >= 0.06 else 0.0)
    scores["arima"] += 0.5 if peak_count <= 3.0 else -0.5

    scores["patchtst"] += 1.5 if acf_seasonal >= 0.12 else 0.0
    scores["patchtst"] += 1.5 if 2.0 <= peak_count <= 4.0 else (0.5 if peak_count == 5.0 else -0.5)
    scores["patchtst"] += 1.0 if peak_spacing_cv <= 0.22 else 0.0
    scores["patchtst"] += 0.8 if changepoint_count <= 2.0 else 0.0
    scores["patchtst"] += 0.5 if residual_exceed_ratio <= 0.05 else 0.0

    scores["itransformer"] += 2.0 if changepoint_count >= 3.0 else (1.0 if changepoint_count >= 2.0 else 0.0)
    scores["itransformer"] += 1.2 if cusum_max >= 90.0 else (0.5 if cusum_max >= 70.0 else 0.0)
    scores["itransformer"] += 0.8 if monotone_duration >= 0.12 else 0.0
    scores["itransformer"] += 0.5 if acf_seasonal >= 0.18 else 0.0
    scores["itransformer"] += 0.5 if peak_count >= 4.0 else 0.0

    scores["chronos2"] += 2.0 if residual_exceed_ratio >= 0.06 else (1.0 if residual_exceed_ratio >= 0.05 else 0.0)
    scores["chronos2"] += 1.2 if peak_count >= 5.0 else 0.0
    scores["chronos2"] += 1.0 if peak_spacing_cv >= 0.30 else 0.0
    scores["chronos2"] += 0.8 if quality_saturation_ratio >= 0.05 or quality_quantization_score >= 0.16 else 0.0
    scores["chronos2"] += 0.5 if acf1 <= 0.88 else 0.0
    return scores


def _select_prediction_model_by_heuristic(history_values: Sequence[float]) -> tuple[str, dict[str, Any], str]:
    feature_snapshot = _compute_routing_feature_snapshot(history_values)
    acf1 = float(feature_snapshot["acf1"])
    acf_seasonal = float(feature_snapshot["acf_seasonal"])
    changepoint_count = float(feature_snapshot["changepoint_count"])
    residual_exceed_ratio = float(feature_snapshot["residual_exceed_ratio"])
    peak_count = float(feature_snapshot["peak_count"])
    peak_spacing_cv = float(feature_snapshot["peak_spacing_cv"])
    monotone_duration = float(feature_snapshot["monotone_duration"])
    cusum_max = float(feature_snapshot["cusum_max"])
    dominant_pattern = str(feature_snapshot["dominant_pattern"])
    quality_issue = (
        float(feature_snapshot["quality_saturation_ratio"]) >= 0.05
        or float(feature_snapshot["quality_quantization_score"]) >= 0.16
    )

    if residual_exceed_ratio >= 0.06 or quality_issue or (peak_count >= 5.0 and peak_spacing_cv >= 0.30):
        return (
            "chronos2",
            feature_snapshot,
            "The window is irregular or noisy, so I avoid brittle classical assumptions and prefer a robust foundation forecaster.",
        )
    if acf1 >= 0.93 and abs(acf_seasonal) <= 0.15 and changepoint_count <= 1.0 and residual_exceed_ratio <= 0.04 and peak_count <= 3.0:
        return (
            "arima",
            feature_snapshot,
            "The series remains stable with strong short-lag autocorrelation and limited structural change, so a linear seasonal forecaster is sufficient.",
        )
    if changepoint_count >= 3.0 or (cusum_max >= 90.0 and monotone_duration >= 0.10):
        return (
            "itransformer",
            feature_snapshot,
            "The series shows regime shifts and longer-range structural drift, so I prefer a model that can absorb global dependency changes.",
        )
    if acf_seasonal >= 0.12 and 2.0 <= peak_count <= 5.0 and peak_spacing_cv <= 0.25:
        return (
            "patchtst",
            feature_snapshot,
            "The window shows repeatable local seasonal motifs with reasonably regular peaks, so a patch-based temporal model is a good fit.",
        )

    scores = _heuristic_routing_scores(feature_snapshot)
    selected_model = max(
        scores.items(),
        key=lambda item: (
            float(item[1]),
            {"arima": 0, "patchtst": 1, "itransformer": 2, "chronos2": 3}[str(item[0])],
        ),
    )[0]
    fallback_reason_by_model = {
        "arima": "The diagnostics still look comparatively stable, so I choose the simplest structured forecaster.",
        "patchtst": "The diagnostics suggest local repeating motifs, so I choose the patch-based temporal model.",
        "itransformer": "The diagnostics indicate structural drift, so I choose the global-dependency forecaster.",
        "chronos2": "The diagnostics remain hard to explain with simple assumptions, so I choose the robust foundation forecaster.",
    }
    if dominant_pattern == "oscillation" and selected_model == "arima":
        selected_model = "chronos2"
    return selected_model, feature_snapshot, fallback_reason_by_model[selected_model]


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
    return format_predictions_to_string(pred_df, last_timestamp)


def build_feature_tool_results(values: list[float]) -> list[ToolResult]:
    return [
        ToolResult(tool_name=name, tool_output=builder(values))
        for name, builder in FEATURE_TOOL_BUILDERS
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


def _normalize_turn3_target_mode(turn3_target_mode: str | None) -> str:
    mode = str(turn3_target_mode or DEFAULT_TURN3_TARGET_MODE).strip().lower()
    if mode not in SUPPORTED_TURN3_TARGET_MODES:
        raise ValueError(
            f"Unsupported turn3_target_mode={turn3_target_mode!r}. "
            f"Expected one of {sorted(SUPPORTED_TURN3_TARGET_MODES)}."
        )
    return mode


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
    turn3_target_mode: str = DEFAULT_TURN3_TARGET_MODE,
) -> dict[str, Any]:
    turn3_target_mode = _normalize_turn3_target_mode(turn3_target_mode)
    base_values = _require_prediction_values(base_prediction_text, forecast_horizon, source_name=f"{model_name}_base")
    reward_model = sample.get("reward_model", {}) if isinstance(sample.get("reward_model"), dict) else {}
    ground_truth_values = _extract_prediction_values(str(reward_model.get("ground_truth", "") or ""))
    if len(ground_truth_values) < forecast_horizon:
        raise ValueError(
            f"ground_truth length must be at least forecast_horizon={forecast_horizon}, got {len(ground_truth_values)}"
        )
    ground_truth_values = ground_truth_values[:forecast_horizon]
    best_delta_summary = _summarize_refine_delta(base_values, base_values)
    if turn3_target_mode == TURN3_TARGET_MODE_PAPER_STRICT:
        return {
            "turn3_target_type": "validated_keep",
            "refine_ops": [],
            "refine_ops_signature": "none",
            "refine_gain_mse": 0.0,
            "refine_gain_mae": 0.0,
            "turn3_trigger_reason": "evidence_consistent",
            "base_teacher_prediction_text": _prediction_text_from_values(base_values),
            "refined_prediction_text": _prediction_text_from_values(base_values),
            **best_delta_summary,
        }

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


def _format_refine_ops_for_reflection(refine_ops: list[str]) -> str:
    if not refine_ops:
        return "keep the selected-model forecast unchanged"

    name_map = {
        "isolated_spike_smoothing": "smooth an isolated spike",
        "flat_tail_repair": "repair a flat tail",
        "local_level_adjust": "adjust a local level shift",
        "local_slope_adjust": "adjust a local slope change",
        "amplitude_clip": "clip an implausible amplitude excursion",
    }
    phrases = [name_map.get(op, op.replace("_", " ")) for op in refine_ops]
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _format_phrase_list(phrases: Sequence[str]) -> str:
    unique_phrases = [phrase for phrase in dict.fromkeys(phrases) if phrase]
    if not unique_phrases:
        return ""
    if len(unique_phrases) == 1:
        return unique_phrases[0]
    if len(unique_phrases) == 2:
        return f"{unique_phrases[0]} and {unique_phrases[1]}"
    return ", ".join(unique_phrases[:-1]) + f", and {unique_phrases[-1]}"


def _format_routing_metric_value(metric_name: str, metric_value: Any) -> str:
    if isinstance(metric_value, str):
        return metric_value
    try:
        value = float(metric_value)
    except Exception:
        return str(metric_value)
    if metric_name in {"Changepoint Count", "Peak Count"}:
        return f"{value:.1f}"
    return f"{value:.4f}"


def _build_routing_metric_items(
    *,
    history_values: Sequence[float],
    selected_feature_tools: Sequence[str],
) -> list[str]:
    metric_items: list[tuple[str, Any]] = []
    selected = set(selected_feature_tools or [])
    values = [float(value) for value in history_values]

    if "extract_basic_statistics" in selected:
        basic = extract_basic_statistics(values)
        metric_items.extend(
            [
                ("ACF(1)", basic.get("acf1", 0.0)),
                ("ACF(seasonal)", basic.get("acf_seasonal", 0.0)),
            ]
        )

    if "extract_within_channel_dynamics" in selected:
        dynamics = extract_within_channel_dynamics(values)
        metric_items.extend(
            [
                ("Changepoint Count", dynamics.get("changepoint_count", 0.0)),
                ("Peak Count", dynamics.get("peak_count", 0.0)),
            ]
        )

    if "extract_forecast_residuals" in selected:
        residuals = extract_forecast_residuals(values)
        metric_items.append(("Exceed Ratio", residuals.get("residual_exceed_ratio", 0.0)))

    if "extract_event_summary" in selected:
        events = extract_event_summary(values)
        dominant_idx = int(float(events.get("event_dominant_pattern", 0.0) or 0.0))
        dominant_idx = min(max(dominant_idx, 0), len(EVENT_PATTERN_NAMES) - 1)
        metric_items.append(("Dominant Pattern", EVENT_PATTERN_NAMES[dominant_idx]))

    return [
        f"{metric_name}={_format_routing_metric_value(metric_name, metric_value)}"
        for metric_name, metric_value in metric_items
    ]


def build_routing_reflection(
    *,
    model_name: str,
    history_values: Sequence[float],
    selected_feature_tools: Sequence[str],
    decision_reason: str = "",
) -> str:
    metric_items = _build_routing_metric_items(
        history_values=history_values,
        selected_feature_tools=selected_feature_tools,
    )
    if metric_items:
        reason_text = str(decision_reason or "").strip()
        if reason_text:
            reason_text = f"\n{reason_text}"
        return (
            f"Observed diagnostics: {'; '.join(metric_items)}.\n"
            f"Decision: {model_name}.\n"
            f"{reason_text.lstrip()}\n"
            f"I will call predict_time_series with {model_name}."
        )

    evidence_phrases = [
        FEATURE_TOOL_LABELS.get(tool_name, tool_name.replace("_", " "))
        for tool_name in selected_feature_tools
    ]
    evidence_text = _format_phrase_list(evidence_phrases)
    model_rationale = ROUTING_MODEL_RATIONALES.get(model_name, "the observed temporal structure")
    if evidence_text:
        return (
            f"Decision: {model_name}.\n"
            f"I reviewed the diagnostic evidence from {evidence_text}. "
            f"The observed series is most compatible with {model_rationale}, "
            f"so I will call predict_time_series with {model_name}."
        )
    return (
        f"Decision: {model_name}.\n"
        f"The observed series is most compatible with {model_rationale}, "
        f"so I will call predict_time_series with {model_name}."
    )


def build_final_answer(
    prediction_values: list[float],
    *,
    turn3_target_type: str,
    model_name: str,
    refine_ops: list[str],
) -> str:
    if turn3_target_type == "local_refine":
        reflection = (
            f"The selected {model_name} forecast is mostly consistent with the diagnostics, "
            f"but I apply a small local correction to {_format_refine_ops_for_reflection(refine_ops)} "
            "while preserving the overall trajectory."
        )
    else:
        reflection = (
            f"The selected {model_name} forecast is consistent with the diagnostic evidence, "
            "so I keep the forecast unchanged."
        )

    return (
        f"<think>\n{reflection}\n</think>\n"
        f"<answer>\n{_prediction_text_from_values(prediction_values)}\n</answer>"
    )


def _make_stage_record(
    *,
    shared_fields: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    turn_stage: str,
    turn_stage_order: int,
    trajectory_turn_count: int,
    current_required_feature_tools: list[str],
    completed_feature_tools_before_turn: list[str],
    history_analysis_count_before_turn: int,
    paper_turn3_required: bool,
    diagnostic_batch_index: int = -1,
) -> dict[str, Any]:
    return {
        **shared_fields,
        "messages": messages,
        "tools": copy.deepcopy(tools) if tools is not None else None,
        "turn_stage": turn_stage,
        "turn_stage_order": int(turn_stage_order),
        "trajectory_turn_count": int(trajectory_turn_count),
        "current_required_feature_tools": list(current_required_feature_tools),
        "completed_feature_tools_before_turn": list(completed_feature_tools_before_turn),
        "history_analysis_count_before_turn": int(history_analysis_count_before_turn),
        "paper_turn3_required": bool(paper_turn3_required),
        "diagnostic_batch_index": int(diagnostic_batch_index),
    }


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
            try:
                _require_prediction_values(
                    cached_teacher_prediction,
                    forecast_horizon,
                    source_name=f"{model_name}_cached_teacher",
                )
                return cached_teacher_prediction, "reference_teacher_cached"
            except Exception:
                pass
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


def build_sft_records(
    sample: dict[str, Any],
    *,
    turn3_target_mode: str = DEFAULT_TURN3_TARGET_MODE,
) -> list[dict[str, Any]]:
    turn3_target_mode = _normalize_turn3_target_mode(turn3_target_mode)
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
    feature_result_by_name = {result.tool_name: result for result in feature_results}
    diagnostic_tool_batches = plan_diagnostic_tool_batches(selected_feature_tools, max_parallel_calls=5)

    reference_teacher_model = _normalize_teacher_model(sample.get("reference_teacher_model"))
    second_teacher_model = _normalize_teacher_model(sample.get("teacher_eval_second_best_model"))
    selected_prediction_model, routing_feature_snapshot, routing_policy_reason = _select_prediction_model_by_heuristic(
        history_values
    )
    routing_policy_source = "heuristic_rule_based"

    def _resolve_selected_prediction(model_name: str) -> tuple[str, str]:
        return _resolve_prediction_text(
            sample=sample,
            historical_data=historical_data,
            data_source=data_source,
            target_column=target_column,
            forecast_horizon=forecast_horizon,
            model_name=model_name,
            allow_cached_reference=(model_name == reference_teacher_model),
        )

    try:
        base_prediction_text, base_prediction_source = _resolve_selected_prediction(selected_prediction_model)
    except Exception as primary_exc:
        fallback_models: list[str] = []
        for fallback_model in [reference_teacher_model, second_teacher_model]:
            fallback_model = _normalize_teacher_model(fallback_model)
            if fallback_model == selected_prediction_model or fallback_model in fallback_models:
                continue
            fallback_models.append(fallback_model)

        for fallback_model in fallback_models:
            try:
                base_prediction_text, base_prediction_source = _resolve_selected_prediction(fallback_model)
                selected_prediction_model = fallback_model
                routing_policy_source = "heuristic_runtime_fallback"
                routing_policy_reason = (
                    f"{routing_policy_reason} The planned model was unavailable during dataset construction, "
                    f"so I fallback to {fallback_model}."
                ).strip()
                break
            except Exception:
                continue
        else:
            raise primary_exc

    turn3_target = _build_turn3_target(
        sample=sample,
        history_values=history_values,
        base_prediction_text=base_prediction_text,
        forecast_horizon=forecast_horizon,
        model_name=selected_prediction_model,
        selected_feature_tools=selected_feature_tools,
        turn3_target_mode=turn3_target_mode,
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
    trajectory_turn_count = len(diagnostic_tool_batches) + 2
    source_sample_index = int(sample.get("index", -1))
    shared_fields = {
        "source_sample_index": source_sample_index,
        "source_uid": str(sample.get("uid") or ""),
        "data_source": data_source,
        "target_column": target_column,
        "forecast_horizon": forecast_horizon,
        "lookback_window": lookback_window,
        "reference_teacher_model": reference_teacher_model,
        "selected_prediction_model": selected_prediction_model,
        "routing_policy_source": routing_policy_source,
        "routing_policy_reason": routing_policy_reason,
        "base_prediction_source": base_prediction_source,
        "selected_feature_tools": list(selected_feature_tools),
        "selected_feature_tool_count": len(selected_feature_tools),
        "selected_feature_tool_signature": _feature_tool_signature(selected_feature_tools),
        "turn3_target_mode": turn3_target_mode,
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

    records: list[dict[str, Any]] = []
    history_analysis: list[str] = []
    completed_feature_tools: list[str] = []
    feature_call_serial = 1
    turn_stage_order = 0

    for batch_idx, batch_tool_names in enumerate(diagnostic_tool_batches):
        turn_prompt = build_runtime_user_prompt(
            data_source=data_source,
            target_column=target_column,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            time_series_data=historical_data,
            history_analysis=history_analysis,
            prediction_results=None,
            required_feature_tools=batch_tool_names,
            completed_feature_tools=completed_feature_tools,
            turn_stage="diagnostic",
        )
        feature_tool_calls = [
            _make_tool_call(tool_name=tool_name, arguments={}, call_id=f"call_{feature_call_serial + idx}_{tool_name}")
            for idx, tool_name in enumerate(batch_tool_names)
        ]
        diagnostic_tools = [copy.deepcopy(FEATURE_TOOL_SCHEMA_BY_NAME[name]) for name in batch_tool_names]
        records.append(
            _make_stage_record(
                shared_fields=shared_fields,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": turn_prompt},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": feature_tool_calls,
                    },
                ],
                tools=diagnostic_tools,
                turn_stage="diagnostic",
                turn_stage_order=turn_stage_order,
                trajectory_turn_count=trajectory_turn_count,
                current_required_feature_tools=list(batch_tool_names),
                completed_feature_tools_before_turn=list(completed_feature_tools),
                history_analysis_count_before_turn=len(history_analysis),
                paper_turn3_required=False,
                diagnostic_batch_index=batch_idx,
            )
        )
        for tool_name in batch_tool_names:
            result = feature_result_by_name[tool_name]
            history_analysis.append(result.tool_output)
            completed_feature_tools.append(tool_name)
        feature_call_serial += len(batch_tool_names)
        turn_stage_order += 1

    routing_prompt = build_runtime_user_prompt(
        data_source=data_source,
        target_column=target_column,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        time_series_data=historical_data,
        history_analysis=history_analysis,
        prediction_results=None,
        required_feature_tools=selected_feature_tools,
        completed_feature_tools=selected_feature_tools,
        turn_stage="routing",
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
        turn_stage="refinement",
    )

    prediction_tool_call = _make_tool_call(
        tool_name="predict_time_series",
        arguments={"model_name": selected_prediction_model},
        call_id="call_predict_time_series",
    )
    records.append(
        _make_stage_record(
            shared_fields=shared_fields,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": routing_prompt},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": build_routing_reflection(
                        model_name=selected_prediction_model,
                        history_values=history_values,
                        selected_feature_tools=selected_feature_tools,
                        decision_reason=routing_policy_reason,
                    ),
                    "tool_calls": [prediction_tool_call],
                },
            ],
            tools=[copy.deepcopy(PREDICT_TIMESERIES_TOOL_SCHEMA)],
            turn_stage="routing",
            turn_stage_order=turn_stage_order,
            trajectory_turn_count=trajectory_turn_count,
            current_required_feature_tools=list(selected_feature_tools),
            completed_feature_tools_before_turn=list(selected_feature_tools),
            history_analysis_count_before_turn=len(history_analysis),
            paper_turn3_required=False,
        )
    )
    turn_stage_order += 1
    records.append(
        _make_stage_record(
            shared_fields=shared_fields,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": turn_3_prompt},
                {
                    "role": "assistant",
                    "content": build_final_answer(
                        final_prediction_values,
                        turn3_target_type=turn3_target["turn3_target_type"],
                        model_name=selected_prediction_model,
                        refine_ops=turn3_target["refine_ops"],
                    ),
                },
            ],
            tools=None,
            turn_stage="refinement",
            turn_stage_order=turn_stage_order,
            trajectory_turn_count=trajectory_turn_count,
            current_required_feature_tools=list(selected_feature_tools),
            completed_feature_tools_before_turn=list(selected_feature_tools),
            history_analysis_count_before_turn=len(history_analysis),
            paper_turn3_required=True,
        )
    )
    return records


def build_sft_record(
    sample: dict[str, Any],
    *,
    turn3_target_mode: str = DEFAULT_TURN3_TARGET_MODE,
) -> dict[str, Any]:
    records = build_sft_records(sample, turn3_target_mode=turn3_target_mode)
    if not records:
        raise ValueError("build_sft_records returned no records")
    for record in records:
        if str(record.get("turn_stage") or "").strip().lower() == "refinement":
            compatible = dict(record)
            compatible["sample_index"] = int(sample.get("index", -1))
            return compatible
    compatible = dict(records[-1])
    compatible["sample_index"] = int(sample.get("index", -1))
    return compatible


def convert_jsonl_to_sft_parquet(
    *,
    input_path: str | Path,
    output_path: str | Path,
    max_samples: int = -1,
    turn3_target_mode: str = DEFAULT_TURN3_TARGET_MODE,
) -> pd.DataFrame:
    turn3_target_mode = _normalize_turn3_target_mode(turn3_target_mode)
    input_path = Path(input_path)
    output_path = Path(output_path)
    records: list[dict[str, Any]] = []
    source_sample_count = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if max_samples > 0 and source_sample_count >= max_samples:
                break
            if not line.strip():
                continue
            sample = json.loads(line)
            stage_records = build_sft_records(sample, turn3_target_mode=turn3_target_mode)
            for record in stage_records:
                row = dict(record)
                row["sample_index"] = len(records)
                records.append(row)
            source_sample_count += 1
            if (line_idx + 1) % 500 == 0:
                print(f"Processed {line_idx + 1} samples from {input_path}")

    dataframe = pd.DataFrame(records)
    _validate_paper_turn3_protocol(dataframe, split_name=input_path.stem, output_path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    checked_count = int(_summarize_paper_turn3_protocol(dataframe).get("turn3_protocol_checked_count", 0))
    print(
        f"Wrote {len(dataframe)} SFT records from {source_sample_count} source samples "
        f"({checked_count} refinement targets) to {output_path}"
    )
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

    group_column = "source_sample_index" if "source_sample_index" in dataframe.columns else "sample_index"
    if group_column not in dataframe.columns:
        return dataframe

    source_frame = _source_level_frame(dataframe)
    local_refine_mask = source_frame["turn3_target_type"] == "local_refine"
    validated_keep_mask = source_frame["turn3_target_type"] == "validated_keep"
    local_refine_df = source_frame.loc[local_refine_mask].copy()
    validated_keep_df = source_frame.loc[validated_keep_mask].copy()
    other_df = source_frame.loc[~(local_refine_mask | validated_keep_mask)].copy()

    local_count = len(local_refine_df)
    keep_count = len(validated_keep_df)
    if local_count <= 0 or keep_count <= 0:
        return dataframe

    max_keep_count = int(np.floor(local_count * (1.0 - min_local_refine_ratio) / min_local_refine_ratio))
    max_keep_count = max(max_keep_count, 0)
    if keep_count <= max_keep_count:
        return dataframe

    validated_keep_df = validated_keep_df.sort_values(group_column).reset_index(drop=True)
    if max_keep_count <= 0:
        rebalanced_keep_df = validated_keep_df.iloc[0:0].copy()
    else:
        take_positions = np.linspace(0, keep_count - 1, num=max_keep_count, dtype=int)
        rebalanced_keep_df = validated_keep_df.iloc[take_positions].copy()

    kept_source_ids = set(local_refine_df[group_column].tolist())
    kept_source_ids.update(rebalanced_keep_df[group_column].tolist())
    kept_source_ids.update(other_df[group_column].tolist())

    balanced = dataframe.loc[dataframe[group_column].isin(kept_source_ids)].copy()
    sort_columns = [col for col in (group_column, "turn_stage_order", "sample_index") if col in balanced.columns]
    if sort_columns:
        balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return balanced


def _rebalance_train_stage_records(
    dataframe: pd.DataFrame,
    *,
    stage_repeat_factors: dict[str, int] | None,
) -> pd.DataFrame:
    if dataframe.empty or not stage_repeat_factors:
        return dataframe
    if "turn_stage" not in dataframe.columns:
        return dataframe

    normalized_factors = {
        str(stage).strip().lower(): max(1, int(factor))
        for stage, factor in stage_repeat_factors.items()
        if str(stage).strip()
    }
    if not normalized_factors:
        return dataframe

    repeated_frames: list[pd.DataFrame] = []
    for _, row in dataframe.iterrows():
        stage = str(row.get("turn_stage") or "").strip().lower()
        repeat_factor = normalized_factors.get(stage, 1)
        row_frame = pd.DataFrame([row.to_dict()])
        repeated_frames.extend(row_frame.copy() for _ in range(max(1, repeat_factor)))

    balanced = pd.concat(repeated_frames, ignore_index=True)
    sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in balanced.columns]
    if sort_columns:
        balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if "sample_index" in balanced.columns:
        balanced["sample_index"] = list(range(len(balanced)))
    return balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ETTh1 OT step-wise runtime-aligned SFT parquet from teacher-curated jsonl samples."
    )
    parser.add_argument(
        "--train-jsonl",
        default=str(DEFAULT_TRAIN_JSONL),
        help="Teacher-curated train jsonl path.",
    )
    parser.add_argument(
        "--val-jsonl",
        default=str(DEFAULT_VAL_JSONL),
        help="Teacher-curated val jsonl path.",
    )
    parser.add_argument(
        "--test-jsonl",
        default=str(DEFAULT_TEST_JSONL),
        help="Optional teacher-curated test jsonl path.",
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
        "--turn3-target-mode",
        choices=sorted(SUPPORTED_TURN3_TARGET_MODES),
        default=DEFAULT_TURN3_TARGET_MODE,
        help="Turn-3 target generation mode. Default keeps the selected forecast unchanged for paper-aligned SFT.",
    )
    parser.add_argument(
        "--train-min-local-refine-ratio",
        type=float,
        default=0.30,
        help="Minimum desired local_refine ratio in train parquet. Set <=0 to disable train rebalancing.",
    )
    parser.add_argument(
        "--train-stage-repeat-factors",
        default=json.dumps(DEFAULT_TRAIN_STAGE_REPEAT_FACTORS, ensure_ascii=False),
        help=(
            "JSON object of train-only stage repeat factors, e.g. "
            '\'{"diagnostic":1,"routing":1,"refinement":1}\'. '
            "Use {} to disable stage reweighting."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    turn3_target_mode = _normalize_turn3_target_mode(args.turn3_target_mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_train_min_local_refine_ratio = (
        0.0
        if turn3_target_mode == TURN3_TARGET_MODE_PAPER_STRICT
        else float(args.train_min_local_refine_ratio)
    )
    raw_train_stage_repeat_factors = json.loads(str(args.train_stage_repeat_factors))
    train_stage_repeat_factors = {
        str(stage).strip().lower(): max(1, int(factor))
        for stage, factor in dict(raw_train_stage_repeat_factors).items()
        if str(stage).strip()
    }

    source_metadata_paths: list[Path] = []
    for split_path in (Path(args.train_jsonl), Path(args.val_jsonl), Path(args.test_jsonl)):
        if not split_path.exists():
            continue
        _, source_metadata_path = validate_sibling_metadata(
            split_path,
            expected_kind=(DATASET_KIND_TEACHER_CURATED_SFT, DATASET_KIND_RUNTIME_SFT_PARQUET),
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
        turn3_target_mode=turn3_target_mode,
    )
    train_df = _rebalance_train_turn3_targets(
        train_df_raw,
        min_local_refine_ratio=effective_train_min_local_refine_ratio,
    )
    train_df = _rebalance_train_stage_records(
        train_df,
        stage_repeat_factors=train_stage_repeat_factors,
    )
    if len(train_df) != len(train_df_raw) or not train_df.equals(train_df_raw):
        _validate_paper_turn3_protocol(train_df, split_name="train", output_path=output_dir / "train.parquet")
        train_df.to_parquet(output_dir / "train.parquet", index=False)
        print(
            f"Rebalanced train.parquet turn3_target_type distribution: "
            f"{_distribution_from_series(train_df['turn3_target_type'])}"
        )
        print(
            f"Rebalanced train.parquet turn_stage distribution: "
            f"{_distribution_from_series(train_df['turn_stage'])}"
        )
    val_df = convert_jsonl_to_sft_parquet(
        input_path=args.val_jsonl,
        output_path=output_dir / "val.parquet",
        max_samples=args.max_val_samples,
        turn3_target_mode=turn3_target_mode,
    )

    test_count = 0
    test_df: pd.DataFrame | None = None
    test_path = Path(args.test_jsonl)
    if test_path.exists():
        test_df = convert_jsonl_to_sft_parquet(
            input_path=test_path,
            output_path=output_dir / "test.parquet",
            max_samples=args.max_test_samples,
            turn3_target_mode=turn3_target_mode,
        )
        test_count = len(test_df)

    train_protocol_summary = _summarize_paper_turn3_protocol(train_df)
    val_protocol_summary = _summarize_paper_turn3_protocol(val_df)
    test_protocol_summary = _summarize_paper_turn3_protocol(test_df) if test_df is not None and len(test_df) > 0 else {}

    metadata_kwargs = dict(
        dataset_kind=DATASET_KIND_RUNTIME_SFT_PARQUET,
        pipeline_stage="runtime_stepwise_sft",
        turn3_protocol="paper_think_answer_xml",
        turn3_target_mode=turn3_target_mode,
        train_samples_before_balance=len(train_df_raw),
        train_samples=len(train_df),
        train_source_samples_before_balance=len(_source_level_frame(train_df_raw)),
        train_source_samples=len(_source_level_frame(train_df)),
        val_samples=len(val_df),
        val_source_samples=len(_source_level_frame(val_df)),
        test_samples=test_count,
        test_source_samples=len(_source_level_frame(test_df)) if test_df is not None else 0,
        requested_train_min_local_refine_ratio=float(args.train_min_local_refine_ratio),
        train_min_local_refine_ratio=effective_train_min_local_refine_ratio,
        train_stage_repeat_factors=train_stage_repeat_factors,
        source_train_jsonl=str(Path(args.train_jsonl)),
        source_val_jsonl=str(Path(args.val_jsonl)),
        source_test_jsonl=str(test_path),
        source_curated_metadata_path=str(source_metadata_paths[0]) if source_metadata_paths else "",
        train_turn_stage_distribution=_distribution_from_series(train_df["turn_stage"]),
        val_turn_stage_distribution=_distribution_from_series(val_df["turn_stage"]),
        train_turn3_target_type_distribution_before_balance=_distribution_from_series(
            _source_level_frame(train_df_raw)["turn3_target_type"]
        ),
        train_reference_teacher_model_distribution=_distribution_from_series(
            _source_level_frame(train_df)["reference_teacher_model"]
        ),
        train_selected_prediction_model_distribution=_distribution_from_series(
            _source_level_frame(train_df)["selected_prediction_model"]
        ),
        train_turn3_target_type_distribution=_distribution_from_series(_source_level_frame(train_df)["turn3_target_type"]),
        train_turn3_trigger_reason_distribution=_distribution_from_series(_source_level_frame(train_df)["turn3_trigger_reason"]),
        train_refine_ops_signature_distribution=_distribution_from_series(_source_level_frame(train_df)["refine_ops_signature"]),
        train_selected_feature_tool_signature_distribution=_distribution_from_series(
            _source_level_frame(train_df)["selected_feature_tool_signature"]
        ),
        train_base_prediction_source_distribution=_distribution_from_series(
            _source_level_frame(train_df)["base_prediction_source"]
        ),
        train_turn3_protocol_checked_count=int(train_protocol_summary.get("turn3_protocol_checked_count", 0)),
        train_turn3_protocol_skipped_count=int(train_protocol_summary.get("turn3_protocol_skipped_count", 0)),
        train_turn3_protocol_valid_count=int(train_protocol_summary.get("turn3_protocol_valid_count", 0)),
        train_turn3_protocol_invalid_count=int(train_protocol_summary.get("turn3_protocol_invalid_count", 0)),
        train_turn3_protocol_valid_ratio=float(train_protocol_summary.get("turn3_protocol_valid_ratio", 0.0)),
        train_turn3_protocol_reason_distribution=dict(train_protocol_summary.get("turn3_protocol_reason_distribution", {})),
        val_reference_teacher_model_distribution=_distribution_from_series(
            _source_level_frame(val_df)["reference_teacher_model"]
        ),
        val_selected_prediction_model_distribution=_distribution_from_series(
            _source_level_frame(val_df)["selected_prediction_model"]
        ),
        val_turn3_target_type_distribution=_distribution_from_series(_source_level_frame(val_df)["turn3_target_type"]),
        val_turn3_trigger_reason_distribution=_distribution_from_series(_source_level_frame(val_df)["turn3_trigger_reason"]),
        val_refine_ops_signature_distribution=_distribution_from_series(_source_level_frame(val_df)["refine_ops_signature"]),
        val_selected_feature_tool_signature_distribution=_distribution_from_series(
            _source_level_frame(val_df)["selected_feature_tool_signature"]
        ),
        val_base_prediction_source_distribution=_distribution_from_series(
            _source_level_frame(val_df)["base_prediction_source"]
        ),
        val_turn3_protocol_checked_count=int(val_protocol_summary.get("turn3_protocol_checked_count", 0)),
        val_turn3_protocol_skipped_count=int(val_protocol_summary.get("turn3_protocol_skipped_count", 0)),
        val_turn3_protocol_valid_count=int(val_protocol_summary.get("turn3_protocol_valid_count", 0)),
        val_turn3_protocol_invalid_count=int(val_protocol_summary.get("turn3_protocol_invalid_count", 0)),
        val_turn3_protocol_valid_ratio=float(val_protocol_summary.get("turn3_protocol_valid_ratio", 0.0)),
        val_turn3_protocol_reason_distribution=dict(val_protocol_summary.get("turn3_protocol_reason_distribution", {})),
    )
    if test_df is not None and len(test_df) > 0:
        metadata_kwargs.update(
            test_turn_stage_distribution=_distribution_from_series(test_df["turn_stage"]),
            test_reference_teacher_model_distribution=_distribution_from_series(
                _source_level_frame(test_df)["reference_teacher_model"]
            ),
            test_selected_prediction_model_distribution=_distribution_from_series(
                _source_level_frame(test_df)["selected_prediction_model"]
            ),
            test_turn3_target_type_distribution=_distribution_from_series(
                _source_level_frame(test_df)["turn3_target_type"]
            ),
            test_turn3_trigger_reason_distribution=_distribution_from_series(
                _source_level_frame(test_df)["turn3_trigger_reason"]
            ),
            test_refine_ops_signature_distribution=_distribution_from_series(
                _source_level_frame(test_df)["refine_ops_signature"]
            ),
            test_selected_feature_tool_signature_distribution=_distribution_from_series(
                _source_level_frame(test_df)["selected_feature_tool_signature"]
            ),
            test_base_prediction_source_distribution=_distribution_from_series(
                _source_level_frame(test_df)["base_prediction_source"]
            ),
            test_turn3_protocol_checked_count=int(test_protocol_summary.get("turn3_protocol_checked_count", 0)),
            test_turn3_protocol_skipped_count=int(test_protocol_summary.get("turn3_protocol_skipped_count", 0)),
            test_turn3_protocol_valid_count=int(test_protocol_summary.get("turn3_protocol_valid_count", 0)),
            test_turn3_protocol_invalid_count=int(test_protocol_summary.get("turn3_protocol_invalid_count", 0)),
            test_turn3_protocol_valid_ratio=float(test_protocol_summary.get("turn3_protocol_valid_ratio", 0.0)),
            test_turn3_protocol_reason_distribution=dict(test_protocol_summary.get("turn3_protocol_reason_distribution", {})),
        )

    _write_metadata(
        output_dir,
        **metadata_kwargs,
    )


if __name__ == "__main__":
    main()
