from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from verl.utils.chain_debug import short_text

from recipe.time_series_forecast.diagnostic_policy import FEATURE_TOOL_ORDER


def expected_prediction_count(forecast_horizon: Any) -> int:
    return int(forecast_horizon or 96)


def current_turn_stage(
    *,
    prediction_results: Any,
    executed_feature_tool_names: list[str],
    required_feature_tool_names: list[str] | None = None,
) -> str:
    if prediction_results is not None:
        return "refinement"
    required_names = normalize_required_feature_tool_names(
        list(required_feature_tool_names or []),
        list(executed_feature_tool_names or []),
    )
    if required_names:
        executed = set(executed_feature_tool_names or [])
        if any(name not in executed for name in required_names):
            return "diagnostic"
        return "routing"
    if executed_feature_tool_names:
        return "routing"
    return "diagnostic"


def required_step_budget(
    *,
    absolute_step_budget: Any,
    configured_max_steps: Any,
    max_prediction_attempts: int,
) -> int:
    if absolute_step_budget is not None:
        return max(int(absolute_step_budget), 1)

    try:
        configured = int(configured_max_steps or 0)
    except (TypeError, ValueError):
        configured = 0

    return max(configured, 1 + max_prediction_attempts + 1, 1)


def normalize_required_feature_tool_names(
    required_feature_tools: list[str],
    executed_feature_tool_names: list[str],
) -> list[str]:
    required = [str(name) for name in required_feature_tools if str(name).strip()]
    if not required:
        return list(executed_feature_tool_names)
    return [name for name in FEATURE_TOOL_ORDER if name in required] or required


def feature_tool_signature(feature_tool_sequence: list[str]) -> str:
    if not feature_tool_sequence:
        return "none"
    return "->".join(feature_tool_sequence)


def analysis_state_signature(executed_feature_tool_names: list[str]) -> str:
    return "|".join(executed_feature_tool_names) if executed_feature_tool_names else "none"


def analysis_coverage_ratio(
    executed_feature_tool_names: list[str],
    required_feature_tool_names: list[str] | None = None,
) -> float:
    required_names = normalize_required_feature_tool_names(
        list(required_feature_tool_names or []),
        list(executed_feature_tool_names or []),
    )
    if not required_names:
        return 1.0 if executed_feature_tool_names else 0.0
    executed = set(executed_feature_tool_names or [])
    covered = sum(1 for name in required_names if name in executed)
    return float(covered / max(len(required_names), 1))


def sample_uid_text(sample_uid: Any) -> str:
    if sample_uid is None:
        return ""
    return str(sample_uid)


def series_preview(values: list[float], preview_size: int = 3) -> str:
    if not values:
        return ""
    head = ", ".join(f"{float(value):.4f}" for value in values[:preview_size])
    if len(values) <= preview_size:
        return head
    tail = ", ".join(f"{float(value):.4f}" for value in values[-preview_size:])
    return f"{head} ... {tail}"


def finite_or_nan(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(numeric) or np.isinf(numeric):
        return float("nan")
    return numeric


def shared_reward_tracking_fields(
    *,
    sample_uid: Any,
    prediction_attempt_count: int,
    prediction_call_count: int,
    illegal_turn3_tool_call_count: int,
    prediction_requested_model: str,
    prediction_model_defaulted: bool,
    prediction_tool_error: str,
    prediction_step_index: Any,
    prediction_turn_stage: str,
    final_answer_step_index: Any,
    feature_tool_sequence: list[str],
    required_feature_tools: list[str],
    executed_feature_tool_names: list[str],
    history_analysis: list[str],
    required_step_budget: int,
) -> dict[str, Any]:
    required_names = normalize_required_feature_tool_names(required_feature_tools, executed_feature_tool_names)
    missing_required = [name for name in required_names if name not in set(executed_feature_tool_names)]
    return {
        "sample_uid": sample_uid_text(sample_uid),
        "prediction_attempt_count": int(prediction_attempt_count),
        "prediction_call_count": int(prediction_call_count),
        "illegal_turn3_tool_call_count": int(illegal_turn3_tool_call_count),
        "prediction_requested_model": prediction_requested_model or "",
        "prediction_model_defaulted": bool(prediction_model_defaulted),
        "prediction_tool_error": prediction_tool_error or "",
        "prediction_step_index": int(prediction_step_index or 0),
        "prediction_turn_stage": prediction_turn_stage or "",
        "final_answer_step_index": int(final_answer_step_index or 0),
        "feature_tool_count": int(len(feature_tool_sequence)),
        "feature_tool_signature": feature_tool_signature(feature_tool_sequence),
        "required_feature_tool_signature": "->".join(required_names) if required_names else "none",
        "required_feature_tool_count": int(len(required_names)),
        "missing_required_feature_tool_count": int(len(missing_required)),
        "analysis_state_signature": analysis_state_signature(executed_feature_tool_names),
        "analysis_coverage_ratio": analysis_coverage_ratio(
            executed_feature_tool_names,
            required_names,
        ),
        "history_analysis_count": int(len(history_analysis)),
        "required_step_budget": int(required_step_budget),
    }


def compute_series_metrics(
    candidate_values: list[float],
    reference_values: list[float],
    *,
    normalize_for_reward_fn: Callable[[list[float], list[float]], tuple[list[float], list[float]]],
) -> tuple[float, float, float, float]:
    if not candidate_values or not reference_values:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    min_len = min(len(candidate_values), len(reference_values))
    if min_len <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    candidate_slice = candidate_values[:min_len]
    reference_slice = reference_values[:min_len]
    candidate_arr = np.asarray(candidate_slice, dtype=float)
    reference_arr = np.asarray(reference_slice, dtype=float)

    orig_mse = float(np.mean((candidate_arr - reference_arr) ** 2))
    orig_mae = float(np.mean(np.abs(candidate_arr - reference_arr)))
    norm_candidate, norm_reference = normalize_for_reward_fn(candidate_slice, reference_slice)
    norm_candidate_arr = np.asarray(norm_candidate, dtype=float)
    norm_reference_arr = np.asarray(norm_reference, dtype=float)
    norm_mse = float(np.mean((norm_candidate_arr - norm_reference_arr) ** 2))
    norm_mae = float(np.mean(np.abs(norm_candidate_arr - norm_reference_arr)))
    return (orig_mse, orig_mae, norm_mse, norm_mae)


def collect_refinement_metrics(
    *,
    ground_truth: str,
    prediction_results: str | None,
    final_answer: str | None,
    extract_values_fn: Callable[[str], list[float]],
    extract_ground_truth_values_fn: Callable[[str], list[float]],
    normalize_for_reward_fn: Callable[[list[float], list[float]], tuple[list[float], list[float]]],
) -> dict[str, Any]:
    gt_values = extract_ground_truth_values_fn(ground_truth) if ground_truth else []
    selected_values = extract_values_fn(prediction_results or "") if prediction_results else []
    final_values = extract_values_fn(final_answer or "") if final_answer else []

    selected_orig_mse, selected_orig_mae, selected_norm_mse, selected_norm_mae = compute_series_metrics(
        selected_values,
        gt_values,
        normalize_for_reward_fn=normalize_for_reward_fn,
    )
    final_vs_selected_mse, final_vs_selected_mae, final_vs_selected_norm_mse, final_vs_selected_norm_mae = (
        compute_series_metrics(
            final_values,
            selected_values,
            normalize_for_reward_fn=normalize_for_reward_fn,
        )
    )

    refinement_delta_orig_mse = float("nan")
    refinement_delta_orig_mae = float("nan")
    if not np.isnan(selected_orig_mse) and final_answer and ground_truth:
        final_orig_mse, final_orig_mae, _, _ = compute_series_metrics(
            final_values,
            gt_values,
            normalize_for_reward_fn=normalize_for_reward_fn,
        )
        if not np.isnan(final_orig_mse):
            refinement_delta_orig_mse = float(selected_orig_mse - final_orig_mse)
        if not np.isnan(final_orig_mae):
            refinement_delta_orig_mae = float(selected_orig_mae - final_orig_mae)

    refinement_changed = False
    refinement_change_mean_abs = float("nan")
    refinement_change_max_abs = float("nan")
    selected_forecast_len_match = bool(selected_values and final_values and len(selected_values) == len(final_values))
    selected_forecast_exact_copy = False
    refinement_compare_len = 0
    refinement_changed_value_count = 0
    refinement_first_changed_index = -1
    if selected_values and final_values:
        refinement_compare_len = min(len(selected_values), len(final_values))
        if refinement_compare_len > 0:
            abs_deltas = [abs(selected_values[idx] - final_values[idx]) for idx in range(refinement_compare_len)]
            refinement_change_mean_abs = float(np.mean(abs_deltas))
            refinement_change_max_abs = float(np.max(abs_deltas))
            changed_positions = [
                idx
                for idx in range(refinement_compare_len)
                if abs(selected_values[idx] - final_values[idx]) > 1e-8
            ]
            refinement_changed_value_count = int(len(changed_positions))
            if changed_positions:
                refinement_first_changed_index = int(changed_positions[0])
            refinement_changed = bool(changed_positions)
            if len(selected_values) != len(final_values):
                refinement_changed = True
                refinement_change_max_abs = max(refinement_change_max_abs, 0.0)
            selected_forecast_exact_copy = bool(
                len(selected_values) == len(final_values) and refinement_changed_value_count == 0
            )

    refinement_improved = bool(not np.isnan(refinement_delta_orig_mse) and refinement_delta_orig_mse > 1e-8)
    refinement_degraded = bool(not np.isnan(refinement_delta_orig_mse) and refinement_delta_orig_mse < -1e-8)

    return {
        "selected_forecast_pred_len": int(len(selected_values)),
        "selected_forecast_orig_mse": selected_orig_mse,
        "selected_forecast_orig_mae": selected_orig_mae,
        "selected_forecast_norm_mse": selected_norm_mse,
        "selected_forecast_norm_mae": selected_norm_mae,
        "selected_forecast_preview": series_preview(selected_values),
        "selected_forecast_len_match": bool(selected_forecast_len_match),
        "selected_forecast_exact_copy": bool(selected_forecast_exact_copy),
        "final_vs_selected_mse": final_vs_selected_mse,
        "final_vs_selected_mae": final_vs_selected_mae,
        "final_vs_selected_norm_mse": final_vs_selected_norm_mse,
        "final_vs_selected_norm_mae": final_vs_selected_norm_mae,
        "refinement_delta_orig_mse": refinement_delta_orig_mse,
        "refinement_delta_orig_mae": refinement_delta_orig_mae,
        "refinement_compare_len": int(refinement_compare_len),
        "refinement_changed_value_count": int(refinement_changed_value_count),
        "refinement_first_changed_index": int(refinement_first_changed_index),
        "refinement_change_mean_abs": refinement_change_mean_abs,
        "refinement_change_max_abs": refinement_change_max_abs,
        "refinement_changed": bool(refinement_changed),
        "refinement_improved": refinement_improved,
        "refinement_degraded": refinement_degraded,
        "final_answer_preview": series_preview(final_values),
    }


def build_turn_debug_payload(
    *,
    request_id: str,
    sample_index: Any,
    sample_uid: Any,
    step_index: int,
    turn_stage: str,
    tool_call_names: list[str],
    prompt_text: str,
    response_text: str,
    generation_stop_reason: str,
    generation_finish_reason: str,
    workflow_status: str,
    workflow_message: str,
    reward_extra_info: dict[str, Any],
    feature_tool_sequence: list[str],
    required_feature_tools: list[str],
    executed_feature_tool_names: list[str],
    history_analysis: list[str],
    prediction_requested_model: str,
    prediction_model_used: str,
    prediction_model_defaulted: bool,
    prediction_tool_error: str,
    prediction_attempt_count: int,
    prediction_call_count: int,
    prediction_step_index: Any,
    prediction_turn_stage: str,
    final_answer_step_index: Any,
    illegal_turn3_tool_call_count: int,
    final_answer_reject_reason: str,
    final_answer_parse_mode: str,
    required_step_budget: int,
) -> dict[str, Any]:
    include_full_text = str(os.getenv("TS_CHAIN_DEBUG_INCLUDE_TEXT", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    required_names = normalize_required_feature_tool_names(required_feature_tools, executed_feature_tool_names)
    missing_required = [name for name in required_names if name not in set(executed_feature_tool_names)]
    payload = {
        "request_id": request_id,
        "sample_index": sample_index,
        "sample_uid": sample_uid_text(sample_uid or reward_extra_info.get("sample_uid")),
        "step_index": int(step_index),
        "turn_stage": turn_stage,
        "tool_call_sequence": tool_call_names,
        "tool_call_count": int(len(tool_call_names)),
        "executed_tool_count": int(reward_extra_info.get("executed_tool_count", 0) or 0),
        "executed_tool_sequence": reward_extra_info.get("executed_tool_sequence", ""),
        "rejected_tool_call_count": int(reward_extra_info.get("rejected_tool_call_count", 0) or 0),
        "feature_tool_signature": reward_extra_info.get(
            "feature_tool_signature",
            feature_tool_signature(feature_tool_sequence),
        ),
        "feature_tool_count": int(reward_extra_info.get("feature_tool_count", len(feature_tool_sequence)) or 0),
        "required_feature_tool_signature": reward_extra_info.get(
            "required_feature_tool_signature",
            "->".join(required_names) if required_names else "none",
        ),
        "required_feature_tool_count": int(
            reward_extra_info.get("required_feature_tool_count", len(required_names)) or 0
        ),
        "missing_required_feature_tool_count": int(
            reward_extra_info.get("missing_required_feature_tool_count", len(missing_required)) or 0
        ),
        "analysis_state_signature": reward_extra_info.get(
            "analysis_state_signature",
            analysis_state_signature(executed_feature_tool_names),
        ),
        "analysis_coverage_ratio": reward_extra_info.get(
            "analysis_coverage_ratio",
            analysis_coverage_ratio(executed_feature_tool_names, required_names),
        ),
        "history_analysis_count": int(reward_extra_info.get("history_analysis_count", len(history_analysis)) or 0),
        "prediction_requested_model": prediction_requested_model or "",
        "prediction_model_used": prediction_model_used or "",
        "prediction_model_defaulted": bool(prediction_model_defaulted),
        "prediction_tool_error": prediction_tool_error or "",
        "prediction_attempt_count": int(prediction_attempt_count),
        "prediction_call_count": int(prediction_call_count),
        "prediction_step_index": int(prediction_step_index or 0),
        "prediction_turn_stage": prediction_turn_stage or "",
        "final_answer_step_index": int(reward_extra_info.get("final_answer_step_index", final_answer_step_index or 0) or 0),
        "illegal_turn3_tool_call_count": int(illegal_turn3_tool_call_count),
        "required_step_budget": int(reward_extra_info.get("required_step_budget", required_step_budget)),
        "workflow_status": workflow_status,
        "workflow_violation_reason": workflow_message,
        "generation_stop_reason": generation_stop_reason,
        "generation_finish_reason": generation_finish_reason,
        "final_answer_reject_reason": final_answer_reject_reason or "",
        "final_answer_parse_mode": final_answer_parse_mode or "",
        "prompt_char_len": int(len(prompt_text)),
        "response_char_len": int(len(response_text)),
        "response_line_count": int(len([line for line in response_text.splitlines() if line.strip()])),
        "selected_forecast_preview": reward_extra_info.get("selected_forecast_preview", ""),
        "final_answer_preview": reward_extra_info.get("final_answer_preview", ""),
        "selected_forecast_orig_mse": finite_or_nan(reward_extra_info.get("selected_forecast_orig_mse")),
        "final_vs_selected_mse": finite_or_nan(reward_extra_info.get("final_vs_selected_mse")),
        "refinement_delta_orig_mse": finite_or_nan(reward_extra_info.get("refinement_delta_orig_mse")),
        "selected_forecast_len_match": bool(reward_extra_info.get("selected_forecast_len_match", False)),
        "selected_forecast_exact_copy": bool(reward_extra_info.get("selected_forecast_exact_copy", False)),
        "refinement_compare_len": int(reward_extra_info.get("refinement_compare_len", 0) or 0),
        "refinement_changed_value_count": int(reward_extra_info.get("refinement_changed_value_count", 0) or 0),
        "refinement_first_changed_index": int(reward_extra_info.get("refinement_first_changed_index", -1) or -1),
        "refinement_changed": bool(reward_extra_info.get("refinement_changed", False)),
        "refinement_improved": bool(reward_extra_info.get("refinement_improved", False)),
        "refinement_degraded": bool(reward_extra_info.get("refinement_degraded", False)),
        "raw_response_tail": short_text("\n".join(response_text.splitlines()[-10:]), limit=400),
    }
    if include_full_text:
        payload["prompt_text"] = prompt_text
        payload["response_text"] = response_text
    return payload


def build_prediction_tool_debug_payload(
    *,
    model_name: str,
    prediction_requested_model: str,
    prediction_model_defaulted: bool,
    prediction_attempt_count: int,
    prediction_step_index: Any,
    prediction_call_count: int,
    analysis_state_signature_value: str,
    feature_tool_signature_value: str,
    forecast_horizon: Any,
    prediction_results: str,
    extract_values_fn: Callable[[str], list[float]],
    success: bool,
    error: str,
) -> dict[str, Any]:
    preview = ""
    if success and prediction_results:
        preview = series_preview(extract_values_fn(prediction_results))
    return {
        "model_name": model_name,
        "prediction_requested_model": prediction_requested_model or "",
        "prediction_model_defaulted": bool(prediction_model_defaulted),
        "prediction_attempt_count": int(prediction_attempt_count),
        "prediction_step_index": int(prediction_step_index or 0),
        "prediction_call_count": int(prediction_call_count),
        "analysis_state_signature": analysis_state_signature_value,
        "feature_tool_signature": feature_tool_signature_value,
        "prediction_length": int(forecast_horizon or 0),
        "prediction_preview": preview,
        "success": bool(success),
        "error": error,
    }


__all__ = [
    "analysis_coverage_ratio",
    "analysis_state_signature",
    "build_prediction_tool_debug_payload",
    "build_turn_debug_payload",
    "collect_refinement_metrics",
    "compute_series_metrics",
    "current_turn_stage",
    "expected_prediction_count",
    "feature_tool_signature",
    "finite_or_nan",
    "normalize_required_feature_tool_names",
    "required_step_budget",
    "sample_uid_text",
    "series_preview",
    "shared_reward_tracking_fields",
]
