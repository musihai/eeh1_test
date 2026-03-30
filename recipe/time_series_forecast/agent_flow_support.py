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


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def summarize_debug_diagnosis(
    *,
    reward_score: Any = None,
    workflow_status: str = "",
    workflow_message: str = "",
    format_failure_reason: str = "",
    final_answer_reject_reason: str = "",
    prediction_tool_error: str = "",
    prediction_call_count: Any = None,
    illegal_turn3_tool_call_count: Any = None,
    missing_required_feature_tool_count: Any = None,
    selected_forecast_exact_copy: bool = False,
    refinement_degraded: bool = False,
    prediction_model_defaulted: bool = False,
) -> dict[str, Any]:
    score = finite_or_nan(reward_score)
    prediction_calls = safe_int(prediction_call_count, default=-1)
    illegal_calls = safe_int(illegal_turn3_tool_call_count, default=0)
    missing_required = safe_int(missing_required_feature_tool_count, default=0)
    workflow_status_text = str(workflow_status or "").strip().lower()
    workflow_message_text = str(workflow_message or "").strip().lower()
    reject_reason = str(final_answer_reject_reason or "").strip()
    format_reason = str(format_failure_reason or "").strip()
    tool_error = str(prediction_tool_error or "").strip()

    if tool_error:
        return {
            "debug_bucket": "tool_error",
            "debug_reason": "prediction_tool_error",
            "debug_severity": 100,
        }

    if workflow_status_text == "rejected":
        if "copy input" in workflow_message_text or "future timestamps" in workflow_message_text:
            reason = "copied_input"
        elif illegal_calls > 0:
            reason = "illegal_refinement_tool_call"
        elif prediction_calls >= 0 and prediction_calls != 1:
            reason = "prediction_call_count_invalid"
        elif missing_required > 0 or "diagnostic" in workflow_message_text:
            reason = "missing_required_diagnostics"
        else:
            reason = "workflow_rejected"
        return {
            "debug_bucket": "workflow_violation",
            "debug_reason": reason,
            "debug_severity": 95,
        }

    primary_reason = reject_reason or (format_reason if format_reason and format_reason != "ok" else "")
    if primary_reason:
        if primary_reason.startswith("missing_answer_close_tag"):
            reason = "missing_answer_close_tag"
        elif primary_reason.startswith("missing_answer_block"):
            reason = "missing_answer_block"
        elif primary_reason.startswith("missing_think"):
            reason = "missing_think_block"
        elif primary_reason.startswith("extra_text_outside_tags"):
            reason = "extra_text_outside_tags"
        elif primary_reason.startswith("invalid_answer_shape"):
            reason = "invalid_answer_shape"
        elif primary_reason.startswith("length_mismatch"):
            reason = "length_mismatch"
        elif primary_reason.startswith("empty_solution"):
            reason = "empty_solution"
        else:
            reason = primary_reason
        return {
            "debug_bucket": "format_failure",
            "debug_reason": reason,
            "debug_severity": 85,
        }

    if np.isfinite(score) and score <= -0.99:
        return {
            "debug_bucket": "hard_failure",
            "debug_reason": "negative_one_reward",
            "debug_severity": 80,
        }

    if refinement_degraded:
        return {
            "debug_bucket": "quality_regression",
            "debug_reason": "refinement_degraded",
            "debug_severity": 60,
        }

    if prediction_model_defaulted:
        return {
            "debug_bucket": "policy_gap",
            "debug_reason": "prediction_model_defaulted",
            "debug_severity": 30,
        }

    if missing_required > 0:
        return {
            "debug_bucket": "policy_gap",
            "debug_reason": "missing_required_diagnostics",
            "debug_severity": 30,
        }

    return {
        "debug_bucket": "ok",
        "debug_reason": "ok",
        "debug_severity": 0,
    }


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
    diagnosis = summarize_debug_diagnosis(
        reward_score=reward_extra_info.get("trainer_seq_score") or reward_extra_info.get("score"),
        workflow_status=workflow_status,
        workflow_message=workflow_message,
        format_failure_reason=str(reward_extra_info.get("format_failure_reason", "")),
        final_answer_reject_reason=final_answer_reject_reason or "",
        prediction_tool_error=prediction_tool_error or "",
        prediction_call_count=prediction_call_count,
        illegal_turn3_tool_call_count=illegal_turn3_tool_call_count,
        missing_required_feature_tool_count=reward_extra_info.get(
            "missing_required_feature_tool_count",
            len(missing_required),
        ),
        selected_forecast_exact_copy=bool(reward_extra_info.get("selected_forecast_exact_copy", False)),
        refinement_degraded=bool(reward_extra_info.get("refinement_degraded", False)),
        prediction_model_defaulted=bool(prediction_model_defaulted),
    )
    payload = {
        "request_id": request_id,
        "sample_index": sample_index,
        "sample_uid": sample_uid_text(sample_uid or reward_extra_info.get("sample_uid")),
        "global_step": safe_int(reward_extra_info.get("global_step"), default=-1),
        "validate": bool(reward_extra_info.get("validate", False)),
        "run_name": str(reward_extra_info.get("run_name", "") or ""),
        "step_index": int(step_index),
        "turn_stage": turn_stage,
        **diagnosis,
        "tool_call_sequence": tool_call_names,
        "tool_call_count": int(len(tool_call_names)),
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
        "pred_len": safe_int(reward_extra_info.get("pred_len"), default=-1),
        "expected_len": safe_int(
            reward_extra_info.get("expected_len", reward_extra_info.get("gt_len")),
            default=-1,
        ),
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
        "strict_score": finite_or_nan(reward_extra_info.get("strict_score")),
        "recovered_score": finite_or_nan(reward_extra_info.get("recovered_score")),
        "recovery_gap": finite_or_nan(reward_extra_info.get("recovery_gap")),
        "raw_overrun_penalty": finite_or_nan(reward_extra_info.get("raw_overrun_penalty")),
        "strict_parse_mode": str(reward_extra_info.get("strict_parse_mode", "") or ""),
        "recovered_parse_mode": str(reward_extra_info.get("recovered_parse_mode", "") or ""),
        "answer_line_count": safe_int(reward_extra_info.get("answer_line_count"), default=-1),
        "expected_answer_line_count": safe_int(reward_extra_info.get("expected_answer_line_count"), default=-1),
        "wrote_expected_rows_before_stop": bool(reward_extra_info.get("wrote_expected_rows_before_stop", False)),
        "think_token_len": safe_int(reward_extra_info.get("think_token_len"), default=-1),
        "answer_token_len": safe_int(reward_extra_info.get("answer_token_len"), default=-1),
        "turn3_horizon_clamped": bool(reward_extra_info.get("turn3_horizon_clamped", False)),
        "turn3_horizon_clamp_reason": str(reward_extra_info.get("turn3_horizon_clamp_reason", "") or ""),
        "turn3_horizon_clamp_discarded_lines": safe_int(
            reward_extra_info.get("turn3_horizon_clamp_discarded_lines"),
            default=0,
        ),
        "turn3_horizon_clamp_valid_prefix_lines": safe_int(
            reward_extra_info.get("turn3_horizon_clamp_valid_prefix_lines"),
            default=0,
        ),
        "turn3_horizon_clamp_raw_answer_lines": safe_int(
            reward_extra_info.get("turn3_horizon_clamp_raw_answer_lines"),
            default=0,
        ),
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
    "safe_int",
    "sample_uid_text",
    "series_preview",
    "shared_reward_tracking_fields",
    "summarize_debug_diagnosis",
]
