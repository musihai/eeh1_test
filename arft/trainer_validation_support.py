from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import torch

from recipe.time_series_forecast.agent_flow_support import summarize_debug_diagnosis
from verl import DataProto


def evaluate_validation_reward_manager(
    val_reward_fn,
    batch: DataProto,
    *,
    auto_await_fn: Callable,
) -> dict[str, torch.Tensor | dict[str, list]]:
    """Run validation reward through the unified reward-loop manager interface."""
    if val_reward_fn is None:
        raise ValueError("val_reward_fn must be provided for validation.")

    run_single = getattr(val_reward_fn, "run_single", None)
    if run_single is None:
        raise TypeError(f"Unsupported val_reward_fn type: {type(val_reward_fn)!r}")

    run_single_sync = auto_await_fn(run_single)
    reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    reward_extra_info: dict[str, list] = defaultdict(list)

    for sample_idx in range(len(batch)):
        sample_batch = batch.select_idxs([sample_idx])
        result = run_single_sync(sample_batch)
        if not isinstance(result, dict) or "reward_score" not in result:
            raise TypeError(
                "Reward manager run_single must return a dict containing 'reward_score'. "
                f"Got: {type(result)!r}"
            )

        reward_score = float(result["reward_score"])
        response_length = sample_batch.batch["responses"].shape[-1]
        valid_response_length = response_length

        if "attention_mask" in sample_batch.batch.keys():
            attention_mask = sample_batch.batch["attention_mask"][0]
            if "prompts" in sample_batch.batch.keys():
                prompt_length = sample_batch.batch["prompts"].shape[-1]
                valid_response_length = int(attention_mask[prompt_length:].sum().item())
            else:
                valid_response_length = int(attention_mask[-response_length:].sum().item())

        valid_response_length = max(1, min(valid_response_length, response_length))
        reward_tensor[sample_idx, valid_response_length - 1] = reward_score

        sample_extra = result.get("reward_extra_info", {}) or {}
        for key in list(reward_extra_info.keys()):
            if key not in sample_extra:
                reward_extra_info[key].append(None)
        for key, value in sample_extra.items():
            if key not in reward_extra_info:
                reward_extra_info[key] = [None] * sample_idx
            reward_extra_info[key].append(value)

    return {
        "reward_tensor": reward_tensor,
        "reward_extra_info": reward_extra_info,
    }


def to_float_list(values, n: int, default: float = float("nan")) -> list[float]:
    if values is None:
        return [default] * n
    out: list[float] = []
    for i in range(n):
        try:
            out.append(float(values[i]))
        except Exception:
            out.append(default)
    return out


def to_bool_list(values, n: int, default: bool = False) -> list[bool]:
    if values is None:
        return [default] * n
    out: list[bool] = []
    for i in range(n):
        try:
            value = values[i]
            if isinstance(value, str):
                out.append(value.strip().lower() in {"1", "true", "yes", "on"})
            else:
                out.append(bool(value))
        except Exception:
            out.append(default)
    return out


def to_str_list(values, n: int, default: str = "") -> list[str]:
    if values is None:
        return [default] * n
    out: list[str] = []
    for i in range(n):
        try:
            value = values[i]
            if value is None:
                out.append(default)
            else:
                out.append(str(value))
        except Exception:
            out.append(default)
    return out


def percentile(values: list[float], q: float) -> float:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.percentile(np.asarray(finite, dtype=np.float64), q))


def top_counter_items(counter: Counter, limit: int = 5) -> dict[str, int]:
    if not counter:
        return {}
    ordered = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return {str(key): int(value) for key, value in ordered[:limit]}


def build_compact_validation_debug_summary(*, global_step: int, agg_row: dict[str, Any]) -> dict[str, Any]:
    summary_keys = (
        "total_samples",
        "validation_reward_mean",
        "validation_reward_min",
        "validation_reward_max",
        "reward_negative_one_ratio",
        "final_answer_accept_ratio",
        "success_ratio",
        "format_failure_ratio",
        "workflow_rejected_ratio",
        "tool_error_ratio",
        "prediction_call_not_once_ratio",
        "missing_required_feature_tool_ratio",
        "illegal_turn3_tool_call_ratio",
        "prediction_model_defaulted_ratio",
        "refinement_degraded_ratio",
        "analysis_coverage_ratio_mean",
        "strict_score_mean",
        "recovered_score_mean",
        "recovery_gap_mean",
        "raw_overrun_penalty_mean",
        "answer_line_count_mean",
        "think_token_len_mean",
        "answer_token_len_mean",
        "wrote_expected_rows_before_stop_ratio",
        "response_token_len_mean",
        "response_token_len_p90",
        "orig_mse_mean",
        "selected_forecast_orig_mse_mean",
        "final_vs_selected_mse_mean",
        "refinement_delta_orig_mse_mean",
    )
    summary = {"global_step": int(global_step)}
    run_name = str(agg_row.get("run_name") or "").strip()
    if run_name:
        summary["run_name"] = run_name
    for key in summary_keys:
        value = agg_row.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if np.isfinite(numeric):
                summary[key] = numeric

    for key in (
        "selected_model_offline_best_agreement_ratio",
        "selected_vs_reference_teacher_orig_mse_regret_mean",
        "selected_vs_reference_teacher_orig_mse_regret_p50",
        "selected_vs_reference_teacher_orig_mse_regret_p90",
        "final_vs_reference_teacher_orig_mse_regret_mean",
        "invalid_tool_call_name_ratio",
        "tool_call_json_decode_error_ratio",
    ):
        value = agg_row.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if np.isfinite(numeric):
                summary[key] = numeric

    for key in (
        "debug_bucket_distribution",
        "debug_reason_distribution",
        "format_failure_reason_distribution",
        "final_answer_reject_reason_distribution",
        "workflow_status_distribution",
        "selected_model_distribution",
        "run_name_distribution",
        "invalid_tool_call_name_distribution",
        "refinement_decision_distribution",
    ):
        value = agg_row.get(key)
        if isinstance(value, dict) and value:
            summary[key] = value

    return summary


def extract_values_from_text(text: str) -> list[float]:
    values: list[float] = []
    if not text:
        return values
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    body = answer_match.group(1) if answer_match else text
    for line in str(body).splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.search(r"(-?\d+\.?\d*)$", line)
        if not match:
            continue
        try:
            values.append(float(match.group(1)))
        except Exception:
            continue
    return values


def tail_lines(text: str, max_lines: int = 10) -> list[str]:
    if not text:
        return []
    return str(text).splitlines()[-max_lines:]


def normalized_mse_mae(pred_values: list[float], gt_values: list[float]) -> tuple[float, float]:
    if not pred_values or not gt_values:
        return float("nan"), float("nan")
    n = min(len(pred_values), len(gt_values))
    if n <= 0:
        return float("nan"), float("nan")
    pred = np.asarray(pred_values[:n], dtype=np.float64)
    gt = np.asarray(gt_values[:n], dtype=np.float64)
    mu = float(np.nanmean(gt))
    std = max(float(np.nanstd(gt)), 1e-8)
    pred_n = (pred - mu) / std
    gt_n = (gt - mu) / std
    diff = pred_n - gt_n
    return float(np.mean(diff**2)), float(np.mean(np.abs(diff)))


def orig_mse_mae(pred_values: list[float], gt_values: list[float]) -> tuple[float, float]:
    if not pred_values or not gt_values:
        return float("nan"), float("nan")
    n = min(len(pred_values), len(gt_values))
    if n <= 0:
        return float("nan"), float("nan")
    pred = np.asarray(pred_values[:n], dtype=np.float64)
    gt = np.asarray(gt_values[:n], dtype=np.float64)
    diff = pred - gt
    return float(np.mean(diff**2)), float(np.mean(np.abs(diff)))


def write_min_eval_debug_files(
    *,
    global_steps: int,
    sample_uids: list,
    sample_outputs: list,
    sample_gts: list,
    sample_scores: list[float],
    reward_extra_infos_dict: dict[str, list],
    debug_dir: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = len(sample_scores)
    if n <= 0:
        return {}, []

    pred_len = to_float_list(reward_extra_infos_dict.get("pred_len"), n)
    expected_len = to_float_list(
        reward_extra_infos_dict.get("expected_len") or reward_extra_infos_dict.get("gt_len"),
        n,
    )
    orig_mse = to_float_list(reward_extra_infos_dict.get("orig_mse") or reward_extra_infos_dict.get("raw_mse"), n)
    orig_mae = to_float_list(reward_extra_infos_dict.get("orig_mae") or reward_extra_infos_dict.get("raw_mae"), n)
    norm_mse = to_float_list(reward_extra_infos_dict.get("norm_mse"), n)
    norm_mae = to_float_list(reward_extra_infos_dict.get("norm_mae"), n)
    has_answer_tag = to_bool_list(reward_extra_infos_dict.get("has_answer_tag"), n)
    has_answer_close = to_bool_list(reward_extra_infos_dict.get("has_answer_close"), n)
    was_clipped = to_bool_list(reward_extra_infos_dict.get("was_clipped"), n)
    format_failure_reason = to_str_list(reward_extra_infos_dict.get("format_failure_reason"), n)
    final_answer_reject_reason = to_str_list(reward_extra_infos_dict.get("final_answer_reject_reason"), n)
    strict_length_match = to_bool_list(reward_extra_infos_dict.get("strict_length_match"), n)
    trainer_seq_score = to_float_list(
        reward_extra_infos_dict.get("trainer_seq_score") or reward_extra_infos_dict.get("score"),
        n,
        default=float("nan"),
    )
    generation_stop_reason = to_str_list(reward_extra_infos_dict.get("generation_stop_reason"), n)
    generation_finish_reason = to_str_list(reward_extra_infos_dict.get("generation_finish_reason"), n)
    strict_score = to_float_list(
        reward_extra_infos_dict.get("strict_score") or reward_extra_infos_dict.get("score"),
        n,
        default=float("nan"),
    )
    recovered_score = to_float_list(
        reward_extra_infos_dict.get("recovered_score") or reward_extra_infos_dict.get("score"),
        n,
        default=float("nan"),
    )
    recovery_gap = to_float_list(reward_extra_infos_dict.get("recovery_gap"), n, default=float("nan"))
    raw_overrun_penalty = to_float_list(
        reward_extra_infos_dict.get("raw_overrun_penalty"),
        n,
        default=float("nan"),
    )
    answer_line_count = to_float_list(reward_extra_infos_dict.get("answer_line_count"), n, default=float("nan"))
    expected_answer_line_count = to_float_list(
        reward_extra_infos_dict.get("expected_answer_line_count"),
        n,
        default=float("nan"),
    )
    wrote_expected_rows_before_stop = to_bool_list(
        reward_extra_infos_dict.get("wrote_expected_rows_before_stop"),
        n,
    )
    think_token_len = to_float_list(reward_extra_infos_dict.get("think_token_len"), n, default=float("nan"))
    answer_token_len = to_float_list(reward_extra_infos_dict.get("answer_token_len"), n, default=float("nan"))
    turn3_horizon_clamped = to_bool_list(reward_extra_infos_dict.get("turn3_horizon_clamped"), n)
    turn3_horizon_clamp_discarded_lines = to_float_list(
        reward_extra_infos_dict.get("turn3_horizon_clamp_discarded_lines"),
        n,
        default=float("nan"),
    )
    turn3_horizon_clamp_valid_prefix_lines = to_float_list(
        reward_extra_infos_dict.get("turn3_horizon_clamp_valid_prefix_lines"),
        n,
        default=float("nan"),
    )
    turn3_horizon_clamp_raw_answer_lines = to_float_list(
        reward_extra_infos_dict.get("turn3_horizon_clamp_raw_answer_lines"),
        n,
        default=float("nan"),
    )
    turn3_horizon_clamp_reason = to_str_list(reward_extra_infos_dict.get("turn3_horizon_clamp_reason"), n)
    validate_flag = to_bool_list(reward_extra_infos_dict.get("validate"), n)
    run_name = to_str_list(reward_extra_infos_dict.get("run_name"), n)
    workflow_violation_reason = to_str_list(reward_extra_infos_dict.get("workflow_violation_reason"), n)
    selected_model = to_str_list(
        reward_extra_infos_dict.get("selected_model")
        or reward_extra_infos_dict.get("prediction_model_used")
        or reward_extra_infos_dict.get("output_source"),
        n,
        default="unknown",
    )
    selected_forecast_orig_mse = to_float_list(
        reward_extra_infos_dict.get("selected_forecast_orig_mse"),
        n,
        default=float("nan"),
    )
    selected_forecast_exact_copy = to_bool_list(reward_extra_infos_dict.get("selected_forecast_exact_copy"), n)
    final_vs_selected_mse = to_float_list(
        reward_extra_infos_dict.get("final_vs_selected_mse"),
        n,
        default=float("nan"),
    )
    refinement_delta_orig_mse = to_float_list(
        reward_extra_infos_dict.get("refinement_delta_orig_mse"),
        n,
        default=float("nan"),
    )
    refinement_changed = to_bool_list(reward_extra_infos_dict.get("refinement_changed"), n)
    refinement_improved = to_bool_list(reward_extra_infos_dict.get("refinement_improved"), n)
    refinement_degraded = to_bool_list(reward_extra_infos_dict.get("refinement_degraded"), n)
    analysis_coverage_ratio = to_float_list(
        reward_extra_infos_dict.get("analysis_coverage_ratio"),
        n,
        default=float("nan"),
    )
    feature_tool_count = to_float_list(reward_extra_infos_dict.get("feature_tool_count"), n, default=float("nan"))
    required_feature_tool_count = to_float_list(
        reward_extra_infos_dict.get("required_feature_tool_count"),
        n,
        default=float("nan"),
    )
    missing_required_feature_tool_count = to_float_list(
        reward_extra_infos_dict.get("missing_required_feature_tool_count"),
        n,
        default=float("nan"),
    )
    prediction_call_count = to_float_list(
        reward_extra_infos_dict.get("prediction_call_count"),
        n,
        default=float("nan"),
    )
    prediction_step_index = to_float_list(
        reward_extra_infos_dict.get("prediction_step_index"),
        n,
        default=float("nan"),
    )
    final_answer_step_index = to_float_list(
        reward_extra_infos_dict.get("final_answer_step_index"),
        n,
        default=float("nan"),
    )
    required_step_budget = to_float_list(
        reward_extra_infos_dict.get("required_step_budget"),
        n,
        default=float("nan"),
    )
    response_token_len = to_float_list(
        reward_extra_infos_dict.get("response_token_len"),
        n,
        default=float("nan"),
    )
    illegal_turn3_tool_call_count = to_float_list(
        reward_extra_infos_dict.get("illegal_turn3_tool_call_count"),
        n,
        default=float("nan"),
    )
    prediction_model_defaulted = to_bool_list(reward_extra_infos_dict.get("prediction_model_defaulted"), n)
    prediction_requested_model = to_str_list(reward_extra_infos_dict.get("prediction_requested_model"), n)
    tool_call_sequence = to_str_list(reward_extra_infos_dict.get("tool_call_sequence"), n, default="none")
    analysis_state_signature = to_str_list(
        reward_extra_infos_dict.get("analysis_state_signature"),
        n,
        default="none",
    )
    workflow_status = to_str_list(reward_extra_infos_dict.get("workflow_status"), n)
    turn_stage = to_str_list(reward_extra_infos_dict.get("turn_stage"), n)
    prediction_tool_error = to_str_list(reward_extra_infos_dict.get("prediction_tool_error"), n)
    selected_forecast_preview = to_str_list(reward_extra_infos_dict.get("selected_forecast_preview"), n)
    final_answer_preview = to_str_list(reward_extra_infos_dict.get("final_answer_preview"), n)
    offline_best_model = to_str_list(reward_extra_infos_dict.get("offline_best_model"), n)
    offline_margin = to_float_list(reward_extra_infos_dict.get("offline_margin"), n, default=float("nan"))
    reference_teacher_error = to_float_list(
        reward_extra_infos_dict.get("reference_teacher_error"),
        n,
        default=float("nan"),
    )
    reference_teacher_error_band = to_str_list(reward_extra_infos_dict.get("reference_teacher_error_band"), n)
    refinement_decision_name = to_str_list(reward_extra_infos_dict.get("refinement_decision_name"), n)
    raw_tool_call_block_count = to_float_list(
        reward_extra_infos_dict.get("raw_tool_call_block_count"),
        n,
        default=float("nan"),
    )
    raw_tool_call_name_sequence = to_str_list(
        reward_extra_infos_dict.get("raw_tool_call_name_sequence"),
        n,
        default="none",
    )
    invalid_tool_call_name_count = to_float_list(
        reward_extra_infos_dict.get("invalid_tool_call_name_count"),
        n,
        default=float("nan"),
    )
    invalid_tool_call_name_sequence = to_str_list(
        reward_extra_infos_dict.get("invalid_tool_call_name_sequence"),
        n,
        default="none",
    )
    tool_call_json_decode_error_count = to_float_list(
        reward_extra_infos_dict.get("tool_call_json_decode_error_count"),
        n,
        default=float("nan"),
    )
    tool_call_missing_name_count = to_float_list(
        reward_extra_infos_dict.get("tool_call_missing_name_count"),
        n,
        default=float("nan"),
    )

    pred_len_arr = np.asarray(pred_len, dtype=np.float64)
    expected_len_arr = np.asarray(expected_len, dtype=np.float64)
    score_arr = np.asarray(
        [
            float(value) if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) else np.nan
            for value in sample_scores
        ],
        dtype=np.float64,
    )
    valid_pred_len = np.isfinite(pred_len_arr)
    valid_expected_len = np.isfinite(expected_len_arr)
    is_94_95 = np.logical_or(pred_len_arr == 94, pred_len_arr == 95)
    missing_close = np.asarray(
        [reason.startswith("missing_answer_close_tag") for reason in format_failure_reason],
        dtype=bool,
    )
    invalid_answer_shape = np.asarray(
        [
            reason.startswith("invalid_answer_shape") or reject_reason.startswith("invalid_answer_shape")
            for reason, reject_reason in zip(format_failure_reason, final_answer_reject_reason, strict=False)
        ],
        dtype=bool,
    )
    explicit_length_mismatch = np.asarray(
        [
            reason.startswith("length_mismatch") or reject_reason.startswith("length_mismatch")
            for reason, reject_reason in zip(format_failure_reason, final_answer_reject_reason, strict=False)
        ],
        dtype=bool,
    )
    exact_expected_match = np.logical_and.reduce([valid_pred_len, valid_expected_len, pred_len_arr == expected_len_arr])
    final_answer_accept = np.logical_and.reduce(
        [
            np.asarray(has_answer_tag, dtype=bool),
            np.asarray(has_answer_close, dtype=bool),
            exact_expected_match,
            np.asarray([reason == "" for reason in final_answer_reject_reason], dtype=bool),
        ]
    )
    workflow_rejected_mask = np.asarray([status == "rejected" for status in workflow_status], dtype=bool)
    reward_negative_one_mask = np.logical_and(np.isfinite(score_arr), score_arr <= -0.99)
    length_mismatch_mask = np.logical_or(
        np.logical_and.reduce([valid_pred_len, valid_expected_len, pred_len_arr != expected_len_arr]),
        np.logical_or(invalid_answer_shape, explicit_length_mismatch),
    )
    success_mask = np.logical_and.reduce(
        [
            final_answer_accept,
            np.logical_not(workflow_rejected_mask),
            np.asarray([np.isfinite(v) for v in orig_mse], dtype=bool),
            np.asarray([np.isfinite(v) for v in orig_mae], dtype=bool),
        ]
    )

    success_orig_mse_values = [orig_mse[i] for i in range(n) if success_mask[i] and np.isfinite(orig_mse[i])]
    success_orig_mae_values = [orig_mae[i] for i in range(n) if success_mask[i] and np.isfinite(orig_mae[i])]
    success_norm_mse_values = [norm_mse[i] for i in range(n) if success_mask[i] and np.isfinite(norm_mse[i])]
    success_norm_mae_values = [norm_mae[i] for i in range(n) if success_mask[i] and np.isfinite(norm_mae[i])]
    reward_values = [float(v) for v in score_arr if np.isfinite(v)]
    trainer_seq_values = [float(v) for v in trainer_seq_score if np.isfinite(v)]
    selected_forecast_orig_mse_values = [float(v) for v in selected_forecast_orig_mse if np.isfinite(v)]
    final_vs_selected_mse_values = [float(v) for v in final_vs_selected_mse if np.isfinite(v)]
    refinement_delta_orig_mse_values = [float(v) for v in refinement_delta_orig_mse if np.isfinite(v)]
    analysis_coverage_ratio_values = [float(v) for v in analysis_coverage_ratio if np.isfinite(v)]
    strict_score_values = [float(v) for v in strict_score if np.isfinite(v)]
    recovered_score_values = [float(v) for v in recovered_score if np.isfinite(v)]
    recovery_gap_values = [float(v) for v in recovery_gap if np.isfinite(v)]
    raw_overrun_penalty_values = [float(v) for v in raw_overrun_penalty if np.isfinite(v)]
    answer_line_count_values = [float(v) for v in answer_line_count if np.isfinite(v)]
    turn3_horizon_clamp_discarded_values = [
        float(v) for v in turn3_horizon_clamp_discarded_lines if np.isfinite(v)
    ]
    turn3_horizon_clamp_valid_prefix_values = [
        float(v) for v in turn3_horizon_clamp_valid_prefix_lines if np.isfinite(v)
    ]
    turn3_horizon_clamp_raw_answer_values = [
        float(v) for v in turn3_horizon_clamp_raw_answer_lines if np.isfinite(v)
    ]
    think_token_len_values = [float(v) for v in think_token_len if np.isfinite(v)]
    answer_token_len_values = [float(v) for v in answer_token_len if np.isfinite(v)]
    feature_tool_count_values = [float(v) for v in feature_tool_count if np.isfinite(v)]
    required_feature_tool_count_values = [float(v) for v in required_feature_tool_count if np.isfinite(v)]
    missing_required_feature_tool_count_values = [
        float(v) for v in missing_required_feature_tool_count if np.isfinite(v)
    ]
    prediction_call_count_values = [float(v) for v in prediction_call_count if np.isfinite(v)]
    response_token_len_values = [float(v) for v in response_token_len if np.isfinite(v)]
    illegal_turn3_tool_call_count_values = [
        float(v) for v in illegal_turn3_tool_call_count if np.isfinite(v)
    ]
    raw_tool_call_block_count_values = [float(v) for v in raw_tool_call_block_count if np.isfinite(v)]
    valid_teacher_agreement_values = [
        float(str(selected_model[i]).strip().lower() == str(offline_best_model[i]).strip().lower())
        for i in range(n)
        if str(selected_model[i]).strip() and str(offline_best_model[i]).strip()
    ]
    selected_vs_reference_teacher_regret_values = [
        float(selected_forecast_orig_mse[i] - reference_teacher_error[i])
        for i in range(n)
        if np.isfinite(selected_forecast_orig_mse[i]) and np.isfinite(reference_teacher_error[i])
    ]
    final_vs_reference_teacher_regret_values = [
        float(orig_mse[i] - reference_teacher_error[i])
        for i in range(n)
        if np.isfinite(orig_mse[i]) and np.isfinite(reference_teacher_error[i])
    ]

    debug_bucket: list[str] = []
    debug_reason: list[str] = []
    debug_severity: list[int] = []
    for i in range(n):
        diagnosis = summarize_debug_diagnosis(
            reward_score=score_arr[i] if np.isfinite(score_arr[i]) else trainer_seq_score[i],
            workflow_status=workflow_status[i],
            workflow_message=workflow_violation_reason[i],
            format_failure_reason=format_failure_reason[i],
            final_answer_reject_reason=final_answer_reject_reason[i],
            prediction_tool_error=prediction_tool_error[i],
            prediction_call_count=prediction_call_count[i],
            illegal_turn3_tool_call_count=illegal_turn3_tool_call_count[i],
            missing_required_feature_tool_count=missing_required_feature_tool_count[i],
            selected_forecast_exact_copy=selected_forecast_exact_copy[i],
            refinement_degraded=refinement_degraded[i],
            prediction_model_defaulted=prediction_model_defaulted[i],
        )
        debug_bucket.append(str(diagnosis["debug_bucket"]))
        debug_reason.append(str(diagnosis["debug_reason"]))
        debug_severity.append(int(diagnosis["debug_severity"]))

    debug_bucket_counter = Counter(debug_bucket)
    debug_reason_counter = Counter(debug_reason)
    format_failure_reason_counter = Counter(reason for reason in format_failure_reason if reason)
    final_answer_reject_counter = Counter(reason for reason in final_answer_reject_reason if reason)
    generation_stop_reason_counter = Counter(reason for reason in generation_stop_reason if reason)
    generation_finish_reason_counter = Counter(reason for reason in generation_finish_reason if reason)
    turn3_horizon_clamp_reason_counter = Counter(reason for reason in turn3_horizon_clamp_reason if reason)
    selected_model_counter = Counter(model for model in selected_model if model)
    workflow_status_counter = Counter(status for status in workflow_status if status)
    turn_stage_counter = Counter(stage for stage in turn_stage if stage)
    run_name_counter = Counter(name for name in run_name if name)
    refinement_decision_counter = Counter(name for name in refinement_decision_name if name)
    prediction_tool_error_count = int(sum(1 for value in prediction_tool_error if value))
    invalid_tool_call_name_counter: Counter[str] = Counter()
    for value in invalid_tool_call_name_sequence:
        for name in str(value or "").split("->"):
            normalized = str(name).strip()
            if normalized and normalized != "none":
                invalid_tool_call_name_counter[normalized] += 1

    format_failure_mask = np.asarray([bucket == "format_failure" for bucket in debug_bucket], dtype=bool)
    tool_error_mask = np.asarray([bucket == "tool_error" for bucket in debug_bucket], dtype=bool)
    quality_regression_mask = np.asarray([bucket == "quality_regression" for bucket in debug_bucket], dtype=bool)
    prediction_call_not_once_mask = np.asarray(
        [np.isfinite(value) and int(value) != 1 for value in prediction_call_count],
        dtype=bool,
    )
    missing_required_feature_tool_mask = np.asarray(
        [np.isfinite(value) and int(value) > 0 for value in missing_required_feature_tool_count],
        dtype=bool,
    )
    invalid_tool_call_name_mask = np.asarray(
        [np.isfinite(value) and int(value) > 0 for value in invalid_tool_call_name_count],
        dtype=bool,
    )
    tool_call_json_decode_error_mask = np.asarray(
        [np.isfinite(value) and int(value) > 0 for value in tool_call_json_decode_error_count],
        dtype=bool,
    )
    unique_run_names = list(run_name_counter.keys())
    aggregate_run_name = unique_run_names[0] if len(unique_run_names) == 1 else ""

    total = float(n)
    agg_row = {
        "step": int(global_steps),
        "total_samples": int(n),
        "run_name": aggregate_run_name,
        "validation_reward_mean": float(np.mean(reward_values)) if reward_values else float("nan"),
        "validation_reward_min": float(np.min(reward_values)) if reward_values else float("nan"),
        "validation_reward_max": float(np.max(reward_values)) if reward_values else float("nan"),
        "reward_negative_one_ratio": float(np.mean(reward_negative_one_mask.astype(np.float64))),
        "final_answer_accept_ratio": float(np.mean(final_answer_accept.astype(np.float64))),
        "success_ratio": float(np.mean(success_mask.astype(np.float64))),
        "format_failure_ratio": float(np.mean(format_failure_mask.astype(np.float64))),
        "workflow_rejected_ratio": float(np.mean(workflow_rejected_mask.astype(np.float64))),
        "tool_error_ratio": float(np.mean(tool_error_mask.astype(np.float64))),
        "prediction_call_not_once_ratio": float(np.mean(prediction_call_not_once_mask.astype(np.float64))),
        "missing_required_feature_tool_ratio": float(np.mean(missing_required_feature_tool_mask.astype(np.float64))),
        "invalid_tool_call_name_ratio": float(np.mean(invalid_tool_call_name_mask.astype(np.float64))),
        "tool_call_json_decode_error_ratio": float(np.mean(tool_call_json_decode_error_mask.astype(np.float64))),
        "illegal_turn3_tool_call_ratio": float(
            np.mean((np.asarray(illegal_turn3_tool_call_count_values, dtype=np.float64) > 0).astype(np.float64))
        )
        if illegal_turn3_tool_call_count_values
        else float("nan"),
        "prediction_model_defaulted_ratio": float(np.mean(np.asarray(prediction_model_defaulted, dtype=np.float64))),
        "selected_forecast_exact_copy_ratio": float(
            np.mean(np.asarray(selected_forecast_exact_copy, dtype=np.float64))
        ),
        "refinement_improved_ratio": float(np.mean(np.asarray(refinement_improved, dtype=np.float64))),
        "refinement_degraded_ratio": float(np.mean(np.asarray(refinement_degraded, dtype=np.float64))),
        "pred_len_94_95_ratio": float(np.sum(is_94_95) / total),
        "missing_answer_close_tag_count": int(np.sum(missing_close)),
        "invalid_answer_shape_count": int(np.sum(invalid_answer_shape)),
        "length_mismatch_ratio": float(np.mean(length_mismatch_mask.astype(np.float64))),
        "was_clipped_count": int(np.sum(np.asarray(was_clipped, dtype=bool))),
        "orig_mse_mean": float(np.mean(success_orig_mse_values)) if success_orig_mse_values else float("nan"),
        "orig_mse_p50": percentile(success_orig_mse_values, 50),
        "orig_mse_p90": percentile(success_orig_mse_values, 90),
        "orig_mae_mean": float(np.mean(success_orig_mae_values)) if success_orig_mae_values else float("nan"),
        "norm_mse_mean": float(np.mean(success_norm_mse_values)) if success_norm_mse_values else float("nan"),
        "norm_mae_mean": float(np.mean(success_norm_mae_values)) if success_norm_mae_values else float("nan"),
        "trainer_seq_score_mean": float(np.mean(trainer_seq_values)) if trainer_seq_values else float("nan"),
        "selected_forecast_orig_mse_mean": float(np.mean(selected_forecast_orig_mse_values))
        if selected_forecast_orig_mse_values
        else float("nan"),
        "selected_model_offline_best_agreement_ratio": float(np.mean(valid_teacher_agreement_values))
        if valid_teacher_agreement_values
        else float("nan"),
        "selected_vs_reference_teacher_orig_mse_regret_mean": float(np.mean(selected_vs_reference_teacher_regret_values))
        if selected_vs_reference_teacher_regret_values
        else float("nan"),
        "selected_vs_reference_teacher_orig_mse_regret_p50": percentile(
            selected_vs_reference_teacher_regret_values,
            50,
        ),
        "selected_vs_reference_teacher_orig_mse_regret_p90": percentile(
            selected_vs_reference_teacher_regret_values,
            90,
        ),
        "final_vs_reference_teacher_orig_mse_regret_mean": float(np.mean(final_vs_reference_teacher_regret_values))
        if final_vs_reference_teacher_regret_values
        else float("nan"),
        "final_vs_selected_mse_mean": float(np.mean(final_vs_selected_mse_values))
        if final_vs_selected_mse_values
        else float("nan"),
        "refinement_delta_orig_mse_mean": float(np.mean(refinement_delta_orig_mse_values))
        if refinement_delta_orig_mse_values
        else float("nan"),
        "analysis_coverage_ratio_mean": float(np.mean(analysis_coverage_ratio_values))
        if analysis_coverage_ratio_values
        else float("nan"),
        "strict_score_mean": float(np.mean(strict_score_values)) if strict_score_values else float("nan"),
        "recovered_score_mean": float(np.mean(recovered_score_values)) if recovered_score_values else float("nan"),
        "recovery_gap_mean": float(np.mean(recovery_gap_values)) if recovery_gap_values else float("nan"),
        "raw_overrun_penalty_mean": float(np.mean(raw_overrun_penalty_values))
        if raw_overrun_penalty_values
        else float("nan"),
        "answer_line_count_mean": float(np.mean(answer_line_count_values)) if answer_line_count_values else float("nan"),
        "turn3_horizon_clamped_ratio": float(np.mean(np.asarray(turn3_horizon_clamped, dtype=np.float64))),
        "turn3_horizon_clamp_discarded_lines_mean": float(np.mean(turn3_horizon_clamp_discarded_values))
        if turn3_horizon_clamp_discarded_values
        else float("nan"),
        "turn3_horizon_clamp_valid_prefix_lines_mean": float(np.mean(turn3_horizon_clamp_valid_prefix_values))
        if turn3_horizon_clamp_valid_prefix_values
        else float("nan"),
        "turn3_horizon_clamp_raw_answer_lines_mean": float(np.mean(turn3_horizon_clamp_raw_answer_values))
        if turn3_horizon_clamp_raw_answer_values
        else float("nan"),
        "think_token_len_mean": float(np.mean(think_token_len_values)) if think_token_len_values else float("nan"),
        "answer_token_len_mean": float(np.mean(answer_token_len_values)) if answer_token_len_values else float("nan"),
        "wrote_expected_rows_before_stop_ratio": float(
            np.mean(np.asarray(wrote_expected_rows_before_stop, dtype=np.float64))
        ),
        "feature_tool_count_mean": float(np.mean(feature_tool_count_values)) if feature_tool_count_values else float("nan"),
        "required_feature_tool_count_mean": float(np.mean(required_feature_tool_count_values))
        if required_feature_tool_count_values
        else float("nan"),
        "missing_required_feature_tool_count_mean": float(np.mean(missing_required_feature_tool_count_values))
        if missing_required_feature_tool_count_values
        else float("nan"),
        "prediction_call_count_mean": float(np.mean(prediction_call_count_values))
        if prediction_call_count_values
        else float("nan"),
        "raw_tool_call_block_count_mean": float(np.mean(raw_tool_call_block_count_values))
        if raw_tool_call_block_count_values
        else float("nan"),
        "response_token_len_mean": float(np.mean(response_token_len_values))
        if response_token_len_values
        else float("nan"),
        "response_token_len_p90": percentile(response_token_len_values, 90),
        "prediction_tool_error_count": prediction_tool_error_count,
        "debug_bucket_distribution": {str(k): int(v) for k, v in sorted(debug_bucket_counter.items())},
        "debug_reason_distribution": {str(k): int(v) for k, v in sorted(debug_reason_counter.items())},
        "selected_model_distribution": {str(k): int(v) for k, v in sorted(selected_model_counter.items())},
        "format_failure_reason_distribution": top_counter_items(format_failure_reason_counter, limit=8),
        "final_answer_reject_reason_distribution": top_counter_items(final_answer_reject_counter, limit=8),
        "generation_stop_reason_distribution": top_counter_items(generation_stop_reason_counter, limit=6),
        "generation_finish_reason_distribution": top_counter_items(generation_finish_reason_counter, limit=6),
        "turn3_horizon_clamp_reason_distribution": top_counter_items(turn3_horizon_clamp_reason_counter, limit=6),
        "workflow_status_distribution": {str(k): int(v) for k, v in sorted(workflow_status_counter.items())},
        "turn_stage_distribution": {str(k): int(v) for k, v in sorted(turn_stage_counter.items())},
        "run_name_distribution": {str(k): int(v) for k, v in sorted(run_name_counter.items())},
        "invalid_tool_call_name_distribution": top_counter_items(invalid_tool_call_name_counter, limit=8),
        "refinement_decision_distribution": {str(k): int(v) for k, v in sorted(refinement_decision_counter.items())},
        "patchtst_share": float(selected_model_counter.get("patchtst", 0) / total),
        "itransformer_share": float(selected_model_counter.get("itransformer", 0) / total),
        "arima_share": float(selected_model_counter.get("arima", 0) / total),
        "chronos2_share": float(selected_model_counter.get("chronos2", 0) / total),
        "unknown_model_share": float(selected_model_counter.get("unknown", 0) / total),
    }

    def _float_or_nan(value: Any) -> float:
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            return float(value)
        return float("nan")

    def _int_or_default(value: Any, default: int = -1) -> int:
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            return int(value)
        return default

    def _sample_id(i: int) -> str:
        return str(sample_uids[i]) if i < len(sample_uids) else f"sample_{i}"

    def _failure_sort_key(i: int) -> tuple[Any, ...]:
        reward_value = _float_or_nan(score_arr[i])
        if np.isnan(reward_value):
            reward_value = 999999.0
        return (-int(debug_severity[i]), reward_value, str(debug_reason[i]), _sample_id(i))

    def _success_sort_key(i: int) -> tuple[Any, ...]:
        mse_value = _float_or_nan(orig_mse[i])
        if np.isnan(mse_value):
            mse_value = 999999.0
        reward_value = _float_or_nan(score_arr[i])
        if np.isnan(reward_value):
            reward_value = -999999.0
        return (mse_value, -reward_value, _sample_id(i))

    def _build_base_row(i: int, category: str) -> dict[str, Any]:
        output_text = str(sample_outputs[i]) if i < len(sample_outputs) else ""
        pred_value = _int_or_default(pred_len_arr[i])
        expected_value = _int_or_default(expected_len_arr[i])
        len_gap = pred_value - expected_value if pred_value >= 0 and expected_value >= 0 else -1
        return {
            "step": int(global_steps),
            "category": category,
            "sample_id": _sample_id(i),
            "debug_bucket": debug_bucket[i],
            "debug_reason": debug_reason[i],
            "debug_severity": int(debug_severity[i]),
            "reward_score": _float_or_nan(score_arr[i]),
            "trainer_seq_score": _float_or_nan(trainer_seq_score[i]),
            "selected_model": selected_model[i],
            "offline_best_model": offline_best_model[i] if offline_best_model[i] else "",
            "offline_margin": _float_or_nan(offline_margin[i]),
            "reference_teacher_error": _float_or_nan(reference_teacher_error[i]),
            "reference_teacher_error_band": reference_teacher_error_band[i] if reference_teacher_error_band[i] else "",
            "selected_model_matches_offline_best": bool(
                str(selected_model[i]).strip()
                and str(offline_best_model[i]).strip()
                and str(selected_model[i]).strip().lower() == str(offline_best_model[i]).strip().lower()
            ),
            "selected_vs_reference_teacher_orig_mse_regret": (
                _float_or_nan(selected_forecast_orig_mse[i] - reference_teacher_error[i])
                if np.isfinite(selected_forecast_orig_mse[i]) and np.isfinite(reference_teacher_error[i])
                else float("nan")
            ),
            "final_vs_reference_teacher_orig_mse_regret": (
                _float_or_nan(orig_mse[i] - reference_teacher_error[i])
                if np.isfinite(orig_mse[i]) and np.isfinite(reference_teacher_error[i])
                else float("nan")
            ),
            "prediction_requested_model": prediction_requested_model[i] if prediction_requested_model[i] else "",
            "prediction_model_defaulted": bool(prediction_model_defaulted[i]),
            "prediction_tool_error": prediction_tool_error[i] if prediction_tool_error[i] else "",
            "workflow_status": workflow_status[i] if workflow_status[i] else "",
            "workflow_violation_reason": workflow_violation_reason[i] if workflow_violation_reason[i] else "",
            "format_failure_reason": format_failure_reason[i] if format_failure_reason[i] else "",
            "final_answer_reject_reason": final_answer_reject_reason[i] if final_answer_reject_reason[i] else "",
            "final_answer_accept": bool(final_answer_accept[i]),
            "was_clipped": bool(was_clipped[i]),
            "turn_stage": turn_stage[i] if turn_stage[i] else "",
            "pred_len": pred_value,
            "expected_len": expected_value,
            "len_gap": int(len_gap),
            "response_token_len": _int_or_default(response_token_len[i]),
            "generation_stop_reason": generation_stop_reason[i] if generation_stop_reason[i] else "",
            "generation_finish_reason": generation_finish_reason[i] if generation_finish_reason[i] else "",
            "strict_score": _float_or_nan(strict_score[i]),
            "recovered_score": _float_or_nan(recovered_score[i]),
            "recovery_gap": _float_or_nan(recovery_gap[i]),
            "raw_overrun_penalty": _float_or_nan(raw_overrun_penalty[i]),
            "answer_line_count": _int_or_default(answer_line_count[i]),
            "expected_answer_line_count": _int_or_default(expected_answer_line_count[i]),
            "wrote_expected_rows_before_stop": bool(wrote_expected_rows_before_stop[i]),
            "think_token_len": _int_or_default(think_token_len[i]),
            "answer_token_len": _int_or_default(answer_token_len[i]),
            "turn3_horizon_clamped": bool(turn3_horizon_clamped[i]),
            "turn3_horizon_clamp_reason": turn3_horizon_clamp_reason[i] if turn3_horizon_clamp_reason[i] else "",
            "turn3_horizon_clamp_discarded_lines": _int_or_default(turn3_horizon_clamp_discarded_lines[i]),
            "turn3_horizon_clamp_valid_prefix_lines": _int_or_default(turn3_horizon_clamp_valid_prefix_lines[i]),
            "turn3_horizon_clamp_raw_answer_lines": _int_or_default(turn3_horizon_clamp_raw_answer_lines[i]),
            "validate": bool(validate_flag[i]),
            "run_name": run_name[i] if run_name[i] else "",
            "feature_tool_count": _int_or_default(feature_tool_count[i]),
            "required_feature_tool_count": _int_or_default(required_feature_tool_count[i]),
            "missing_required_feature_tool_count": _int_or_default(missing_required_feature_tool_count[i]),
            "analysis_coverage_ratio": _float_or_nan(analysis_coverage_ratio[i]),
            "prediction_call_count": _int_or_default(prediction_call_count[i]),
            "raw_tool_call_block_count": _int_or_default(raw_tool_call_block_count[i]),
            "raw_tool_call_name_sequence": raw_tool_call_name_sequence[i] if raw_tool_call_name_sequence[i] else "none",
            "invalid_tool_call_name_count": _int_or_default(invalid_tool_call_name_count[i]),
            "invalid_tool_call_name_sequence": (
                invalid_tool_call_name_sequence[i] if invalid_tool_call_name_sequence[i] else "none"
            ),
            "tool_call_json_decode_error_count": _int_or_default(tool_call_json_decode_error_count[i]),
            "tool_call_missing_name_count": _int_or_default(tool_call_missing_name_count[i]),
            "prediction_step_index": _int_or_default(prediction_step_index[i]),
            "final_answer_step_index": _int_or_default(final_answer_step_index[i]),
            "required_step_budget": _int_or_default(required_step_budget[i]),
            "illegal_turn3_tool_call_count": _int_or_default(illegal_turn3_tool_call_count[i]),
            "tool_call_sequence": tool_call_sequence[i] if tool_call_sequence[i] else "none",
            "analysis_state_signature": analysis_state_signature[i] if analysis_state_signature[i] else "",
            "refinement_decision_name": refinement_decision_name[i] if refinement_decision_name[i] else "",
            "selected_forecast_orig_mse": _float_or_nan(selected_forecast_orig_mse[i]),
            "final_vs_selected_mse": _float_or_nan(final_vs_selected_mse[i]),
            "refinement_delta_orig_mse": _float_or_nan(refinement_delta_orig_mse[i]),
            "selected_forecast_exact_copy": bool(selected_forecast_exact_copy[i]),
            "refinement_changed": bool(refinement_changed[i]),
            "refinement_improved": bool(refinement_improved[i]),
            "refinement_degraded": bool(refinement_degraded[i]),
            "selected_forecast_preview": selected_forecast_preview[i] if selected_forecast_preview[i] else "",
            "final_answer_preview": final_answer_preview[i] if final_answer_preview[i] else "",
            "raw_model_output_tail": tail_lines(output_text, 10),
        }

    def _build_row(i: int, category: str) -> dict[str, Any]:
        row = _build_base_row(i, category)
        if category == "near_miss_94_95":
            output_text = str(sample_outputs[i]) if i < len(sample_outputs) else ""
            gt_text = str(sample_gts[i]) if i < len(sample_gts) else ""
            pred_values = extract_values_from_text(output_text)
            gt_values = extract_values_from_text(gt_text)
            if pred_values and gt_values and len(pred_values) < len(gt_values):
                filled_pred = list(pred_values)
                filled_pred.extend([filled_pred[-1]] * (len(gt_values) - len(filled_pred)))
                filled_orig_mse, filled_orig_mae = orig_mse_mae(filled_pred, gt_values)
                filled_norm_mse, filled_norm_mae = normalized_mse_mae(filled_pred, gt_values)
            else:
                filled_orig_mse, filled_orig_mae = float("nan"), float("nan")
                filled_norm_mse, filled_norm_mae = float("nan"), float("nan")
            row["filled_orig_mse"] = float(filled_orig_mse)
            row["filled_orig_mae"] = float(filled_orig_mae)
            row["filled_norm_mse"] = float(filled_norm_mse)
            row["filled_norm_mae"] = float(filled_norm_mae)
        return row

    sample_rows: list[dict[str, Any]] = []
    chosen_indices: set[int] = set()

    def _append_rows(
        *,
        indices: list[int],
        category: str,
        limit: int,
        unique_reason: bool,
        sort_key: Callable[[int], tuple[Any, ...]],
    ) -> None:
        ordered = sorted([idx for idx in indices if idx not in chosen_indices], key=sort_key)
        added = 0
        seen_reasons: set[str] = set()
        for idx in ordered:
            reason = debug_reason[idx]
            if unique_reason and reason in seen_reasons:
                continue
            sample_rows.append(_build_row(idx, category))
            chosen_indices.add(idx)
            seen_reasons.add(reason)
            added += 1
            if added >= limit:
                return
        if unique_reason and added < limit:
            for idx in ordered:
                if idx in chosen_indices:
                    continue
                sample_rows.append(_build_row(idx, category))
                chosen_indices.add(idx)
                added += 1
                if added >= limit:
                    return

    near_miss_indices = [i for i in range(n) if is_94_95[i]]
    critical_failure_indices = [
        i
        for i in range(n)
        if debug_bucket[i] in {"tool_error", "workflow_violation", "format_failure", "hard_failure"}
    ]
    quality_regression_indices = [i for i in range(n) if quality_regression_mask[i]]
    success_indices = [i for i in range(n) if success_mask[i]]

    _append_rows(
        indices=near_miss_indices,
        category="near_miss_94_95",
        limit=2,
        unique_reason=False,
        sort_key=_failure_sort_key,
    )
    _append_rows(
        indices=critical_failure_indices,
        category="critical_failure",
        limit=6,
        unique_reason=True,
        sort_key=_failure_sort_key,
    )
    _append_rows(
        indices=quality_regression_indices,
        category="quality_regression",
        limit=2,
        unique_reason=False,
        sort_key=_failure_sort_key,
    )
    _append_rows(
        indices=success_indices,
        category="best_success",
        limit=2,
        unique_reason=False,
        sort_key=_success_sort_key,
    )

    debug_dir = debug_dir or os.getenv("TS_MIN_DEBUG_DIR", os.path.join(os.getcwd(), "logs", "debug"))
    os.makedirs(debug_dir, exist_ok=True)
    agg_file = os.getenv("TS_MIN_EVAL_AGG_FILE", os.path.join(debug_dir, "eval_step_aggregate.jsonl"))
    sample_file = os.getenv("TS_MIN_EVAL_SAMPLE_FILE", os.path.join(debug_dir, "eval_step_samples.jsonl"))

    with open(agg_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(agg_row, ensure_ascii=False) + "\n")

    with open(sample_file, "a", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return agg_row, sample_rows


__all__ = [
    "build_compact_validation_debug_summary",
    "evaluate_validation_reward_manager",
    "extract_values_from_text",
    "normalized_mse_mae",
    "orig_mse_mae",
    "percentile",
    "tail_lines",
    "to_bool_list",
    "to_float_list",
    "to_str_list",
    "write_min_eval_debug_files",
]
