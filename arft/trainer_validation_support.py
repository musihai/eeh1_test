from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import torch

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
        return

    pred_len = to_float_list(reward_extra_infos_dict.get("pred_len"), n)
    expected_len = to_float_list(
        reward_extra_infos_dict.get("expected_len") or reward_extra_infos_dict.get("gt_len"),
        n,
    )
    orig_mse = to_float_list(
        reward_extra_infos_dict.get("orig_mse") or reward_extra_infos_dict.get("raw_mse"),
        n,
    )
    orig_mae = to_float_list(
        reward_extra_infos_dict.get("orig_mae") or reward_extra_infos_dict.get("raw_mae"),
        n,
    )
    norm_mse = to_float_list(reward_extra_infos_dict.get("norm_mse"), n)
    norm_mae = to_float_list(reward_extra_infos_dict.get("norm_mae"), n)
    has_answer_tag = to_bool_list(reward_extra_infos_dict.get("has_answer_tag"), n)
    has_answer_close = to_bool_list(reward_extra_infos_dict.get("has_answer_close"), n)
    was_clipped = to_bool_list(reward_extra_infos_dict.get("was_clipped"), n)
    format_failure_reason = to_str_list(reward_extra_infos_dict.get("format_failure_reason"), n)
    final_answer_reject_reason = to_str_list(
        reward_extra_infos_dict.get("final_answer_reject_reason"),
        n,
    )
    length_hard_fail = to_float_list(reward_extra_infos_dict.get("length_hard_fail"), n, default=0.0)
    strict_length_match = to_bool_list(reward_extra_infos_dict.get("strict_length_match"), n)
    trainer_seq_score = to_float_list(
        reward_extra_infos_dict.get("trainer_seq_score") or reward_extra_infos_dict.get("score"),
        n,
        default=float("nan"),
    )
    generation_stop_reason = to_str_list(reward_extra_infos_dict.get("generation_stop_reason"), n)
    generation_finish_reason = to_str_list(reward_extra_infos_dict.get("generation_finish_reason"), n)
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
    selected_forecast_len_match = to_bool_list(
        reward_extra_infos_dict.get("selected_forecast_len_match"),
        n,
    )
    selected_forecast_exact_copy = to_bool_list(
        reward_extra_infos_dict.get("selected_forecast_exact_copy"),
        n,
    )
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
    refinement_compare_len = to_float_list(
        reward_extra_infos_dict.get("refinement_compare_len"),
        n,
        default=float("nan"),
    )
    refinement_changed_value_count = to_float_list(
        reward_extra_infos_dict.get("refinement_changed_value_count"),
        n,
        default=float("nan"),
    )
    refinement_first_changed_index = to_float_list(
        reward_extra_infos_dict.get("refinement_first_changed_index"),
        n,
        default=float("nan"),
    )
    refinement_change_mean_abs = to_float_list(
        reward_extra_infos_dict.get("refinement_change_mean_abs"),
        n,
        default=float("nan"),
    )
    refinement_change_max_abs = to_float_list(
        reward_extra_infos_dict.get("refinement_change_max_abs"),
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
    feature_tool_count = to_float_list(
        reward_extra_infos_dict.get("feature_tool_count"),
        n,
        default=float("nan"),
    )
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
    tool_call_count = to_float_list(
        reward_extra_infos_dict.get("tool_call_count"),
        n,
        default=float("nan"),
    )
    response_token_len = to_float_list(
        reward_extra_infos_dict.get("response_token_len"),
        n,
        default=float("nan"),
    )
    history_analysis_count = to_float_list(
        reward_extra_infos_dict.get("history_analysis_count"),
        n,
        default=float("nan"),
    )
    illegal_turn3_tool_call_count = to_float_list(
        reward_extra_infos_dict.get("illegal_turn3_tool_call_count"),
        n,
        default=float("nan"),
    )
    prediction_model_defaulted = to_bool_list(
        reward_extra_infos_dict.get("prediction_model_defaulted"),
        n,
    )
    prediction_requested_model = to_str_list(
        reward_extra_infos_dict.get("prediction_requested_model"),
        n,
    )
    feature_tool_signature = to_str_list(
        reward_extra_infos_dict.get("feature_tool_signature"),
        n,
        default="none",
    )
    required_feature_tool_signature = to_str_list(
        reward_extra_infos_dict.get("required_feature_tool_signature"),
        n,
        default="none",
    )
    tool_call_sequence = to_str_list(
        reward_extra_infos_dict.get("tool_call_sequence"),
        n,
        default="none",
    )
    analysis_state_signature = to_str_list(
        reward_extra_infos_dict.get("analysis_state_signature"),
        n,
        default="none",
    )
    workflow_status = to_str_list(
        reward_extra_infos_dict.get("workflow_status"),
        n,
    )
    turn_stage = to_str_list(
        reward_extra_infos_dict.get("turn_stage"),
        n,
    )
    prediction_tool_error = to_str_list(
        reward_extra_infos_dict.get("prediction_tool_error"),
        n,
    )
    selected_forecast_preview = to_str_list(
        reward_extra_infos_dict.get("selected_forecast_preview"),
        n,
    )
    final_answer_preview = to_str_list(
        reward_extra_infos_dict.get("final_answer_preview"),
        n,
    )

    pred_len_arr = np.asarray(pred_len, dtype=np.float64)
    expected_len_arr = np.asarray(expected_len, dtype=np.float64)
    valid_pred_len = np.isfinite(pred_len_arr)
    valid_expected_len = np.isfinite(expected_len_arr)

    is_96 = pred_len_arr == 96
    is_94_95 = np.logical_or(pred_len_arr == 94, pred_len_arr == 95)
    is_lt_96 = pred_len_arr < 96
    is_gt_96 = pred_len_arr > 96
    missing_close = np.asarray([reason == "missing_answer_close_tag" for reason in format_failure_reason], dtype=bool)
    invalid_answer_shape = np.asarray(
        [reason.startswith("invalid_answer_shape") for reason in final_answer_reject_reason],
        dtype=bool,
    )
    strict_length_match_arr = np.asarray(strict_length_match, dtype=bool)
    exact_expected_match = np.logical_and.reduce([valid_pred_len, valid_expected_len, pred_len_arr == expected_len_arr])
    final_answer_accept = np.logical_and.reduce(
        [
            np.asarray(has_answer_tag, dtype=bool),
            np.asarray(has_answer_close, dtype=bool),
            exact_expected_match,
            np.asarray([reason == "" for reason in final_answer_reject_reason], dtype=bool),
        ]
    )

    success_mask = np.logical_and.reduce(
        [
            final_answer_accept,
            np.asarray([np.isfinite(v) for v in orig_mse], dtype=bool),
            np.asarray([np.isfinite(v) for v in orig_mae], dtype=bool),
        ]
    )

    success_orig_mse_values = [orig_mse[i] for i in range(n) if success_mask[i] and np.isfinite(orig_mse[i])]
    success_orig_mae_values = [orig_mae[i] for i in range(n) if success_mask[i] and np.isfinite(orig_mae[i])]
    success_norm_mse_values = [norm_mse[i] for i in range(n) if success_mask[i] and np.isfinite(norm_mse[i])]
    success_norm_mae_values = [norm_mae[i] for i in range(n) if success_mask[i] and np.isfinite(norm_mae[i])]
    reward_values = [float(v) for v in sample_scores if isinstance(v, (int, float)) and np.isfinite(v)]
    length_hard_fail_values = [float(v) for v in length_hard_fail if np.isfinite(v)]
    trainer_seq_values = [float(v) for v in trainer_seq_score if np.isfinite(v)]
    final_answer_reject_counter = Counter(reason for reason in final_answer_reject_reason if reason)
    format_failure_reason_counter = Counter(reason for reason in format_failure_reason if reason)
    generation_stop_reason_counter = Counter(reason for reason in generation_stop_reason if reason)
    generation_finish_reason_counter = Counter(reason for reason in generation_finish_reason if reason)
    selected_model_counter = Counter(model for model in selected_model if model)
    feature_tool_signature_counter = Counter(signature for signature in feature_tool_signature if signature)
    required_feature_tool_signature_counter = Counter(signature for signature in required_feature_tool_signature if signature)
    tool_call_sequence_counter = Counter((signature if signature else "none") for signature in tool_call_sequence)
    analysis_state_signature_counter = Counter(signature for signature in analysis_state_signature if signature)
    prediction_requested_model_counter = Counter(model for model in prediction_requested_model if model)
    workflow_status_counter = Counter(status for status in workflow_status if status)
    turn_stage_counter = Counter(stage for stage in turn_stage if stage)
    prediction_tool_error_count = int(sum(1 for value in prediction_tool_error if value))

    selected_forecast_orig_mse_values = [
        float(v) for v in selected_forecast_orig_mse if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    final_vs_selected_mse_values = [
        float(v) for v in final_vs_selected_mse if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    refinement_delta_orig_mse_values = [
        float(v) for v in refinement_delta_orig_mse if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    refinement_compare_len_values = [
        float(v) for v in refinement_compare_len if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    refinement_changed_value_count_values = [
        float(v) for v in refinement_changed_value_count if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    refinement_first_changed_index_values = [
        float(v) for v in refinement_first_changed_index if isinstance(v, (int, float)) and np.isfinite(v) and v >= 0
    ]
    refinement_change_mean_abs_values = [
        float(v) for v in refinement_change_mean_abs if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    refinement_change_max_abs_values = [
        float(v) for v in refinement_change_max_abs if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    analysis_coverage_ratio_values = [
        float(v) for v in analysis_coverage_ratio if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    feature_tool_count_values = [
        float(v) for v in feature_tool_count if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    required_feature_tool_count_values = [
        float(v) for v in required_feature_tool_count if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    missing_required_feature_tool_count_values = [
        float(v) for v in missing_required_feature_tool_count if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    prediction_call_count_values = [
        float(v) for v in prediction_call_count if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    required_step_budget_values = [
        float(v) for v in required_step_budget if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    tool_call_count_values = [
        float(v) for v in tool_call_count if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    response_token_len_values = [
        float(v) for v in response_token_len if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    history_analysis_count_values = [
        float(v) for v in history_analysis_count if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    illegal_turn3_tool_call_count_values = [
        float(v) for v in illegal_turn3_tool_call_count if isinstance(v, (int, float)) and np.isfinite(v)
    ]

    total = float(n)
    known_model_shares = {
        "patchtst_share": float(selected_model_counter.get("patchtst", 0) / total),
        "itransformer_share": float(selected_model_counter.get("itransformer", 0) / total),
        "arima_share": float(selected_model_counter.get("arima", 0) / total),
        "chronos2_share": float(selected_model_counter.get("chronos2", 0) / total),
        "unknown_model_share": float(selected_model_counter.get("unknown", 0) / total),
    }
    agg_row = {
        "step": int(global_steps),
        "total_samples": int(n),
        "exact_96_ratio": float(np.sum(is_96) / total),
        "pred_len_96_ratio": float(np.sum(is_96) / total),
        "pred_len_94_95_ratio": float(np.sum(is_94_95) / total),
        "pred_len_lt_96_ratio": float(np.sum(np.logical_and(is_lt_96, valid_pred_len)) / total),
        "pred_len_gt_96_ratio": float(np.sum(np.logical_and(is_gt_96, valid_pred_len)) / total),
        "has_answer_tag_ratio": float(np.mean(np.asarray(has_answer_tag, dtype=np.float64))),
        "has_answer_close_ratio": float(np.mean(np.asarray(has_answer_close, dtype=np.float64))),
        "final_answer_accept_ratio": float(np.mean(final_answer_accept.astype(np.float64))),
        "strict_length_match_ratio": float(np.mean(strict_length_match_arr.astype(np.float64))),
        "missing_answer_close_tag_count": int(np.sum(missing_close)),
        "invalid_answer_shape_count": int(np.sum(invalid_answer_shape)),
        "was_clipped_count": int(np.sum(np.asarray(was_clipped, dtype=bool))),
        "orig_mse_mean": float(np.mean(success_orig_mse_values)) if success_orig_mse_values else float("nan"),
        "orig_mse_p50": percentile(success_orig_mse_values, 50),
        "orig_mse_p90": percentile(success_orig_mse_values, 90),
        "orig_mae_mean": float(np.mean(success_orig_mae_values)) if success_orig_mae_values else float("nan"),
        "orig_mae_p50": percentile(success_orig_mae_values, 50),
        "orig_mae_p90": percentile(success_orig_mae_values, 90),
        "norm_mse_mean": float(np.mean(success_norm_mse_values)) if success_norm_mse_values else float("nan"),
        "norm_mse_p50": percentile(success_norm_mse_values, 50),
        "norm_mse_p90": percentile(success_norm_mse_values, 90),
        "norm_mae_mean": float(np.mean(success_norm_mae_values)) if success_norm_mae_values else float("nan"),
        "norm_mae_p50": percentile(success_norm_mae_values, 50),
        "norm_mae_p90": percentile(success_norm_mae_values, 90),
        "success_raw_mse_mean": float(np.mean(success_orig_mse_values)) if success_orig_mse_values else float("nan"),
        "success_raw_mse_p50": percentile(success_orig_mse_values, 50),
        "success_raw_mse_p90": percentile(success_orig_mse_values, 90),
        "success_raw_mae_mean": float(np.mean(success_orig_mae_values)) if success_orig_mae_values else float("nan"),
        "success_raw_mae_p50": percentile(success_orig_mae_values, 50),
        "success_raw_mae_p90": percentile(success_orig_mae_values, 90),
        "validation_reward_mean": float(np.mean(reward_values)) if reward_values else float("nan"),
        "length_hard_fail_ratio": float(np.mean(length_hard_fail_values)) if length_hard_fail_values else float("nan"),
        "length_hard_fail_mean": float(np.mean(length_hard_fail_values)) if length_hard_fail_values else float("nan"),
        "trainer_seq_score_mean": float(np.mean(trainer_seq_values)) if trainer_seq_values else float("nan"),
        "selected_forecast_orig_mse_mean": float(np.mean(selected_forecast_orig_mse_values))
        if selected_forecast_orig_mse_values
        else float("nan"),
        "selected_forecast_len_match_ratio": float(np.mean(np.asarray(selected_forecast_len_match, dtype=np.float64))),
        "selected_forecast_exact_copy_ratio": float(
            np.mean(np.asarray(selected_forecast_exact_copy, dtype=np.float64))
        ),
        "final_vs_selected_mse_mean": float(np.mean(final_vs_selected_mse_values))
        if final_vs_selected_mse_values
        else float("nan"),
        "refinement_delta_orig_mse_mean": float(np.mean(refinement_delta_orig_mse_values))
        if refinement_delta_orig_mse_values
        else float("nan"),
        "refinement_compare_len_mean": float(np.mean(refinement_compare_len_values))
        if refinement_compare_len_values
        else float("nan"),
        "refinement_changed_value_count_mean": float(np.mean(refinement_changed_value_count_values))
        if refinement_changed_value_count_values
        else float("nan"),
        "refinement_first_changed_index_mean": float(np.mean(refinement_first_changed_index_values))
        if refinement_first_changed_index_values
        else float("nan"),
        "refinement_change_mean_abs_mean": float(np.mean(refinement_change_mean_abs_values))
        if refinement_change_mean_abs_values
        else float("nan"),
        "refinement_change_max_abs_mean": float(np.mean(refinement_change_max_abs_values))
        if refinement_change_max_abs_values
        else float("nan"),
        "refinement_changed_ratio": float(np.mean(np.asarray(refinement_changed, dtype=np.float64))),
        "refinement_improved_ratio": float(np.mean(np.asarray(refinement_improved, dtype=np.float64))),
        "refinement_degraded_ratio": float(np.mean(np.asarray(refinement_degraded, dtype=np.float64))),
        "analysis_coverage_ratio_mean": float(np.mean(analysis_coverage_ratio_values))
        if analysis_coverage_ratio_values
        else float("nan"),
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
        "required_step_budget_mean": float(np.mean(required_step_budget_values))
        if required_step_budget_values
        else float("nan"),
        "tool_call_count_mean": float(np.mean(tool_call_count_values)) if tool_call_count_values else float("nan"),
        "response_token_len_mean": float(np.mean(response_token_len_values))
        if response_token_len_values
        else float("nan"),
        "history_analysis_count_mean": float(np.mean(history_analysis_count_values))
        if history_analysis_count_values
        else float("nan"),
        "no_tool_call_ratio": float(
            np.mean((np.asarray(tool_call_count_values, dtype=np.float64) <= 0.0).astype(np.float64))
        )
        if tool_call_count_values
        else float("nan"),
        "no_history_analysis_ratio": float(
            np.mean((np.asarray(history_analysis_count_values, dtype=np.float64) <= 0.0).astype(np.float64))
        )
        if history_analysis_count_values
        else float("nan"),
        "illegal_turn3_tool_call_ratio": float(
            np.mean((np.asarray(illegal_turn3_tool_call_count_values, dtype=np.float64) > 0).astype(np.float64))
        )
        if illegal_turn3_tool_call_count_values
        else float("nan"),
        "prediction_model_defaulted_ratio": float(np.mean(np.asarray(prediction_model_defaulted, dtype=np.float64))),
        "prediction_tool_error_count": prediction_tool_error_count,
        "selected_model_distribution": {str(k): int(v) for k, v in sorted(selected_model_counter.items())},
        "format_failure_reason_distribution": {str(k): int(v) for k, v in sorted(format_failure_reason_counter.items())},
        "final_answer_reject_reason_distribution": {
            str(k): int(v) for k, v in sorted(final_answer_reject_counter.items())
        },
        "generation_stop_reason_distribution": {
            str(k): int(v) for k, v in sorted(generation_stop_reason_counter.items())
        },
        "generation_finish_reason_distribution": {
            str(k): int(v) for k, v in sorted(generation_finish_reason_counter.items())
        },
        "feature_tool_signature_distribution": {
            str(k): int(v) for k, v in sorted(feature_tool_signature_counter.items())
        },
        "required_feature_tool_signature_distribution": {
            str(k): int(v) for k, v in sorted(required_feature_tool_signature_counter.items())
        },
        "tool_call_sequence_distribution": {
            str(k): int(v) for k, v in sorted(tool_call_sequence_counter.items())
        },
        "analysis_state_signature_distribution": {
            str(k): int(v) for k, v in sorted(analysis_state_signature_counter.items())
        },
        "prediction_requested_model_distribution": {
            str(k): int(v) for k, v in sorted(prediction_requested_model_counter.items())
        },
        "workflow_status_distribution": {str(k): int(v) for k, v in sorted(workflow_status_counter.items())},
        "turn_stage_distribution": {str(k): int(v) for k, v in sorted(turn_stage_counter.items())},
    }
    agg_row.update(known_model_shares)

    all_indices = list(range(n))
    success_indices = [i for i in all_indices if success_mask[i]]
    near_miss_indices = [i for i in all_indices if is_94_95[i]]
    failure_indices = [i for i in all_indices if i not in set(success_indices)]

    rng = random.Random(int(global_steps) + 20260318)

    def _sample(indices: list[int], k: int) -> list[int]:
        if len(indices) <= k:
            return indices
        return rng.sample(indices, k)

    pick_success = _sample(success_indices, 10)
    pick_failure = _sample(failure_indices, 10)
    pick_nearmiss = _sample(near_miss_indices, 10)

    sample_rows: list[dict] = []

    def _build_base_row(i: int, category: str) -> dict:
        output_text = str(sample_outputs[i]) if i < len(sample_outputs) else ""
        return {
            "step": int(global_steps),
            "category": category,
            "sample_id": str(sample_uids[i]) if i < len(sample_uids) else f"sample_{i}",
            "selected_model": selected_model[i],
            "pred_len": int(pred_len_arr[i]) if np.isfinite(pred_len_arr[i]) else -1,
            "expected_len": int(expected_len_arr[i]) if np.isfinite(expected_len_arr[i]) else -1,
            "orig_mse": float(orig_mse[i]) if np.isfinite(orig_mse[i]) else float("nan"),
            "orig_mae": float(orig_mae[i]) if np.isfinite(orig_mae[i]) else float("nan"),
            "norm_mse": float(norm_mse[i]) if np.isfinite(norm_mse[i]) else float("nan"),
            "norm_mae": float(norm_mae[i]) if np.isfinite(norm_mae[i]) else float("nan"),
            "raw_mse": float(orig_mse[i]) if np.isfinite(orig_mse[i]) else float("nan"),
            "raw_mae": float(orig_mae[i]) if np.isfinite(orig_mae[i]) else float("nan"),
            "has_answer_tag": bool(has_answer_tag[i]),
            "has_answer_close": bool(has_answer_close[i]),
            "failure_reason": format_failure_reason[i] if format_failure_reason[i] else "",
            "final_answer_reject_reason": final_answer_reject_reason[i] if final_answer_reject_reason[i] else "",
            "generation_stop_reason": generation_stop_reason[i] if generation_stop_reason[i] else "",
            "generation_finish_reason": generation_finish_reason[i] if generation_finish_reason[i] else "",
            "strict_length_match": bool(strict_length_match_arr[i]),
            "length_hard_fail": bool(length_hard_fail[i]),
            "trainer_seq_score": float(trainer_seq_score[i]) if np.isfinite(trainer_seq_score[i]) else float("nan"),
            "selected_forecast_orig_mse": float(selected_forecast_orig_mse[i])
            if np.isfinite(selected_forecast_orig_mse[i])
            else float("nan"),
            "selected_forecast_len_match": bool(selected_forecast_len_match[i]),
            "selected_forecast_exact_copy": bool(selected_forecast_exact_copy[i]),
            "final_vs_selected_mse": float(final_vs_selected_mse[i])
            if np.isfinite(final_vs_selected_mse[i])
            else float("nan"),
            "refinement_delta_orig_mse": float(refinement_delta_orig_mse[i])
            if np.isfinite(refinement_delta_orig_mse[i])
            else float("nan"),
            "refinement_compare_len": int(refinement_compare_len[i]) if np.isfinite(refinement_compare_len[i]) else -1,
            "refinement_changed_value_count": int(refinement_changed_value_count[i])
            if np.isfinite(refinement_changed_value_count[i])
            else -1,
            "refinement_first_changed_index": int(refinement_first_changed_index[i])
            if np.isfinite(refinement_first_changed_index[i])
            else -1,
            "refinement_change_mean_abs": float(refinement_change_mean_abs[i])
            if np.isfinite(refinement_change_mean_abs[i])
            else float("nan"),
            "refinement_change_max_abs": float(refinement_change_max_abs[i])
            if np.isfinite(refinement_change_max_abs[i])
            else float("nan"),
            "refinement_changed": bool(refinement_changed[i]),
            "refinement_improved": bool(refinement_improved[i]),
            "refinement_degraded": bool(refinement_degraded[i]),
            "analysis_coverage_ratio": float(analysis_coverage_ratio[i])
            if np.isfinite(analysis_coverage_ratio[i])
            else float("nan"),
            "feature_tool_count": int(feature_tool_count[i]) if np.isfinite(feature_tool_count[i]) else -1,
            "required_feature_tool_count": int(required_feature_tool_count[i])
            if np.isfinite(required_feature_tool_count[i])
            else -1,
            "missing_required_feature_tool_count": int(missing_required_feature_tool_count[i])
            if np.isfinite(missing_required_feature_tool_count[i])
            else -1,
            "prediction_call_count": int(prediction_call_count[i]) if np.isfinite(prediction_call_count[i]) else -1,
            "prediction_step_index": int(prediction_step_index[i]) if np.isfinite(prediction_step_index[i]) else -1,
            "final_answer_step_index": int(final_answer_step_index[i]) if np.isfinite(final_answer_step_index[i]) else -1,
            "required_step_budget": int(required_step_budget[i]) if np.isfinite(required_step_budget[i]) else -1,
            "tool_call_count": int(tool_call_count[i]) if np.isfinite(tool_call_count[i]) else -1,
            "response_token_len": int(response_token_len[i]) if np.isfinite(response_token_len[i]) else -1,
            "history_analysis_count": int(history_analysis_count[i]) if np.isfinite(history_analysis_count[i]) else -1,
            "illegal_turn3_tool_call_count": int(illegal_turn3_tool_call_count[i])
            if np.isfinite(illegal_turn3_tool_call_count[i])
            else -1,
            "prediction_requested_model": prediction_requested_model[i] if prediction_requested_model[i] else "",
            "prediction_model_defaulted": bool(prediction_model_defaulted[i]),
            "feature_tool_signature": feature_tool_signature[i] if feature_tool_signature[i] else "",
            "required_feature_tool_signature": required_feature_tool_signature[i] if required_feature_tool_signature[i] else "",
            "tool_call_sequence": tool_call_sequence[i] if tool_call_sequence[i] else "none",
            "analysis_state_signature": analysis_state_signature[i] if analysis_state_signature[i] else "",
            "workflow_status": workflow_status[i] if workflow_status[i] else "",
            "turn_stage": turn_stage[i] if turn_stage[i] else "",
            "prediction_tool_error": prediction_tool_error[i] if prediction_tool_error[i] else "",
            "selected_forecast_preview": selected_forecast_preview[i] if selected_forecast_preview[i] else "",
            "final_answer_preview": final_answer_preview[i] if final_answer_preview[i] else "",
            "raw_model_output_tail": tail_lines(output_text, 10),
        }

    for i in pick_success:
        row = _build_base_row(i, "success")
        output_text = str(sample_outputs[i]) if i < len(sample_outputs) else ""
        gt_text = str(sample_gts[i]) if i < len(sample_gts) else ""
        row["pred_values"] = extract_values_from_text(output_text)
        row["gt_values"] = extract_values_from_text(gt_text)
        sample_rows.append(row)

    for i in pick_failure:
        row = _build_base_row(i, "failure")
        sample_rows.append(row)

    for i in pick_nearmiss:
        row = _build_base_row(i, "near_miss_94_95")
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
        row["filled_raw_mse"] = float(filled_orig_mse)
        row["filled_raw_mae"] = float(filled_orig_mae)
        sample_rows.append(row)

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
