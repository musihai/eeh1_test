from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np

from recipe.time_series_forecast.reward_metrics import (
    PREDICTION_ERROR_SCORE_WEIGHT,
    compute_change_point_score,
    compute_format_score,
    compute_length_penalty,
    compute_length_score,
    compute_mse_score,
    compute_norm_mse_score,
    compute_recovery_penalty,
    compute_season_trend_score,
    decompose,
    find_change_points,
    infer_format_failure_reason,
    mean_squared_error,
    mean_squared_error_season_trend,
    moving_avg,
    normalize_for_reward,
    series_decomp,
)
from recipe.time_series_forecast.reward_protocol import (
    canonicalize_forecast_values,
    clamp_turn3_answer_horizon,
    count_numeric_only_lines,
    detect_suffix_repetition,
    extract_answer,
    extract_answer_region,
    extract_forecast_block,
    extract_ground_truth_values,
    extract_strict_protocol_answer,
    extract_tail_lines,
    extract_values_from_time_series_string,
    infer_answer_shape_failure,
    is_plain_forecast_block_response,
    looks_like_forecast_answer,
    normalized_nonempty_lines,
    parse_final_answer_protocol,
    recover_protocol_answer,
    trailing_text_after_close,
)
from verl.utils.chain_debug import append_chain_debug, chain_debug_enabled, short_text


# Paper-aligned reward uses fixed-weight multi-view aggregation: accuracy,
# seasonal/trend consistency, turning-point alignment, format validity, and
# length consistency all participate directly in the terminal score.
ENABLE_CHANGE_POINT_SCORE = True
ENABLE_SEASON_TREND_SCORE = True
TURN3_SUCCESS_SAMPLE_RATE = max(int(os.getenv("TS_TURN3_SUCCESS_SAMPLE_RATE", "100") or 100), 1)


def _bool_from_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _should_attach_protocol_compare(extra_info_dict: dict, nested_reward_info: dict) -> bool:
    validate_value = extra_info_dict.get("validate")
    if validate_value is None:
        validate_value = nested_reward_info.get("validate")
    return chain_debug_enabled() or _bool_from_value(validate_value)


def _count_answer_lines(text: str) -> int:
    return len(normalized_nonempty_lines(extract_answer_region(text)))


def append_turn3_generation_debug(
    *,
    data_source: str,
    solution_str: Optional[str],
    ground_truth: str,
    sample_uid: Optional[str],
    output_source: Optional[str],
    format_score: float,
    length_score: float,
    length_penalty: float,
    raw_overrun_penalty: float = 0.0,
    recovery_penalty: float,
    mse_score: float,
    change_point_score: float,
    season_trend_score: float,
    final_score: float,
    orig_mse: float,
    orig_mae: float,
    norm_mse: float,
    norm_mae: float,
    raw_mse: float,
    raw_mae: float,
    length_hard_fail: bool,
    strict_length_match: bool,
    format_parse_mode: str,
    raw_protocol_reject_reason: str,
    was_recovered: bool,
    format_failure_reason: str,
    has_answer_tag: bool,
    has_answer_open: bool,
    has_answer_close: bool,
    pred_values: list[float],
    gt_values: list[float],
) -> None:
    pred_len = len(pred_values)
    gt_len = len(gt_values)
    len_gap = abs(pred_len - gt_len) if gt_len > 0 else pred_len
    is_failure = bool(format_score < 0 or recovery_penalty > 0 or (gt_len > 0 and pred_len != gt_len))
    raw_text = solution_str or ""
    parsed_answer = extract_answer(raw_text)
    answer_region = extract_answer_region(raw_text)
    numeric_line_count, non_numeric_line_count = count_numeric_only_lines(answer_region)
    suffix_repetition_detected, suffix_repetition_period, suffix_repetition_repeats = detect_suffix_repetition(
        pred_values
    )
    trailing_after_close = trailing_text_after_close(raw_text)
    value_unique_ratio = (
        float(len({round(float(v), 6) for v in pred_values}) / pred_len) if pred_len > 0 else float("nan")
    )
    was_clipped = bool(raw_protocol_reject_reason == "missing_answer_close_tag")
    under_generation = bool(gt_len > 0 and pred_len < gt_len)
    over_generation = bool(gt_len > 0 and pred_len > gt_len)
    exact_generation = bool(gt_len > 0 and pred_len == gt_len)

    payload = {
        "data_source": data_source,
        "sample_uid": sample_uid,
        "output_source": output_source,
        "is_failure": is_failure,
        "was_clipped": was_clipped,
        "format_failure_reason": format_failure_reason,
        "has_answer_tag": bool(has_answer_tag),
        "has_answer_open": bool(has_answer_open),
        "has_answer_close": bool(has_answer_close),
        "answer_open_count": int(raw_text.count("<answer>")),
        "answer_close_count": int(raw_text.count("</answer>")),
        "think_open_count": int(raw_text.count("<think>")),
        "think_close_count": int(raw_text.count("</think>")),
        "raw_pred_len": pred_len,
        "pred_len": pred_len,
        "gt_len": gt_len,
        "num_values": pred_len,
        "len_gap": int(len_gap),
        "under_generation": under_generation,
        "over_generation": over_generation,
        "exact_generation": exact_generation,
        "format_score": float(format_score),
        "format_parse_mode": format_parse_mode,
        "raw_protocol_reject_reason": raw_protocol_reject_reason,
        "was_recovered": bool(was_recovered),
        "length_score": float(length_score),
        "length_penalty": float(length_penalty),
        "raw_overrun_penalty": float(raw_overrun_penalty),
        "recovery_penalty": float(recovery_penalty),
        "mse_score": float(mse_score),
        "change_point_score": float(change_point_score),
        "season_trend_score": float(season_trend_score),
        "final_score": float(final_score),
        "orig_mse": orig_mse,
        "orig_mae": orig_mae,
        "norm_mse": norm_mse,
        "norm_mae": norm_mae,
        "raw_mse": raw_mse,
        "raw_mae": raw_mae,
        "length_hard_fail": bool(length_hard_fail),
        "strict_length_match": bool(strict_length_match),
        "answer_region_line_count": int(len([line for line in answer_region.splitlines() if line.strip()])),
        "numeric_only_line_count": int(numeric_line_count),
        "non_numeric_line_count": int(non_numeric_line_count),
        "suffix_repetition_detected": bool(suffix_repetition_detected),
        "suffix_repetition_period": int(suffix_repetition_period),
        "suffix_repetition_repeats": int(suffix_repetition_repeats),
        "value_unique_ratio": value_unique_ratio,
        "trailing_text_after_close_len": int(len(trailing_after_close)),
        "trailing_text_after_close_head": short_text(trailing_after_close, 200),
        "raw_text_tail": extract_tail_lines(raw_text, 10),
    }

    if is_failure:
        payload.update(
            {
                "raw_model_output": raw_text,
                "parsed_answer_text": parsed_answer,
                "parsed_values": pred_values,
                "ground_truth_values": gt_values,
                "ground_truth_text": ground_truth,
                "_force_log": True,
            }
        )
    else:
        payload.update(
            {
                "raw_model_output_head": short_text(raw_text, 500),
                "parsed_answer_head": short_text(parsed_answer, 500),
                "parsed_values_head": pred_values[:20],
                "ground_truth_values_head": gt_values[:20],
                "_sample_rate": TURN3_SUCCESS_SAMPLE_RATE,
            }
        )

    append_chain_debug("turn3_generation_debug", payload)


def _compute_score_impl(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    allow_recovery: bool = False,
    *,
    emit_debug: bool = True,
    raw_overrun_penalty: float = 0.0,
) -> dict:
    """Compute the paper-aligned composite reward for time series prediction."""
    extra_info_dict = extra_info or {}
    nested_reward_info = extra_info_dict.get("reward_extra_info")
    if not isinstance(nested_reward_info, dict):
        nested_reward_info = {}

    def _resolve_extra_value(*keys: str):
        for key in keys:
            value = extra_info_dict.get(key)
            if value is not None:
                return value
        for key in keys:
            value = nested_reward_info.get(key)
            if value is not None:
                return value
        return None

    passthrough_extra_keys = (
        "sample_uid",
        "final_answer_reject_reason",
        "generation_stop_reason",
        "generation_finish_reason",
        "prediction_model_used",
        "prediction_attempt_count",
        "prediction_call_count",
        "illegal_turn3_tool_call_count",
        "prediction_requested_model",
        "prediction_model_defaulted",
        "prediction_tool_error",
        "prediction_step_index",
        "final_answer_step_index",
        "feature_tool_count",
        "feature_tool_signature",
        "required_feature_tool_count",
        "required_feature_tool_signature",
        "missing_required_feature_tool_count",
        "analysis_state_signature",
        "analysis_coverage_ratio",
        "history_analysis_count",
        "tool_call_count",
        "tool_call_sequence",
        "turn_stage",
        "workflow_status",
        "workflow_violation_reason",
        "global_step",
        "validate",
        "run_name",
        "required_step_budget",
        "prompt_char_len",
        "response_char_len",
        "response_token_len",
        "answer_line_count",
        "expected_answer_line_count",
        "wrote_expected_rows_before_stop",
        "think_token_len",
        "answer_token_len",
        "turn3_horizon_clamped",
        "turn3_horizon_clamp_reason",
        "turn3_horizon_clamp_discarded_lines",
        "turn3_horizon_clamp_valid_prefix_lines",
        "turn3_horizon_clamp_raw_answer_lines",
        "selected_forecast_orig_mse",
        "selected_forecast_orig_mae",
        "selected_forecast_norm_mse",
        "selected_forecast_norm_mae",
        "selected_forecast_preview",
        "selected_forecast_len_match",
        "selected_forecast_exact_copy",
        "final_answer_preview",
        "final_vs_selected_mse",
        "final_vs_selected_mae",
        "refinement_delta_orig_mse",
        "refinement_delta_orig_mae",
        "refinement_compare_len",
        "refinement_changed_value_count",
        "refinement_first_changed_index",
        "refinement_change_mean_abs",
        "refinement_change_max_abs",
        "refinement_changed",
        "refinement_improved",
        "refinement_degraded",
    )
    passthrough_extra_info = {
        key: value for key in passthrough_extra_keys if (value := _resolve_extra_value(key)) is not None
    }

    selected_model = str(_resolve_extra_value("prediction_model_used", "selected_model", "output_source") or "unknown")
    output_source = _resolve_extra_value("output_source", "prediction_model_used", "selected_model")
    sample_uid = _resolve_extra_value("uid", "sample_uid")

    if solution_str is None:
        append_turn3_generation_debug(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            sample_uid=sample_uid,
            output_source=output_source,
            format_score=-1.0,
            length_score=0.0,
            length_penalty=0.0,
            recovery_penalty=0.0,
            mse_score=0.0,
            change_point_score=0.0,
            season_trend_score=0.0,
            final_score=-1.0,
            orig_mse=float("nan"),
            orig_mae=float("nan"),
            norm_mse=float("nan"),
            norm_mae=float("nan"),
            raw_mse=float("nan"),
            raw_mae=float("nan"),
            length_hard_fail=False,
            strict_length_match=False,
            format_parse_mode="rejected_empty_solution",
            raw_protocol_reject_reason="empty_solution",
            was_recovered=False,
            format_failure_reason="empty_solution",
            has_answer_tag=False,
            has_answer_open=False,
            has_answer_close=False,
            pred_values=[],
            gt_values=[],
        )
        append_chain_debug(
            "reward_compute",
            {
                "data_source": data_source,
                "sample_uid": sample_uid,
                "selected_model": selected_model,
                "output_source": output_source,
                **passthrough_extra_info,
                "format_score": -1.0,
                "length_score": 0.0,
                "length_penalty": 0.0,
                "recovery_penalty": 0.0,
                "mse_score": 0.0,
                "change_point_score": 0.0,
                "season_trend_score": 0.0,
                "orig_mse": float("nan"),
                "orig_mae": float("nan"),
                "norm_mse": float("nan"),
                "norm_mae": float("nan"),
                "raw_mse": float("nan"),
                "raw_mae": float("nan"),
                "final_score": -1.0,
                "raw_pred_len": 0,
                "pred_len": 0,
                "gt_len": 0,
                "has_answer_tag": False,
                "has_answer_open": False,
                "has_answer_close": False,
                "first_5_pred_values": [],
                "first_5_gt_values": [],
                "raw_model_output_head": "",
                "parsed_answer_text_head": "",
                "parsed_values": [],
                "format_parse_mode": "rejected_empty_solution",
                "raw_protocol_reject_reason": "empty_solution",
                "was_recovered": False,
                "format_failure_reason": "empty_solution",
                "length_hard_fail": False,
                "strict_length_match": False,
            },
        )
        return {
            "score": -1.0,
            "trainer_seq_score": -1.0,
            "format_score": -1.0,
            "length_score": 0.0,
            "length_penalty": 0.0,
            "recovery_penalty": 0.0,
            "mse_score": 0.0,
            "change_point_score": 0.0,
            "season_trend_score": 0.0,
            "orig_mse": float("nan"),
            "orig_mae": float("nan"),
            "norm_mse": float("nan"),
            "norm_mae": float("nan"),
            "raw_mse": float("nan"),
            "raw_mae": float("nan"),
            "format_parse_mode": "rejected_empty_solution",
            "raw_protocol_reject_reason": "empty_solution",
            "was_recovered": False,
            "format_failure_reason": "empty_solution",
            "pred_len": 0,
            "gt_len": 0,
            "has_answer_tag": False,
            "has_answer_open": False,
            "has_answer_close": False,
            "missing_answer_close_tag": False,
            "was_clipped": False,
            "selected_model": selected_model,
            "len_gap": 0,
            "under_generation": False,
            "over_generation": False,
            "exact_generation": False,
            "length_hard_fail": False,
            "strict_length_match": False,
            **passthrough_extra_info,
        }

    has_answer_tag = bool(re.search(r"<answer>(.*?)</answer>", solution_str or "", re.DOTALL))
    has_answer_open = bool("<answer>" in (solution_str or ""))
    has_answer_close = bool("</answer>" in (solution_str or ""))
    raw_pred_values = extract_values_from_time_series_string(solution_str or "")
    gt_values = extract_ground_truth_values(ground_truth)
    gt_len = len(gt_values)
    protocol_expected_len = gt_len if gt_len > 0 else max(len(raw_pred_values), 1)
    canonical_answer, format_parse_mode, raw_protocol_reject_reason = parse_final_answer_protocol(
        solution_str,
        protocol_expected_len,
        allow_recovery=allow_recovery,
    )
    scoring_solution = f"<answer>\n{canonical_answer}\n</answer>" if canonical_answer is not None else solution_str
    pred_values = extract_values_from_time_series_string(scoring_solution)
    pred_len = len(pred_values)
    len_gap = abs(pred_len - gt_len) if gt_len > 0 else pred_len
    under_generation = bool(gt_len > 0 and pred_len < gt_len)
    over_generation = bool(gt_len > 0 and pred_len > gt_len)
    exact_generation = bool(gt_len > 0 and pred_len == gt_len)
    strict_length_match = bool(gt_len > 0 and pred_len == gt_len)
    length_hard_fail = False
    change_point_score = 0.0
    season_trend_score = 0.0

    format_score = 0.0 if canonical_answer is not None else -1.0
    length_score = 0.0
    length_penalty = 0.0
    recovery_penalty = 0.0
    mse_score = 0.0
    orig_mse: float = float("nan")
    orig_mae: float = float("nan")
    norm_mse: float = float("nan")
    norm_mae: float = float("nan")
    format_failure_reason = "ok"
    was_recovered = bool(format_parse_mode.startswith("recovered_"))
    missing_answer_close_tag = bool(raw_protocol_reject_reason == "missing_answer_close_tag")

    if format_score < 0:
        format_failure_reason = raw_protocol_reject_reason or infer_format_failure_reason(
            solution_str,
            expected_len=protocol_expected_len,
            allow_recovery=allow_recovery,
        )
    elif was_recovered:
        format_failure_reason = raw_protocol_reject_reason or "recovered_format_failure"
        recovery_penalty = compute_recovery_penalty(raw_protocol_reject_reason or "", len(raw_pred_values), gt_len)

    if format_score >= 0 and gt_len > 0 and pred_len > 0:
        min_len = min(pred_len, gt_len)
        pred_slice = pred_values[:min_len]
        gt_slice = gt_values[:min_len]
        orig_mse = float(mean_squared_error(gt_slice, pred_slice))
        orig_mae = float(np.mean(np.abs(np.asarray(gt_slice) - np.asarray(pred_slice))))
        norm_pred, norm_gt = normalize_for_reward(pred_slice, gt_slice)
        norm_mse = float(mean_squared_error(norm_gt, norm_pred))
        norm_mae = float(np.mean(np.abs(np.asarray(norm_gt) - np.asarray(norm_pred))))

    if format_score < 0:
        score = -1.0
    else:
        score = float(format_score)
        if gt_len > 0 and pred_len != gt_len:
            format_failure_reason = f"length_mismatch:{pred_len}!={gt_len}"
            length_penalty = compute_length_penalty(pred_len, gt_len)
        length_score = compute_length_score(scoring_solution, ground_truth)
        if not np.isnan(orig_mse) and not np.isnan(norm_mse):
            mse_score = compute_norm_mse_score(norm_mse)
        if ENABLE_CHANGE_POINT_SCORE:
            change_point_score = float(compute_change_point_score(scoring_solution, ground_truth))
        if ENABLE_SEASON_TREND_SCORE:
            season_trend_score = float(compute_season_trend_score(scoring_solution, ground_truth))
        score += length_score
        score += mse_score
        score += change_point_score
        score += season_trend_score
        score -= length_penalty
        score -= raw_overrun_penalty
        score -= recovery_penalty

    result = {
        "score": float(score),
        "trainer_seq_score": float(score),
        "format_score": float(format_score),
        "length_score": float(length_score),
        "length_penalty": float(length_penalty),
        "raw_overrun_penalty": float(raw_overrun_penalty),
        "recovery_penalty": float(recovery_penalty),
        "mse_score": float(mse_score),
        "change_point_score": float(change_point_score),
        "season_trend_score": float(season_trend_score),
        "orig_mse": orig_mse,
        "orig_mae": orig_mae,
        "norm_mse": norm_mse,
        "norm_mae": norm_mae,
        "raw_mse": orig_mse,
        "raw_mae": orig_mae,
        "format_parse_mode": format_parse_mode,
        "raw_protocol_reject_reason": raw_protocol_reject_reason or "",
        "was_recovered": was_recovered,
        "format_failure_reason": format_failure_reason,
        "pred_len": int(pred_len),
        "gt_len": int(gt_len),
        "has_answer_tag": bool(has_answer_tag),
        "has_answer_open": bool(has_answer_open),
        "has_answer_close": bool(has_answer_close),
        "missing_answer_close_tag": missing_answer_close_tag,
        "was_clipped": missing_answer_close_tag,
        "selected_model": selected_model,
        "len_gap": int(len_gap),
        "under_generation": under_generation,
        "over_generation": over_generation,
        "exact_generation": exact_generation,
        "length_hard_fail": bool(length_hard_fail),
        "strict_length_match": bool(strict_length_match),
        **passthrough_extra_info,
    }

    if emit_debug:
        append_chain_debug(
            "reward_compute",
            {
                "data_source": data_source,
                "sample_uid": sample_uid,
                "selected_model": selected_model,
                "output_source": output_source,
                **passthrough_extra_info,
                "format_score": float(format_score),
                "length_score": float(length_score),
                "length_penalty": float(length_penalty),
                "raw_overrun_penalty": float(raw_overrun_penalty),
                "recovery_penalty": float(recovery_penalty),
                "mse_score": float(mse_score),
                "change_point_score": float(change_point_score),
                "season_trend_score": float(season_trend_score),
                "orig_mse": orig_mse,
                "orig_mae": orig_mae,
                "norm_mse": norm_mse,
                "norm_mae": norm_mae,
                "raw_mse": orig_mse,
                "raw_mae": orig_mae,
                "final_score": float(score),
                "raw_pred_len": pred_len,
                "pred_len": pred_len,
                "gt_len": gt_len,
                "len_gap": int(len_gap),
                "under_generation": under_generation,
                "over_generation": over_generation,
                "exact_generation": exact_generation,
                "has_answer_tag": has_answer_tag,
                "has_answer_open": has_answer_open,
                "has_answer_close": has_answer_close,
                "first_5_pred_values": pred_values[:5],
                "first_5_gt_values": gt_values[:5],
                "raw_model_output_head": short_text(solution_str, 500),
                "parsed_answer_text_head": short_text(extract_answer(solution_str), 500),
                "raw_model_output_tail_10_lines": extract_tail_lines(solution_str, 10),
                "parsed_answer_tail_10_lines": extract_tail_lines(extract_answer(solution_str), 10),
                "parsed_values": pred_values[:50],
                "format_parse_mode": format_parse_mode,
                "raw_protocol_reject_reason": raw_protocol_reject_reason or "",
                "was_recovered": was_recovered,
                "format_failure_reason": format_failure_reason,
                "length_hard_fail": bool(length_hard_fail),
                "strict_length_match": bool(strict_length_match),
            },
        )

        append_turn3_generation_debug(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            sample_uid=sample_uid,
            output_source=output_source,
            format_score=float(format_score),
            length_score=float(length_score),
            length_penalty=float(length_penalty),
            raw_overrun_penalty=float(raw_overrun_penalty),
            recovery_penalty=float(recovery_penalty),
            mse_score=float(mse_score),
            change_point_score=float(change_point_score),
            season_trend_score=float(season_trend_score),
            final_score=float(score),
            orig_mse=orig_mse,
            orig_mae=orig_mae,
            norm_mse=norm_mse,
            norm_mae=norm_mae,
            raw_mse=orig_mse,
            raw_mae=orig_mae,
            length_hard_fail=bool(length_hard_fail),
            strict_length_match=bool(strict_length_match),
            format_parse_mode=format_parse_mode,
            raw_protocol_reject_reason=raw_protocol_reject_reason or "",
            was_recovered=was_recovered,
            format_failure_reason=format_failure_reason,
            has_answer_tag=has_answer_tag,
            has_answer_open=has_answer_open,
            has_answer_close=has_answer_close,
            pred_values=pred_values,
            gt_values=gt_values,
        )

    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    allow_recovery: bool = False,
) -> dict:
    """Compute the paper-aligned composite reward for time series prediction."""
    extra_info_dict = extra_info or {}
    nested_reward_info = extra_info_dict.get("reward_extra_info")
    if not isinstance(nested_reward_info, dict):
        nested_reward_info = {}

    raw_solution = str(solution_str or "")
    materialized_solution = extra_info_dict.get("materialized_solution_str")
    if materialized_solution is None:
        materialized_solution = nested_reward_info.get("materialized_solution_str")
    scoring_solution = str(materialized_solution or raw_solution)
    clamp_info = {
        "applied": False,
        "reason": "",
        "raw_answer_line_count": 0,
        "valid_prefix_line_count": 0,
        "discarded_line_count": 0,
    }
    expected_len = len(extract_ground_truth_values(ground_truth))
    if expected_len > 0:
        scoring_solution, clamp_info = clamp_turn3_answer_horizon(scoring_solution, expected_len)
    raw_answer_line_count = int(clamp_info.get("raw_answer_line_count") or 0)
    raw_overrun_penalty = (
        float(compute_length_penalty(raw_answer_line_count, expected_len))
        if bool(clamp_info.get("applied", False)) and expected_len > 0 and raw_answer_line_count > expected_len
        else 0.0
    )

    result = _compute_score_impl(
        data_source=data_source,
        solution_str=scoring_solution,
        ground_truth=ground_truth,
        extra_info=extra_info,
        allow_recovery=allow_recovery,
        emit_debug=True,
        raw_overrun_penalty=raw_overrun_penalty,
    )

    clamp_fields = {
        "used_materialized_solution": bool(str(materialized_solution or "").strip()),
        "turn3_horizon_clamped": bool(clamp_info.get("applied", False)),
        "turn3_horizon_clamp_reason": str(clamp_info.get("reason") or ""),
        "turn3_horizon_clamp_discarded_lines": int(clamp_info.get("discarded_line_count") or 0),
        "turn3_horizon_clamp_valid_prefix_lines": int(clamp_info.get("valid_prefix_line_count") or 0),
        "turn3_horizon_clamp_raw_answer_lines": int(clamp_info.get("raw_answer_line_count") or 0),
        "raw_overrun_penalty": float(raw_overrun_penalty),
    }
    result.update(clamp_fields)

    if _should_attach_protocol_compare(extra_info_dict, nested_reward_info):
        strict_result = (
            result
            if not allow_recovery
            else _compute_score_impl(
                data_source=data_source,
                solution_str=scoring_solution,
                ground_truth=ground_truth,
                extra_info=extra_info,
                allow_recovery=False,
                emit_debug=False,
                raw_overrun_penalty=raw_overrun_penalty,
            )
        )
        recovered_result = (
            result
            if allow_recovery
            else _compute_score_impl(
                data_source=data_source,
                solution_str=scoring_solution,
                ground_truth=ground_truth,
                extra_info=extra_info,
                allow_recovery=True,
                emit_debug=False,
                raw_overrun_penalty=raw_overrun_penalty,
            )
        )
        strict_score = float(strict_result["score"])
        recovered_score = float(recovered_result["score"])
        expected_line_count = int(recovered_result.get("gt_len") or strict_result.get("gt_len") or 0)
        answer_line_count = _count_answer_lines(scoring_solution or "")
        raw_response_answer_line_count = _count_answer_lines(raw_solution)
        protocol_compare = {
            "strict_score": strict_score,
            "recovered_score": recovered_score,
            "recovery_gap": float(recovered_score - strict_score),
            "strict_parse_mode": str(strict_result.get("format_parse_mode") or ""),
            "recovered_parse_mode": str(recovered_result.get("format_parse_mode") or ""),
            "strict_reject_reason": str(strict_result.get("raw_protocol_reject_reason") or ""),
            "recovered_reject_reason": str(recovered_result.get("raw_protocol_reject_reason") or ""),
            "answer_line_count": int(answer_line_count),
            "raw_response_answer_line_count": int(raw_response_answer_line_count),
            "expected_answer_line_count": expected_line_count,
            "wrote_expected_rows_before_stop": bool(expected_line_count > 0 and answer_line_count >= expected_line_count),
            "raw_overrun_penalty": float(raw_overrun_penalty),
            **clamp_fields,
        }
        result.update(protocol_compare)
        append_chain_debug(
            "reward_protocol_compare",
            {
                "data_source": data_source,
                "sample_uid": result.get("sample_uid"),
                "selected_model": result.get("selected_model"),
                "global_step": result.get("global_step"),
                "validate": result.get("validate"),
                "run_name": result.get("run_name"),
                "strict_score": strict_score,
                "recovered_score": recovered_score,
                "recovery_gap": float(recovered_score - strict_score),
                "strict_parse_mode": protocol_compare["strict_parse_mode"],
                "recovered_parse_mode": protocol_compare["recovered_parse_mode"],
                "strict_reject_reason": protocol_compare["strict_reject_reason"],
                "recovered_reject_reason": protocol_compare["recovered_reject_reason"],
                "answer_line_count": int(answer_line_count),
                "expected_answer_line_count": expected_line_count,
                "wrote_expected_rows_before_stop": bool(protocol_compare["wrote_expected_rows_before_stop"]),
                "raw_overrun_penalty": float(raw_overrun_penalty),
                **clamp_fields,
            },
        )

    return result


__all__ = [
    "ENABLE_CHANGE_POINT_SCORE",
    "ENABLE_SEASON_TREND_SCORE",
    "append_turn3_generation_debug",
    "canonicalize_forecast_values",
    "compute_format_score",
    "compute_length_penalty",
    "compute_length_score",
    "compute_mse_score",
    "compute_score",
    "count_numeric_only_lines",
    "decompose",
    "detect_suffix_repetition",
    "extract_answer",
    "extract_answer_region",
    "extract_forecast_block",
    "extract_ground_truth_values",
    "extract_strict_protocol_answer",
    "extract_tail_lines",
    "extract_values_from_time_series_string",
    "find_change_points",
    "infer_answer_shape_failure",
    "infer_format_failure_reason",
    "is_plain_forecast_block_response",
    "looks_like_forecast_answer",
    "mean_squared_error",
    "mean_squared_error_season_trend",
    "moving_avg",
    "normalize_for_reward",
    "normalized_nonempty_lines",
    "parse_final_answer_protocol",
    "recover_protocol_answer",
    "series_decomp",
    "trailing_text_after_close",
]
