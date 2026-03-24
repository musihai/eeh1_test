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
from verl.utils.chain_debug import append_chain_debug, short_text


ENABLE_CHANGE_POINT_SCORE = True
ENABLE_SEASON_TREND_SCORE = True
TURN3_SUCCESS_SAMPLE_RATE = max(int(os.getenv("TS_TURN3_SUCCESS_SAMPLE_RATE", "100") or 100), 1)


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
    is_failure = bool(format_score < 0 or (gt_len > 0 and pred_len != gt_len))
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
        "required_step_budget",
        "prompt_char_len",
        "response_char_len",
        "response_token_len",
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
                "format_score": -1.0,
                "length_score": 0.0,
                "length_penalty": 0.0,
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
    elif gt_len > 0 and pred_len > 0:
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
            mse_score = (1.0 / (1.0 + np.log1p(norm_mse))) * PREDICTION_ERROR_SCORE_WEIGHT
        score += length_score
        score += mse_score
        score -= length_penalty

    if ENABLE_CHANGE_POINT_SCORE and format_score >= 0:
        try:
            change_point_score = compute_change_point_score(scoring_solution, ground_truth)
            score += change_point_score
        except Exception as error:
            print(f"[DEBUG] Error in compute_change_point_score: {error}")

    if ENABLE_SEASON_TREND_SCORE and format_score >= 0:
        try:
            season_trend_score = compute_season_trend_score(scoring_solution, ground_truth)
            score += season_trend_score
        except Exception as error:
            print(f"[DEBUG] Error in compute_season_trend_score: {error}")

    append_chain_debug(
        "reward_compute",
        {
            "data_source": data_source,
            "sample_uid": sample_uid,
            "selected_model": selected_model,
            "output_source": output_source,
            "format_score": float(format_score),
            "length_score": float(length_score),
            "length_penalty": float(length_penalty),
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

    return {
        "score": float(score),
        "trainer_seq_score": float(score),
        "format_score": float(format_score),
        "length_score": float(length_score),
        "length_penalty": float(length_penalty),
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


__all__ = [
    "ENABLE_CHANGE_POINT_SCORE",
    "ENABLE_SEASON_TREND_SCORE",
    "append_turn3_generation_debug",
    "canonicalize_forecast_values",
    "compute_change_point_score",
    "compute_format_score",
    "compute_length_penalty",
    "compute_length_score",
    "compute_mse_score",
    "compute_score",
    "compute_season_trend_score",
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
