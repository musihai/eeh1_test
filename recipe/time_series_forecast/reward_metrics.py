from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from recipe.time_series_forecast.reward_protocol import (
    extract_ground_truth_values,
    extract_values_from_time_series_string,
    parse_final_answer_protocol,
)

PREDICTION_ERROR_SCORE_WEIGHT = 0.6
SEASON_COMPONENT_SCORE_WEIGHT = 0.0125
TREND_COMPONENT_SCORE_WEIGHT = 0.0375
CHANGE_POINT_COMPONENT_SCORE_WEIGHT = 0.025
STRUCTURAL_TIE_BREAK_SCALE = 0.2
STRUCTURAL_TIE_BREAK_MAX_NORM_MSE = 0.5


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


def decompose(x: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose time series into seasonal and trend components."""
    x = np.array(x)
    x = torch.from_numpy(x).float()
    x = x.unsqueeze(0).unsqueeze(2)

    kernel_size = min(25, len(x.squeeze()) // 2)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    model = series_decomp(kernel_size=kernel_size)
    season, trend = model(x)
    return season.squeeze(0).squeeze(1).detach().numpy(), trend.squeeze(0).squeeze(1).detach().numpy()


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mean_squared_error_season_trend(y_true: List[float], y_pred: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(y_true) < 5 or len(y_pred) < 5:
        return None, None

    try:
        season_true, trend_true = decompose(y_true)
        season_pred, trend_pred = decompose(y_pred)
        mse_season = float(np.mean((season_true - season_pred) ** 2))
        mse_trend = float(np.mean((trend_true - trend_pred) ** 2))
        return mse_season, mse_trend
    except Exception:
        return None, None


def normalize_for_reward(x_list: List[float], y_list: List[float]) -> Tuple[List[float], List[float]]:
    """Normalize prediction and ground truth using ground truth statistics."""
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    mu = np.nanmean(y_array)
    std = max(np.nanstd(y_array), 1e-8)
    return ((x_array - mu) / std).tolist(), ((y_array - mu) / std).tolist()


def compute_format_score(
    solution_str: str,
    expected_len: Optional[int] = None,
    *,
    allow_recovery: bool = False,
) -> float:
    if solution_str is None:
        return -1.0

    try:
        inferred_len = int(expected_len or 0)
        if inferred_len <= 0:
            inferred_len = len(extract_values_from_time_series_string(solution_str or ""))
        answer_text, _, _ = parse_final_answer_protocol(
            solution_str,
            max(inferred_len, 1),
            allow_recovery=allow_recovery,
        )
        return 0.0 if answer_text is not None else -1.0
    except Exception as error:
        print(f"[DEBUG] Error in compute_format_score: {error}")
        return -1.0


def infer_format_failure_reason(
    solution_str: Optional[str],
    expected_len: Optional[int] = None,
    *,
    allow_recovery: bool = False,
) -> str:
    if solution_str is None:
        return "empty_solution"
    inferred_len = int(expected_len or 0)
    if inferred_len <= 0:
        inferred_len = len(extract_values_from_time_series_string(solution_str or ""))
    answer_text, _, reject_reason = parse_final_answer_protocol(
        solution_str,
        max(inferred_len, 1),
        allow_recovery=allow_recovery,
    )
    if answer_text is not None:
        return "ok"
    return reject_reason or "unknown_format_failure"


def compute_length_score(solution_str: str, ground_truth: str) -> float:
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)
    if not gt_values:
        return 0.0
    return 0.1 if len(pred_values) == len(gt_values) else 0.0


def compute_length_penalty(pred_len: int, gt_len: int) -> float:
    if gt_len <= 0:
        return 0.0

    len_gap = abs(pred_len - gt_len)
    if len_gap == 0:
        penalty = 0.0
    elif len_gap == 1:
        penalty = 0.03
    elif len_gap == 2:
        penalty = 0.05
    elif len_gap <= 5:
        penalty = 0.08
    elif len_gap <= 20:
        penalty = 0.12
    elif len_gap <= 100:
        penalty = 0.18
    else:
        penalty = 0.24

    if pred_len > gt_len:
        penalty += min((pred_len - gt_len) / gt_len, 1.0) * 0.05

    return float(penalty)


def compute_mse_score(solution_str: str, ground_truth: str) -> float:
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)

    if not gt_values or not pred_values:
        return 0.0

    min_len = min(len(pred_values), len(gt_values))
    if min_len == 0:
        return 0.0

    pred_slice = pred_values[:min_len]
    gt_slice = gt_values[:min_len]
    norm_pred, norm_gt = normalize_for_reward(pred_slice, gt_slice)
    norm_mse = float(mean_squared_error(norm_gt, norm_pred))
    score = 1.0 / (1.0 + np.log1p(norm_mse))
    return score * PREDICTION_ERROR_SCORE_WEIGHT


def compute_season_trend_score(solution_str: str, ground_truth: str) -> float:
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)

    if not gt_values or not pred_values or len(gt_values) < 5:
        return 0.0

    min_len = min(len(pred_values), len(gt_values))
    if min_len < 5:
        return 0.0

    norm_pred, norm_gt = normalize_for_reward(pred_values[:min_len], gt_values[:min_len])
    mse_season, mse_trend = mean_squared_error_season_trend(norm_gt, norm_pred)
    if mse_season is None or mse_trend is None:
        return 0.0

    score_season = 1.0 / (1.0 + np.log1p(mse_season))
    score_trend = 1.0 / (1.0 + np.log1p(mse_trend))
    length_ratio = min(len(pred_values) / len(gt_values), 1.0)
    return (
        SEASON_COMPONENT_SCORE_WEIGHT * score_season
        + TREND_COMPONENT_SCORE_WEIGHT * score_trend
    ) * length_ratio


def compute_structural_tie_break_gate(norm_mse: float) -> float:
    try:
        numeric_norm_mse = float(norm_mse)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric_norm_mse) or numeric_norm_mse >= STRUCTURAL_TIE_BREAK_MAX_NORM_MSE:
        return 0.0
    closeness = max(0.0, 1.0 - (numeric_norm_mse / STRUCTURAL_TIE_BREAK_MAX_NORM_MSE))
    return float(STRUCTURAL_TIE_BREAK_SCALE * closeness)


def find_change_points(data: List[float]) -> Tuple[List[int], List[int]]:
    if len(data) < 5:
        return [], []

    change_point_max = []
    change_point_min = []
    data_mirror_forward = [data[2], data[1]]
    data_mirror_backward = [data[-2], data[-1]]
    data_new = data_mirror_forward + list(data) + data_mirror_backward

    for i in range(len(data)):
        if (
            data_new[i + 2] >= data_new[i]
            and data_new[i + 2] >= data_new[i + 1]
            and data_new[i + 2] >= data_new[i + 3]
            and data_new[i + 2] >= data_new[i + 4]
        ):
            change_point_max.append(i)

        if (
            data_new[i + 2] <= data_new[i]
            and data_new[i + 2] <= data_new[i + 1]
            and data_new[i + 2] <= data_new[i + 3]
            and data_new[i + 2] <= data_new[i + 4]
        ):
            change_point_min.append(i)

    return change_point_max, change_point_min


def compute_change_point_score(solution_str: str, ground_truth: str, tolerance: int = 2) -> float:
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)

    if not gt_values or not pred_values:
        return 0.0

    min_len = min(len(pred_values), len(gt_values))
    if min_len < 5:
        return 0.0

    gt_max, gt_min = find_change_points(gt_values[:min_len])
    pred_max, pred_min = find_change_points(pred_values[:min_len])

    def count_hits_with_tolerance(pred_points: List[int], gt_points: List[int], tol: int) -> int:
        hits = 0
        used_gt = set()
        for point in pred_points:
            for gt_point in gt_points:
                if gt_point not in used_gt and abs(point - gt_point) <= tol:
                    hits += 1
                    used_gt.add(gt_point)
                    break
        return hits

    max_hits = count_hits_with_tolerance(pred_max, gt_max, tolerance)
    min_hits = count_hits_with_tolerance(pred_min, gt_min, tolerance)
    max_score = max_hits / len(gt_max) * CHANGE_POINT_COMPONENT_SCORE_WEIGHT if gt_max else 0.0
    min_score = min_hits / len(gt_min) * CHANGE_POINT_COMPONENT_SCORE_WEIGHT if gt_min else 0.0
    length_ratio = min(len(pred_values) / len(gt_values), 1.0)
    return (max_score + min_score) * length_ratio


__all__ = [
    "CHANGE_POINT_COMPONENT_SCORE_WEIGHT",
    "PREDICTION_ERROR_SCORE_WEIGHT",
    "SEASON_COMPONENT_SCORE_WEIGHT",
    "STRUCTURAL_TIE_BREAK_MAX_NORM_MSE",
    "STRUCTURAL_TIE_BREAK_SCALE",
    "TREND_COMPONENT_SCORE_WEIGHT",
    "compute_change_point_score",
    "compute_format_score",
    "compute_length_penalty",
    "compute_length_score",
    "compute_mse_score",
    "compute_season_trend_score",
    "compute_structural_tie_break_gate",
    "decompose",
    "find_change_points",
    "infer_format_failure_reason",
    "mean_squared_error",
    "mean_squared_error_season_trend",
    "moving_avg",
    "normalize_for_reward",
    "series_decomp",
]
