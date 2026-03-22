import os
import string
import re
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from verl.utils.chain_debug import append_chain_debug, short_text


# Paper-aligned composite reward:
# - normalized/log-transformed MSE as the main term
# - structural alignment and season/trend consistency as shaping terms
# - format validity and output-length consistency as auxiliary constraints
ENABLE_CHANGE_POINT_SCORE = True
ENABLE_SEASON_TREND_SCORE = True
TURN3_SUCCESS_SAMPLE_RATE = max(int(os.getenv("TS_TURN3_SUCCESS_SAMPLE_RATE", "100") or 100), 1)


def turn3_generation_debug_file() -> str:
    override = os.getenv("TS_TURN3_DEBUG_FILE", "").strip()
    if override:
        return override

    chain_file = os.getenv("TS_CHAIN_DEBUG_FILE", "/tmp/ts_chain_debug.jsonl")
    chain_dir = os.path.dirname(chain_file) or "."
    chain_name = os.path.basename(chain_file)
    if chain_name.startswith("ts_chain_debug"):
        suffix = chain_name[len("ts_chain_debug"):]
        return os.path.join(chain_dir, f"turn3_generation_debug{suffix}")
    return os.path.join(chain_dir, "turn3_generation_debug.jsonl")


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
    pred_values: List[float],
    gt_values: List[float],
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
        float(len({round(float(v), 6) for v in pred_values}) / pred_len)
        if pred_len > 0
        else float("nan")
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
        "_debug_file": turn3_generation_debug_file(),
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


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
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
        super(series_decomp, self).__init__()
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

    season = season.squeeze(0).squeeze(1).detach().numpy()
    trend = trend.squeeze(0).squeeze(1).detach().numpy()
    return season, trend


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate mean squared error."""
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mean_squared_error_season_trend(y_true: List[float], y_pred: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Calculate MSE for seasonal and trend components separately.
    
    Returns:
        Tuple of (mse_season, mse_trend), or (None, None) if decomposition fails.
    """
    if len(y_true) < 5 or len(y_pred) < 5:
        # Not enough data for decomposition
        return None, None
    
    try:
        season_true, trend_true = decompose(y_true)
        season_pred, trend_pred = decompose(y_pred)
        mse_season = float(np.mean((season_true - season_pred) ** 2))
        mse_trend = float(np.mean((trend_true - trend_pred) ** 2))
        return mse_season, mse_trend
    except Exception:
        return None, None


def extract_answer(text: str) -> str:
    """Extract content within <answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else text


def normalized_nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def extract_forecast_block(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = (
        str(text)
        .replace("<|im_end|>", "\n")
        .replace("<answer>", "\n")
        .replace("</answer>", "\n")
        .strip()
    )
    lines = cleaned.splitlines()
    collected: List[str] = []
    started = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        is_timestamp_value = re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+-?\d+\.?\d*$", line) is not None
        is_numeric_value = re.match(r"^-?\d+\.?\d*$", line) is not None
        if not started:
            if is_timestamp_value or is_numeric_value:
                started = True
                collected.append(line)
            continue

        if is_timestamp_value or is_numeric_value:
            collected.append(line)
            continue

        break

    if not collected:
        return None
    return "\n".join(collected)


def canonicalize_forecast_values(values: List[float], expected_len: int) -> str:
    return "\n".join(f"{float(value):.4f}" for value in values[:expected_len])


def looks_like_forecast_answer(answer_text: Optional[str], expected_len: int) -> bool:
    if not answer_text:
        return False
    lines = normalized_nonempty_lines(answer_text)
    values = extract_values_from_time_series_string(answer_text)
    if len(lines) != expected_len or len(values) != expected_len:
        return False
    for line in lines:
        if re.fullmatch(r"-?\d+(?:\.\d+)?", line) is None:
            return False
    return True


def infer_answer_shape_failure(answer_text: str, expected_len: int) -> str:
    lines = normalized_nonempty_lines(answer_text)
    values = extract_values_from_time_series_string(answer_text)
    if not lines:
        return "empty_answer_block"
    if len(lines) != expected_len:
        return f"invalid_answer_shape:lines={len(lines)},expected={expected_len}"
    if len(values) != expected_len:
        return f"invalid_answer_shape:values={len(values)},expected={expected_len}"
    for line in lines:
        if re.fullmatch(r"-?\d+(?:\.\d+)?", line) is None:
            return "invalid_answer_shape:non_numeric_line"
    return "invalid_answer_shape:unknown"


def is_plain_forecast_block_response(response_text: Optional[str]) -> bool:
    forecast_block = extract_forecast_block(response_text)
    if not forecast_block or response_text is None:
        return False

    cleaned = (
        str(response_text)
        .replace("<|im_end|>", "\n")
        .replace("<think>", "\n")
        .replace("</think>", "\n")
        .replace("<answer>", "\n")
        .replace("</answer>", "\n")
        .strip()
    )
    return normalized_nonempty_lines(cleaned) == normalized_nonempty_lines(forecast_block)


def extract_strict_protocol_answer(solution_str: Optional[str], expected_len: int) -> Tuple[Optional[str], Optional[str]]:
    if solution_str is None:
        return None, "empty_solution"

    if "<answer>" in solution_str and "</answer>" not in solution_str:
        return None, "missing_answer_close_tag"
    if "</answer>" in solution_str and "<answer>" not in solution_str:
        return None, "missing_answer_open_tag"
    if "<answer>" not in solution_str and "</answer>" not in solution_str:
        return None, "missing_answer_block"

    protocol_match = re.fullmatch(
        r"\s*(?:<think>.*?</think>\s*)?<answer>(.*?)</answer>\s*",
        solution_str,
        re.DOTALL,
    )
    if not protocol_match:
        return None, "extra_text_outside_tags"

    candidate = protocol_match.group(1).strip()
    if looks_like_forecast_answer(candidate, expected_len):
        return candidate, None
    return None, infer_answer_shape_failure(candidate, expected_len)


def recover_protocol_answer(
    solution_str: Optional[str],
    reject_reason: str,
    expected_len: int,
) -> Tuple[Optional[str], Optional[str]]:
    if not solution_str:
        return None, None
    if "<tool_call>" in solution_str or "</tool_call>" in solution_str:
        return None, None

    candidate_sources: List[Tuple[str, str]] = []
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if answer_match:
        candidate_sources.append(("answer_block", answer_match.group(1)))
    elif "<answer>" in solution_str:
        candidate_sources.append(("answer_block", solution_str.split("<answer>", 1)[1]))

    if is_plain_forecast_block_response(solution_str):
        forecast_block = extract_forecast_block(solution_str)
        if forecast_block:
            candidate_sources.append(("plain_forecast_block", forecast_block))

    seen_sources: set[str] = set()
    for source_name, source_text in candidate_sources:
        normalized_source = source_text.strip()
        if not normalized_source or normalized_source in seen_sources:
            continue
        seen_sources.add(normalized_source)

        values = extract_values_from_time_series_string(normalized_source)
        if len(values) < expected_len:
            continue

        canonical_answer = canonicalize_forecast_values(values, expected_len)
        if not looks_like_forecast_answer(canonical_answer, expected_len):
            continue

        return canonical_answer, f"recovered_{reject_reason}_{source_name}"

    return None, None


def parse_final_answer_protocol(
    solution_str: Optional[str],
    expected_len: int,
    *,
    allow_recovery: bool = True,
) -> Tuple[Optional[str], str, Optional[str]]:
    strict_answer, reject_reason = extract_strict_protocol_answer(solution_str, expected_len)
    if strict_answer is not None:
        return strict_answer, "strict_protocol", None

    reject_reason = reject_reason or "unknown_format_failure"
    if allow_recovery:
        recovered_answer, parse_mode = recover_protocol_answer(solution_str, reject_reason, expected_len)
        if recovered_answer is not None and parse_mode is not None:
            return recovered_answer, parse_mode, reject_reason

    return None, f"rejected_{reject_reason}", reject_reason


def extract_tail_lines(text: Optional[str], max_lines: int = 10) -> List[str]:
    if text is None:
        return []
    lines = str(text).splitlines()
    if max_lines <= 0:
        return []
    return lines[-max_lines:]


def extract_answer_region(text: Optional[str]) -> str:
    if text is None:
        return ""
    raw_text = str(text)
    open_tag = "<answer>"
    close_tag = "</answer>"
    start = raw_text.rfind(open_tag)
    if start < 0:
        return ""
    start += len(open_tag)
    end = raw_text.find(close_tag, start)
    if end < 0:
        return raw_text[start:].strip()
    return raw_text[start:end].strip()


def count_numeric_only_lines(text: str) -> Tuple[int, int]:
    numeric_count = 0
    non_numeric_count = 0
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"-?\d+(?:\.\d+)?", line):
            numeric_count += 1
        else:
            non_numeric_count += 1
    return numeric_count, non_numeric_count


def trailing_text_after_close(text: Optional[str]) -> str:
    if text is None:
        return ""
    raw_text = str(text)
    close_tag = "</answer>"
    if close_tag not in raw_text:
        return ""
    return raw_text.split(close_tag, 1)[1].strip()


def detect_suffix_repetition(
    values: List[float],
    *,
    max_period: int = 8,
    min_repeats: int = 2,
    atol: float = 1e-8,
) -> Tuple[bool, int, int]:
    n = len(values)
    if n < 2:
        return False, 0, 0

    def _segments_equal(a: List[float], b: List[float]) -> bool:
        if len(a) != len(b):
            return False
        return all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))

    best_period = 0
    best_repeats = 0
    upper_period = min(max_period, n // min_repeats)
    for period in range(1, upper_period + 1):
        base = values[-period:]
        repeats = 1
        cursor = n - 2 * period
        while cursor >= 0:
            segment = values[cursor : cursor + period]
            if not _segments_equal(segment, base):
                break
            repeats += 1
            cursor -= period
        if repeats >= min_repeats and repeats * period > best_repeats * max(best_period, 1):
            best_period = period
            best_repeats = repeats
    if best_repeats >= min_repeats:
        return True, best_period, best_repeats
    return False, 0, 0


def extract_values_from_time_series_string(text: str) -> List[float]:
    """
    Extract numeric values from time series string format.
    
    Expected formats:
    1. "2017-05-01 00:00:00 11.588" (timestamp + value)
    2. "| 1 | 11.588 |" (table format)
    3. Just numbers on each line
    
    Args:
        text: Text containing time series data
        
    Returns:
        List of float values
    """
    raw_text = text
    text = extract_answer(text)
    values = []
    
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Pattern 1: timestamp format "2017-05-01 00:00:00 11.588"
        match = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(-?\d+\.?\d*)', line)
        if match:
            try:
                values.append(float(match.group(1)))
                continue
            except ValueError:
                pass
        
        # Pattern 2: table format "| 1 | 11.588 |"
        match = re.search(r'\|\s*\d+\s*\|\s*(-?\d+\.?\d*)\s*\|', line)
        if match:
            try:
                values.append(float(match.group(1)))
                continue
            except ValueError:
                pass
        
        # Pattern 3: just the last number on the line
        match = re.search(r'(-?\d+\.?\d*)$', line)
        if match:
            try:
                values.append(float(match.group(1)))
                continue
            except ValueError:
                pass
    
    if os.getenv("TS_REWARD_PARSE_DEBUG", "0").lower() in {"1", "true", "yes", "on"}:
        append_chain_debug(
            "reward_parse_extract",
            {
                "has_answer_tag": bool(re.search(r"<answer>(.*?)</answer>", raw_text or "", re.DOTALL)),
                "raw_text_head": short_text(raw_text, 300),
                "parsed_answer_head": short_text(text, 300),
                "num_values": len(values),
                "values_head": values[:10],
            },
        )
    return values


def extract_ground_truth_values(text: str) -> List[float]:
    """Extract ground truth values from time series string."""
    return extract_values_from_time_series_string(text)


def normalize_for_reward(x_list: List[float], y_list: List[float]) -> Tuple[List[float], List[float]]:
    """
    Normalize both prediction and ground truth for fair comparison.
    
    Uses GROUND TRUTH's mean and std as normalization parameters.
    This ensures stable normalization regardless of prediction quality.
    """
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    
    # Use ground truth's statistics for stable normalization
    mu = np.nanmean(y_array)
    std = np.nanstd(y_array)
    std = max(std, 1e-8)  # Avoid division by zero
    
    x_result = (x_array - mu) / std
    y_result = (y_array - mu) / std
    
    return x_result.tolist(), y_result.tolist()


def compute_format_score(solution_str: str, expected_len: Optional[int] = None) -> float:
    """
    Check if the solution follows the required format with <answer> tags.
    
    Returns:
        0.0 if format is correct, -1.0 otherwise
    """
    if solution_str is None:
        return -1.0
    
    try:
        inferred_len = int(expected_len or 0)
        if inferred_len <= 0:
            inferred_len = len(extract_values_from_time_series_string(solution_str or ""))
        answer_text, _, _ = parse_final_answer_protocol(solution_str, max(inferred_len, 1), allow_recovery=True)
        if answer_text is not None:
            return 0.0
        return -1.0
    except Exception as e:
        print(f"[DEBUG] Error in compute_format_score: {e}")
        return -1.0


def infer_format_failure_reason(solution_str: Optional[str], expected_len: Optional[int] = None) -> str:
    if solution_str is None:
        return "empty_solution"
    inferred_len = int(expected_len or 0)
    if inferred_len <= 0:
        inferred_len = len(extract_values_from_time_series_string(solution_str or ""))
    answer_text, _, reject_reason = parse_final_answer_protocol(solution_str, max(inferred_len, 1), allow_recovery=True)
    if answer_text is not None:
        return "ok"
    return reject_reason or "unknown_format_failure"


def compute_length_score(solution_str: str, ground_truth: str) -> float:
    """
    Reward for generating the correct number of predictions.
    
    Returns:
        0.1 if prediction length exactly matches ground truth length, 0.0 otherwise
    """
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
    """
    Compute score based on MSE between prediction and ground truth.
    
    Uses log transformation: score = 1 / (1 + log(1 + mse))
    
    This design ensures:
    1. MSE=0 -> score=1 (perfect prediction)
    2. MSE=1 -> score=0.59
    3. MSE=10 -> score=0.29
    4. MSE=100 -> score=0.18
    5. MSE=1000 -> score=0.13
    
    Key benefits for GRPO training:
    - Log compression: even very large MSE still produces meaningful scores
    - Always differentiable: smooth gradient for optimization
    - Maintains ranking: lower MSE always gets higher score
    - Good discrimination: even poor predictions have distinguishable rewards
    
    If prediction is shorter than ground truth:
    - Calculate MSE on available predictions
    - Apply length penalty: score * (pred_len / gt_len)
    
    Returns:
        Score in range [0, 0.6]
    """
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)
    
    if not gt_values or not pred_values:
        return 0.0

    pred_len = len(pred_values)
    gt_len = len(gt_values)
    
    min_len = min(pred_len, gt_len)
    if min_len == 0:
        return 0.0
    
    pred_slice = pred_values[:min_len]
    gt_slice = gt_values[:min_len]
    orig_mse = float(mean_squared_error(gt_slice, pred_slice))
    norm_pred, norm_gt = normalize_for_reward(pred_slice, gt_slice)
    norm_mse = float(mean_squared_error(norm_gt, norm_pred))
    score = 1.0 / (1.0 + np.log1p(norm_mse))

    return score * 0.6


def compute_season_trend_score(solution_str: str, ground_truth: str) -> float:
    """
    Compute score based on seasonal and trend decomposition.
    
    Uses log compression (consistent with compute_mse_score).
    
    Returns:
        Score up to 0.2 (0.05 for season + 0.15 for trend)
        Returns 0.0 if decomposition fails (instead of giving undeserved points)
    """
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)
    
    if not gt_values or not pred_values:
        return 0.0
    
    if len(gt_values) < 5:
        return 0.0
    
    # Use minimum length for comparison
    min_len = min(len(pred_values), len(gt_values))
    if min_len < 5:
        return 0.0
    
    # Normalize using ground truth's statistics
    norm_pred, norm_gt = normalize_for_reward(pred_values[:min_len], gt_values[:min_len])
    
    # Calculate MSE for seasonal and trend components
    mse_season, mse_trend = mean_squared_error_season_trend(norm_gt, norm_pred)
    
    # If decomposition failed, return 0 (not undeserved points)
    if mse_season is None or mse_trend is None:
        return 0.0
    
    # Transform to scores using log compression (consistent with compute_mse_score)
    score_season = 1.0 / (1.0 + np.log1p(mse_season))
    score_trend = 1.0 / (1.0 + np.log1p(mse_trend))
    
    # Apply length penalty if prediction is shorter
    length_ratio = min(len(pred_values) / len(gt_values), 1.0)
    
    return (0.05 * score_season + 0.15 * score_trend) * length_ratio


def find_change_points(data: List[float]) -> Tuple[List[int], List[int]]:
    """
    Find local maxima and minima (change points) in the data.
    
    Returns:
        Tuple of (max_indices, min_indices)
    """
    if len(data) < 5:
        return [], []
    
    change_point_max = []
    change_point_min = []
    
    # Mirror padding
    data_mirror_forward = [data[2], data[1]]
    data_mirror_backward = [data[-2], data[-1]]
    data_new = data_mirror_forward + list(data) + data_mirror_backward
    
    for i in range(len(data)):
        # Check for local maximum
        if (data_new[i+2] >= data_new[i] and 
            data_new[i+2] >= data_new[i+1] and 
            data_new[i+2] >= data_new[i+3] and 
            data_new[i+2] >= data_new[i+4]):
            change_point_max.append(i)
        
        # Check for local minimum
        if (data_new[i+2] <= data_new[i] and 
            data_new[i+2] <= data_new[i+1] and 
            data_new[i+2] <= data_new[i+3] and 
            data_new[i+2] <= data_new[i+4]):
            change_point_min.append(i)
    
    return change_point_max, change_point_min


def compute_change_point_score(solution_str: str, ground_truth: str, tolerance: int = 2) -> float:
    """
    Compute score based on correctly predicted change points (local max/min).
    
    Args:
        solution_str: Model's prediction string
        ground_truth: Ground truth string
        tolerance: Allow change point to be off by this many time steps (default: 2)
    
    Returns:
        Score up to 0.2 (0.1 for max + 0.1 for min)
    """
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)
    
    if not gt_values or not pred_values:
        return 0.0
    
    # Use minimum length for comparison
    min_len = min(len(pred_values), len(gt_values))
    if min_len < 5:
        return 0.0
    
    gt_max, gt_min = find_change_points(gt_values[:min_len])
    pred_max, pred_min = find_change_points(pred_values[:min_len])
    
    def count_hits_with_tolerance(pred_points: List[int], gt_points: List[int], tol: int) -> int:
        """Count predicted points that are within tolerance of a ground truth point."""
        hits = 0
        used_gt = set()
        for p in pred_points:
            for g in gt_points:
                if g not in used_gt and abs(p - g) <= tol:
                    hits += 1
                    used_gt.add(g)
                    break
        return hits
    
    # Count correct predictions with tolerance
    max_hits = count_hits_with_tolerance(pred_max, gt_max, tolerance)
    min_hits = count_hits_with_tolerance(pred_min, gt_min, tolerance)
    
    # Calculate scores
    max_score = max_hits / len(gt_max) * 0.1 if gt_max else 0.0
    min_score = min_hits / len(gt_min) * 0.1 if gt_min else 0.0
    
    # Apply length penalty if prediction is shorter
    length_ratio = min(len(pred_values) / len(gt_values), 1.0)
    
    return (max_score + min_score) * length_ratio


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None
) -> dict:
    """
    Compute the paper-aligned composite reward for time series prediction.

    The score is composed of:
    1. Format validity: -1.0 or 0.0
    2. Length consistency: exact-match bonus plus mild mismatch penalty
    3. Prediction error reward: normalized/log-transformed MSE as the main term
    4. Structural alignment reward: local turning point consistency
    5. Trend/seasonality reward: decomposition consistency
    
    Args:
        data_source: Source identifier (unused but kept for compatibility)
        solution_str: Model's solution string with <answer> tags
        ground_truth: Ground truth time series string
        extra_info: Optional additional information (unused)
        
    Returns:
        Total reward score
    """
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
        "prediction_model_used",
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
        allow_recovery=True,
    )
    scoring_solution = (
        f"<answer>\n{canonical_answer}\n</answer>"
        if canonical_answer is not None
        else solution_str
    )
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
            solution_str, expected_len=protocol_expected_len
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
            mse_score = (1.0 / (1.0 + np.log1p(norm_mse))) * 0.6
        score += length_score
        score += mse_score
        score -= length_penalty

    if ENABLE_CHANGE_POINT_SCORE and format_score >= 0:
        try:
            change_point_score = compute_change_point_score(scoring_solution, ground_truth)
            score += change_point_score
        except Exception as e:
            print(f"[DEBUG] Error in compute_change_point_score: {e}")

    if ENABLE_SEASON_TREND_SCORE and format_score >= 0:
        try:
            season_trend_score = compute_season_trend_score(scoring_solution, ground_truth)
            score += season_trend_score
        except Exception as e:
            print(f"[DEBUG] Error in compute_season_trend_score: {e}")

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
