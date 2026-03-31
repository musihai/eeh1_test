from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from recipe.time_series_forecast.reward_protocol import extract_answer_region, normalized_nonempty_lines
from recipe.time_series_forecast.time_series_io import parse_time_series_string


def _series_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "start": 0.0,
            "end": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "net_change": 0.0,
            "mean_abs_step": 0.0,
        }
    array = np.asarray(list(values), dtype=float)
    return {
        "start": float(array[0]),
        "end": float(array[-1]),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "net_change": float(array[-1] - array[0]),
        "mean_abs_step": float(np.mean(np.abs(np.diff(array)))) if len(array) >= 2 else 0.0,
    }


def _direction_label(value: float, *, eps: float = 1e-6) -> str:
    if value > eps:
        return "up"
    if value < -eps:
        return "down"
    return "flat"


def compute_candidate_visible_metrics(
    *,
    historical_data: str,
    target_column: str,
    candidates: Sequence[Mapping[str, Any]],
    default_candidate_id: str,
) -> dict[str, dict[str, Any]]:
    _, history_values = parse_time_series_string(str(historical_data or ""), target_column=target_column)
    recent_values = history_values[-24:] if len(history_values) >= 24 else history_values
    recent_stats = _series_stats(recent_values)
    recent_direction = _direction_label(recent_stats["net_change"])

    output: dict[str, dict[str, Any]] = {}
    default_metrics: dict[str, Any] | None = None
    default_candidate_key = str(default_candidate_id or "").strip()

    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        prediction_text = str(candidate.get("prediction_text") or candidate.get("compact_prediction_text") or "")
        _, prediction_values = parse_time_series_string(prediction_text)
        candidate_stats = _series_stats(prediction_values)
        candidate_direction = _direction_label(candidate_stats["net_change"])
        metrics = {
            "path_type": str(candidate.get("path_type") or ""),
            "model_name": str(candidate.get("model_name") or ""),
            "candidate_kind": str(candidate.get("candidate_kind") or ""),
            "level_gap": float(abs(candidate_stats["start"] - recent_stats["end"])),
            "mean_gap": float(abs(candidate_stats["mean"] - recent_stats["mean"])),
            "trend_gap": float(abs(candidate_stats["net_change"] - recent_stats["net_change"])),
            "step_gap": float(abs(candidate_stats["mean_abs_step"] - recent_stats["mean_abs_step"])),
            "volatility_gap": float(abs(candidate_stats["std"] - recent_stats["std"])),
            "recent_direction": recent_direction,
            "candidate_direction": candidate_direction,
            "direction_match": bool(candidate_direction == recent_direction),
        }
        output[candidate_id] = metrics
        if candidate_id == default_candidate_key:
            default_metrics = metrics

    if default_metrics is None and output:
        first_key = next(iter(output))
        default_metrics = output[first_key]

    if default_metrics is None:
        return output

    for metrics in output.values():
        metrics["level_gain_vs_default"] = float(default_metrics["level_gap"] - metrics["level_gap"])
        metrics["mean_gain_vs_default"] = float(default_metrics["mean_gap"] - metrics["mean_gap"])
        metrics["trend_gain_vs_default"] = float(default_metrics["trend_gap"] - metrics["trend_gap"])
        metrics["step_gain_vs_default"] = float(default_metrics["step_gap"] - metrics["step_gap"])
        metrics["volatility_gain_vs_default"] = float(default_metrics["volatility_gap"] - metrics["volatility_gap"])
        metrics["direction_gain_vs_default"] = int(metrics["direction_match"]) - int(default_metrics["direction_match"])

    return output


def parse_candidate_selection_protocol(
    response_text: str,
    *,
    allowed_candidate_ids: Sequence[str],
) -> tuple[str | None, str | None, str | None, str | None]:
    text = str(response_text or "")
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if think_match is None:
        return None, None, "rejected_missing_think_block", "missing_think_block"
    if answer_match is None:
        return think_match.group(1).strip(), None, "rejected_missing_answer_block", "missing_answer_block"

    answer_lines = normalized_nonempty_lines(extract_answer_region(text))
    if len(answer_lines) != 1:
        return think_match.group(1).strip(), None, "rejected_invalid_candidate_shape", "invalid_candidate_shape"

    answer_line = answer_lines[0].strip()
    if answer_line.lower().startswith("candidate_id="):
        candidate_id = answer_line.split("=", 1)[1].strip()
    else:
        candidate_id = answer_line
    allowed = {str(item).strip() for item in allowed_candidate_ids if str(item).strip()}
    if candidate_id not in allowed:
        return think_match.group(1).strip(), None, "rejected_invalid_candidate_id", "invalid_candidate_id"
    return think_match.group(1).strip(), candidate_id, "candidate_selection_protocol", None


def materialize_candidate_selection(
    *,
    response_text: str,
    candidate_prediction_text_map: Mapping[str, str],
) -> tuple[str | None, str | None, str | None, str | None]:
    think_text, candidate_id, parse_mode, reject_reason = parse_candidate_selection_protocol(
        response_text,
        allowed_candidate_ids=list(candidate_prediction_text_map.keys()),
    )
    if candidate_id is None:
        return None, None, parse_mode, reject_reason
    prediction_text = str(candidate_prediction_text_map.get(candidate_id) or "").strip()
    if not prediction_text:
        return None, None, "rejected_missing_candidate_prediction", "missing_candidate_prediction"
    think = (think_text or "").strip() or f"I select {candidate_id} as the best visible candidate."
    final_answer = f"<think>\n{think}\n</think>\n<answer>\n{prediction_text}\n</answer>"
    return final_answer, candidate_id, parse_mode, None


__all__ = [
    "compute_candidate_visible_metrics",
    "materialize_candidate_selection",
    "parse_candidate_selection_protocol",
]
