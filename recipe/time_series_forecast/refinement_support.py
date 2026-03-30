from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np

from recipe.time_series_forecast.reward_protocol import extract_answer_region, normalized_nonempty_lines
from recipe.time_series_forecast.utils import parse_time_series_string


REFINEMENT_OP_DESCRIPTIONS: dict[str, str] = {
    "isolated_spike_smoothing": "smooth isolated one-step spikes",
    "local_level_adjust": "shift only the local level near the tail",
    "local_slope_adjust": "adjust only the local slope near the tail",
}

REFINEMENT_KEEP_DECISION = "keep_baseline"


def filter_refinement_candidates_for_model(
    candidate_refinements: Sequence[tuple[Sequence[float], Sequence[str]]],
    prediction_model_used: str | None = None,
) -> list[tuple[list[float], list[str]]]:
    model_name = str(prediction_model_used or "").strip().lower()
    filtered: list[tuple[list[float], list[str]]] = []
    for candidate_values, candidate_ops in candidate_refinements:
        ops = [str(op).strip() for op in candidate_ops if str(op).strip()]
        if not ops:
            continue
        if model_name in {"arima", "itransformer"} and "local_level_adjust" in ops:
            continue
        if model_name == "itransformer" and "isolated_spike_smoothing" in ops:
            continue
        filtered.append(([float(value) for value in candidate_values], ops))
    return filtered


def _median_abs_deviation(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(list(values), dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return mad


def _compute_diff_mad(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 1e-6
    diffs = np.diff(np.asarray(list(values), dtype=float))
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))
    return max(mad, 1e-6)


def _detect_isolated_spikes(values: Sequence[float], diff_mad: float) -> list[int]:
    if len(values) < 3:
        return []
    threshold = max(3.0 * diff_mad, 1e-4)
    spike_indices: list[int] = []
    for idx in range(1, len(values) - 1):
        left_delta = float(values[idx] - values[idx - 1])
        right_delta = float(values[idx + 1] - values[idx])
        if abs(left_delta) <= threshold or abs(right_delta) <= threshold:
            continue
        if left_delta * right_delta < 0:
            spike_indices.append(idx)
    return spike_indices


def _smooth_isolated_spikes(values: Sequence[float]) -> tuple[list[float], list[str]]:
    smoothed = [float(value) for value in values]
    diff_mad = _compute_diff_mad(smoothed)
    spike_indices = _detect_isolated_spikes(smoothed, diff_mad)
    if not spike_indices:
        return smoothed, []
    for idx in spike_indices:
        smoothed[idx] = 0.5 * (smoothed[idx - 1] + smoothed[idx + 1])
    return smoothed, ["isolated_spike_smoothing"]


def _adjust_local_level(values: Sequence[float], history_values: Sequence[float]) -> tuple[list[float], list[str]]:
    window = min(12, len(values), len(history_values))
    if window < 4:
        return [float(value) for value in values], []
    adjusted = [float(value) for value in values]
    pred_tail = np.asarray(adjusted[-window:], dtype=float)
    history_tail = np.asarray(list(history_values)[-window:], dtype=float)
    level_gap = float(np.mean(history_tail) - np.mean(pred_tail))
    history_scale = max(_median_abs_deviation(history_values), 1e-4)
    if abs(level_gap) <= max(1.5 * history_scale, 1e-4):
        return adjusted, []
    correction = 0.5 * level_gap
    adjusted[-window:] = [float(value + correction) for value in adjusted[-window:]]
    return adjusted, ["local_level_adjust"]


def _adjust_local_slope(values: Sequence[float], history_values: Sequence[float]) -> tuple[list[float], list[str]]:
    window = min(12, len(values), len(history_values))
    if window < 4:
        return [float(value) for value in values], []
    adjusted = [float(value) for value in values]
    pred_tail = np.asarray(adjusted[-window:], dtype=float)
    history_tail = np.asarray(list(history_values)[-window:], dtype=float)
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


def generate_local_refinement_candidates(
    values: Sequence[float],
    history_values: Sequence[float],
) -> list[tuple[list[float], list[str]]]:
    candidate_builders = [
        lambda current: _smooth_isolated_spikes(current),
        lambda current: _adjust_local_level(current, history_values),
        lambda current: _adjust_local_slope(current, history_values),
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


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


def build_refinement_support_payload(
    *,
    base_values: Sequence[float],
    history_values: Sequence[float],
    selected_feature_tools: Sequence[str] | None = None,
    candidate_refinements: Sequence[tuple[list[float], list[str]]] | None = None,
    prediction_model_used: str | None = None,
) -> dict[str, Any]:
    selected_tools = _dedupe(selected_feature_tools or [])
    candidates = list(candidate_refinements) if candidate_refinements is not None else generate_local_refinement_candidates(
        base_values,
        history_values,
    )
    candidates = filter_refinement_candidates_for_model(candidates, prediction_model_used=prediction_model_used)
    candidate_adjustments = _dedupe(op for _values, ops in candidates for op in ops)
    edit_support_signals: dict[str, list[str]] = {adjustment: [] for adjustment in candidate_adjustments}

    if "extract_forecast_residuals" in selected_tools:
        if "isolated_spike_smoothing" in edit_support_signals:
            edit_support_signals["isolated_spike_smoothing"].append("residual_spike_signal")
        if "local_level_adjust" in edit_support_signals:
            edit_support_signals["local_level_adjust"].append("residual_level_shift")
        if "local_slope_adjust" in edit_support_signals:
            edit_support_signals["local_slope_adjust"].append("residual_drift_signal")

    if "extract_within_channel_dynamics" in selected_tools:
        if "isolated_spike_smoothing" in edit_support_signals:
            edit_support_signals["isolated_spike_smoothing"].append("irregular_local_spike")
        if "local_slope_adjust" in edit_support_signals:
            edit_support_signals["local_slope_adjust"].append("local_slope_shift")

    edit_support_signals = {
        adjustment: _dedupe(signals) or ["none"]
        for adjustment, signals in edit_support_signals.items()
    }
    keep_support_signals = ["evidence_consistent"] if not candidate_adjustments else ["none"]
    flat_support_signals: list[str] = []
    for signals in edit_support_signals.values():
        flat_support_signals.extend(signals)
    deduped_supports = _dedupe(flat_support_signals)
    if not deduped_supports:
        deduped_supports = ["evidence_consistent"]

    return {
        "observed_tools": selected_tools,
        "support_signals": deduped_supports,
        "keep_support_signals": keep_support_signals,
        "edit_support_signals": edit_support_signals,
        "candidate_adjustments": candidate_adjustments or ["none"],
        # Candidate local edits are optional heuristics, not mandatory repairs.
        # Keeping the selected forecast must remain legal even when edit options exist.
        "keep_baseline_allowed": True,
    }


def refinement_decision_name(refine_ops: Sequence[str] | None) -> str:
    ops = _dedupe(refine_ops or [])
    if not ops:
        return REFINEMENT_KEEP_DECISION
    return "->".join(ops)


def render_prediction_text_from_reference(values: Sequence[float], reference_prediction_text: str) -> str:
    forecast_horizon = len(values)
    reference_timestamps, _ = parse_time_series_string(reference_prediction_text)
    normalized_timestamps = [str(ts).strip() for ts in reference_timestamps[:forecast_horizon]]
    if len(normalized_timestamps) != forecast_horizon or not all(normalized_timestamps):
        raise ValueError("reference_prediction_text does not contain enough timestamps for rendering")
    return "\n".join(
        f"{normalized_timestamps[idx]} {float(values[idx]):.4f}"
        for idx in range(forecast_horizon)
    )


def build_refinement_candidate_prediction_text_map(
    *,
    base_prediction_text: str,
    candidate_refinements: Sequence[tuple[Sequence[float], Sequence[str]]],
    prediction_model_used: str | None = None,
) -> dict[str, str]:
    _timestamps, base_values = parse_time_series_string(base_prediction_text)
    if not base_values:
        raise ValueError("base_prediction_text must contain forecast values")
    filtered_candidates = filter_refinement_candidates_for_model(
        candidate_refinements,
        prediction_model_used=prediction_model_used,
    )
    mapping: dict[str, str] = {
        REFINEMENT_KEEP_DECISION: render_prediction_text_from_reference(base_values, base_prediction_text),
    }
    for refined_values, refine_ops in filtered_candidates:
        decision = refinement_decision_name(refine_ops)
        if decision in mapping:
            continue
        mapping[decision] = render_prediction_text_from_reference(refined_values, base_prediction_text)
    return mapping


def parse_refinement_decision_protocol(
    response_text: str,
    *,
    allowed_decisions: Sequence[str],
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
        return think_match.group(1).strip(), None, "rejected_invalid_decision_shape", "invalid_decision_shape"

    answer_line = answer_lines[0].strip()
    decision = answer_line.split("=", 1)[1].strip() if answer_line.lower().startswith("decision=") else answer_line
    allowed = {str(item).strip() for item in allowed_decisions if str(item).strip()}
    if decision not in allowed:
        return think_match.group(1).strip(), None, "rejected_invalid_decision_name", "invalid_decision_name"
    return think_match.group(1).strip(), decision, "refinement_decision_protocol", None


def materialize_refinement_decision(
    *,
    response_text: str,
    candidate_prediction_text_map: Mapping[str, str],
) -> tuple[str | None, str | None, str | None, str | None]:
    think_text, decision, parse_mode, reject_reason = parse_refinement_decision_protocol(
        response_text,
        allowed_decisions=list(candidate_prediction_text_map.keys()),
    )
    if decision is None:
        return None, None, parse_mode, reject_reason
    prediction_text = str(candidate_prediction_text_map.get(decision) or "").strip()
    if not prediction_text:
        return None, None, "rejected_missing_decision_prediction", "missing_decision_prediction"
    think = (think_text or "").strip() or f"I choose {decision} from the refinement evidence."
    final_answer = f"<think>\n{think}\n</think>\n<answer>\n{prediction_text}\n</answer>"
    return final_answer, decision, parse_mode, None


__all__ = [
    "REFINEMENT_KEEP_DECISION",
    "REFINEMENT_OP_DESCRIPTIONS",
    "build_refinement_candidate_prediction_text_map",
    "build_refinement_support_payload",
    "filter_refinement_candidates_for_model",
    "generate_local_refinement_candidates",
    "materialize_refinement_decision",
    "parse_refinement_decision_protocol",
    "refinement_decision_name",
    "render_prediction_text_from_reference",
]
