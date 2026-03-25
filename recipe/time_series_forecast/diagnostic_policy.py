from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recipe.time_series_forecast.diagnostic_features import (
    extract_basic_statistics,
    extract_data_quality,
    extract_event_summary,
    extract_forecast_residuals,
    extract_within_channel_dynamics,
)

FEATURE_TOOL_ORDER = (
    "extract_basic_statistics",
    "extract_within_channel_dynamics",
    "extract_forecast_residuals",
    "extract_data_quality",
    "extract_event_summary",
)

EVENT_PATTERN_NAMES = ("rise", "fall", "flat", "oscillation")
MODEL_ORDER = ("arima", "patchtst", "itransformer", "chronos2")
MODEL_TOOL_REQUIREMENTS = {
    "arima": (
        "extract_basic_statistics",
        "extract_within_channel_dynamics",
        "extract_forecast_residuals",
    ),
    "patchtst": (
        "extract_basic_statistics",
        "extract_within_channel_dynamics",
        "extract_event_summary",
    ),
    "itransformer": (
        "extract_basic_statistics",
        "extract_within_channel_dynamics",
        "extract_event_summary",
    ),
    "chronos2": (
        "extract_basic_statistics",
        "extract_within_channel_dynamics",
        "extract_forecast_residuals",
        "extract_data_quality",
    ),
}


@dataclass(frozen=True)
class DiagnosticPlan:
    tool_names: tuple[str, ...]
    primary_model: str
    runner_up_model: str
    score_gap: float
    rationale: str
    feature_snapshot: dict[str, Any]


def _clean_history_values(values: list[float] | tuple[float, ...]) -> list[float]:
    cleaned: list[float] = []
    for value in values or []:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric == numeric and abs(numeric) != float("inf"):
            cleaned.append(numeric)
    return cleaned


def _event_pattern_name(raw_idx: Any) -> str:
    try:
        idx = int(float(raw_idx or 0.0))
    except (TypeError, ValueError):
        idx = 0
    idx = min(max(idx, 0), len(EVENT_PATTERN_NAMES) - 1)
    return EVENT_PATTERN_NAMES[idx]


def _compute_feature_snapshot(values: list[float]) -> dict[str, Any]:
    basic = extract_basic_statistics(values)
    dynamics = extract_within_channel_dynamics(values)
    residuals = extract_forecast_residuals(values)
    quality = extract_data_quality(values)
    events = extract_event_summary(values)
    return {
        "acf1": float(basic.get("acf1", 0.0)),
        "acf_seasonal": float(basic.get("acf_seasonal", 0.0)),
        "cusum_max": float(basic.get("cusum_max", 0.0)),
        "changepoint_count": float(dynamics.get("changepoint_count", 0.0)),
        "peak_count": float(dynamics.get("peak_count", 0.0)),
        "peak_spacing_cv": float(dynamics.get("peak_spacing_cv", 0.0)),
        "monotone_duration": float(dynamics.get("monotone_duration", 0.0)),
        "residual_exceed_ratio": float(residuals.get("residual_exceed_ratio", 0.0)),
        "residual_concentration": float(residuals.get("residual_concentration", 0.0)),
        "quality_quantization_score": float(quality.get("quality_quantization_score", 0.0)),
        "quality_saturation_ratio": float(quality.get("quality_saturation_ratio", 0.0)),
        "event_segment_count": float(events.get("event_segment_count", 0.0)),
        "dominant_pattern": _event_pattern_name(events.get("event_dominant_pattern", 0.0)),
    }


def _heuristic_model_scores(feature_snapshot: dict[str, Any]) -> dict[str, float]:
    acf1 = float(feature_snapshot.get("acf1", 0.0))
    acf_seasonal = float(feature_snapshot.get("acf_seasonal", 0.0))
    cusum_max = float(feature_snapshot.get("cusum_max", 0.0))
    changepoint_count = float(feature_snapshot.get("changepoint_count", 0.0))
    peak_count = float(feature_snapshot.get("peak_count", 0.0))
    peak_spacing_cv = float(feature_snapshot.get("peak_spacing_cv", 0.0))
    monotone_duration = float(feature_snapshot.get("monotone_duration", 0.0))
    residual_exceed_ratio = float(feature_snapshot.get("residual_exceed_ratio", 0.0))
    quality_quantization_score = float(feature_snapshot.get("quality_quantization_score", 0.0))
    quality_saturation_ratio = float(feature_snapshot.get("quality_saturation_ratio", 0.0))

    scores = {model_name: 0.0 for model_name in MODEL_ORDER}
    scores["arima"] += 0.10
    scores["patchtst"] += 0.15
    scores["chronos2"] -= 0.50

    scores["arima"] += 1.50 if acf1 >= 0.92 else (0.75 if acf1 >= 0.88 else 0.0)
    scores["arima"] += 0.75 if changepoint_count <= 1.0 else 0.0
    scores["arima"] += 0.50 if residual_exceed_ratio <= 0.05 else 0.0

    scores["patchtst"] += 1.00 if 2.0 <= peak_count <= 5.0 else 0.0
    scores["patchtst"] += 0.75 if peak_spacing_cv <= 0.30 else 0.0
    scores["patchtst"] += 0.50 if acf_seasonal >= 0.05 else 0.0

    scores["itransformer"] += 1.25 if changepoint_count >= 2.0 else 0.0
    scores["itransformer"] += 0.75 if cusum_max >= 70.0 else 0.0
    scores["itransformer"] += 0.50 if monotone_duration >= 0.10 else 0.0

    scores["chronos2"] += 0.75 if residual_exceed_ratio >= 0.08 else 0.0
    scores["chronos2"] += 0.50 if quality_saturation_ratio >= 0.08 or quality_quantization_score >= 0.24 else 0.0
    scores["chronos2"] += 0.50 if peak_count >= 6.0 and peak_spacing_cv >= 0.35 else 0.0
    return scores


def _select_primary_model(feature_snapshot: dict[str, Any], scores: dict[str, float]) -> str:
    acf1 = float(feature_snapshot.get("acf1", 0.0))
    changepoint_count = float(feature_snapshot.get("changepoint_count", 0.0))
    residual_exceed_ratio = float(feature_snapshot.get("residual_exceed_ratio", 0.0))
    peak_count = float(feature_snapshot.get("peak_count", 0.0))
    peak_spacing_cv = float(feature_snapshot.get("peak_spacing_cv", 0.0))
    monotone_duration = float(feature_snapshot.get("monotone_duration", 0.0))
    cusum_max = float(feature_snapshot.get("cusum_max", 0.0))
    quality_issue = (
        float(feature_snapshot.get("quality_saturation_ratio", 0.0)) >= 0.08
        or float(feature_snapshot.get("quality_quantization_score", 0.0)) >= 0.24
    )

    if quality_issue or (residual_exceed_ratio >= 0.085 and peak_count >= 6.0 and peak_spacing_cv >= 0.40):
        return "chronos2"
    if acf1 >= 0.93 and changepoint_count <= 1.0 and residual_exceed_ratio <= 0.05 and peak_count <= 4.0:
        return "arima"
    if changepoint_count >= 3.0 and (cusum_max >= 70.0 or monotone_duration >= 0.10):
        return "itransformer"
    if 2.0 <= peak_count <= 5.0 and peak_spacing_cv <= 0.30:
        return "patchtst"
    return _ordered_models_by_score(scores)[0][0]


def _ordered_models_by_score(scores: dict[str, float]) -> list[tuple[str, float]]:
    index_by_model = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    return sorted(scores.items(), key=lambda item: (-float(item[1]), index_by_model.get(item[0], 99), item[0]))


def _ordered_tool_names(tool_names: set[str]) -> list[str]:
    return [name for name in FEATURE_TOOL_ORDER if name in tool_names]


def _evidence_phrases(feature_snapshot: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    if float(feature_snapshot.get("acf1", 0.0)) >= 0.88:
        phrases.append("strong short-lag autocorrelation")
    if float(feature_snapshot.get("acf_seasonal", 0.0)) >= 0.05 or (
        2.0 <= float(feature_snapshot.get("peak_count", 0.0)) <= 5.0
        and float(feature_snapshot.get("peak_spacing_cv", 0.0)) <= 0.30
    ):
        phrases.append("repeatable local seasonality")
    if float(feature_snapshot.get("changepoint_count", 0.0)) >= 2.0 or float(feature_snapshot.get("cusum_max", 0.0)) >= 70.0:
        phrases.append("structural drift")
    if float(feature_snapshot.get("residual_exceed_ratio", 0.0)) >= 0.08:
        phrases.append("irregular residual excursions")
    if (
        float(feature_snapshot.get("quality_quantization_score", 0.0)) >= 0.24
        or float(feature_snapshot.get("quality_saturation_ratio", 0.0)) >= 0.08
    ):
        phrases.append("possible data-quality stress")
    dominant_pattern = str(feature_snapshot.get("dominant_pattern", "") or "")
    if dominant_pattern in {"rise", "fall", "oscillation"}:
        phrases.append(f"{dominant_pattern}-dominant event structure")
    return list(dict.fromkeys(phrases))


def _tool_focus_phrase(tool_name: str) -> str:
    focus_by_tool = {
        "extract_basic_statistics": "baseline autocorrelation and seasonal strength",
        "extract_within_channel_dynamics": "local peaks, changepoints, and slope persistence",
        "extract_forecast_residuals": "linear predictability and residual stress",
        "extract_data_quality": "quantization, saturation, and corruption signals",
        "extract_event_summary": "segment-level rise, fall, and oscillation patterns",
    }
    return focus_by_tool.get(tool_name, tool_name.replace("_", " "))


def _build_plan_rationale(
    *,
    tool_names: list[str],
    primary_model: str,
    runner_up_model: str,
    score_gap: float,
    feature_snapshot: dict[str, Any],
) -> str:
    evidence = _evidence_phrases(feature_snapshot)
    if evidence:
        evidence_text = ", ".join(evidence[:3])
        opener = f"The window shows {evidence_text}."
    else:
        opener = "The window mixes stable and shifting behavior."

    tool_focus = ", ".join(_tool_focus_phrase(name) for name in tool_names)
    if score_gap <= 0.75 and runner_up_model != primary_model:
        decision_text = "I need a broader diagnostic pass to separate the competing temporal explanations before routing."
    else:
        decision_text = "I will gather the minimum evidence needed to establish a confident routing state."
    return f"{opener} {decision_text} I will inspect {tool_focus}."


def build_diagnostic_plan(values: list[float] | tuple[float, ...]) -> DiagnosticPlan:
    history_values = _clean_history_values(values)
    if not history_values:
        rationale = (
            "The history is unavailable or too short, so I start with baseline autocorrelation and seasonality evidence first."
        )
        return DiagnosticPlan(
            tool_names=("extract_basic_statistics",),
            primary_model="patchtst",
            runner_up_model="arima",
            score_gap=0.0,
            rationale=rationale,
            feature_snapshot={},
        )

    feature_snapshot = _compute_feature_snapshot(history_values)
    scores = _heuristic_model_scores(feature_snapshot)
    primary_model = _select_primary_model(feature_snapshot, scores)
    primary_score = float(scores.get(primary_model, 0.0))
    ranked_models = [
        (model_name, model_score)
        for model_name, model_score in _ordered_models_by_score(scores)
        if model_name != primary_model
    ]
    runner_up_model, runner_up_score = ranked_models[0] if ranked_models else (primary_model, primary_score)
    score_gap = float(primary_score - runner_up_score)

    selected_tools = {"extract_basic_statistics"}
    selected_tools.update(MODEL_TOOL_REQUIREMENTS.get(primary_model, ()))
    if score_gap <= 0.75 and runner_up_model != primary_model:
        selected_tools.update(MODEL_TOOL_REQUIREMENTS.get(runner_up_model, ()))

    if (
        float(feature_snapshot.get("quality_quantization_score", 0.0)) >= 0.24
        or float(feature_snapshot.get("quality_saturation_ratio", 0.0)) >= 0.08
    ):
        selected_tools.add("extract_data_quality")

    if (
        float(feature_snapshot.get("event_segment_count", 0.0)) >= 4.0
        or str(feature_snapshot.get("dominant_pattern", "") or "") in {"rise", "fall", "oscillation"}
    ):
        selected_tools.add("extract_event_summary")

    if len(selected_tools) == 1:
        selected_tools.add("extract_within_channel_dynamics")

    ordered_tools = _ordered_tool_names(selected_tools)
    rationale = _build_plan_rationale(
        tool_names=ordered_tools,
        primary_model=primary_model,
        runner_up_model=runner_up_model,
        score_gap=score_gap,
        feature_snapshot=feature_snapshot,
    )
    return DiagnosticPlan(
        tool_names=tuple(ordered_tools),
        primary_model=primary_model,
        runner_up_model=runner_up_model,
        score_gap=score_gap,
        rationale=rationale,
        feature_snapshot=feature_snapshot,
    )


def select_feature_tool_names(values: list[float]) -> list[str]:
    return list(build_diagnostic_plan(values).tool_names)


def plan_diagnostic_tool_batches(
    required_tool_names: list[str] | tuple[str, ...],
    *,
    max_parallel_calls: int = 5,
) -> list[list[str]]:
    required = [name for name in FEATURE_TOOL_ORDER if name in set(required_tool_names or [])]
    if not required:
        required = ["extract_basic_statistics"]

    batch_cap = max(1, int(max_parallel_calls or 1))
    batches: list[list[str]] = []
    remaining = list(required)

    while remaining:
        batches.append(remaining[:batch_cap])
        remaining = remaining[batch_cap:]

    return [batch for batch in batches if batch]
