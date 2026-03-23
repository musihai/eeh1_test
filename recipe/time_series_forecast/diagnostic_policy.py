from __future__ import annotations

from recipe.time_series_forecast.utils import (
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


def select_feature_tool_names(values: list[float]) -> list[str]:
    basic_features = extract_basic_statistics(values)
    dynamics_features = extract_within_channel_dynamics(values)
    residual_features = extract_forecast_residuals(values)
    quality_features = extract_data_quality(values)
    event_features = extract_event_summary(values)

    selected_tool_names = {"extract_basic_statistics"}

    strong_dynamics = (
        float(dynamics_features.get("changepoint_count", 0.0)) >= 1.0
        or float(dynamics_features.get("peak_count", 0.0)) >= 3.0
        or float(dynamics_features.get("slope_second_diff_max", 0.0)) >= max(float(basic_features.get("mad", 0.0)), 1e-3)
        or float(dynamics_features.get("monotone_duration", 0.0)) >= 0.35
    )
    residual_difficulty = (
        abs(float(residual_features.get("residual_acf1", 0.0))) >= 0.2
        or float(residual_features.get("residual_exceed_ratio", 0.0)) >= 0.08
        or float(residual_features.get("residual_concentration", 0.0)) >= 0.35
    )
    quality_issue = (
        float(quality_features.get("quality_quantization_score", 0.0)) >= 0.25
        or float(quality_features.get("quality_saturation_ratio", 0.0)) >= 0.08
        or float(quality_features.get("quality_constant_channel_ratio", 0.0)) > 0.0
        or float(quality_features.get("quality_dropout_ratio", 0.0)) > 0.0
    )
    eventful_window = (
        float(event_features.get("event_segment_count", 0.0)) >= 4.0
        or int(float(event_features.get("event_dominant_pattern", 0.0))) == 3
        or abs(float(basic_features.get("acf_seasonal", 0.0))) >= 0.25
    )

    if strong_dynamics:
        selected_tool_names.add("extract_within_channel_dynamics")
    if residual_difficulty:
        selected_tool_names.add("extract_forecast_residuals")
    if quality_issue:
        selected_tool_names.add("extract_data_quality")
    if eventful_window:
        selected_tool_names.add("extract_event_summary")

    if len(selected_tool_names) == 1:
        fallback_tool = (
            "extract_event_summary"
            if abs(float(basic_features.get("acf_seasonal", 0.0))) >= 0.2
            else "extract_within_channel_dynamics"
        )
        selected_tool_names.add(fallback_tool)

    return [name for name in FEATURE_TOOL_ORDER if name in selected_tool_names]


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
