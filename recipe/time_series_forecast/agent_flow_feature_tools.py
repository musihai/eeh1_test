from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from recipe.time_series_forecast.diagnostic_features import (
    extract_basic_statistics,
    extract_data_quality,
    extract_event_summary,
    extract_forecast_residuals,
    extract_within_channel_dynamics,
    format_basic_statistics,
    format_data_quality,
    format_event_summary,
    format_forecast_residuals,
    format_within_channel_dynamics,
)


@dataclass(frozen=True)
class FeatureToolSpec:
    name: str
    state_attr: str
    extractor: Callable[[list[float]], dict]
    formatter: Callable[[dict], str]
    success_log: str


FEATURE_TOOL_SPECS = {
    "extract_basic_statistics": FeatureToolSpec(
        name="extract_basic_statistics",
        state_attr="basic_statistics",
        extractor=extract_basic_statistics,
        formatter=format_basic_statistics,
        success_log="Basic statistics extraction completed",
    ),
    "extract_within_channel_dynamics": FeatureToolSpec(
        name="extract_within_channel_dynamics",
        state_attr="within_channel_dynamics",
        extractor=extract_within_channel_dynamics,
        formatter=format_within_channel_dynamics,
        success_log="Within-channel dynamics extraction completed",
    ),
    "extract_forecast_residuals": FeatureToolSpec(
        name="extract_forecast_residuals",
        state_attr="forecast_residuals",
        extractor=extract_forecast_residuals,
        formatter=format_forecast_residuals,
        success_log="Forecast residuals extraction completed",
    ),
    "extract_data_quality": FeatureToolSpec(
        name="extract_data_quality",
        state_attr="data_quality",
        extractor=extract_data_quality,
        formatter=format_data_quality,
        success_log="Data quality extraction completed",
    ),
    "extract_event_summary": FeatureToolSpec(
        name="extract_event_summary",
        state_attr="event_summary",
        extractor=extract_event_summary,
        formatter=format_event_summary,
        success_log="Event summary extraction completed",
    ),
}


__all__ = ["FEATURE_TOOL_SPECS", "FeatureToolSpec"]
