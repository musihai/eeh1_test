# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from recipe.time_series_forecast.time_series_io import compact_historical_data_for_prompt
from recipe.time_series_forecast.refinement_support import REFINEMENT_OP_DESCRIPTIONS

def _build_dataset_description(
    data_source: Optional[str],
    target_column: Optional[str],
) -> str:
    dataset_name = data_source or "ETTh1"
    target_name = target_column or "OT"

    if dataset_name.lower() == "etth1":
        return (
            "This dataset is ETTh1, an hourly electricity transformer temperature benchmark. "
            f"The forecasting target is `{target_name}`, with six aligned load-related covariates "
            "(`HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`) observed over the same horizon. "
            "The target series usually shows strong daily seasonality, medium-term trend drift, "
            "flat stretches, and occasional abrupt level changes. Use both target dynamics and "
            "cross-channel evidence when judging which forecasting expert is appropriate."
        )

    return (
        f"This is a time-series forecasting task for dataset `{dataset_name}` with forecast target "
        f"`{target_name}`. Use the target series together with any provided covariates when reasoning about model choice."
    )


def build_timeseries_system_prompt(
    data_source: Optional[str] = None,
    target_column: Optional[str] = None,
) -> str:
    dataset_description = _build_dataset_description(data_source=data_source, target_column=target_column)
    target_name = target_column or "OT"
    dataset_name = data_source or "ETTh1"

    return f"""You are a time series forecasting agent.

## Dataset Description
{dataset_description}

## Task Context
- Dataset: {dataset_name}
- Forecast target: {target_name}
- Task type: multivariate forecasting with a single target channel

## CRITICAL RULES
- Follow the CURRENT user turn instructions only.
- Use tool calls only when tool schemas are available in the current turn.
- If no tool schema is available, NEVER emit `<tool_call>`.
- If you emit `<tool_call>`, the content inside it must be strict JSON with double-quoted strings.
- When the user asks for the final forecast, output ONLY the required `<think>...</think><answer>...</answer>` blocks.
- Never add markdown, bullets, or explanatory prose outside the requested output format.
"""


TIMESERIES_SYSTEM_PROMPT = build_timeseries_system_prompt()


def truncate_time_series_data(data: str, recent_rows: int = 48) -> str:
    lines = [line for line in data.strip().split("\n") if line.strip()]
    if recent_rows <= 0 or len(lines) <= recent_rows:
        return data

    tail_lines = lines[-recent_rows:]
    omitted = len(lines) - recent_rows
    return f"... ({omitted} earlier rows omitted) ...\n" + "\n".join(tail_lines)


def _normalize_turn_stage(turn_stage: Optional[str]) -> str:
    stage = str(turn_stage or "").strip().lower()
    if stage in {"diagnostic", "routing", "refinement"}:
        return stage
    return ""


ROUTING_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("acf1", "acf1"),
    ("acf_seasonal", "acf_seasonal"),
    ("cusum_max", "cusum_max"),
    ("changepoint_count", "changepoints"),
    ("peak_count", "peak_count"),
    ("peak_spacing_cv", "peak_spacing_cv"),
    ("monotone_duration", "monotone_duration"),
    ("residual_exceed_ratio", "residual_exceed_ratio"),
    ("quality_quantization_score", "quantization_score"),
    ("quality_saturation_ratio", "saturation_ratio"),
    ("dominant_pattern", "dominant_pattern"),
)

ROUTING_TOOL_FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("extract_basic_statistics", ("acf1", "acf_seasonal", "cusum_max")),
    ("extract_within_channel_dynamics", ("changepoint_count", "peak_count", "peak_spacing_cv", "monotone_duration")),
    ("extract_forecast_residuals", ("residual_exceed_ratio",)),
    ("extract_data_quality", ("quality_quantization_score", "quality_saturation_ratio")),
    ("extract_event_summary", ("dominant_pattern",)),
)

ROUTING_REASON_GUIDE = """### Routing Decision Guide
- `arima`: stable linear trend or seasonality, high acf, few changepoints, low residual excursions.
- `patchtst`: repeatable local motifs, regular peaks, moderate seasonal structure.
- `itransformer`: broader structural drift, multiple changepoints, long monotone segments.
- `chronos2`: irregular, noisy, weakly structured, or zero-shot windows."""


def _format_routing_field_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 100:
        return f"{numeric:.1f}"
    return f"{numeric:.4f}"


def _routing_tool_payload(
    payload: Mapping[str, Mapping[str, Any]],
    tool_name: str,
) -> Mapping[str, Any] | None:
    value = payload.get(tool_name)
    return value if isinstance(value, Mapping) else None


def _routing_signal_value(condition: bool | None) -> str:
    if condition is None:
        return "unknown"
    return "yes" if bool(condition) else "no"


def _routing_support_codes_from_payload(
    payload: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    basic = _routing_tool_payload(payload, "extract_basic_statistics")
    dynamics = _routing_tool_payload(payload, "extract_within_channel_dynamics")
    residuals = _routing_tool_payload(payload, "extract_forecast_residuals")
    quality = _routing_tool_payload(payload, "extract_data_quality")
    events = _routing_tool_payload(payload, "extract_event_summary")

    acf1 = float(basic.get("acf1", 0.0)) if basic else 0.0
    acf_seasonal = float(basic.get("acf_seasonal", 0.0)) if basic else 0.0
    cusum_max = float(basic.get("cusum_max", 0.0)) if basic else 0.0
    changepoint_count = float(dynamics.get("changepoint_count", 0.0)) if dynamics else 0.0
    peak_count = float(dynamics.get("peak_count", 0.0)) if dynamics else 0.0
    peak_spacing_cv = float(dynamics.get("peak_spacing_cv", 0.0)) if dynamics else 0.0
    monotone_duration = float(dynamics.get("monotone_duration", 0.0)) if dynamics else 0.0
    residual_exceed_ratio = float(residuals.get("residual_exceed_ratio", 0.0)) if residuals else 0.0
    quality_quantization_score = float(quality.get("quality_quantization_score", 0.0)) if quality else 0.0
    quality_saturation_ratio = float(quality.get("quality_saturation_ratio", 0.0)) if quality else 0.0
    dominant_pattern = str(events.get("dominant_pattern", "unknown")) if events else "unknown"

    supports = {
        "arima": [],
        "patchtst": [],
        "itransformer": [],
        "chronos2": [],
    }

    if basic and acf1 >= 0.93:
        supports["arima"].append("acf1_high")
    if basic and acf_seasonal >= 0.05:
        supports["arima"].append("seasonality_visible")
        supports["patchtst"].append("seasonality_visible")
    if dynamics and changepoint_count <= 1.0:
        supports["arima"].append("few_breaks")
    if residuals and residual_exceed_ratio <= 0.05:
        supports["arima"].append("residual_low")

    if dynamics and 2.0 <= peak_count <= 5.0:
        supports["patchtst"].append("repeatable_peaks")
    if dynamics and peak_spacing_cv <= 0.30:
        supports["patchtst"].append("peak_spacing_regular")
    if events and dominant_pattern == "oscillation":
        supports["patchtst"].append("oscillation")

    if dynamics and changepoint_count >= 3.0:
        supports["itransformer"].append("multi_breaks")
    if basic and cusum_max >= 70.0:
        supports["itransformer"].append("drift_high")
    if dynamics and monotone_duration >= 0.10:
        supports["itransformer"].append("long_monotone_segment")

    if residuals and residual_exceed_ratio >= 0.08:
        supports["chronos2"].append("residual_high")
    if quality and (quality_saturation_ratio >= 0.08 or quality_quantization_score >= 0.24):
        supports["chronos2"].append("quality_stress")
    if dynamics and peak_count >= 6.0 and peak_spacing_cv >= 0.35:
        supports["chronos2"].append("irregular_local_dynamics")
    return supports


def build_routing_evidence_card(
    *,
    routing_feature_payload: Mapping[str, Mapping[str, Any]] | None,
    completed_feature_tools: Sequence[str] | None = None,
) -> str:
    payload = dict(routing_feature_payload or {})
    completed = [str(name) for name in (completed_feature_tools or []) if str(name).strip()]
    lines: list[str] = []
    lines.append("### Routing Evidence Card")
    lines.append(f"observed_tools=[{', '.join(completed) if completed else 'none'}]")
    lines.append("expert_support_signals:")
    supports = _routing_support_codes_from_payload(payload)
    for model_name in ("arima", "patchtst", "itransformer", "chronos2"):
        values = supports[model_name]
        lines.append(f"- {model_name}=[{', '.join(values) if values else 'none'}]")

    missing_tools = [tool_name for tool_name, _field_names in ROUTING_TOOL_FIELD_GROUPS if tool_name not in payload]
    lines.append(f"missing_tool_groups=[{', '.join(missing_tools) if missing_tools else 'none'}]")
    lines.append(ROUTING_REASON_GUIDE)
    return "\n".join(lines)


def build_refinement_evidence_card(
    *,
    refinement_feature_payload: Mapping[str, Any] | None,
    prediction_model_used: Optional[str] = None,
) -> str:
    payload = dict(refinement_feature_payload or {})
    observed_tools = [str(item) for item in payload.get("observed_tools", []) if str(item).strip()]
    support_signals = [str(item) for item in payload.get("support_signals", []) if str(item).strip()]
    keep_support_signals = [str(item) for item in payload.get("keep_support_signals", []) if str(item).strip()]
    raw_edit_support_signals = payload.get("edit_support_signals", {}) or {}
    edit_support_signals = {
        str(name).strip(): [str(item) for item in values if str(item).strip()]
        for name, values in dict(raw_edit_support_signals).items()
        if str(name).strip()
    }
    candidate_adjustments = [str(item) for item in payload.get("candidate_adjustments", []) if str(item).strip()]
    keep_baseline_allowed = bool(payload.get("keep_baseline_allowed", True))

    lines = ["### Refinement Evidence Card"]
    if prediction_model_used:
        lines.append(f"selected_model={prediction_model_used}")
    lines.append(f"observed_tools=[{', '.join(observed_tools) if observed_tools else 'none'}]")
    lines.append(f"keep_support=[{', '.join(keep_support_signals) if keep_support_signals else 'none'}]")
    lines.append(f"support_signals=[{', '.join(support_signals) if support_signals else 'evidence_consistent'}]")
    decision_options = ["keep_baseline"]
    if candidate_adjustments and candidate_adjustments != ["none"]:
        lines.append("edit_support:")
        for adjustment in candidate_adjustments:
            description = REFINEMENT_OP_DESCRIPTIONS.get(adjustment, adjustment.replace("_", " "))
            support_values = edit_support_signals.get(adjustment, ["none"])
            lines.append(f"- {adjustment}={description}; support=[{', '.join(support_values)}]")
            decision_options.append(adjustment)
    else:
        lines.append("candidate_adjustments=[none]")
    lines.append(f"decision_options=[{', '.join(decision_options)}]")
    lines.append(f"keep_baseline_allowed={'yes' if keep_baseline_allowed else 'no'}")
    return "\n".join(lines)


def get_runtime_turn_info(
    history_analysis: Sequence[str] | None,
    prediction_results: Optional[str],
    forecast_horizon: int = 96,
    turn_stage: Optional[str] = None,
) -> tuple[str, str]:
    stage = _normalize_turn_stage(turn_stage)
    if not stage:
        has_predictions = bool(prediction_results)
        has_history = bool(list(history_analysis or []))
        if has_predictions:
            stage = "refinement"
        elif has_history:
            stage = "routing"
        else:
            stage = "diagnostic"

    if stage == "diagnostic":
        return "Diagnostic", (
            "Plan what evidence to collect from the historical series, then use feature-extraction tools only. "
            "You may emit multiple feature-tool calls in parallel in this turn. "
            "Do NOT call predict_time_series yet."
        )
    if stage == "routing":
        return "Routing", "Choose one forecasting expert from the current analysis state and call predict_time_series exactly once."
    return "Refinement", (
        "Review the selected model forecast against the diagnostics, then produce the final forecast. "
        "Do not forecast again from scratch. Reuse the provided forecast grid, keep the horizon fixed, "
        "and output only <think>...</think><answer>...</answer>."
    )


def build_runtime_user_prompt(
    *,
    data_source: str,
    target_column: str,
    lookback_window: int,
    forecast_horizon: int,
    time_series_data: str,
    history_analysis: Sequence[str] | None = None,
    prediction_results: Optional[str] = None,
    prediction_model_used: Optional[str] = None,
    available_feature_tools: Sequence[str] | None = None,
    completed_feature_tools: Sequence[str] | None = None,
    routing_feature_payload: Mapping[str, Mapping[str, Any]] | None = None,
    refinement_feature_payload: Mapping[str, Any] | None = None,
    turn_stage: Optional[str] = None,
) -> str:
    history_records = list(history_analysis or [])
    available_diagnostic_tools = [str(name) for name in (available_feature_tools or []) if str(name).strip()]
    completed_tools = [str(name) for name in (completed_feature_tools or []) if str(name).strip()]
    history_text = "\n".join(history_records) if history_records else "No previous analysis performed."
    compact_history = compact_historical_data_for_prompt(
        time_series_data,
        target_column=target_column,
    )
    stage_name, action = get_runtime_turn_info(
        history_records,
        prediction_results,
        forecast_horizon=forecast_horizon,
        turn_stage=turn_stage,
    )
    normalized_stage = _normalize_turn_stage(turn_stage) or stage_name.lower()

    if normalized_stage == "refinement":
        model_line = f"### Selected Forecast Model: {prediction_model_used}\n" if prediction_model_used else ""
        truncated_history = truncate_time_series_data(
            compact_history,
            recent_rows=min(max(int(lookback_window or 96) // 2, 24), int(lookback_window or 96)),
        )
        refinement_card = build_refinement_evidence_card(
            refinement_feature_payload=refinement_feature_payload,
            prediction_model_used=prediction_model_used,
        )
        return f"""**[Stage: {stage_name}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows
{model_line}{refinement_card}

### Analysis Summary
{history_text}

### Recent Historical Window
{truncated_history}

### Prediction Tool Output
{prediction_results}

**Instructions**:
1. Do NOT call more tools. No tool schema is available in this turn.
2. Treat "Prediction Tool Output" as the base forecast from the selected model.
3. Make the refinement decision from the Refinement Evidence Card first, then verify it against the recent historical window.
4. Choose exactly one decision from `decision_options`.
5. Use `keep_support` to decide whether the selected forecast already matches the evidence card.
6. `keep_baseline` is always allowed when the selected forecast already matches the evidence card.
7. Only choose a local edit when that exact edit's `support=[...]` line clearly justifies changing the selected forecast.
8. Output exactly one `<think>...</think>` block followed immediately by one `<answer>...</answer>` block.
9. `<think>` should briefly explain whether the selected forecast already matches the evidence card or why one local edit is needed.
10. `<answer>` must contain exactly one non-empty line in the form `decision=<name>`.
11. Do not output forecast rows, markdown, bullets, JSON, or extra commentary inside `<answer>`.
12. Stop immediately after `</answer>`."""

    if normalized_stage == "diagnostic":
        available_tools_text = ", ".join(available_diagnostic_tools) if available_diagnostic_tools else "extract_basic_statistics"
        return f"""**[Stage: {stage_name}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows
### Historical Data
{compact_history}

### Analysis History
{history_text}

### Diagnostic Tool Schemas Available This Turn
{available_tools_text}

**Instructions**:
- This is the planning and diagnostic stage.
- First decide what evidence you need from the current state before routing.
- Stay in the diagnostic stage and call only the feature-extraction tools exposed in this turn.
- You may call one or more feature tools in the same assistant turn.
- Each `<tool_call>` block must contain strict JSON only, such as `{{"name":"extract_basic_statistics","arguments":{{}}}}`.
- Use the tool outputs to characterize statistical properties, temporal patterns, and possible non-stationarity.
- Do NOT call predict_time_series in this turn.
"""

    if normalized_stage == "routing":
        evidence_card = build_routing_evidence_card(
            routing_feature_payload=routing_feature_payload,
            completed_feature_tools=completed_tools,
        )
        return f"""**[Stage: {stage_name}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows
{evidence_card}

**Instructions**:
- Do NOT call feature extraction tools again.
- Make the routing decision from the structured evidence card only.
- Do NOT restate the evidence in long prose.
- The `predict_time_series` call must be emitted as strict JSON inside `<tool_call>...</tool_call>`.
- Use the exact function name `predict_time_series`. Never emit placeholders such as `tool_name`.
- Choose exactly one model_name from `patchtst`, `itransformer`, `arima`, `chronos2`.
- Call predict_time_series exactly once with your chosen model_name.
- Valid example:
<tool_call>
{{"name":"predict_time_series","arguments":{{"model_name":"arima"}}}}
</tool_call>
"""

    raise ValueError(f"Unsupported runtime turn stage: {normalized_stage}")

# OpenAI-compatible tool schemas for TimeSeriesForecast actions.
PREDICT_TIMESERIES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "predict_time_series",
        "description": (
            "Call this only in the routing turn after completing your diagnostic analysis turn. "
            "Models: 'patchtst' (local temporal patterns with long-range dependencies), "
            "'itransformer' (cross-channel dependencies or broader structural interactions), "
            "'arima' (linear trends and stable seasonality), "
            "'chronos2' (irregular, noisy, or zero-shot scenarios)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Forecasting expert to invoke in the routing stage.",
                    "enum": ["patchtst", "itransformer", "arima", "chronos2"]
                }
            },
            "required": ["model_name"],
        },
    },
}

EXTRACT_BASIC_STATISTICS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_basic_statistics",
        "description": (
            "Extract core statistical features including median, MAD, autocorrelation, "
            "spectral features, CUSUM, quantile kurtosis, and other univariate summary signals."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_WITHIN_CHANNEL_DYNAMICS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_within_channel_dynamics",
        "description": (
            "Extract within-channel dynamics including changepoints, slopes, flatlines, "
            "peaks, entropy, and run-lengths."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_FORECAST_RESIDUALS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_forecast_residuals",
        "description": (
            "Extract AR residual diagnostics including mean, max, exceedance, ACF, "
            "and concentration."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_DATA_QUALITY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_data_quality",
        "description": (
            "Extract data quality metrics including quantization, saturation, "
            "constant channels, and dropout."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_EVENT_SUMMARY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_event_summary",
        "description": (
            "Extract event summary including segment count, rise/fall/flat/oscillation patterns."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

TIMESERIES_TOOL_SCHEMAS = [
    EXTRACT_BASIC_STATISTICS_SCHEMA,
    EXTRACT_WITHIN_CHANNEL_DYNAMICS_SCHEMA,
    EXTRACT_FORECAST_RESIDUALS_SCHEMA,
    EXTRACT_DATA_QUALITY_SCHEMA,
    EXTRACT_EVENT_SUMMARY_SCHEMA,
    PREDICT_TIMESERIES_TOOL_SCHEMA,
]

FEATURE_TOOL_SCHEMAS = [
    EXTRACT_BASIC_STATISTICS_SCHEMA,
    EXTRACT_WITHIN_CHANNEL_DYNAMICS_SCHEMA,
    EXTRACT_FORECAST_RESIDUALS_SCHEMA,
    EXTRACT_DATA_QUALITY_SCHEMA,
    EXTRACT_EVENT_SUMMARY_SCHEMA,
]
