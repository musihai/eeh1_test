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

from typing import Optional, Sequence


def _build_dataset_description(
    data_source: Optional[str],
    target_column: Optional[str],
) -> str:
    dataset_name = data_source or "ETTh1"
    target_name = target_column or "OT"

    if dataset_name.lower() == "etth1":
        return (
            "This dataset is ETTh1, an hourly electricity transformer temperature benchmark. "
            f"The forecasting target is the single variable `{target_name}`. The series usually "
            "shows strong daily seasonality, medium-term trend drift, flat stretches, and occasional "
            "abrupt level changes. This is a univariate forecasting task, so focus on temporal "
            "patterns in the target itself rather than cross-channel reasoning."
        )

    return (
        f"This is a time-series forecasting task for dataset `{dataset_name}` with single target "
        f"`{target_name}`. Focus on seasonality, trend, local dynamics, and data quality of the target series."
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
- Task type: single-variable forecasting

## CRITICAL RULES
- Follow the CURRENT user turn instructions only.
- Use tool calls only when tool schemas are available in the current turn.
- If no tool schema is available, NEVER emit `<tool_call>`.
- When the user asks for the final forecast, output ONLY the required `<answer>...</answer>` block.
- Never add markdown, bullets, or explanatory prose outside the requested output format.
"""


TIMESERIES_SYSTEM_PROMPT = build_timeseries_system_prompt()


def truncate_time_series_data(data: str, head: int = 5, tail: int = 5) -> str:
    lines = data.strip().split("\n")
    if len(lines) <= head + tail:
        return data

    head_lines = lines[:head]
    tail_lines = lines[-tail:]
    omitted = len(lines) - head - tail
    return "\n".join(head_lines) + f"\n... ({omitted} rows omitted) ...\n" + "\n".join(tail_lines)


def get_runtime_turn_info(
    history_analysis: Sequence[str] | None,
    prediction_results: Optional[str],
    forecast_horizon: int = 96,
    required_feature_tools: Sequence[str] | None = None,
    completed_feature_tools: Sequence[str] | None = None,
) -> tuple[int, str]:
    has_predictions = bool(prediction_results)
    required = [str(name) for name in (required_feature_tools or []) if str(name).strip()]
    completed = {str(name) for name in (completed_feature_tools or []) if str(name).strip()}
    missing = [name for name in required if name not in completed]

    if not has_predictions and missing:
        if len(missing) == 1:
            return 1, f"Call the remaining feature extraction tool `{missing[0]}`. Do NOT call predict_time_series yet."
        missing_text = ", ".join(f"`{name}`" for name in missing)
        return 1, f"Call the remaining feature extraction tools in parallel when possible: {missing_text}. Do NOT call predict_time_series yet."
    if not has_predictions:
        return 2, "Call predict_time_series with your chosen model (e.g., 'chronos2')."
    return 3, (
        "Output your final answer in <answer>...</answer> format using the selected model prediction as the base forecast. "
        "Keep it unchanged if it is already consistent; if you refine it, keep the change small, local, and evidence-based. "
        f"Do NOT output any text outside <answer>. <answer> must contain exactly {forecast_horizon} lines, one numeric value per line, no timestamps. "
        f"Stop immediately after the {forecast_horizon}th value. Do not generate any extra value. "
        f"After the {forecast_horizon}th value, immediately close with </answer>, and do not output any text after it. "
        "Do not emit <think>."
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
    required_feature_tools: Sequence[str] | None = None,
    completed_feature_tools: Sequence[str] | None = None,
) -> str:
    history_records = list(history_analysis or [])
    required_tools = [str(name) for name in (required_feature_tools or []) if str(name).strip()]
    completed_tools = [str(name) for name in (completed_feature_tools or []) if str(name).strip()]
    completed_tool_set = set(completed_tools)
    missing_tools = [name for name in required_tools if name not in completed_tool_set]
    history_text = "\n".join(history_records) if history_records else "No previous analysis performed."
    turn_num, action = get_runtime_turn_info(
        history_records,
        prediction_results,
        forecast_horizon=forecast_horizon,
        required_feature_tools=required_tools,
        completed_feature_tools=completed_tools,
    )

    if prediction_results:
        model_line = f"### Selected Forecast Model: {prediction_model_used}\n" if prediction_model_used else ""
        return f"""**[Turn {turn_num}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows
{model_line}### Analysis Summary
{history_text}

### Prediction Tool Output
{prediction_results}

**Instructions**:
1. Do NOT call more tools.
2. No tool schema is available in this turn. Any <tool_call> output is invalid.
3. Treat "Prediction Tool Output" as the base forecast produced by the selected model.
4. If the base forecast is already consistent, keep it unchanged. If refinement is needed, keep it small, local, and evidence-based.
5. Do NOT rewrite the forecast arbitrarily or ignore the selected-model forecast.
6. Output ONLY <answer>...</answer>. Do not include anything else.
7. <answer> must contain EXACTLY {forecast_horizon} lines.
8. Each line must be a single float value only (no timestamp, no extra words, no markdown, no bullets).
9. Stop immediately after the {forecast_horizon}th value, and do NOT generate any extra value.
10. After the {forecast_horizon}th value, immediately output </answer>, with no extra text after it.

<answer>
[Final prediction after optional small refinement]
</answer>"""

    if missing_tools:
        required_text = ", ".join(required_tools) if required_tools else "extract_basic_statistics"
        completed_text = ", ".join(completed_tools) if completed_tools else "none"
        missing_text = ", ".join(missing_tools)
        return f"""**[Turn {turn_num}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows
### Historical Data
{time_series_data}

### Analysis History
{history_text}

### Required Diagnostic Tools
{required_text}

### Completed Diagnostic Tools
{completed_text}

### Remaining Diagnostic Tools
{missing_text}

**Instructions**:
- Stay in Turn 1 until ALL required diagnostic tools have been called.
- Call the remaining diagnostic tools in this turn.
- If multiple required tools remain, emit multiple tool calls in the same assistant turn.
- Do NOT call predict_time_series until every required diagnostic tool above has been executed.
"""

    if history_records:
        return f"""**[Turn {turn_num}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows

### Analysis Summary
{history_text}

### Completed Diagnostic Tools
{", ".join(completed_tools) if completed_tools else "none"}

**Instructions**:
- Do NOT call feature extraction tools again.
- Choose the model from the completed diagnostics, not from a fixed prior.
- `patchtst`: smooth windows with strong local periodicity.
- `arima`: stable linear trend and seasonality.
- `chronos2`: irregular or noisy windows as a robust fallback.
- `itransformer`: regime changes or longer-range dependencies.
- Call predict_time_series exactly once with your chosen model_name.
"""

    prediction_text = "No predictions available yet. Call predict_time_series to generate forecasts."
    return f"""**[Turn {turn_num}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows
### Historical Data
{time_series_data}

### Analysis History
{history_text}

### Model Predictions
{prediction_text}

**Check your current state and act accordingly:**
- If any required diagnostic tools are still missing -> Call the remaining feature extraction tools. Do NOT call predict_time_series yet.
- Only after all required diagnostic tools are completed and "Model Predictions" is empty -> Call predict_time_series with model_name.
"""

# OpenAI-compatible tool schemas for TimeSeriesForecast actions.
PREDICT_TIMESERIES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "predict_time_series",
        "description": (
            "PREREQUISITE: You must have completed all required feature extraction tools before calling this tool. "
            "Do NOT call this until the runtime user message indicates the diagnostic stage is complete. "
            "Models: 'patchtst' (smooth windows with local periodicity), "
            "'arima' (stable trend/seasonality), 'chronos2' (irregular or noisy windows), "
            "'itransformer' (regime changes or longer-range dependencies). "
            "Choose based on extracted features rather than a fixed preference."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": (
                        "Model to use. Choose based on the extracted features rather than a fixed preference."
                    ),
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
