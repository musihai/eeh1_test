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

    return f"""You are a time series forecasting agent. This is a MULTI-TURN interaction.

## Dataset Description
{dataset_description}

## Task Context
- Dataset: {dataset_name}
- Forecast target: {target_name}
- Task type: single-variable forecasting

## Workflow (MUST follow this order across turns)

**Turn 1 - Feature Extraction ONLY**:
Call one or more feature extraction tools. Do NOT call predict_time_series yet.
- `extract_basic_statistics`: median, MAD, autocorrelation, spectral features, CUSUM, distribution shape
- `extract_within_channel_dynamics`: changepoints, slopes, peaks, monotone runs
- `extract_forecast_residuals`: simple autoregressive residual diagnostics
- `extract_data_quality`: quantization, saturation, constant stretches, dropout
- `extract_event_summary`: rise/fall/flat/oscillation segment summary

**Turn 2 - Prediction**:
After seeing feature results in "Analysis History", call `predict_time_series` with the chosen model:
- 'patchtst': Usually strong for ETTh1-like seasonal univariate patterns
- 'arima': Good when trend/seasonality is stable and mostly linear
- 'chronos2': Good fallback for irregular or noisy windows
- 'itransformer': Available, but usually less preferred for single-variable inputs

**Turn 3 - Final Output**:
Reflect on feature analysis and model predictions, refine unreasonable results, and output the final forecast.

## Output Format (Turn 3 only)
Your response MUST contain ONLY the two tags below, in this order, with no extra text before/between/after them.
If you would use the model predictions directly, paste the actual predicted values inside <answer>.
<think>[Reflect predictions, note any adjustments]</think>
<answer>
2017-05-05 00:00:00 12.3450
...
</answer>

STRICT OUTPUT CONSTRAINTS (Turn 3):
- <answer> must contain EXACTLY 96 lines.
- Each line must be one forecast item in the format: YYYY-MM-DD HH:MM:SS value
- One value per line; no table, no bullets, no prose.
- Do not omit closing </answer>.
- Do not output markdown/code fences.

## CRITICAL RULES
- Turn 1: Feature extraction ONLY. Do NOT call predict_time_series.
- Turn 2: Call predict_time_series ONLY after features are extracted.
- Turn 3: Output answer ONLY after predictions are available.
- Do NOT output anything outside <think> and <answer>. Missing <answer> tags is incorrect.
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
) -> tuple[int, str]:
    has_features = bool(history_analysis)
    has_predictions = bool(prediction_results)

    if not has_features:
        return 1, "Call feature extraction tools (e.g., extract_basic_statistics). Do NOT call predict_time_series yet."
    if not has_predictions:
        return 2, "Call predict_time_series with your chosen model (e.g., 'chronos2')."
    return 3, (
        "Output your final answer in <think>...</think><answer>...</answer> format using the model predictions. "
        "Do NOT output any text outside these tags. <answer> must contain exactly 96 lines, one forecast value per line."
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
    conversation_has_tool_history: bool = False,
) -> str:
    history_records = list(history_analysis or [])
    history_text = "\n".join(history_records) if history_records else "No previous analysis performed."
    turn_num, action = get_runtime_turn_info(history_records, prediction_results)

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
2. Reflect briefly on whether the forecast values are consistent with the compact analysis summary.
3. Output the final refined prediction.
4. Output ONLY <think>...</think> and <answer>...</answer>. Do not include anything else.
5. <answer> must contain EXACTLY {forecast_horizon} lines.
6. Each line must be: timestamp + single float value (no extra words, no markdown, no bullets).

<think>
[Reflection on the consistency between feature evidence and forecast values]
</think>

<answer>
[Final prediction after reflection and refinement]
</answer>"""

    if history_records:
        return f"""**[Turn {turn_num}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows

### Analysis Summary
{history_text}

**Instructions**:
- Do NOT call feature extraction tools again.
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
- If "Analysis History" is empty -> Call feature extraction tools. Do NOT call predict_time_series yet.
- If "Analysis History" has features but "Model Predictions" is empty -> Call predict_time_series with model_name.
"""

# OpenAI-compatible tool schemas for TimeSeriesForecast actions.
PREDICT_TIMESERIES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "predict_time_series",
        "description": (
            "PREREQUISITE: You must have called feature extraction tools first (check 'Analysis History' is not empty). "
            "Do NOT call this on Turn 1 - extract features first! "
            "Models: 'patchtst' (preferred for ETTh1-like univariate seasonality), "
            "'arima' (stable trend/seasonality), 'chronos2' (irregular windows), "
            "'itransformer' (available but usually less helpful for single-variable inputs)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": (
                        "Model to use. Prefer 'patchtst' or 'arima' for regular ETTh1 OT patterns."
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
