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
            "Inspect the historical series with feature-extraction tools only. "
            "You may emit multiple feature-tool calls in parallel in this turn. "
            "Do NOT call predict_time_series yet."
        )
    if stage == "routing":
        return "Routing", "Call predict_time_series exactly once with the model chosen from the current diagnostic memory state."
    return "Refinement", (
        "Output your final answer in <think>...</think><answer>...</answer> format using the selected model prediction as the base forecast. "
        "Use <think> to briefly explain whether you keep the base forecast unchanged or apply a small local refinement. "
        "Keep the forecast unchanged if it is already consistent; if you refine it, keep the change small, local, and evidence-based. "
        f"Do NOT output any text outside <think> and <answer>. <answer> must contain exactly {forecast_horizon} lines, one numeric value per line, no timestamps. "
        f"Stop immediately after the {forecast_horizon}th value and close with </answer>. Do not generate any extra value or any text after </answer>."
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
    turn_stage: Optional[str] = None,
) -> str:
    history_records = list(history_analysis or [])
    available_diagnostic_tools = [str(name) for name in (required_feature_tools or []) if str(name).strip()]
    completed_tools = [str(name) for name in (completed_feature_tools or []) if str(name).strip()]
    history_text = "\n".join(history_records) if history_records else "No previous analysis performed."
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
            time_series_data,
            recent_rows=min(max(int(lookback_window or 96) // 2, 24), int(lookback_window or 96)),
        )
        return f"""**[Stage: {stage_name}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows
{model_line}### Analysis Summary
{history_text}

### Recent Historical Window
{truncated_history}

### Prediction Tool Output
{prediction_results}

**Instructions**:
1. Do NOT call more tools.
2. No tool schema is available in this turn. Any <tool_call> output is invalid.
3. Treat "Prediction Tool Output" as the base forecast produced by the selected model.
4. If the base forecast is already consistent, keep it unchanged. If refinement is needed, keep it small, local, and evidence-based.
5. Do NOT rewrite the forecast arbitrarily or ignore the selected-model forecast.
6. Output ONLY one <think>...</think> block followed immediately by one <answer>...</answer> block.
7. <think> must briefly state whether you keep the base forecast or apply a small local refinement, and why.
8. <answer> must contain EXACTLY {forecast_horizon} lines.
9. Each answer line must be a single float value only (no timestamp, no extra words, no markdown, no bullets).
10. Stop immediately after the {forecast_horizon}th value, then output </answer>, with no extra text after it.
11. Your reply must start with <think>. Do NOT copy placeholder text or template scaffolding into the final answer."""

    if normalized_stage == "diagnostic":
        available_tools_text = ", ".join(available_diagnostic_tools) if available_diagnostic_tools else "extract_basic_statistics"
        return f"""**[Stage: {stage_name}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows
### Historical Data
{time_series_data}

### Analysis History
{history_text}

### Diagnostic Tool Schemas Available This Turn
{available_tools_text}

**Instructions**:
- Stay in the diagnostic stage and inspect the series through feature-extraction tools only.
- You may call one or more feature tools in the same assistant turn.
- Do NOT call predict_time_series in this turn.
"""

    if normalized_stage == "routing":
        return f"""**[Stage: {stage_name}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows
### Historical Data
{time_series_data}

### Analysis Summary
{history_text}

### Completed Diagnostic Tools
{", ".join(completed_tools) if completed_tools else "none"}

**Instructions**:
- Do NOT call feature extraction tools again.
- Choose the model from the completed diagnostics and maintained memory state, not from a fixed default.
- Compare the selected model against at least one plausible alternative in your reasoning before you call the tool.
- Match the model to the observed evidence: `patchtst` for local motifs and regular seasonality, `arima` for stable autocorrelation structure, `chronos2` for irregular or quality-stressed windows, and `itransformer` for broader structural drift.
- Call predict_time_series exactly once with your chosen model_name.
"""

    raise ValueError(f"Unsupported runtime turn stage: {normalized_stage}")

# OpenAI-compatible tool schemas for TimeSeriesForecast actions.
PREDICT_TIMESERIES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "predict_time_series",
        "description": (
            "Call this only in the routing turn after completing your diagnostic analysis turn. "
            "Models: 'patchtst' (smooth windows with local periodicity), "
            "'arima' (stable trend/seasonality), 'chronos2' (irregular or noisy windows), "
            "'itransformer' (regime changes or longer-range dependencies). "
            "Choose from the maintained diagnostic state rather than a fixed preference."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": (
                        "Model to use. Choose from the diagnostic evidence and maintained state rather than a fixed preference."
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

FEATURE_TOOL_SCHEMAS = [
    EXTRACT_BASIC_STATISTICS_SCHEMA,
    EXTRACT_WITHIN_CHANNEL_DYNAMICS_SCHEMA,
    EXTRACT_FORECAST_RESIDUALS_SCHEMA,
    EXTRACT_DATA_QUALITY_SCHEMA,
    EXTRACT_EVENT_SUMMARY_SCHEMA,
]
