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

from recipe.time_series_forecast.time_series_io import compact_historical_data_for_prompt, get_last_timestamp

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
        "Convert the selected model forecast into the final protocol. Do NOT forecast again from scratch. "
        "Choose exactly one of two actions: KEEP the base forecast unchanged, or apply one small local refinement on the same 96-row template. "
        "KEEP means the final answer must copy the provided forecast rows exactly. "
        "Reuse the provided timestamps exactly, stop at the last provided row, and output only <think>...</think><answer>...</answer>."
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
        terminal_timestamp = get_last_timestamp(prediction_results or "")
        prediction_lines = [line.strip() for line in str(prediction_results or "").splitlines() if line.strip()]
        terminal_row = prediction_lines[-1] if prediction_lines else ""
        terminal_line = (
            f"### Final Allowed Forecast Timestamp: {terminal_timestamp}\n" if terminal_timestamp else ""
        )
        terminal_row_line = (
            f"### Final Required Forecast Row: {terminal_row}\n" if terminal_row else ""
        )
        truncated_history = truncate_time_series_data(
            compact_history,
            recent_rows=min(max(int(lookback_window or 96) // 2, 24), int(lookback_window or 96)),
        )
        return f"""**[Stage: {stage_name}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows
{model_line}### Analysis Summary
{history_text}

### Recent Historical Window
{truncated_history}

{terminal_line}{terminal_row_line}### Prediction Tool Output
{prediction_results}

**Instructions**:
1. Do NOT call more tools.
2. No tool schema is available in this turn. Any <tool_call> output is invalid.
3. Treat "Prediction Tool Output" as the canonical base forecast produced by the selected model. It is the only legal row template for <answer>.
4. You must choose exactly one action:
   KEEP: copy the 96 forecast rows from "Prediction Tool Output" exactly once into <answer>, with identical timestamps and identical numeric values.
   LOCAL_REFINE: keep the same 96 timestamps and row order, but change only a small contiguous set of numeric values. Most rows should remain identical to the base forecast.
5. If unsure, choose KEEP. Flat or repeated tail values are allowed and do NOT by themselves justify adding rows or inventing new timestamps.
6. Reuse the timestamps from "Prediction Tool Output" in the same order. If you refine, adjust values only. Do NOT alter, renumber, skip, or invent timestamps.
7. If <think> says KEEP, then <answer> must be an exact row-for-row copy of "Prediction Tool Output". Do NOT reorder rows, restart from an earlier timestamp, or splice in rows from another part of the sequence.
8. Output ONLY one <think>...</think> block followed immediately by one <answer>...</answer> block.
9. <think> must be one short sentence that explicitly says either KEEP or LOCAL_REFINE and why.
10. <answer> must contain EXACTLY {forecast_horizon} lines, one forecast item per line, and must use `YYYY-MM-DD HH:MM:SS value`.
11. Do NOT emit any timestamp that does not already appear in "Prediction Tool Output". Never synthesize timestamps such as `24:00:00` or `25:00:00`.
12. The final answer must end at the last timestamp shown in "Final Allowed Forecast Timestamp". Do NOT emit any later timestamp and do NOT add extra rows beyond that terminal row.
13. The {forecast_horizon}th row must be exactly the line shown in "Final Required Forecast Row". After you write that exact row, immediately output </answer>.
14. Do NOT restart from row 1, do NOT duplicate the sequence, and do NOT continue a flat last value past the terminal timestamp.
15. Do not add extra words, markdown, bullets, or commentary on answer lines.
        16. Stop immediately after </answer>, with no extra text after it.
        17. Your reply must start with <think>. Do NOT copy placeholder text or template scaffolding into the final answer."""

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
        return f"""**[Stage: {stage_name}] Action: {action}**
### Lookback Window: {lookback_window} rows
### Forecast Horizon: {forecast_horizon} rows
### Historical Data
{compact_history}

### Analysis Summary
{history_text}

### Completed Diagnostic Tools
{", ".join(completed_tools) if completed_tools else "none"}

**Instructions**:
- Do NOT call feature extraction tools again.
- Use the accumulated analysis history to choose the forecasting expert whose inductive bias best matches the observed evidence.
- The `predict_time_series` call must be emitted as strict JSON inside `<tool_call>...</tool_call>`.
- Forecasting expert guidance:
  - `patchtst`: local temporal patterns with long-range dependencies.
  - `itransformer`: cross-channel dependencies or broader structural interactions.
  - `arima`: linear trends and stable seasonality.
  - `chronos2`: irregular, noisy, or zero-shot scenarios.
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
