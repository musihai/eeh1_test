from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_TIMESTAMP_PREFIX_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})(?:\s+|$)(?P<body>.*)$"
)
_NAMED_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*(-?\d+(?:\.\d+)?)")
_TRAILING_VALUE_RE = re.compile(r"(-?\d+(?:\.\d+)?)$")
_PLAIN_VALUE_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class TimeSeriesTaskSpec:
    raw_prompt: str
    historical_data: str
    data_source: Optional[str] = None
    task_type: Optional[str] = None
    target_column: Optional[str] = None
    lookback_window: Optional[int] = None
    forecast_horizon: Optional[int] = None


@dataclass(frozen=True)
class ParsedTimeSeriesRecords:
    timestamps: list[Optional[str]]
    target_values: list[float]
    rows: list[dict[str, float]]
    feature_columns: list[str]


def _extract_int_field(prompt_text: str, field_name: str) -> Optional[int]:
    pattern = rf"(?im)^{re.escape(field_name)}[ \t]*:[ \t]*(\d+)[ \t]*$"
    match = re.search(pattern, prompt_text)
    if not match:
        return None
    return int(match.group(1))


def _extract_text_field(prompt_text: str, field_name: str) -> Optional[str]:
    pattern = rf"(?im)^{re.escape(field_name)}[ \t]*:[ \t]*(.+?)[ \t]*$"
    match = re.search(pattern, prompt_text)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def extract_historical_data_block(prompt_text: str) -> str:
    prompt_text = prompt_text.strip()
    if not prompt_text:
        return ""

    multiline_marker = re.search(r"(?im)^Historical Data:[ \t]*$", prompt_text)
    if multiline_marker:
        return prompt_text[multiline_marker.end():].strip()

    inline_marker = re.search(r"(?im)^Historical Data:[ \t]*(.+)$", prompt_text)
    if inline_marker:
        return prompt_text[inline_marker.start(1):].strip()

    return prompt_text


def parse_task_prompt(prompt_text: str, data_source: Optional[str] = None) -> TimeSeriesTaskSpec:
    raw_prompt = prompt_text.strip()
    historical_data = extract_historical_data_block(raw_prompt)
    task_type_match = re.search(r"(?im)^\[Task\]\s*(.+?)\s*$", raw_prompt)
    task_type = task_type_match.group(1).strip() if task_type_match else None
    if task_type is None:
        if re.search(r"(?i)multivariate", raw_prompt):
            task_type = "multivariate time-series forecasting"
        elif re.search(r"(?i)single-variable", raw_prompt):
            task_type = "single-variable time-series forecasting"
        elif re.search(r"(?i)time-series forecasting", raw_prompt):
            task_type = "time-series forecasting"

    return TimeSeriesTaskSpec(
        raw_prompt=raw_prompt,
        historical_data=historical_data,
        data_source=data_source,
        task_type=task_type,
        target_column=_extract_text_field(raw_prompt, "Target Column"),
        lookback_window=_extract_int_field(raw_prompt, "Lookback Window"),
        forecast_horizon=_extract_int_field(raw_prompt, "Forecast Horizon"),
    )


def _parse_named_values(body: str) -> dict[str, float]:
    matches = _NAMED_VALUE_RE.findall(body)
    if not matches:
        return {}

    named_values = {}
    for name, value in matches:
        try:
            named_values[name] = float(value)
        except ValueError:
            continue

    return named_values


def _select_named_value(body: str, target_column: Optional[str]) -> Optional[float]:
    named_values = _parse_named_values(body)
    if not named_values:
        return None

    if target_column and target_column in named_values:
        return named_values[target_column]
    if "target" in named_values:
        return named_values["target"]
    if "OT" in named_values:
        return named_values["OT"]

    first_key = next(iter(named_values))
    return named_values[first_key]


def parse_time_series_feature_records(
    text: str,
    target_column: Optional[str] = None,
) -> ParsedTimeSeriesRecords:
    """Parse timestamps, target values, and per-row named features from historical data."""
    historical_data = extract_historical_data_block(text)
    timestamps: list[Optional[str]] = []
    target_values: list[float] = []
    rows: list[dict[str, float]] = []
    feature_columns: list[str] = []
    feature_seen: set[str] = set()

    def _register_feature_names(ordered_names: list[str]) -> None:
        for name in ordered_names:
            if name not in feature_seen:
                feature_columns.append(name)
                feature_seen.add(name)

    inferred_target_name = target_column or "target"

    for raw_line in historical_data.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        timestamp: Optional[str] = None
        body = line
        ts_match = _TIMESTAMP_PREFIX_RE.match(line)
        if ts_match:
            timestamp = ts_match.group("timestamp")
            body = ts_match.group("body").strip()

        named_matches = _NAMED_VALUE_RE.findall(body)
        named_values = _parse_named_values(body)
        if named_values:
            target_value = _select_named_value(body, target_column)
            if target_value is None:
                continue
            timestamps.append(timestamp)
            target_values.append(target_value)
            rows.append(named_values)
            _register_feature_names([name for name, _ in named_matches])
            continue

        if _PLAIN_VALUE_RE.match(body):
            scalar = float(body)
            timestamps.append(timestamp)
            target_values.append(scalar)
            rows.append({inferred_target_name: scalar})
            _register_feature_names([inferred_target_name])
            continue

        trailing_match = _TRAILING_VALUE_RE.search(body)
        if trailing_match:
            scalar = float(trailing_match.group(1))
            timestamps.append(timestamp)
            target_values.append(scalar)
            rows.append({inferred_target_name: scalar})
            _register_feature_names([inferred_target_name])

    return ParsedTimeSeriesRecords(
        timestamps=timestamps,
        target_values=target_values,
        rows=rows,
        feature_columns=feature_columns,
    )


def parse_time_series_records(
    text: str,
    target_column: Optional[str] = None,
) -> tuple[list[Optional[str]], list[float]]:
    """
    Parse time-series records from a prompt or raw historical-data block.

    Supported line formats:
    - 2016-08-29 11:00:00 25.8170
    - 2016-08-29 11:00:00 OT=25.8170
    - 2016-08-29 11:00:00 HUFL=1.0 OT=25.8170
    - 25.8170
    """
    parsed = parse_time_series_feature_records(text, target_column=target_column)
    return parsed.timestamps, parsed.target_values
