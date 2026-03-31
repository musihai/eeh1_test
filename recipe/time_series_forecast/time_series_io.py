from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from recipe.time_series_forecast.config_utils import get_default_lengths
from recipe.time_series_forecast.task_protocol import parse_time_series_feature_records, parse_time_series_records


DEFAULT_LOOKBACK_WINDOW, DEFAULT_FORECAST_HORIZON = get_default_lengths()
SYNTHETIC_TIMESTAMP_ANCHOR = pd.Timestamp("2000-01-01 00:00:00")


def parse_time_series_string(
    data_str: str,
    target_column: Optional[str] = None,
) -> Tuple[List[Optional[str]], List[float]]:
    """
    Parse time series text into timestamps and values.

    Supports full task prompts and raw historical-data blocks.
    Supported record formats include:
        "2017-05-01 00:00:00 11.588"
        "2017-05-01 00:00:00 OT=11.588"
        "2017-05-01 00:00:00 HUFL=1.0 OT=11.588"
    """
    return parse_time_series_records(data_str, target_column=target_column)


def infer_frequency(timestamps: List[pd.Timestamp]) -> Optional[pd.Timedelta]:
    """Infer the most common frequency from a list of timestamps."""
    if len(timestamps) < 2:
        return None

    diffs = []
    for i in range(1, len(timestamps)):
        diff = timestamps[i] - timestamps[i - 1]
        if diff > pd.Timedelta(0):
            diffs.append(diff)

    if not diffs:
        return None

    diff_counts = Counter(diffs)
    return diff_counts.most_common(1)[0][0]


def parse_time_series_to_dataframe(
    data_str: str,
    series_id: str = "series_0",
    default_freq: str = "1h",
    target_column: Optional[str] = None,
    include_covariates: bool = False,
) -> pd.DataFrame:
    """
    Parse time series string into a DataFrame suitable for prediction tools.

    Handles:
    1. All timestamps present
    2. No timestamps
    3. Partial timestamps with inferred frequency
    4. Single timestamp with default frequency
    """
    parsed = parse_time_series_feature_records(data_str, target_column=target_column)
    timestamps = parsed.timestamps
    values = parsed.target_values

    if not values:
        raise ValueError("No valid data points found in the input string")

    valid_timestamps = [ts for ts in timestamps if ts is not None]
    valid_ts_indices = [i for i, ts in enumerate(timestamps) if ts is not None]

    datetime_list = []

    if len(valid_timestamps) == len(timestamps) and len(valid_timestamps) > 0:
        datetime_list = [pd.to_datetime(ts) for ts in timestamps]
    elif len(valid_timestamps) == 0:
        base_time = SYNTHETIC_TIMESTAMP_ANCHOR
        freq = pd.Timedelta(default_freq)
        datetime_list = [base_time + freq * i for i in range(len(values))]
    elif len(valid_timestamps) >= 2:
        parsed_valid_ts = [pd.to_datetime(ts) for ts in valid_timestamps]
        inferred_freq = infer_frequency(parsed_valid_ts)
        if inferred_freq is None:
            inferred_freq = pd.Timedelta(default_freq)

        first_valid_idx = valid_ts_indices[0]
        first_valid_ts = pd.to_datetime(valid_timestamps[0])
        for i in range(len(values)):
            offset = i - first_valid_idx
            datetime_list.append(first_valid_ts + inferred_freq * offset)
    else:
        valid_idx = valid_ts_indices[0]
        anchor_ts = pd.to_datetime(valid_timestamps[0])
        freq = pd.Timedelta(default_freq)
        for i in range(len(values)):
            offset = i - valid_idx
            datetime_list.append(anchor_ts + freq * offset)

    data = {
        "id": [series_id] * len(values),
        "timestamp": datetime_list,
        "target": values,
    }

    if include_covariates:
        target_name = target_column or "target"
        feature_columns = list(parsed.feature_columns or [target_name])
        for column_name in feature_columns:
            column_values: list[float] = []
            for row_idx, row in enumerate(parsed.rows):
                if column_name in row:
                    column_values.append(float(row[column_name]))
                    continue
                if column_name == target_name:
                    column_values.append(float(values[row_idx]))
                    continue
                raise ValueError(
                    f"Missing feature '{column_name}' in historical row {row_idx}. "
                    "Multivariate prompts must provide all declared feature columns on every row."
                )
            data[column_name] = column_values

    frame = pd.DataFrame(data)
    if include_covariates:
        frame.attrs["feature_columns"] = list(parsed.feature_columns or [target_column or "target"])
        frame.attrs["target_column"] = target_column or "target"
    return frame


def format_predictions_to_string(
    pred_df: pd.DataFrame,
    last_timestamp: str = None,
    freq_hours: int = 1,
) -> str:
    """Format prediction DataFrame into the canonical timestamp-value string."""
    lines = []
    value_col = None

    for col in pred_df.columns:
        col_str = str(col)
        if col_str == "0.5" or col_str == "target_0.5":
            value_col = col
            break

    if value_col is None and "predictions" in pred_df.columns:
        value_col = "predictions"

    if value_col is None:
        numeric_cols = pred_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            value_col = numeric_cols[-1]
        else:
            raise ValueError("No numeric prediction column found in DataFrame")

    if "timestamp" in pred_df.columns:
        for _, row in pred_df.iterrows():
            ts = row["timestamp"]
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, pd.Timestamp) else str(ts)
            value = row[value_col]
            try:
                lines.append(f"{ts_str} {float(value):.4f}")
            except (ValueError, TypeError):
                lines.append(f"{ts_str} {value}")
    else:
        base_time = pd.to_datetime(last_timestamp) if last_timestamp else SYNTHETIC_TIMESTAMP_ANCHOR
        for i, (_, row) in enumerate(pred_df.iterrows()):
            ts = base_time + pd.Timedelta(hours=(i + 1) * freq_hours)
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            value = row[value_col]
            try:
                lines.append(f"{ts_str} {float(value):.4f}")
            except (ValueError, TypeError):
                lines.append(f"{ts_str} {value}")

    return "\n".join(lines)


def compact_prediction_tool_output_from_string(
    prediction_text: str,
    *,
    model_name: Optional[str] = None,
    freq_hours: int = 1,
) -> str:
    """Compress timestamped prediction text into a compact tool-response format."""
    timestamps, values = parse_time_series_string(prediction_text)
    if not values:
        return prediction_text

    start_timestamp = next((ts for ts in timestamps if ts), None)
    lines: list[str] = []
    if model_name:
        lines.append(f"Model: {model_name}")
    if start_timestamp:
        lines.append(f"Start Timestamp: {start_timestamp}")
    lines.append(f"Frequency Hours: {freq_hours}")
    lines.append("Forecast Values:")
    lines.extend(f"{float(value):.4f}" for value in values)
    return "\n".join(lines)


def compact_prediction_selection_preview_from_string(
    prediction_text: str,
    *,
    model_name: Optional[str] = None,
    freq_hours: int = 1,
    head_values: int = 6,
    tail_values: int = 6,
) -> str:
    """Render a short candidate preview for Turn-3 final selection prompts."""
    timestamps, values = parse_time_series_string(prediction_text)
    if not values:
        return prediction_text

    array = np.asarray(values, dtype=float)
    start_timestamp = next((ts for ts in timestamps if ts), None)
    head = ", ".join(f"{float(value):.4f}" for value in array[:head_values])
    tail = ", ".join(f"{float(value):.4f}" for value in array[-tail_values:])
    mean_abs_step = float(np.mean(np.abs(np.diff(array)))) if len(array) >= 2 else 0.0

    lines: list[str] = []
    if model_name:
        lines.append(f"Model: {model_name}")
    if start_timestamp:
        lines.append(f"Start Timestamp: {start_timestamp}")
    lines.append(f"Frequency Hours: {freq_hours}")
    lines.append(f"Count: {len(array)}")
    lines.append(
        "Summary: "
        f"mean={float(array.mean()):.4f}, std={float(array.std()):.4f}, "
        f"min={float(array.min()):.4f}, max={float(array.max()):.4f}, "
        f"net_change={float(array[-1] - array[0]):.4f}, mean_abs_step={mean_abs_step:.4f}"
    )
    lines.append(f"Head Values: {head}")
    lines.append(f"Tail Values: {tail}")
    return "\n".join(lines)


def compact_historical_data_for_prompt(
    historical_text: str,
    *,
    target_column: Optional[str] = None,
    default_freq: str = "1h",
) -> str:
    """Render historical context in a compact row-matrix format when beneficial."""
    raw_text = str(historical_text or "").strip()
    if not raw_text:
        return raw_text

    try:
        frame = parse_time_series_to_dataframe(
            raw_text,
            target_column=target_column,
            default_freq=default_freq,
            include_covariates=True,
        )
    except Exception:
        return raw_text

    feature_columns = list(frame.attrs.get("feature_columns") or [target_column or "target"])
    header = ",".join(["timestamp", *feature_columns])
    rows: list[str] = [header]
    for _, row in frame.iterrows():
        timestamp = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        values = [f"{float(row[column_name]):.4f}" for column_name in feature_columns]
        rows.append(",".join([timestamp, *values]))

    compact_text = "\n".join(rows)
    if len(compact_text) >= len(raw_text):
        return raw_text
    return compact_text


def compact_historical_selection_context(
    historical_text: str,
    *,
    target_column: Optional[str] = None,
    default_freq: str = "1h",
    recent_rows: int = 24,
) -> str:
    """Build a short target-focused history summary for v19 final selection."""
    raw_text = str(historical_text or "").strip()
    if not raw_text:
        return raw_text

    try:
        frame = parse_time_series_to_dataframe(
            raw_text,
            target_column=target_column,
            default_freq=default_freq,
            include_covariates=True,
        )
    except Exception:
        return raw_text

    target_name = target_column or frame.attrs.get("target_column") or "target"
    if target_name not in frame.columns:
        target_name = "target"

    target_values = frame[target_name].astype(float).to_numpy()
    recent = frame.tail(max(1, int(recent_rows)))

    rows = ["timestamp,target"]
    for _, row in recent.iterrows():
        timestamp = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(f"{timestamp},{float(row[target_name]):.4f}")

    mean_abs_step = float(np.mean(np.abs(np.diff(target_values)))) if len(target_values) >= 2 else 0.0
    summary = (
        "Target Summary: "
        f"count={len(target_values)}, mean={float(target_values.mean()):.4f}, std={float(target_values.std()):.4f}, "
        f"min={float(target_values.min()):.4f}, max={float(target_values.max()):.4f}, "
        f"net_change={float(target_values[-1] - target_values[0]):.4f}, mean_abs_step={mean_abs_step:.4f}"
    )
    return f"{summary}\nRecent Target Rows (last {len(recent)}):\n" + "\n".join(rows)


def format_prediction_tool_output(
    pred_df: pd.DataFrame,
    last_timestamp: str = None,
    *,
    freq_hours: int = 1,
    model_name: Optional[str] = None,
) -> str:
    """Format forecast output for compact tool responses."""
    full_prediction_text = format_predictions_to_string(
        pred_df,
        last_timestamp=last_timestamp,
        freq_hours=freq_hours,
    )
    return compact_prediction_tool_output_from_string(
        full_prediction_text,
        model_name=model_name,
        freq_hours=freq_hours,
    )


def get_last_timestamp(data_str: str) -> Optional[str]:
    """Extract the last timestamp from a time series string."""
    timestamps, _ = parse_time_series_string(data_str)
    if timestamps and timestamps[-1]:
        return timestamps[-1]
    return None


__all__ = [
    "compact_historical_selection_context",
    "compact_historical_data_for_prompt",
    "compact_prediction_selection_preview_from_string",
    "DEFAULT_FORECAST_HORIZON",
    "DEFAULT_LOOKBACK_WINDOW",
    "SYNTHETIC_TIMESTAMP_ANCHOR",
    "compact_prediction_tool_output_from_string",
    "format_prediction_tool_output",
    "format_predictions_to_string",
    "get_last_timestamp",
    "infer_frequency",
    "parse_time_series_string",
    "parse_time_series_to_dataframe",
]
