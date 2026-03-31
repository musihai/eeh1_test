from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from recipe.time_series_forecast.config_utils import (
    ETTH1_COVARIATE_COLUMNS,
    ETTH1_FEATURE_COLUMNS,
    ETTH1_TARGET_COLUMN,
)
from recipe.time_series_forecast.prompts import (
    FEATURE_TOOL_SCHEMAS,
    PREDICT_TIMESERIES_TOOL_SCHEMA,
    ROUTE_TIMESERIES_TOOL_SCHEMA,
    build_routing_evidence_card,
    build_runtime_user_prompt,
    build_timeseries_system_prompt,
)
from recipe.time_series_forecast.refinement_support import (
    REFINEMENT_KEEP_DECISION,
    build_refinement_candidate_prediction_text_map,
    build_refinement_support_payload,
    filter_refinement_candidates_for_model,
    generate_local_refinement_candidates,
    refinement_decision_name,
)
from recipe.time_series_forecast.time_series_io import DEFAULT_FORECAST_HORIZON, DEFAULT_LOOKBACK_WINDOW
from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RL_JSONL,
    DATASET_KIND_RUNTIME_SFT_PARQUET,
    DATASET_KIND_TEACHER_CURATED_SFT,
    HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
    require_multivariate_etth1_metadata,
    validate_sibling_metadata,
)
from recipe.time_series_forecast.dataset_file_utils import write_metadata_file
from recipe.time_series_forecast.diagnostic_policy import (
    build_diagnostic_plan,
    plan_diagnostic_tool_batches,
    select_feature_tool_names,
)
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.utils import (
    extract_basic_statistics,
    extract_data_quality,
    extract_event_summary,
    extract_forecast_residuals,
    extract_within_channel_dynamics,
    format_basic_statistics,
    format_data_quality,
    format_event_summary,
    format_forecast_residuals,
    format_predictions_to_string,
    format_within_channel_dynamics,
    get_last_timestamp,
    parse_time_series_string,
    parse_time_series_to_dataframe,
    predict_time_series_async,
)


SUPPORTED_PREDICTION_MODELS = {"patchtst", "itransformer", "arima", "chronos2"}
ROUTE_DECISION_KEEP_DEFAULT = "keep_default"
ROUTE_DECISION_OVERRIDE = "override"
SUPPORTED_ROUTE_DECISIONS = {
    ROUTE_DECISION_KEEP_DEFAULT,
    ROUTE_DECISION_OVERRIDE,
}
TURN3_TARGET_MODE_PAPER_STRICT = "paper_strict"
TURN3_TARGET_MODE_ENGINEERING_REFINE = "engineering_refine"
ROUTING_LABEL_SOURCE_HEURISTIC = "heuristic"
ROUTING_LABEL_SOURCE_REFERENCE_TEACHER = "reference_teacher"
ROUTING_LABEL_SOURCE_DEFAULT_OVERRIDE = "default_override"
SFT_STAGE_MODE_FULL = "full"
SFT_STAGE_MODE_ROUTING_ONLY = "routing_only"
SFT_STAGE_MODE_REFINEMENT_ONLY = "refinement_only"
SUPPORTED_TURN3_TARGET_MODES = {
    TURN3_TARGET_MODE_PAPER_STRICT,
    TURN3_TARGET_MODE_ENGINEERING_REFINE,
}
SUPPORTED_ROUTING_LABEL_SOURCES = {
    ROUTING_LABEL_SOURCE_HEURISTIC,
    ROUTING_LABEL_SOURCE_REFERENCE_TEACHER,
    ROUTING_LABEL_SOURCE_DEFAULT_OVERRIDE,
}
SUPPORTED_SFT_STAGE_MODES = {
    SFT_STAGE_MODE_FULL,
    SFT_STAGE_MODE_ROUTING_ONLY,
    SFT_STAGE_MODE_REFINEMENT_ONLY,
}
TRAIN_TURN3_REBALANCE_MODE_DOWNSAMPLE_KEEP = "downsample_keep"
TRAIN_TURN3_REBALANCE_MODE_OVERSAMPLE_LOCAL_REFINE = "oversample_local_refine"
SUPPORTED_TRAIN_TURN3_REBALANCE_MODES = {
    TRAIN_TURN3_REBALANCE_MODE_DOWNSAMPLE_KEEP,
    TRAIN_TURN3_REBALANCE_MODE_OVERSAMPLE_LOCAL_REFINE,
}
DEFAULT_TURN3_TARGET_MODE = TURN3_TARGET_MODE_ENGINEERING_REFINE
# Formal step-wise SFT should imitate the offline reference teacher route by
# default; the heuristic router remains available only for ablations/debugging.
DEFAULT_ROUTING_LABEL_SOURCE = ROUTING_LABEL_SOURCE_REFERENCE_TEACHER
DEFAULT_SFT_STAGE_MODE = SFT_STAGE_MODE_FULL
DEFAULT_TRAIN_TURN3_REBALANCE_MODE = TRAIN_TURN3_REBALANCE_MODE_DOWNSAMPLE_KEEP
DEFAULT_OUTPUT_DIR = Path("dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise")
DEFAULT_CURATED_INPUT_DIR = Path("dataset/ett_sft_etth1_runtime_teacher200_paper_same2")
DEFAULT_TRAIN_JSONL = DEFAULT_CURATED_INPUT_DIR / "train_curated.jsonl"
DEFAULT_VAL_JSONL = DEFAULT_CURATED_INPUT_DIR / "val_curated.jsonl"
DEFAULT_TEST_JSONL = DEFAULT_CURATED_INPUT_DIR / "test_curated.jsonl"
DEFAULT_TRAIN_STAGE_REPEAT_FACTORS = {"diagnostic": 1, "routing": 1, "refinement": 1}
DEFAULT_BALANCE_TRAIN_ROUTING_MODELS = True
DEFAULT_TRAIN_PRIORITY_VALIDATED_KEEP_REPEAT_FACTOR = 1
DEFAULT_TRAIN_LOCAL_REFINE_REFINEMENT_REPEAT_FACTOR = 1
DEFAULT_TRAIN_ROUTING_CONFIDENCE_MIN_TIER = "none"
SUPPORTED_TRAIN_ROUTING_CONFIDENCE_MIN_TIERS = {"none", "mid", "high"}
DEFAULT_TRAIN_HIGH_CONFIDENCE_ROUTING_REPEAT_FACTOR = 1
DEFAULT_ROUTING_MID_CONF_MIN_DIAGNOSTIC_PLAN_SCORE_GAP = 0.15
DEFAULT_ROUTING_HIGH_CONF_MIN_DIAGNOSTIC_PLAN_SCORE_GAP = 0.25
DEFAULT_ROUTING_MID_CONF_MIN_TEACHER_EVAL_SCORE_MARGIN = 0.01
DEFAULT_ROUTING_HIGH_CONF_MIN_TEACHER_EVAL_SCORE_MARGIN = 0.03


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    tool_output: str


FEATURE_TOOL_BUILDERS = (
    ("extract_basic_statistics", lambda values: format_basic_statistics(extract_basic_statistics(values))),
    (
        "extract_within_channel_dynamics",
        lambda values: format_within_channel_dynamics(extract_within_channel_dynamics(values)),
    ),
    ("extract_forecast_residuals", lambda values: format_forecast_residuals(extract_forecast_residuals(values))),
    ("extract_data_quality", lambda values: format_data_quality(extract_data_quality(values))),
    ("extract_event_summary", lambda values: format_event_summary(extract_event_summary(values))),
)
FEATURE_TOOL_SCHEMA_BY_NAME = {
    str(schema["function"]["name"]): schema for schema in FEATURE_TOOL_SCHEMAS
}
FEATURE_TOOL_LABELS = {
    "extract_basic_statistics": "basic statistics",
    "extract_within_channel_dynamics": "within-channel dynamics",
    "extract_forecast_residuals": "forecast residual patterns",
    "extract_data_quality": "data quality",
    "extract_event_summary": "event-level changes",
}
ROUTING_FEATURE_TOOLS = {
    "acf1": "extract_basic_statistics",
    "acf_seasonal": "extract_basic_statistics",
    "cusum_max": "extract_basic_statistics",
    "changepoint_count": "extract_within_channel_dynamics",
    "peak_count": "extract_within_channel_dynamics",
    "peak_spacing_cv": "extract_within_channel_dynamics",
    "monotone_duration": "extract_within_channel_dynamics",
    "residual_exceed_ratio": "extract_forecast_residuals",
    "quality_quantization_score": "extract_data_quality",
    "quality_saturation_ratio": "extract_data_quality",
    "dominant_pattern": "extract_event_summary",
}
EVENT_PATTERN_NAMES = ["rise", "fall", "flat", "oscillation"]
ROUTING_MODEL_RATIONALES = {
    "patchtst": "patch-level local patterns and seasonality",
    "itransformer": "global temporal dependencies and trend interactions",
    "arima": "stable linear autocorrelation structure",
    "chronos2": "irregular dynamics that benefit from a strong general forecasting prior",
}
ROUTING_FALLBACK_PREFERENCE = {
    "arima": 3,
    "patchtst": 2,
    "itransformer": 1,
    "chronos2": 0,
}
ROUTING_CONFIDENCE_TIER_RANK = {"low": 0, "mid": 1, "high": 2}


def _select_model_from_scores(scores: dict[str, float]) -> str:
    return max(
        scores.items(),
        key=lambda item: (
            float(item[1]),
            ROUTING_FALLBACK_PREFERENCE.get(str(item[0]), -1),
        ),
    )[0]


def _normalize_train_routing_confidence_min_tier(min_tier: str | None) -> str:
    tier = str(min_tier or DEFAULT_TRAIN_ROUTING_CONFIDENCE_MIN_TIER).strip().lower()
    if tier not in SUPPORTED_TRAIN_ROUTING_CONFIDENCE_MIN_TIERS:
        raise ValueError(
            f"Unsupported train_routing_confidence_min_tier={min_tier!r}. "
            f"Expected one of {sorted(SUPPORTED_TRAIN_ROUTING_CONFIDENCE_MIN_TIERS)}."
        )
    return tier


def _routing_confidence_tier(
    *,
    heuristic_selected_prediction_model: str,
    reference_teacher_model: str,
    diagnostic_plan_score_gap: Any,
    teacher_eval_score_margin: Any,
) -> str:
    agrees = str(heuristic_selected_prediction_model or "").strip().lower() == str(reference_teacher_model or "").strip().lower()
    if not agrees:
        return "low"
    try:
        score_gap = float(diagnostic_plan_score_gap or 0.0)
    except Exception:
        score_gap = 0.0
    try:
        teacher_margin = float(teacher_eval_score_margin or 0.0)
    except Exception:
        teacher_margin = 0.0
    if (
        score_gap >= DEFAULT_ROUTING_HIGH_CONF_MIN_DIAGNOSTIC_PLAN_SCORE_GAP
        and teacher_margin >= DEFAULT_ROUTING_HIGH_CONF_MIN_TEACHER_EVAL_SCORE_MARGIN
    ):
        return "high"
    if (
        score_gap >= DEFAULT_ROUTING_MID_CONF_MIN_DIAGNOSTIC_PLAN_SCORE_GAP
        and teacher_margin >= DEFAULT_ROUTING_MID_CONF_MIN_TEACHER_EVAL_SCORE_MARGIN
    ):
        return "mid"
    return "low"


def _last_assistant_content(messages: Any) -> str:
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    elif isinstance(messages, tuple):
        messages = list(messages)
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            content = msg["content"].strip()
            if content:
                return content
    return ""


def _coerce_bool_flag(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, float) and pd.isna(value):
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _row_requires_paper_turn3_protocol(row: Any) -> bool:
    explicit_flag = row.get("paper_turn3_required") if hasattr(row, "get") else None
    if explicit_flag is not None and not (isinstance(explicit_flag, float) and pd.isna(explicit_flag)):
        return _coerce_bool_flag(explicit_flag, default=False)

    turn_stage = str((row.get("turn_stage") if hasattr(row, "get") else "") or "").strip().lower()
    if turn_stage:
        return turn_stage == "refinement"
    return True


def _source_level_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe

    if "turn_stage" in dataframe.columns:
        refinement_df = dataframe.loc[dataframe["turn_stage"] == "refinement"].copy()
        if not refinement_df.empty:
            sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in refinement_df.columns]
            if sort_columns:
                refinement_df = refinement_df.sort_values(sort_columns, kind="stable")
            return refinement_df.reset_index(drop=True)

    if "source_sample_index" in dataframe.columns:
        sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in dataframe.columns]
        ordered = dataframe.sort_values(sort_columns or ["source_sample_index"], kind="stable")
        return ordered.drop_duplicates(subset=["source_sample_index"], keep="last").reset_index(drop=True)
    return dataframe.reset_index(drop=True)


def _source_sample_unique_count(dataframe: pd.DataFrame | None) -> int:
    if dataframe is None or dataframe.empty:
        return 0
    if "source_sample_index" in dataframe.columns:
        return int(dataframe["source_sample_index"].nunique())
    return int(len(_source_level_frame(dataframe)))


def _stage_mode_allowed_turn_stages(sft_stage_mode: str) -> set[str]:
    mode = _normalize_sft_stage_mode(sft_stage_mode)
    if mode == SFT_STAGE_MODE_ROUTING_ONLY:
        return {"routing"}
    if mode == SFT_STAGE_MODE_REFINEMENT_ONLY:
        return {"refinement"}
    return {"diagnostic", "routing", "refinement"}


def _filter_records_for_stage_mode(records: list[dict[str, Any]], *, sft_stage_mode: str) -> list[dict[str, Any]]:
    allowed_turn_stages = _stage_mode_allowed_turn_stages(sft_stage_mode)
    filtered: list[dict[str, Any]] = []
    for record in records:
        stage = str(record.get("turn_stage") or "").strip().lower()
        if stage in allowed_turn_stages:
            filtered.append(record)
    return filtered


def source_sample_coverage_by_stage(dataframe: pd.DataFrame | None) -> dict[str, int]:
    if dataframe is None or dataframe.empty or "turn_stage" not in dataframe.columns:
        return {}
    coverage: dict[str, int] = {}
    for stage_name, stage_df in dataframe.groupby(dataframe["turn_stage"].astype(str), sort=True):
        if "source_sample_index" in stage_df.columns:
            coverage[str(stage_name)] = int(stage_df["source_sample_index"].nunique())
        else:
            coverage[str(stage_name)] = int(len(stage_df))
    return coverage


def _paper_turn3_protocol_reason(content: str, expected_len: int) -> str:
    if not content:
        return "empty_last_assistant"
    match = re.fullmatch(r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*", content, re.DOTALL)
    if match is None:
        if "<think>" not in content and "</think>" not in content:
            return "missing_think_block"
        if "<think>" in content and "</think>" not in content:
            return "missing_think_close_tag"
        if "</think>" in content and "<think>" not in content:
            return "missing_think_open_tag"
        if "<answer>" not in content and "</answer>" not in content:
            return "missing_answer_block"
        if "<answer>" in content and "</answer>" not in content:
            return "missing_answer_close_tag"
        if "</answer>" in content and "<answer>" not in content:
            return "missing_answer_open_tag"
        return "extra_text_outside_tags"

    answer_text = match.group(2).strip()
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    if len(lines) == 1:
        single_line = lines[0]
        if single_line.lower().startswith("decision=") and single_line.split("=", 1)[1].strip():
            return "ok"
        if single_line.lower().startswith("candidate_id=") and single_line.split("=", 1)[1].strip():
            return "ok"
    values = _extract_prediction_values(answer_text)
    if len(lines) != expected_len:
        return f"invalid_answer_shape:lines={len(lines)},expected={expected_len}"
    if len(values) != expected_len:
        return f"invalid_answer_shape:values={len(values)},expected={expected_len}"
    return "ok"


def _summarize_paper_turn3_protocol(dataframe: pd.DataFrame) -> dict[str, Any]:
    total = int(len(dataframe))
    reason_counter: Counter[str] = Counter()
    invalid_examples: list[dict[str, Any]] = []
    checked_count = 0

    for row_idx, row in dataframe.iterrows():
        if not _row_requires_paper_turn3_protocol(row):
            continue
        checked_count += 1
        expected_len = int(row.get("forecast_horizon", 96) or 96)
        content = _last_assistant_content(row.get("messages"))
        reason = _paper_turn3_protocol_reason(content, expected_len)
        reason_counter[reason] += 1
        if reason != "ok" and len(invalid_examples) < 5:
            invalid_examples.append(
                {
                    "row_index": int(row_idx),
                    "sample_index": int(row.get("sample_index", -1) or -1),
                    "reason": reason,
                    "assistant_head": content[:200],
                }
            )

    valid_count = int(reason_counter.get("ok", 0))
    invalid_count = checked_count - valid_count
    return {
        "turn3_protocol": "paper_think_answer_xml",
        "turn3_protocol_checked_count": int(checked_count),
        "turn3_protocol_skipped_count": int(max(total - checked_count, 0)),
        "turn3_protocol_valid_count": valid_count,
        "turn3_protocol_invalid_count": int(invalid_count),
        "turn3_protocol_valid_ratio": float(valid_count / checked_count) if checked_count > 0 else 0.0,
        "turn3_protocol_reason_distribution": {str(k): int(v) for k, v in sorted(reason_counter.items())},
        "turn3_protocol_invalid_examples": invalid_examples,
    }


def _validate_paper_turn3_protocol(
    dataframe: pd.DataFrame,
    *,
    split_name: str,
    output_path: Path,
    allow_no_refinement: bool = False,
) -> dict[str, Any]:
    summary = _summarize_paper_turn3_protocol(dataframe)
    if int(summary["turn3_protocol_checked_count"]) <= 0:
        if allow_no_refinement:
            return summary
        raise ValueError(
            f"{split_name} SFT parquet at {output_path} does not contain any refinement records to validate."
        )
    if int(summary["turn3_protocol_invalid_count"]) > 0:
        raise ValueError(
            f"{split_name} SFT parquet is not paper-aligned at {output_path}. "
            f"reason_distribution={summary['turn3_protocol_reason_distribution']} "
            f"examples={summary['turn3_protocol_invalid_examples']}"
        )
    return summary


def _normalize_teacher_model(model_name: Any) -> str:
    model = str(model_name or "patchtst").strip().lower()
    return model if model in SUPPORTED_PREDICTION_MODELS else "patchtst"


def _normalize_supported_model_or_empty(model_name: Any) -> str:
    model = str(model_name or "").strip().lower()
    return model if model in SUPPORTED_PREDICTION_MODELS else ""


def _normalize_route_decision(decision: Any) -> str:
    normalized = str(decision or "").strip().lower()
    return normalized if normalized in SUPPORTED_ROUTE_DECISIONS else ""


def _normalize_route_label(route_label: Any) -> str:
    normalized = str(route_label or "").strip().lower()
    if normalized == ROUTE_DECISION_KEEP_DEFAULT:
        return ROUTE_DECISION_KEEP_DEFAULT
    if normalized.startswith("override_to_"):
        override_model = _normalize_supported_model_or_empty(normalized.split("override_to_", 1)[1])
        if not override_model:
            return ""
        return f"override_to_{override_model}"
    return ""


def _resolve_route_override_target(sample: dict[str, Any]) -> dict[str, Any] | None:
    default_expert = _normalize_supported_model_or_empty(sample.get("default_expert") or sample.get("route_default_expert"))
    route_label = _normalize_route_label(sample.get("route_label"))
    route_decision = _normalize_route_decision(sample.get("route_decision"))
    override_model = _normalize_supported_model_or_empty(sample.get("route_override_model"))

    if not default_expert or not route_label:
        return None

    if route_label == ROUTE_DECISION_KEEP_DEFAULT:
        return {
            "default_expert": default_expert,
            "route_label": ROUTE_DECISION_KEEP_DEFAULT,
            "route_decision": ROUTE_DECISION_KEEP_DEFAULT,
            "override_model": "",
            "resolved_model": default_expert,
            "tool_arguments": {"decision": ROUTE_DECISION_KEEP_DEFAULT},
        }

    if not route_decision:
        route_decision = ROUTE_DECISION_OVERRIDE
    if route_decision != ROUTE_DECISION_OVERRIDE:
        return None

    if not override_model:
        override_model = _normalize_supported_model_or_empty(route_label.split("override_to_", 1)[1])
    if not override_model or override_model == default_expert:
        return None

    return {
        "default_expert": default_expert,
        "route_label": route_label,
        "route_decision": ROUTE_DECISION_OVERRIDE,
        "override_model": override_model,
        "resolved_model": override_model,
        "tool_arguments": {
            "decision": ROUTE_DECISION_OVERRIDE,
            "model_name": override_model,
        },
    }


def _normalize_routing_label_source(routing_label_source: str | None) -> str:
    source = str(routing_label_source or DEFAULT_ROUTING_LABEL_SOURCE).strip().lower()
    if source not in SUPPORTED_ROUTING_LABEL_SOURCES:
        raise ValueError(
            f"Unsupported routing_label_source={routing_label_source!r}. "
            f"Expected one of {sorted(SUPPORTED_ROUTING_LABEL_SOURCES)}."
        )
    return source


def _normalize_sft_stage_mode(sft_stage_mode: str | None) -> str:
    mode = str(sft_stage_mode or DEFAULT_SFT_STAGE_MODE).strip().lower()
    if mode not in SUPPORTED_SFT_STAGE_MODES:
        raise ValueError(
            f"Unsupported sft_stage_mode={sft_stage_mode!r}. "
            f"Expected one of {sorted(SUPPORTED_SFT_STAGE_MODES)}."
        )
    return mode


def _normalize_train_turn3_rebalance_mode(train_turn3_rebalance_mode: str | None) -> str:
    mode = str(train_turn3_rebalance_mode or DEFAULT_TRAIN_TURN3_REBALANCE_MODE).strip().lower()
    if mode not in SUPPORTED_TRAIN_TURN3_REBALANCE_MODES:
        raise ValueError(
            f"Unsupported train_turn3_rebalance_mode={train_turn3_rebalance_mode!r}. "
            f"Expected one of {sorted(SUPPORTED_TRAIN_TURN3_REBALANCE_MODES)}."
        )
    return mode


def _sample_has_reference_teacher_model(sample: dict[str, Any]) -> bool:
    for key in ("reference_teacher_model", "offline_best_model"):
        if str(sample.get(key) or "").strip():
            return True
    return False


def _resolve_reference_teacher_model(sample: dict[str, Any]) -> str:
    for key in ("reference_teacher_model", "offline_best_model"):
        value = sample.get(key)
        if str(value or "").strip():
            return _normalize_teacher_model(value)
    return _normalize_teacher_model(None)


def _compute_routing_feature_snapshot(history_values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in history_values]
    basic = extract_basic_statistics(values)
    dynamics = extract_within_channel_dynamics(values)
    residuals = extract_forecast_residuals(values)
    quality = extract_data_quality(values)
    events = extract_event_summary(values)
    dominant_pattern_idx = int(float(events.get("event_dominant_pattern", 0.0) or 0.0))
    dominant_pattern_idx = min(max(dominant_pattern_idx, 0), len(EVENT_PATTERN_NAMES) - 1)
    return {
        "acf1": float(basic.get("acf1", 0.0)),
        "acf_seasonal": float(basic.get("acf_seasonal", 0.0)),
        "cusum_max": float(basic.get("cusum_max", 0.0)),
        "changepoint_count": float(dynamics.get("changepoint_count", 0.0)),
        "peak_count": float(dynamics.get("peak_count", 0.0)),
        "peak_spacing_cv": float(dynamics.get("peak_spacing_cv", 0.0)),
        "monotone_duration": float(dynamics.get("monotone_duration", 0.0)),
        "residual_exceed_ratio": float(residuals.get("residual_exceed_ratio", 0.0)),
        "quality_quantization_score": float(quality.get("quality_quantization_score", 0.0)),
        "quality_saturation_ratio": float(quality.get("quality_saturation_ratio", 0.0)),
        "dominant_pattern": EVENT_PATTERN_NAMES[dominant_pattern_idx],
    }


def _visible_feature_tools(selected_feature_tools: Sequence[str] | None) -> set[str] | None:
    if selected_feature_tools is None:
        return None
    visible = {str(tool_name).strip() for tool_name in selected_feature_tools if str(tool_name).strip()}
    return visible or set()


def _feature_is_visible(visible_tools: set[str] | None, feature_name: str) -> bool:
    if visible_tools is None:
        return True
    required_tool = ROUTING_FEATURE_TOOLS.get(feature_name)
    return required_tool in visible_tools if required_tool else False


def _heuristic_routing_scores(
    feature_snapshot: dict[str, Any],
    *,
    selected_feature_tools: Sequence[str] | None = None,
) -> dict[str, float]:
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
    visible_tools = _visible_feature_tools(selected_feature_tools)

    scores = {model_name: 0.0 for model_name in sorted(SUPPORTED_PREDICTION_MODELS)}
    scores["arima"] += 0.10
    scores["patchtst"] += 0.15
    scores["chronos2"] -= 0.50

    if _feature_is_visible(visible_tools, "acf1"):
        scores["arima"] += 1.50 if acf1 >= 0.92 else (0.75 if acf1 >= 0.88 else 0.0)
    if _feature_is_visible(visible_tools, "changepoint_count"):
        scores["arima"] += 0.75 if changepoint_count <= 1.0 else 0.0
    if _feature_is_visible(visible_tools, "residual_exceed_ratio"):
        scores["arima"] += 0.50 if residual_exceed_ratio <= 0.05 else 0.0

    if _feature_is_visible(visible_tools, "peak_count"):
        scores["patchtst"] += 1.00 if 2.0 <= peak_count <= 5.0 else 0.0
    if _feature_is_visible(visible_tools, "peak_spacing_cv"):
        scores["patchtst"] += 0.75 if peak_spacing_cv <= 0.30 else 0.0
    if _feature_is_visible(visible_tools, "acf_seasonal"):
        scores["patchtst"] += 0.50 if acf_seasonal >= 0.05 else 0.0

    if _feature_is_visible(visible_tools, "changepoint_count"):
        scores["itransformer"] += 1.25 if changepoint_count >= 2.0 else 0.0
    if _feature_is_visible(visible_tools, "cusum_max"):
        scores["itransformer"] += 0.75 if cusum_max >= 70.0 else 0.0
    if _feature_is_visible(visible_tools, "monotone_duration"):
        scores["itransformer"] += 0.50 if monotone_duration >= 0.10 else 0.0

    if _feature_is_visible(visible_tools, "residual_exceed_ratio"):
        scores["chronos2"] += 0.75 if residual_exceed_ratio >= 0.08 else 0.0
    if _feature_is_visible(visible_tools, "quality_saturation_ratio") or _feature_is_visible(visible_tools, "quality_quantization_score"):
        scores["chronos2"] += 0.50 if quality_saturation_ratio >= 0.08 or quality_quantization_score >= 0.24 else 0.0
    if _feature_is_visible(visible_tools, "peak_count") and _feature_is_visible(visible_tools, "peak_spacing_cv"):
        scores["chronos2"] += 0.50 if peak_count >= 6.0 and peak_spacing_cv >= 0.35 else 0.0
    return scores


def _select_prediction_model_by_heuristic(
    history_values: Sequence[float],
    *,
    selected_feature_tools: Sequence[str] | None = None,
) -> tuple[str, dict[str, Any], str]:
    feature_snapshot = _compute_routing_feature_snapshot(history_values)
    visible_tools = _visible_feature_tools(selected_feature_tools)
    acf1 = float(feature_snapshot["acf1"])
    changepoint_count = float(feature_snapshot["changepoint_count"])
    residual_exceed_ratio = float(feature_snapshot["residual_exceed_ratio"])
    peak_count = float(feature_snapshot["peak_count"])
    peak_spacing_cv = float(feature_snapshot["peak_spacing_cv"])
    monotone_duration = float(feature_snapshot["monotone_duration"])
    cusum_max = float(feature_snapshot["cusum_max"])
    quality_issue = (
        _feature_is_visible(visible_tools, "quality_saturation_ratio")
        or _feature_is_visible(visible_tools, "quality_quantization_score")
    ) and (
        float(feature_snapshot["quality_saturation_ratio"]) >= 0.08
        or float(feature_snapshot["quality_quantization_score"]) >= 0.24
    )

    if quality_issue or (
        _feature_is_visible(visible_tools, "residual_exceed_ratio")
        and _feature_is_visible(visible_tools, "peak_count")
        and _feature_is_visible(visible_tools, "peak_spacing_cv")
        and residual_exceed_ratio >= 0.085
        and peak_count >= 6.0
        and peak_spacing_cv >= 0.40
    ):
        return (
            "chronos2",
            feature_snapshot,
            "The diagnostics show strong irregularity, residual stress, or data-quality risk in the current window.",
        )
    if (
        _feature_is_visible(visible_tools, "acf1")
        and _feature_is_visible(visible_tools, "changepoint_count")
        and _feature_is_visible(visible_tools, "residual_exceed_ratio")
        and _feature_is_visible(visible_tools, "peak_count")
        and acf1 >= 0.93
        and changepoint_count <= 1.0
        and residual_exceed_ratio <= 0.05
        and peak_count <= 4.0
    ):
        return (
            "arima",
            feature_snapshot,
            "The diagnostics remain stable, with high short-lag autocorrelation, few structural breaks, and low residual stress.",
        )
    if (
        _feature_is_visible(visible_tools, "changepoint_count")
        and changepoint_count >= 3.0
        and (
            (_feature_is_visible(visible_tools, "cusum_max") and cusum_max >= 70.0)
            or (_feature_is_visible(visible_tools, "monotone_duration") and monotone_duration >= 0.10)
        )
    ):
        return (
            "itransformer",
            feature_snapshot,
            "The diagnostics indicate broader structural change, with multiple changepoints or sustained drift across the window.",
        )
    if (
        _feature_is_visible(visible_tools, "peak_count")
        and _feature_is_visible(visible_tools, "peak_spacing_cv")
        and 2.0 <= peak_count <= 5.0
        and peak_spacing_cv <= 0.30
    ):
        return (
            "patchtst",
            feature_snapshot,
            "The diagnostics indicate repeatable local peaks, stable spacing, and usable seasonal structure in the window.",
        )

    scores = _heuristic_routing_scores(feature_snapshot, selected_feature_tools=selected_feature_tools)
    selected_model = _select_model_from_scores(scores)
    fallback_reason_by_model = {
        "arima": "The diagnostics remain comparatively stable, with structured short-lag behavior and limited break activity.",
        "patchtst": "The diagnostics suggest repeating local structure with relatively stable spacing and seasonal evidence.",
        "itransformer": "The diagnostics indicate broader structural drift and longer-range dependency changes across the window.",
        "chronos2": "The diagnostics remain highly irregular or quality-stressed, with evidence that the window is not cleanly structured.",
    }
    return selected_model, feature_snapshot, fallback_reason_by_model[selected_model]


def _make_tool_call(tool_name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


async def _predict_with_runtime_tools(
    *,
    historical_data: str,
    data_source: str,
    target_column: str,
    forecast_horizon: int,
    model_name: str,
) -> str:
    context_df = parse_time_series_to_dataframe(
        historical_data,
        series_id=data_source or "ETTh1",
        target_column=target_column,
        include_covariates=True,
    )
    pred_df = await predict_time_series_async(
        context_df,
        prediction_length=forecast_horizon,
        model_name=model_name,
    )
    last_timestamp = get_last_timestamp(historical_data)
    return format_predictions_to_string(pred_df, last_timestamp)


def build_feature_tool_results(
    values: list[float],
    *,
    tool_names: Sequence[str] | None = None,
) -> list[ToolResult]:
    allowed_names = {str(name) for name in (tool_names or []) if str(name).strip()}
    return [
        ToolResult(tool_name=name, tool_output=builder(values))
        for name, builder in FEATURE_TOOL_BUILDERS
        if not allowed_names or name in allowed_names
    ]


def _extract_prediction_values(prediction_text: str) -> list[float]:
    values: list[float] = []
    for line in str(prediction_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        parsed = None
        for token in reversed(parts):
            try:
                parsed = float(token)
                break
            except Exception:
                continue
        if parsed is None:
            matches = re.findall(r"-?\d+(?:\.\d+)?", line)
            if matches:
                try:
                    parsed = float(matches[-1])
                except Exception:
                    parsed = None
        if parsed is not None and np.isfinite(parsed):
            values.append(float(parsed))
    return values


def _require_prediction_values(prediction_text: str, forecast_horizon: int, *, source_name: str) -> list[float]:
    values = _extract_prediction_values(prediction_text)
    if len(values) != forecast_horizon:
        raise ValueError(
            f"{source_name} prediction length must equal forecast_horizon={forecast_horizon}, got {len(values)}"
        )
    return values


def _canonical_prediction_text(
    prediction_text: str,
    forecast_horizon: int,
    *,
    history_text: str,
) -> str:
    timestamps, values = parse_time_series_string(prediction_text)
    if len(values) < forecast_horizon:
        raise ValueError(
            f"prediction_text length must equal forecast_horizon={forecast_horizon}, got {len(values)}"
        )

    normalized_values = [float(value) for value in values[:forecast_horizon]]
    normalized_timestamps = [str(ts).strip() for ts in timestamps[:forecast_horizon]]
    if len(normalized_timestamps) == forecast_horizon and all(normalized_timestamps):
        return "\n".join(
            f"{normalized_timestamps[idx]} {normalized_values[idx]:.4f}"
            for idx in range(forecast_horizon)
        )

    pred_df = pd.DataFrame({"target_0.5": normalized_values})
    return format_predictions_to_string(
        pred_df,
        last_timestamp=get_last_timestamp(history_text),
    )


def _prediction_text_from_values(
    values: list[float],
    *,
    reference_prediction_text: str,
    history_text: str,
) -> str:
    forecast_horizon = len(values)
    reference_timestamps, _reference_values = parse_time_series_string(reference_prediction_text)
    normalized_timestamps = [str(ts).strip() for ts in reference_timestamps[:forecast_horizon]]
    if len(normalized_timestamps) == forecast_horizon and all(normalized_timestamps):
        return "\n".join(
            f"{normalized_timestamps[idx]} {float(values[idx]):.4f}"
            for idx in range(forecast_horizon)
        )

    pred_df = pd.DataFrame({"target_0.5": [float(value) for value in values]})
    return format_predictions_to_string(
        pred_df,
        last_timestamp=get_last_timestamp(history_text),
    )


def _feature_tool_signature(selected_feature_tools: list[str]) -> str:
    return "->".join(selected_feature_tools) if selected_feature_tools else "none"


def _normalize_turn3_target_mode(turn3_target_mode: str | None) -> str:
    mode = str(turn3_target_mode or DEFAULT_TURN3_TARGET_MODE).strip().lower()
    if mode not in SUPPORTED_TURN3_TARGET_MODES:
        raise ValueError(
            f"Unsupported turn3_target_mode={turn3_target_mode!r}. "
            f"Expected one of {sorted(SUPPORTED_TURN3_TARGET_MODES)}."
        )
    return mode


def _compute_error_metrics(candidate_values: list[float], ground_truth_values: list[float]) -> tuple[float, float]:
    min_len = min(len(candidate_values), len(ground_truth_values))
    if min_len <= 0:
        return float("inf"), float("inf")
    candidate = np.asarray(candidate_values[:min_len], dtype=float)
    reference = np.asarray(ground_truth_values[:min_len], dtype=float)
    mse = float(np.mean((candidate - reference) ** 2))
    mae = float(np.mean(np.abs(candidate - reference)))
    return mse, mae
def _should_attempt_refinement(
    refinement_support_payload: dict[str, Any],
) -> tuple[bool, list[str]]:
    candidate_adjustments = [
        str(item)
        for item in refinement_support_payload.get("candidate_adjustments", [])
        if str(item).strip() and str(item).strip().lower() != "none"
    ]
    reasons = [
        str(item)
        for item in refinement_support_payload.get("support_signals", [])
        if str(item).strip() and str(item).strip().lower() not in {"evidence_consistent", "none"}
    ]
    return bool(candidate_adjustments and reasons), reasons or ["evidence_consistent"]


def _summarize_refine_delta(base_values: list[float], refined_values: list[float]) -> dict[str, float]:
    if len(base_values) != len(refined_values):
        raise ValueError("base_values and refined_values must have the same length")
    deltas = np.abs(np.asarray(refined_values, dtype=float) - np.asarray(base_values, dtype=float))
    changed_indices = np.where(deltas > 1e-6)[0]
    if changed_indices.size == 0:
        return {
            "refine_changed_value_count": 0,
            "refine_first_changed_index": -1,
            "refine_last_changed_index": -1,
            "refine_changed_span": 0,
            "refine_mean_abs_delta": 0.0,
            "refine_max_abs_delta": 0.0,
        }
    first_idx = int(changed_indices[0])
    last_idx = int(changed_indices[-1])
    return {
        "refine_changed_value_count": int(changed_indices.size),
        "refine_first_changed_index": first_idx,
        "refine_last_changed_index": last_idx,
        "refine_changed_span": int(last_idx - first_idx + 1),
        "refine_mean_abs_delta": float(np.mean(deltas[changed_indices])),
        "refine_max_abs_delta": float(np.max(deltas[changed_indices])),
    }


def _is_meaningful_local_refine(
    *,
    base_values: list[float],
    refined_values: list[float],
    history_values: list[float],
    base_mse: float,
    base_mae: float,
    refined_mse: float,
    refined_mae: float,
) -> tuple[bool, dict[str, float]]:
    delta_summary = _summarize_refine_delta(base_values, refined_values)
    changed_count = int(delta_summary["refine_changed_value_count"])
    if changed_count <= 0:
        return False, delta_summary

    max_changed = min(24, max(4, len(base_values) // 4))
    max_span = min(32, max(8, len(base_values) // 3))
    local_enough = (
        changed_count <= max_changed
        and int(delta_summary["refine_changed_span"]) <= max_span
    )

    mse_gain = float(base_mse - refined_mse)
    mae_gain = float(base_mae - refined_mae)
    meaningful_gain = (
        mse_gain > max(1e-6, 0.005 * max(base_mse, 1e-6))
        or mae_gain > max(1e-6, 0.005 * max(base_mae, 1e-6))
    )
    return bool(local_enough and meaningful_gain), delta_summary


def _build_turn3_target(
    *,
    sample: dict[str, Any],
    historical_data: str,
    history_values: list[float],
    base_prediction_text: str,
    forecast_horizon: int,
    model_name: str,
    selected_feature_tools: list[str],
    turn3_target_mode: str = DEFAULT_TURN3_TARGET_MODE,
) -> dict[str, Any]:
    turn3_target_mode = _normalize_turn3_target_mode(turn3_target_mode)
    base_values = _require_prediction_values(base_prediction_text, forecast_horizon, source_name=f"{model_name}_base")
    reward_model = sample.get("reward_model", {}) if isinstance(sample.get("reward_model"), dict) else {}
    ground_truth_values = _extract_prediction_values(str(reward_model.get("ground_truth", "") or ""))
    if len(ground_truth_values) < forecast_horizon:
        raise ValueError(
            f"ground_truth length must be at least forecast_horizon={forecast_horizon}, got {len(ground_truth_values)}"
        )
    ground_truth_values = ground_truth_values[:forecast_horizon]
    best_delta_summary = _summarize_refine_delta(base_values, base_values)

    candidate_refinements = filter_refinement_candidates_for_model(
        generate_local_refinement_candidates(base_values, history_values),
        prediction_model_used=model_name,
    )
    refinement_support_payload = build_refinement_support_payload(
        base_values=base_values,
        history_values=history_values,
        selected_feature_tools=selected_feature_tools,
        candidate_refinements=candidate_refinements,
        prediction_model_used=model_name,
    )
    attempt_refine, trigger_reasons = _should_attempt_refinement(
        refinement_support_payload,
    )

    base_mse, base_mae = _compute_error_metrics(base_values, ground_truth_values)
    best_target_type = "validated_keep"
    best_values = list(base_values)
    best_ops: list[str] = []
    best_mse = base_mse
    best_mae = base_mae
    best_trigger_reasons = ["evidence_consistent"]

    if attempt_refine:
        candidate_trigger_reasons = trigger_reasons
        for refined_values, refine_ops in candidate_refinements:
            refined_mse, refined_mae = _compute_error_metrics(refined_values, ground_truth_values)
            is_meaningful_refine, delta_summary = _is_meaningful_local_refine(
                base_values=base_values,
                refined_values=refined_values,
                history_values=history_values,
                base_mse=base_mse,
                base_mae=base_mae,
                refined_mse=refined_mse,
                refined_mae=refined_mae,
            )
            if not is_meaningful_refine or not np.isfinite(refined_mse) or refined_mse >= best_mse - 1e-8:
                continue
            best_target_type = "local_refine"
            best_values = refined_values
            best_ops = refine_ops
            best_mse = refined_mse
            best_mae = refined_mae
            best_delta_summary = delta_summary
            best_trigger_reasons = candidate_trigger_reasons
    trigger_reasons = best_trigger_reasons

    refine_gain_mse = float(base_mse - best_mse)
    refine_gain_mae = float(base_mae - best_mae)
    refine_ops_signature = "none" if not best_ops else "->".join(best_ops)
    trigger_reason = "none" if not trigger_reasons else "->".join(dict.fromkeys(trigger_reasons))
    base_teacher_prediction_text = _prediction_text_from_values(
        base_values,
        reference_prediction_text=base_prediction_text,
        history_text=historical_data,
    )
    refined_prediction_text = _prediction_text_from_values(
        best_values,
        reference_prediction_text=base_prediction_text,
        history_text=historical_data,
    )
    candidate_prediction_text_map = build_refinement_candidate_prediction_text_map(
        base_prediction_text=base_teacher_prediction_text,
        candidate_refinements=candidate_refinements,
        prediction_model_used=model_name,
    )
    decision_name = refinement_decision_name(best_ops)

    return {
        "turn3_target_type": best_target_type,
        "refine_ops": list(dict.fromkeys(best_ops)),
        "refine_ops_signature": refine_ops_signature,
        "refinement_decision_name": decision_name,
        "refinement_support_signals": list(refinement_support_payload["support_signals"]),
        "refinement_support_signal_signature": "->".join(refinement_support_payload["support_signals"]),
        "refinement_keep_support_signals": list(refinement_support_payload["keep_support_signals"]),
        "refinement_keep_support_signal_signature": "->".join(refinement_support_payload["keep_support_signals"]),
        "refinement_edit_support_signals": json.dumps(
            refinement_support_payload["edit_support_signals"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "refinement_candidate_adjustments": list(refinement_support_payload["candidate_adjustments"]),
        "refinement_candidate_adjustment_signature": "->".join(refinement_support_payload["candidate_adjustments"]),
        "refinement_candidate_prediction_text_map": json.dumps(candidate_prediction_text_map, ensure_ascii=False),
        "refine_gain_mse": refine_gain_mse,
        "refine_gain_mae": refine_gain_mae,
        "turn3_trigger_reason": trigger_reason,
        "base_teacher_prediction_text": base_teacher_prediction_text,
        "refined_prediction_text": refined_prediction_text,
        **best_delta_summary,
    }


def _format_refine_ops_for_reflection(refine_ops: list[str]) -> str:
    if not refine_ops:
        return "keep the selected-model forecast unchanged"

    name_map = {
        "isolated_spike_smoothing": "smooth an isolated spike",
        "local_level_adjust": "adjust a local level shift",
        "local_slope_adjust": "adjust a local slope change",
    }
    phrases = [name_map.get(op, op.replace("_", " ")) for op in refine_ops]
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _build_diagnostic_reflection(
    *,
    plan_reason: str,
    batch_tool_names: Sequence[str],
    completed_feature_tools: Sequence[str],
) -> str:
    tool_text = ", ".join(str(name) for name in batch_tool_names if str(name).strip()) or "extract_basic_statistics"
    completed_text = ", ".join(str(name) for name in completed_feature_tools if str(name).strip())
    if completed_text:
        return (
            f"{str(plan_reason or '').strip()} I already completed {completed_text}, so I now call {tool_text} "
            "to finish the planned diagnostic evidence before routing."
        ).strip()
    return f"{str(plan_reason or '').strip()} I start by calling {tool_text}.".strip()


def _format_phrase_list(phrases: Sequence[str]) -> str:
    unique_phrases = [phrase for phrase in dict.fromkeys(phrases) if phrase]
    if not unique_phrases:
        return ""
    if len(unique_phrases) == 1:
        return unique_phrases[0]
    if len(unique_phrases) == 2:
        return f"{unique_phrases[0]} and {unique_phrases[1]}"
    return ", ".join(unique_phrases[:-1]) + f", and {unique_phrases[-1]}"


def _build_routing_feature_payload(
    *,
    history_values: Sequence[float],
    selected_feature_tools: Sequence[str],
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    selected = {str(name).strip() for name in selected_feature_tools if str(name).strip()}
    values = [float(value) for value in history_values]

    if "extract_basic_statistics" in selected:
        basic = extract_basic_statistics(values)
        payload["extract_basic_statistics"] = {
            "acf1": float(basic.get("acf1", 0.0)),
            "acf_seasonal": float(basic.get("acf_seasonal", 0.0)),
            "cusum_max": float(basic.get("cusum_max", 0.0)),
        }

    if "extract_within_channel_dynamics" in selected:
        dynamics = extract_within_channel_dynamics(values)
        payload["extract_within_channel_dynamics"] = {
            "changepoint_count": float(dynamics.get("changepoint_count", 0.0)),
            "peak_count": float(dynamics.get("peak_count", 0.0)),
            "peak_spacing_cv": float(dynamics.get("peak_spacing_cv", 0.0)),
            "monotone_duration": float(dynamics.get("monotone_duration", 0.0)),
        }

    if "extract_forecast_residuals" in selected:
        residuals = extract_forecast_residuals(values)
        payload["extract_forecast_residuals"] = {
            "residual_exceed_ratio": float(residuals.get("residual_exceed_ratio", 0.0)),
        }

    if "extract_data_quality" in selected:
        quality = extract_data_quality(values)
        payload["extract_data_quality"] = {
            "quality_quantization_score": float(quality.get("quality_quantization_score", 0.0)),
            "quality_saturation_ratio": float(quality.get("quality_saturation_ratio", 0.0)),
        }

    if "extract_event_summary" in selected:
        events = extract_event_summary(values)
        dominant_pattern_idx = int(float(events.get("event_dominant_pattern", 0.0) or 0.0))
        dominant_pattern_idx = min(max(dominant_pattern_idx, 0), len(EVENT_PATTERN_NAMES) - 1)
        payload["extract_event_summary"] = {
            "dominant_pattern": EVENT_PATTERN_NAMES[dominant_pattern_idx],
        }
    return payload


def build_routing_reflection(
    *,
    model_name: str,
    history_values: Sequence[float],
    selected_feature_tools: Sequence[str],
    decision_reason: str = "",
    include_heuristic_comparison: bool = True,
    route_default_expert: str = "",
    route_decision: str = "",
) -> str:
    del model_name
    del history_values
    del selected_feature_tools
    del decision_reason
    del include_heuristic_comparison
    if str(route_default_expert or "").strip() and str(route_decision or "").strip():
        return "Use the diagnostic evidence to decide whether the default forecaster should be kept or overridden."
    return "Use the diagnostic evidence to choose one forecasting model."


def build_final_answer(
    decision_name: str,
    prediction_text: str,
    turn3_target_mode: str,
    turn3_target_type: str,
    model_name: str,
    refine_ops: list[str],
) -> str:
    if turn3_target_type == "local_refine":
        reflection = (
            f"I compare the selected {model_name} forecast against the diagnostics and make a small local adjustment "
            f"around {_format_refine_ops_for_reflection(refine_ops)} while preserving the overall trajectory."
        )
    else:
        reflection = (
            f"I compare the selected {model_name} forecast against the diagnostics and keep it unchanged because it "
            "already matches the evidence."
        )

    normalized_mode = _normalize_turn3_target_mode(turn3_target_mode)
    if normalized_mode == TURN3_TARGET_MODE_ENGINEERING_REFINE:
        normalized_decision = (str(decision_name or "").strip() or REFINEMENT_KEEP_DECISION)
        return f"<think>\n{reflection}\n</think>\n<answer>\ndecision={normalized_decision}\n</answer>"
    return f"<think>\n{reflection}\n</think>\n<answer>\n{prediction_text}\n</answer>"


def _make_stage_record(
    *,
    shared_fields: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    turn_stage: str,
    turn_stage_order: int,
    trajectory_turn_count: int,
    current_required_feature_tools: list[str],
    completed_feature_tools_before_turn: list[str],
    history_analysis_count_before_turn: int,
    paper_turn3_required: bool,
    diagnostic_batch_index: int = -1,
) -> dict[str, Any]:
    return {
        **shared_fields,
        "messages": messages,
        "tools": copy.deepcopy(tools) if tools is not None else None,
        "turn_stage": turn_stage,
        "turn_stage_order": int(turn_stage_order),
        "trajectory_turn_count": int(trajectory_turn_count),
        "current_required_feature_tools": list(current_required_feature_tools),
        "completed_feature_tools_before_turn": list(completed_feature_tools_before_turn),
        "history_analysis_count_before_turn": int(history_analysis_count_before_turn),
        "paper_turn3_required": bool(paper_turn3_required),
        "diagnostic_batch_index": int(diagnostic_batch_index),
    }


def _resolve_prediction_text(
    *,
    sample: dict[str, Any],
    historical_data: str,
    data_source: str,
    target_column: str,
    forecast_horizon: int,
    model_name: str,
    allow_cached_reference: bool,
) -> tuple[str, str]:
    if allow_cached_reference:
        cached_teacher_prediction = str(sample.get("teacher_prediction_text", "") or "").strip()
        if cached_teacher_prediction:
            try:
                _require_prediction_values(
                    cached_teacher_prediction,
                    forecast_horizon,
                    source_name=f"{model_name}_cached_teacher",
                )
                return cached_teacher_prediction, "reference_teacher_cached"
            except Exception:
                pass
    prediction_text = asyncio.run(
        _predict_with_runtime_tools(
            historical_data=historical_data,
            data_source=data_source,
            target_column=target_column,
            forecast_horizon=forecast_horizon,
            model_name=model_name,
        )
    )
    return prediction_text, "reference_teacher_runtime"


def build_sft_records(
    sample: dict[str, Any],
    *,
    turn3_target_mode: str = DEFAULT_TURN3_TARGET_MODE,
    routing_label_source: str = DEFAULT_ROUTING_LABEL_SOURCE,
    sft_stage_mode: str = DEFAULT_SFT_STAGE_MODE,
) -> list[dict[str, Any]]:
    turn3_target_mode = _normalize_turn3_target_mode(turn3_target_mode)
    sft_stage_mode = _normalize_sft_stage_mode(sft_stage_mode)
    routing_label_source = _normalize_routing_label_source(routing_label_source)
    raw_prompt = sample["raw_prompt"][0]["content"]
    reward_model = sample.get("reward_model", {}) if isinstance(sample.get("reward_model"), dict) else {}
    task_spec = parse_task_prompt(raw_prompt, data_source=sample.get("data_source"))
    data_source = task_spec.data_source or str(sample.get("data_source") or "ETTh1")
    target_column = task_spec.target_column or "OT"
    lookback_window = int(task_spec.lookback_window or 96)
    forecast_horizon = int(task_spec.forecast_horizon or 96)
    historical_data = task_spec.historical_data or raw_prompt
    _, history_values = parse_time_series_string(historical_data, target_column=target_column)
    if not history_values:
        raise ValueError("No valid ETTh1 values found in raw_prompt")

    diagnostic_plan = build_diagnostic_plan(history_values)
    selected_feature_tools = list(diagnostic_plan.tool_names)
    feature_results = build_feature_tool_results(history_values, tool_names=selected_feature_tools)
    feature_result_by_name = {result.tool_name: result for result in feature_results}
    diagnostic_tool_batches = plan_diagnostic_tool_batches(selected_feature_tools, max_parallel_calls=5)

    reference_teacher_model = _resolve_reference_teacher_model(sample)
    second_teacher_model = _normalize_teacher_model(sample.get("teacher_eval_second_best_model"))
    route_override_target = _resolve_route_override_target(sample)
    heuristic_selected_prediction_model, _routing_feature_snapshot, _heuristic_routing_policy_reason = _select_prediction_model_by_heuristic(
        history_values,
        selected_feature_tools=selected_feature_tools,
    )
    routing_confidence_tier = _routing_confidence_tier(
        heuristic_selected_prediction_model=heuristic_selected_prediction_model,
        reference_teacher_model=reference_teacher_model,
        diagnostic_plan_score_gap=diagnostic_plan.score_gap,
        teacher_eval_score_margin=sample.get("teacher_eval_score_margin"),
    )
    selected_prediction_model = heuristic_selected_prediction_model
    routing_policy_reason = ""
    routing_policy_source = "heuristic_rule_based"
    effective_routing_label_source = routing_label_source
    route_default_expert = ""
    route_decision = ""
    route_override_model = ""
    route_tool_name = "predict_time_series"
    route_tool_arguments: dict[str, Any] = {}

    if route_override_target is not None:
        selected_prediction_model = str(route_override_target["resolved_model"])
        effective_routing_label_source = ROUTING_LABEL_SOURCE_DEFAULT_OVERRIDE
        route_default_expert = str(route_override_target["default_expert"])
        route_decision = str(route_override_target["route_decision"])
        route_override_model = str(route_override_target["override_model"])
        route_tool_name = "route_time_series"
        route_tool_arguments = dict(route_override_target["tool_arguments"])
        routing_policy_source = "route_override_bootstrap"
        routing_policy_reason = str(route_override_target["route_label"])
        routing_confidence_tier = str(
            sample.get("route_label_confidence")
            or sample.get("route_bootstrap_confidence_tier")
            or "mid"
        ).strip().lower()
    elif (
        routing_label_source == ROUTING_LABEL_SOURCE_REFERENCE_TEACHER
        and _sample_has_reference_teacher_model(sample)
    ):
        selected_prediction_model = reference_teacher_model
        routing_policy_source = "reference_teacher_offline_best"

    routing_only_mode = sft_stage_mode == SFT_STAGE_MODE_ROUTING_ONLY
    base_prediction_source = "route_label_only" if routing_only_mode else ""
    turn3_target: dict[str, Any] | None = None
    turn3_base_prediction_text = ""
    if not routing_only_mode:
        def _resolve_selected_prediction(model_name: str) -> tuple[str, str]:
            return _resolve_prediction_text(
                sample=sample,
                historical_data=historical_data,
                data_source=data_source,
                target_column=target_column,
                forecast_horizon=forecast_horizon,
                model_name=model_name,
                allow_cached_reference=(model_name == reference_teacher_model),
            )

        try:
            base_prediction_text, base_prediction_source = _resolve_selected_prediction(selected_prediction_model)
        except Exception as primary_exc:
            fallback_models: list[str] = []
            for fallback_model in [reference_teacher_model, second_teacher_model]:
                fallback_model = _normalize_teacher_model(fallback_model)
                if fallback_model == selected_prediction_model or fallback_model in fallback_models:
                    continue
                fallback_models.append(fallback_model)

            for fallback_model in fallback_models:
                try:
                    base_prediction_text, base_prediction_source = _resolve_selected_prediction(fallback_model)
                    selected_prediction_model = fallback_model
                    if route_override_target is not None:
                        route_tool_name = "route_time_series"
                        route_decision = ROUTE_DECISION_OVERRIDE if fallback_model != route_default_expert else ROUTE_DECISION_KEEP_DEFAULT
                        route_override_model = fallback_model if route_decision == ROUTE_DECISION_OVERRIDE else ""
                        route_tool_arguments = (
                            {"decision": ROUTE_DECISION_KEEP_DEFAULT}
                            if route_decision == ROUTE_DECISION_KEEP_DEFAULT
                            else {"decision": ROUTE_DECISION_OVERRIDE, "model_name": fallback_model}
                        )
                        routing_policy_source = "route_override_bootstrap_fallback"
                    else:
                        routing_policy_source = "heuristic_rule_based_fallback"
                    routing_policy_reason = f"fallback_to={fallback_model}"
                    break
                except Exception:
                    continue
            else:
                raise primary_exc

        turn3_target = _build_turn3_target(
            sample=sample,
            historical_data=historical_data,
            history_values=history_values,
            base_prediction_text=base_prediction_text,
            forecast_horizon=forecast_horizon,
            model_name=selected_prediction_model,
            selected_feature_tools=selected_feature_tools,
            turn3_target_mode=turn3_target_mode,
        )
        _require_prediction_values(
            turn3_target["refined_prediction_text"],
            forecast_horizon,
            source_name="turn3_target",
        )
        turn3_base_prediction_text = _canonical_prediction_text(
            base_prediction_text,
            forecast_horizon,
            history_text=historical_data,
        )

    system_prompt = build_timeseries_system_prompt(data_source=data_source, target_column=target_column)
    trajectory_turn_count = len(diagnostic_tool_batches) + (1 if routing_only_mode else 2)
    source_sample_index = int(sample.get("index", -1))
    shared_fields = {
        "source_sample_index": source_sample_index,
        "source_uid": str(sample.get("uid") or ""),
        "sft_stage_mode": sft_stage_mode,
        "data_source": data_source,
        "target_column": target_column,
        "forecast_horizon": forecast_horizon,
        "lookback_window": lookback_window,
        "reference_teacher_model": reference_teacher_model,
        "heuristic_selected_prediction_model": heuristic_selected_prediction_model,
        "heuristic_matches_reference_teacher": (
            str(heuristic_selected_prediction_model or "").strip().lower()
            == str(reference_teacher_model or "").strip().lower()
        ),
        "selected_prediction_model": selected_prediction_model,
        "routing_label_source": effective_routing_label_source,
        "diagnostic_plan_reason": diagnostic_plan.rationale,
        "diagnostic_primary_model": diagnostic_plan.primary_model,
        "diagnostic_runner_up_model": diagnostic_plan.runner_up_model,
        "diagnostic_plan_score_gap": diagnostic_plan.score_gap,
        "routing_confidence_tier": routing_confidence_tier,
        "routing_policy_source": routing_policy_source,
        "routing_policy_reason": routing_policy_reason,
        "route_default_expert": route_default_expert or sample.get("route_default_expert") or sample.get("default_expert"),
        "route_decision": route_decision or sample.get("route_decision"),
        "route_override_model": route_override_model or sample.get("route_override_model"),
        "route_label": (
            ROUTE_DECISION_KEEP_DEFAULT
            if route_decision == ROUTE_DECISION_KEEP_DEFAULT
            else (f"override_to_{route_override_model}" if route_decision == ROUTE_DECISION_OVERRIDE and route_override_model else sample.get("route_label"))
        ),
        "route_label_confidence": sample.get("route_label_confidence"),
        "default_expert": sample.get("default_expert"),
        "default_error": sample.get("default_error"),
        "best_model": sample.get("best_model"),
        "best_error": sample.get("best_error"),
        "improvement_vs_default": sample.get("improvement_vs_default"),
        "improvement_vs_default_rel": sample.get("improvement_vs_default_rel"),
        "base_prediction_source": base_prediction_source,
        "selected_feature_tools": list(selected_feature_tools),
        "selected_feature_tool_count": len(selected_feature_tools),
        "selected_feature_tool_signature": _feature_tool_signature(selected_feature_tools),
        "turn3_target_mode": turn3_target_mode,
        "turn3_target_type": turn3_target["turn3_target_type"] if turn3_target else "",
        "turn3_trigger_reason": turn3_target["turn3_trigger_reason"] if turn3_target else "",
        "refinement_support_signals": turn3_target["refinement_support_signals"] if turn3_target else [],
        "refinement_support_signal_signature": turn3_target["refinement_support_signal_signature"] if turn3_target else "",
        "refinement_keep_support_signals": turn3_target["refinement_keep_support_signals"] if turn3_target else [],
        "refinement_keep_support_signal_signature": (
            turn3_target["refinement_keep_support_signal_signature"] if turn3_target else ""
        ),
        "refinement_edit_support_signals": (
            turn3_target["refinement_edit_support_signals"] if turn3_target else {"__routing_only__": []}
        ),
        "refinement_candidate_adjustments": turn3_target["refinement_candidate_adjustments"] if turn3_target else [],
        "refinement_candidate_adjustment_signature": (
            turn3_target["refinement_candidate_adjustment_signature"] if turn3_target else ""
        ),
        "refinement_decision_name": turn3_target["refinement_decision_name"] if turn3_target else "",
        "refinement_candidate_prediction_text_map": (
            turn3_target["refinement_candidate_prediction_text_map"]
            if turn3_target
            else {"__routing_only__": ""}
        ),
        "refine_ops": turn3_target["refine_ops"] if turn3_target else [],
        "refine_ops_signature": turn3_target["refine_ops_signature"] if turn3_target else "",
        "refine_gain_mse": turn3_target["refine_gain_mse"] if turn3_target else 0.0,
        "refine_gain_mae": turn3_target["refine_gain_mae"] if turn3_target else 0.0,
        "refine_changed_value_count": turn3_target["refine_changed_value_count"] if turn3_target else 0,
        "refine_first_changed_index": turn3_target["refine_first_changed_index"] if turn3_target else -1,
        "refine_last_changed_index": turn3_target["refine_last_changed_index"] if turn3_target else -1,
        "refine_changed_span": turn3_target["refine_changed_span"] if turn3_target else 0,
        "refine_mean_abs_delta": turn3_target["refine_mean_abs_delta"] if turn3_target else 0.0,
        "refine_max_abs_delta": turn3_target["refine_max_abs_delta"] if turn3_target else 0.0,
        "base_teacher_prediction_text": turn3_target["base_teacher_prediction_text"] if turn3_target else "",
        "refined_prediction_text": turn3_target["refined_prediction_text"] if turn3_target else "",
        "teacher_eval_best_score": sample.get("teacher_eval_best_score"),
        "teacher_eval_second_best_model": sample.get("teacher_eval_second_best_model"),
        "teacher_eval_second_best_score": sample.get("teacher_eval_second_best_score"),
        "teacher_eval_score_margin": sample.get("teacher_eval_score_margin"),
        "teacher_eval_scores": sample.get("teacher_eval_scores"),
        "teacher_eval_score_details": sample.get("teacher_eval_score_details"),
        "route_best_model": sample.get("route_best_model"),
        "route_second_best_model": sample.get("route_second_best_model"),
        "route_best_error": sample.get("route_best_error"),
        "route_second_best_error": sample.get("route_second_best_error"),
        "route_margin_abs": sample.get("route_margin_abs"),
        "route_margin_rel": sample.get("route_margin_rel"),
        "route_top2_models": sample.get("route_top2_models"),
        "route_bootstrap_confidence_tier": sample.get("route_bootstrap_confidence_tier"),
    }

    records: list[dict[str, Any]] = []
    history_analysis: list[str] = []
    completed_feature_tools: list[str] = []
    feature_call_serial = 1
    turn_stage_order = 0

    for batch_idx, batch_tool_names in enumerate(diagnostic_tool_batches):
        turn_prompt = build_runtime_user_prompt(
            data_source=data_source,
            target_column=target_column,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            time_series_data=historical_data,
            history_analysis=history_analysis,
            prediction_results=None,
            available_feature_tools=batch_tool_names,
            completed_feature_tools=completed_feature_tools,
            turn_stage="diagnostic",
        )
        feature_tool_calls = [
            _make_tool_call(tool_name=tool_name, arguments={}, call_id=f"call_{feature_call_serial + idx}_{tool_name}")
            for idx, tool_name in enumerate(batch_tool_names)
        ]
        diagnostic_tools = [copy.deepcopy(FEATURE_TOOL_SCHEMA_BY_NAME[name]) for name in batch_tool_names]
        records.append(
            _make_stage_record(
                shared_fields=shared_fields,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": turn_prompt},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": _build_diagnostic_reflection(
                            plan_reason=diagnostic_plan.rationale,
                            batch_tool_names=batch_tool_names,
                            completed_feature_tools=completed_feature_tools,
                        ),
                        "tool_calls": feature_tool_calls,
                    },
                ],
                tools=diagnostic_tools,
                turn_stage="diagnostic",
                turn_stage_order=turn_stage_order,
                trajectory_turn_count=trajectory_turn_count,
                current_required_feature_tools=list(batch_tool_names),
                completed_feature_tools_before_turn=list(completed_feature_tools),
                history_analysis_count_before_turn=len(history_analysis),
                paper_turn3_required=False,
                diagnostic_batch_index=batch_idx,
            )
        )
        for tool_name in batch_tool_names:
            result = feature_result_by_name[tool_name]
            history_analysis.append(result.tool_output)
            completed_feature_tools.append(tool_name)
        feature_call_serial += len(batch_tool_names)
        turn_stage_order += 1

    routing_prompt = build_runtime_user_prompt(
        data_source=data_source,
        target_column=target_column,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        time_series_data=historical_data,
        history_analysis=history_analysis,
        prediction_results=None,
        available_feature_tools=selected_feature_tools,
        completed_feature_tools=selected_feature_tools,
        routing_feature_payload=_build_routing_feature_payload(
            history_values=history_values,
            selected_feature_tools=selected_feature_tools,
        ),
        turn_stage="routing",
        route_default_expert=route_default_expert or None,
    )
    route_tool_call = _make_tool_call(
        tool_name=route_tool_name,
        arguments=route_tool_arguments or {"model_name": selected_prediction_model},
        call_id=f"call_{route_tool_name}",
    )
    records.append(
        _make_stage_record(
            shared_fields=shared_fields,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": routing_prompt},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": build_routing_reflection(
                        model_name=selected_prediction_model,
                        history_values=history_values,
                        selected_feature_tools=selected_feature_tools,
                        decision_reason=routing_policy_reason,
                        include_heuristic_comparison=False,
                        route_default_expert=route_default_expert,
                        route_decision=route_decision,
                    ),
                    "tool_calls": [route_tool_call],
                },
            ],
            tools=[
                copy.deepcopy(
                    ROUTE_TIMESERIES_TOOL_SCHEMA if route_tool_name == "route_time_series" else PREDICT_TIMESERIES_TOOL_SCHEMA
                )
            ],
            turn_stage="routing",
            turn_stage_order=turn_stage_order,
            trajectory_turn_count=trajectory_turn_count,
            current_required_feature_tools=list(selected_feature_tools),
            completed_feature_tools_before_turn=list(selected_feature_tools),
            history_analysis_count_before_turn=len(history_analysis),
            paper_turn3_required=False,
        )
    )
    turn_stage_order += 1
    if routing_only_mode:
        return _filter_records_for_stage_mode(records, sft_stage_mode=sft_stage_mode)

    turn_3_prompt = build_runtime_user_prompt(
        data_source=data_source,
        target_column=target_column,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        time_series_data=historical_data,
        history_analysis=history_analysis,
        prediction_results=turn3_base_prediction_text,
        prediction_model_used=selected_prediction_model,
        available_feature_tools=selected_feature_tools,
        completed_feature_tools=selected_feature_tools,
        refinement_feature_payload=build_refinement_support_payload(
            base_values=_require_prediction_values(
                turn3_base_prediction_text,
                forecast_horizon,
                source_name=f"{selected_prediction_model}_turn3_prompt",
            ),
            history_values=history_values,
            selected_feature_tools=selected_feature_tools,
            prediction_model_used=selected_prediction_model,
        ),
        turn_stage="refinement",
    )
    records.append(
        _make_stage_record(
            shared_fields=shared_fields,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": turn_3_prompt},
                {
                    "role": "assistant",
                    "content": build_final_answer(
                        decision_name=turn3_target["refinement_decision_name"],
                        prediction_text=turn3_target["refined_prediction_text"],
                        turn3_target_mode=turn3_target_mode,
                        turn3_target_type=turn3_target["turn3_target_type"],
                        model_name=selected_prediction_model,
                        refine_ops=turn3_target["refine_ops"],
                    ),
                },
            ],
            tools=None,
            turn_stage="refinement",
            turn_stage_order=turn_stage_order,
            trajectory_turn_count=trajectory_turn_count,
            current_required_feature_tools=list(selected_feature_tools),
            completed_feature_tools_before_turn=list(selected_feature_tools),
            history_analysis_count_before_turn=len(history_analysis),
            paper_turn3_required=True,
        )
    )
    return _filter_records_for_stage_mode(records, sft_stage_mode=sft_stage_mode)


def build_sft_record(
    sample: dict[str, Any],
    *,
    turn3_target_mode: str = DEFAULT_TURN3_TARGET_MODE,
    routing_label_source: str = DEFAULT_ROUTING_LABEL_SOURCE,
    sft_stage_mode: str = DEFAULT_SFT_STAGE_MODE,
) -> dict[str, Any]:
    records = build_sft_records(
        sample,
        turn3_target_mode=turn3_target_mode,
        routing_label_source=routing_label_source,
        sft_stage_mode=sft_stage_mode,
    )
    if not records:
        raise ValueError("build_sft_records returned no records")
    for record in records:
        if str(record.get("turn_stage") or "").strip().lower() == "refinement":
            selected_record = dict(record)
            selected_record["sample_index"] = int(sample.get("index", -1))
            return selected_record
    for record in records:
        if str(record.get("turn_stage") or "").strip().lower() == "routing":
            selected_record = dict(record)
            selected_record["sample_index"] = int(sample.get("index", -1))
            return selected_record
    selected_record = dict(records[-1])
    selected_record["sample_index"] = int(sample.get("index", -1))
    return selected_record


def convert_jsonl_to_sft_parquet(
    *,
    input_path: str | Path,
    output_path: str | Path,
    max_samples: int = -1,
    turn3_target_mode: str = DEFAULT_TURN3_TARGET_MODE,
    routing_label_source: str = DEFAULT_ROUTING_LABEL_SOURCE,
    sft_stage_mode: str = DEFAULT_SFT_STAGE_MODE,
) -> pd.DataFrame:
    turn3_target_mode = _normalize_turn3_target_mode(turn3_target_mode)
    routing_label_source = _normalize_routing_label_source(routing_label_source)
    sft_stage_mode = _normalize_sft_stage_mode(sft_stage_mode)
    input_path = Path(input_path)
    output_path = Path(output_path)
    records: list[dict[str, Any]] = []
    source_sample_count = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if max_samples > 0 and source_sample_count >= max_samples:
                break
            if not line.strip():
                continue
            sample = json.loads(line)
            stage_records = build_sft_records(
                sample,
                turn3_target_mode=turn3_target_mode,
                routing_label_source=routing_label_source,
                sft_stage_mode=sft_stage_mode,
            )
            for record in stage_records:
                row = dict(record)
                row["sample_index"] = len(records)
                records.append(row)
            source_sample_count += 1
            if (line_idx + 1) % 500 == 0:
                print(f"Processed {line_idx + 1} samples from {input_path}")

    dataframe = pd.DataFrame(records)
    _validate_paper_turn3_protocol(
        dataframe,
        split_name=input_path.stem,
        output_path=output_path,
        allow_no_refinement=(sft_stage_mode == SFT_STAGE_MODE_ROUTING_ONLY),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    checked_count = int(_summarize_paper_turn3_protocol(dataframe).get("turn3_protocol_checked_count", 0))
    print(
        f"Wrote {len(dataframe)} SFT records from {source_sample_count} source samples "
        f"({checked_count} refinement targets) to {output_path}"
    )
    return dataframe


def distribution_from_series(values: Iterable[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        if value is None:
            key = "__missing__"
        elif isinstance(value, float) and pd.isna(value):
            key = "__missing__"
        else:
            key = str(value)
        counter[key] += 1
    return {str(k): int(v) for k, v in sorted(counter.items())}


def agreement_ratio_from_frame(
    dataframe: pd.DataFrame,
    *,
    left_column: str,
    right_column: str,
) -> float:
    if dataframe is None or dataframe.empty:
        return 0.0
    if left_column not in dataframe.columns or right_column not in dataframe.columns:
        return 0.0

    comparable = dataframe[[left_column, right_column]].copy()
    comparable[left_column] = comparable[left_column].astype(str).str.strip().str.lower()
    comparable[right_column] = comparable[right_column].astype(str).str.strip().str.lower()
    comparable = comparable[
        (comparable[left_column] != "")
        & (comparable[right_column] != "")
        & (comparable[left_column] != "__missing__")
        & (comparable[right_column] != "__missing__")
    ]
    if comparable.empty:
        return 0.0
    return float((comparable[left_column] == comparable[right_column]).mean())


def _tail_repeat_run_length(values: Sequence[float], *, atol: float = 1e-8) -> int:
    if not values:
        return 0
    run_length = 1
    last_value = float(values[-1])
    for value in reversed(values[:-1]):
        if abs(float(value) - last_value) > atol:
            break
        run_length += 1
    return int(run_length)


def _row_prediction_values_for_turn3_priority(row: Any) -> list[float]:
    for field_name in ("base_teacher_prediction_text", "refined_prediction_text"):
        text = str((row.get(field_name) if hasattr(row, "get") else "") or "").strip()
        if not text:
            continue
        _timestamps, values = parse_time_series_string(text)
        if values:
            return [float(value) for value in values]
    return []


def _is_priority_validated_keep_row(row: Any) -> bool:
    if str((row.get("turn3_target_type") if hasattr(row, "get") else "") or "").strip().lower() != "validated_keep":
        return False
    if str((row.get("selected_prediction_model") if hasattr(row, "get") else "") or "").strip().lower() != "arima":
        return False
    values = _row_prediction_values_for_turn3_priority(row)
    if len(values) < 8:
        return False
    return _tail_repeat_run_length(values) >= 8


def rebalance_train_turn3_targets(
    dataframe: pd.DataFrame,
    *,
    min_local_refine_ratio: float,
    rebalance_mode: str = DEFAULT_TRAIN_TURN3_REBALANCE_MODE,
) -> pd.DataFrame:
    if dataframe.empty or min_local_refine_ratio <= 0:
        return dataframe
    if "turn3_target_type" not in dataframe.columns:
        return dataframe
    rebalance_mode = _normalize_train_turn3_rebalance_mode(rebalance_mode)

    group_column = "source_sample_index" if "source_sample_index" in dataframe.columns else "sample_index"
    if group_column not in dataframe.columns:
        return dataframe

    source_frame = _source_level_frame(dataframe)
    local_refine_mask = source_frame["turn3_target_type"] == "local_refine"
    validated_keep_mask = source_frame["turn3_target_type"] == "validated_keep"
    local_refine_df = source_frame.loc[local_refine_mask].copy()
    validated_keep_df = source_frame.loc[validated_keep_mask].copy()
    other_df = source_frame.loc[~(local_refine_mask | validated_keep_mask)].copy()

    local_count = len(local_refine_df)
    keep_count = len(validated_keep_df)
    other_count = len(other_df)
    if local_count <= 0 or keep_count <= 0:
        return dataframe

    current_ratio = float(local_count / max(local_count + keep_count + other_count, 1))
    if current_ratio >= min_local_refine_ratio:
        return dataframe

    if rebalance_mode == TRAIN_TURN3_REBALANCE_MODE_OVERSAMPLE_LOCAL_REFINE:
        non_local_count = keep_count + other_count
        target_local_count = int(np.ceil(min_local_refine_ratio * non_local_count / max(1.0 - min_local_refine_ratio, 1e-8)))
        additional_local_count = max(target_local_count - local_count, 0)
        if additional_local_count <= 0:
            return dataframe

        local_source_ids = local_refine_df[group_column].tolist()
        if not local_source_ids:
            return dataframe

        source_frames = {
            source_id: dataframe.loc[dataframe[group_column] == source_id].copy()
            for source_id in local_source_ids
        }
        repeated_frames: list[pd.DataFrame] = [dataframe.copy()]
        for repeat_index in range(additional_local_count):
            source_id = local_source_ids[repeat_index % len(local_source_ids)]
            repeated_frames.append(source_frames[source_id].copy())

        balanced = pd.concat(repeated_frames, ignore_index=True)
        sort_columns = [col for col in (group_column, "turn_stage_order", "sample_index") if col in balanced.columns]
        if sort_columns:
            balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        if "sample_index" in balanced.columns:
            balanced["sample_index"] = list(range(len(balanced)))
        return balanced

    max_keep_count = int(np.floor(local_count * (1.0 - min_local_refine_ratio) / min_local_refine_ratio))
    max_keep_count = max(max_keep_count - other_count, 0)
    if keep_count <= max_keep_count:
        return dataframe

    validated_keep_df = validated_keep_df.sort_values(group_column).reset_index(drop=True)
    if max_keep_count <= 0:
        rebalanced_keep_df = validated_keep_df.iloc[0:0].copy()
    else:
        priority_mask = validated_keep_df.apply(_is_priority_validated_keep_row, axis=1)
        priority_keep_df = validated_keep_df.loc[priority_mask].copy()
        regular_keep_df = validated_keep_df.loc[~priority_mask].copy()

        if len(priority_keep_df) >= max_keep_count:
            rebalanced_keep_df = priority_keep_df.iloc[:max_keep_count].copy()
        else:
            keep_frames: list[pd.DataFrame] = [priority_keep_df]
            remaining_count = max_keep_count - len(priority_keep_df)
            if remaining_count > 0 and not regular_keep_df.empty:
                regular_count = len(regular_keep_df)
                if regular_count <= remaining_count:
                    sampled_regular_keep_df = regular_keep_df.copy()
                else:
                    take_positions = np.linspace(0, regular_count - 1, num=remaining_count, dtype=int)
                    sampled_regular_keep_df = regular_keep_df.iloc[take_positions].copy()
                keep_frames.append(sampled_regular_keep_df)
            rebalanced_keep_df = pd.concat(keep_frames, ignore_index=True) if keep_frames else validated_keep_df.iloc[0:0].copy()

    kept_source_ids = set(local_refine_df[group_column].tolist())
    kept_source_ids.update(rebalanced_keep_df[group_column].tolist())
    kept_source_ids.update(other_df[group_column].tolist())

    balanced = dataframe.loc[dataframe[group_column].isin(kept_source_ids)].copy()
    sort_columns = [col for col in (group_column, "turn_stage_order", "sample_index") if col in balanced.columns]
    if sort_columns:
        balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return balanced


def rebalance_refinement_stage_targets(
    dataframe: pd.DataFrame,
    *,
    min_local_refine_ratio: float,
    rebalance_mode: str = DEFAULT_TRAIN_TURN3_REBALANCE_MODE,
) -> pd.DataFrame:
    if dataframe.empty or min_local_refine_ratio <= 0:
        return dataframe
    if "turn_stage" not in dataframe.columns or "turn3_target_type" not in dataframe.columns:
        return dataframe
    rebalance_mode = _normalize_train_turn3_rebalance_mode(rebalance_mode)

    refinement_mask = dataframe["turn_stage"].astype(str).str.lower() == "refinement"
    refinement_df = dataframe.loc[refinement_mask].copy()
    non_refinement_df = dataframe.loc[~refinement_mask].copy()
    if refinement_df.empty:
        return dataframe

    local_refine_df = refinement_df.loc[
        refinement_df["turn3_target_type"].astype(str).str.lower() == "local_refine"
    ].copy()
    validated_keep_df = refinement_df.loc[
        refinement_df["turn3_target_type"].astype(str).str.lower() == "validated_keep"
    ].copy()
    other_df = refinement_df.loc[
        ~refinement_df["turn3_target_type"].astype(str).str.lower().isin({"local_refine", "validated_keep"})
    ].copy()

    local_count = len(local_refine_df)
    keep_count = len(validated_keep_df)
    other_count = len(other_df)
    if local_count <= 0 or keep_count <= 0:
        return dataframe

    current_ratio = float(local_count / max(local_count + keep_count + other_count, 1))
    if current_ratio >= min_local_refine_ratio:
        return dataframe

    if rebalance_mode == TRAIN_TURN3_REBALANCE_MODE_OVERSAMPLE_LOCAL_REFINE:
        non_local_count = keep_count + other_count
        target_local_count = int(np.ceil(min_local_refine_ratio * non_local_count / max(1.0 - min_local_refine_ratio, 1e-8)))
        additional_local_count = max(target_local_count - local_count, 0)
        if additional_local_count <= 0:
            return dataframe

        repeated_frames: list[pd.DataFrame] = [dataframe.copy()]
        for repeat_index in range(additional_local_count):
            repeated_frames.append(local_refine_df.iloc[[repeat_index % local_count]].copy())
        balanced = pd.concat(repeated_frames, ignore_index=True)
    else:
        max_keep_count = int(np.floor(local_count * (1.0 - min_local_refine_ratio) / min_local_refine_ratio))
        max_keep_count = max(max_keep_count - other_count, 0)
        if keep_count <= max_keep_count:
            return dataframe

        sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in validated_keep_df.columns]
        if sort_columns:
            validated_keep_df = validated_keep_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        else:
            validated_keep_df = validated_keep_df.reset_index(drop=True)
        if max_keep_count <= 0:
            kept_validated_keep_df = validated_keep_df.iloc[0:0].copy()
        else:
            priority_mask = validated_keep_df.apply(_is_priority_validated_keep_row, axis=1)
            priority_keep_df = validated_keep_df.loc[priority_mask].copy()
            regular_keep_df = validated_keep_df.loc[~priority_mask].copy()
            if len(priority_keep_df) >= max_keep_count:
                kept_validated_keep_df = priority_keep_df.iloc[:max_keep_count].copy()
            else:
                keep_frames: list[pd.DataFrame] = [priority_keep_df]
                remaining_count = max_keep_count - len(priority_keep_df)
                if remaining_count > 0 and not regular_keep_df.empty:
                    regular_count = len(regular_keep_df)
                    if regular_count <= remaining_count:
                        sampled_regular_keep_df = regular_keep_df.copy()
                    else:
                        take_positions = np.linspace(0, regular_count - 1, num=remaining_count, dtype=int)
                        sampled_regular_keep_df = regular_keep_df.iloc[take_positions].copy()
                    keep_frames.append(sampled_regular_keep_df)
                kept_validated_keep_df = (
                    pd.concat(keep_frames, ignore_index=True)
                    if keep_frames
                    else validated_keep_df.iloc[0:0].copy()
                )

        kept_refinement_df = pd.concat(
            [local_refine_df, kept_validated_keep_df, other_df],
            ignore_index=True,
        )
        balanced = pd.concat([non_refinement_df, kept_refinement_df], ignore_index=True)

    sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in balanced.columns]
    if sort_columns:
        balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if "sample_index" in balanced.columns:
        balanced["sample_index"] = list(range(len(balanced)))
    return balanced


def rebalance_train_stage_records(
    dataframe: pd.DataFrame,
    *,
    stage_repeat_factors: dict[str, int] | None,
) -> pd.DataFrame:
    if dataframe.empty or not stage_repeat_factors:
        return dataframe
    if "turn_stage" not in dataframe.columns:
        return dataframe

    normalized_factors = {
        str(stage).strip().lower(): max(1, int(factor))
        for stage, factor in stage_repeat_factors.items()
        if str(stage).strip()
    }
    if not normalized_factors:
        return dataframe

    repeated_frames: list[pd.DataFrame] = []
    for _, row in dataframe.iterrows():
        stage = str(row.get("turn_stage") or "").strip().lower()
        repeat_factor = normalized_factors.get(stage, 1)
        row_frame = pd.DataFrame([row.to_dict()])
        repeated_frames.extend(row_frame.copy() for _ in range(max(1, repeat_factor)))

    balanced = pd.concat(repeated_frames, ignore_index=True)
    sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in balanced.columns]
    if sort_columns:
        balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if "sample_index" in balanced.columns:
        balanced["sample_index"] = list(range(len(balanced)))
    return balanced


def repeat_priority_validated_keep_refinement_rows(
    dataframe: pd.DataFrame,
    *,
    repeat_factor: int,
) -> pd.DataFrame:
    if dataframe.empty or repeat_factor <= 1:
        return dataframe
    if "turn_stage" not in dataframe.columns:
        return dataframe

    refinement_mask = dataframe["turn_stage"].astype(str).str.lower() == "refinement"
    refinement_df = dataframe.loc[refinement_mask].copy()
    if refinement_df.empty:
        return dataframe

    priority_mask = refinement_df.apply(_is_priority_validated_keep_row, axis=1)
    priority_df = refinement_df.loc[priority_mask].copy()
    if priority_df.empty:
        return dataframe

    repeated_frames: list[pd.DataFrame] = [dataframe.copy()]
    repeated_frames.extend(priority_df.copy() for _ in range(int(repeat_factor) - 1))
    balanced = pd.concat(repeated_frames, ignore_index=True)
    sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in balanced.columns]
    if sort_columns:
        balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if "sample_index" in balanced.columns:
        balanced["sample_index"] = list(range(len(balanced)))
    return balanced


def repeat_local_refine_refinement_rows(
    dataframe: pd.DataFrame,
    *,
    repeat_factor: int,
) -> pd.DataFrame:
    if dataframe.empty or repeat_factor <= 1:
        return dataframe
    if "turn_stage" not in dataframe.columns or "turn3_target_type" not in dataframe.columns:
        return dataframe

    local_refine_mask = (
        dataframe["turn_stage"].astype(str).str.lower().eq("refinement")
        & dataframe["turn3_target_type"].astype(str).str.lower().eq("local_refine")
    )
    local_refine_df = dataframe.loc[local_refine_mask].copy()
    if local_refine_df.empty:
        return dataframe

    repeated_frames: list[pd.DataFrame] = [dataframe.copy()]
    repeated_frames.extend(local_refine_df.copy() for _ in range(int(repeat_factor) - 1))
    balanced = pd.concat(repeated_frames, ignore_index=True)
    sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in balanced.columns]
    if sort_columns:
        balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if "sample_index" in balanced.columns:
        balanced["sample_index"] = list(range(len(balanced)))
    return balanced


def filter_train_routing_records_by_confidence(
    dataframe: pd.DataFrame,
    *,
    min_tier: str,
) -> pd.DataFrame:
    normalized_min_tier = _normalize_train_routing_confidence_min_tier(min_tier)
    if dataframe.empty or normalized_min_tier == "none":
        return dataframe
    if "turn_stage" not in dataframe.columns or "routing_confidence_tier" not in dataframe.columns:
        return dataframe

    min_rank = ROUTING_CONFIDENCE_TIER_RANK[normalized_min_tier]
    keep_rows: list[dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        stage = str(row.get("turn_stage") or "").strip().lower()
        if stage != "routing":
            keep_rows.append(row.to_dict())
            continue
        if str(row.get("routing_label_source") or "").strip().lower() != ROUTING_LABEL_SOURCE_HEURISTIC:
            keep_rows.append(row.to_dict())
            continue
        tier = str(row.get("routing_confidence_tier") or "low").strip().lower()
        tier_rank = ROUTING_CONFIDENCE_TIER_RANK.get(tier, ROUTING_CONFIDENCE_TIER_RANK["low"])
        if tier_rank >= min_rank:
            keep_rows.append(row.to_dict())

    balanced = pd.DataFrame(keep_rows)
    if balanced.empty:
        return balanced
    sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in balanced.columns]
    if sort_columns:
        balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if "sample_index" in balanced.columns:
        balanced["sample_index"] = list(range(len(balanced)))
    return balanced


def repeat_high_confidence_routing_rows(
    dataframe: pd.DataFrame,
    *,
    repeat_factor: int,
) -> pd.DataFrame:
    if dataframe.empty or repeat_factor <= 1:
        return dataframe
    if "turn_stage" not in dataframe.columns or "routing_confidence_tier" not in dataframe.columns:
        return dataframe

    high_conf_mask = (
        dataframe["turn_stage"].astype(str).str.lower().eq("routing")
        & dataframe["routing_confidence_tier"].astype(str).str.lower().eq("high")
        & dataframe["routing_label_source"].astype(str).str.lower().eq(ROUTING_LABEL_SOURCE_HEURISTIC)
    )
    high_conf_df = dataframe.loc[high_conf_mask].copy()
    if high_conf_df.empty:
        return dataframe

    repeated_frames: list[pd.DataFrame] = [dataframe.copy()]
    repeated_frames.extend(high_conf_df.copy() for _ in range(int(repeat_factor) - 1))
    balanced = pd.concat(repeated_frames, ignore_index=True)
    sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in balanced.columns]
    if sort_columns:
        balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if "sample_index" in balanced.columns:
        balanced["sample_index"] = list(range(len(balanced)))
    return balanced


def rebalance_train_routing_model_records(
    dataframe: pd.DataFrame,
    *,
    enabled: bool,
) -> pd.DataFrame:
    if dataframe.empty or not enabled:
        return dataframe
    if "turn_stage" not in dataframe.columns or "selected_prediction_model" not in dataframe.columns:
        return dataframe

    routing_mask = dataframe["turn_stage"].astype(str).str.lower() == "routing"
    routing_df = dataframe.loc[routing_mask].copy()
    non_routing_df = dataframe.loc[~routing_mask].copy()
    if routing_df.empty:
        return dataframe

    class_counts = routing_df["selected_prediction_model"].astype(str).value_counts()
    if class_counts.empty or len(class_counts) <= 1:
        return dataframe

    max_count = int(class_counts.max())
    repeated_frames: list[pd.DataFrame] = [non_routing_df]
    sort_columns = [col for col in ("source_sample_index", "turn_stage_order", "sample_index") if col in routing_df.columns]

    for model_name, class_df in routing_df.groupby("selected_prediction_model", sort=False):
        class_df = class_df.copy()
        class_len = len(class_df)
        repeat_factor, remainder = divmod(max_count, class_len)
        repeated_frames.extend(class_df.copy() for _ in range(max(1, repeat_factor)))
        if remainder > 0:
            repeated_frames.append(class_df.iloc[:remainder].copy())

    balanced = pd.concat(repeated_frames, ignore_index=True)
    if sort_columns:
        balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if "sample_index" in balanced.columns:
        balanced["sample_index"] = list(range(len(balanced)))
    return balanced

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ETTh1 OT step-wise runtime-aligned SFT parquet from teacher-curated or RL jsonl samples."
    )
    parser.add_argument(
        "--train-jsonl",
        default=str(DEFAULT_TRAIN_JSONL),
        help="Teacher-curated or RL train jsonl path.",
    )
    parser.add_argument(
        "--val-jsonl",
        default=str(DEFAULT_VAL_JSONL),
        help="Teacher-curated or RL val jsonl path.",
    )
    parser.add_argument(
        "--test-jsonl",
        default=str(DEFAULT_TEST_JSONL),
        help="Optional teacher-curated or RL test jsonl path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for parquet files.",
    )
    parser.add_argument("--max-train-samples", type=int, default=-1, help="Limit train sample count for debugging.")
    parser.add_argument("--max-val-samples", type=int, default=-1, help="Limit val sample count for debugging.")
    parser.add_argument("--max-test-samples", type=int, default=-1, help="Limit test sample count for debugging.")
    parser.add_argument(
        "--sft-stage-mode",
        choices=sorted(SUPPORTED_SFT_STAGE_MODES),
        default=DEFAULT_SFT_STAGE_MODE,
        help=(
            "Which stage records to keep in the generated parquet. "
            "`full` keeps the full three-stage trajectory; "
            "`routing_only` keeps only routing rows; "
            "`refinement_only` keeps only refinement rows for stage-local Turn-3 training."
        ),
    )
    parser.add_argument(
        "--turn3-target-mode",
        choices=sorted(SUPPORTED_TURN3_TARGET_MODES),
        default=DEFAULT_TURN3_TARGET_MODE,
        help=(
            "Turn-3 target generation mode. `engineering_refine` is the formal default and keeps the "
            "outer <think><answer> XML shell while constraining <answer> to a single structured "
            "`decision=<name>` line. `paper_strict` keeps the strict full-forecast answer target "
            "for ablations/debugging."
        ),
    )
    parser.add_argument(
        "--routing-label-source",
        choices=sorted(SUPPORTED_ROUTING_LABEL_SOURCES),
        default=DEFAULT_ROUTING_LABEL_SOURCE,
        help=(
            "Source for routing labels in step-wise SFT. "
            "`reference_teacher` uses the offline best/reference teacher model and is the formal default; "
            "`heuristic` keeps the rule-based routing teacher for ablations/debugging."
        ),
    )
    parser.add_argument(
        "--train-min-local-refine-ratio",
        type=float,
        default=0.25,
        help="Minimum desired local_refine ratio in train parquet. Set <=0 to disable train rebalancing.",
    )
    parser.add_argument(
        "--train-turn3-rebalance-mode",
        choices=sorted(SUPPORTED_TRAIN_TURN3_REBALANCE_MODES),
        default=DEFAULT_TRAIN_TURN3_REBALANCE_MODE,
        help=(
            "How to satisfy --train-min-local-refine-ratio when train local_refine rows are too scarce. "
            "`downsample_keep` trims validated_keep source groups; "
            "`oversample_local_refine` duplicates local_refine source groups to preserve source coverage."
        ),
    )
    parser.add_argument(
        "--train-stage-repeat-factors",
        default=json.dumps(DEFAULT_TRAIN_STAGE_REPEAT_FACTORS, ensure_ascii=False),
        help=(
            "JSON object of train-only stage repeat factors, e.g. "
            '\'{"diagnostic":1,"routing":1,"refinement":1}\'. '
            "Use {} to disable stage reweighting."
        ),
    )
    parser.add_argument(
        "--balance-train-routing-models",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_BALANCE_TRAIN_ROUTING_MODELS,
        help="Whether to rebalance train routing rows across selected_prediction_model classes.",
    )
    parser.add_argument(
        "--train-priority-validated-keep-repeat-factor",
        type=int,
        default=DEFAULT_TRAIN_PRIORITY_VALIDATED_KEEP_REPEAT_FACTOR,
        help=(
            "Repeat factor for train refinement rows that are validated_keep, selected_prediction_model=arima, "
            "and show a flat repeated tail. Set <=1 to disable."
        ),
    )
    parser.add_argument(
        "--train-local-refine-refinement-repeat-factor",
        type=int,
        default=DEFAULT_TRAIN_LOCAL_REFINE_REFINEMENT_REPEAT_FACTOR,
        help=(
            "Extra repeat factor applied only to train refinement rows with turn3_target_type=local_refine. "
            "Set <=1 to disable."
        ),
    )
    parser.add_argument(
        "--train-routing-confidence-min-tier",
        choices=sorted(SUPPORTED_TRAIN_ROUTING_CONFIDENCE_MIN_TIERS),
        default=DEFAULT_TRAIN_ROUTING_CONFIDENCE_MIN_TIER,
        help=(
            "Optional train-only filter for heuristic routing supervision. "
            "`high` keeps only routing rows whose heuristic choice matches the reference teacher and whose "
            "diagnostic/teacher margins are both strong; `mid` keeps mid+high confidence rows; `none` disables filtering."
        ),
    )
    parser.add_argument(
        "--train-high-confidence-routing-repeat-factor",
        type=int,
        default=DEFAULT_TRAIN_HIGH_CONFIDENCE_ROUTING_REPEAT_FACTOR,
        help=(
            "Extra repeat factor applied only to train routing rows with routing_confidence_tier=high. "
            "Set <=1 to disable."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sft_stage_mode = _normalize_sft_stage_mode(args.sft_stage_mode)
    turn3_target_mode = _normalize_turn3_target_mode(args.turn3_target_mode)
    routing_label_source = _normalize_routing_label_source(args.routing_label_source)
    train_turn3_rebalance_mode = _normalize_train_turn3_rebalance_mode(args.train_turn3_rebalance_mode)
    train_routing_confidence_min_tier = _normalize_train_routing_confidence_min_tier(
        getattr(args, "train_routing_confidence_min_tier", DEFAULT_TRAIN_ROUTING_CONFIDENCE_MIN_TIER)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_train_min_local_refine_ratio = float(args.train_min_local_refine_ratio)
    if sft_stage_mode == SFT_STAGE_MODE_ROUTING_ONLY:
        effective_train_min_local_refine_ratio = 0.0
    raw_train_stage_repeat_factors = json.loads(str(args.train_stage_repeat_factors))
    train_stage_repeat_factors = {
        str(stage).strip().lower(): max(1, int(factor))
        for stage, factor in dict(raw_train_stage_repeat_factors).items()
        if str(stage).strip()
    }

    source_metadata_paths: list[Path] = []
    for split_path in (Path(args.train_jsonl), Path(args.val_jsonl), Path(args.test_jsonl)):
        if not split_path.exists():
            continue
        source_metadata, source_metadata_path = validate_sibling_metadata(
            split_path,
            expected_kind=(
                DATASET_KIND_TEACHER_CURATED_SFT,
                DATASET_KIND_RUNTIME_SFT_PARQUET,
                DATASET_KIND_RL_JSONL,
            ),
        )
        require_multivariate_etth1_metadata(source_metadata, metadata_path=source_metadata_path)
        source_metadata_paths.append(source_metadata_path)
    if source_metadata_paths:
        unique_source_metadata_paths = {str(path) for path in source_metadata_paths}
        if len(unique_source_metadata_paths) != 1:
            raise ValueError(
                "All source jsonl splits must come from the same dataset directory. "
                f"Got metadata files: {sorted(unique_source_metadata_paths)}"
            )

    train_df_raw = convert_jsonl_to_sft_parquet(
        input_path=args.train_jsonl,
        output_path=output_dir / "train.parquet",
        max_samples=args.max_train_samples,
        turn3_target_mode=turn3_target_mode,
        routing_label_source=routing_label_source,
        sft_stage_mode=sft_stage_mode,
    )
    train_df = train_df_raw.copy()
    if sft_stage_mode == SFT_STAGE_MODE_REFINEMENT_ONLY:
        train_df = rebalance_refinement_stage_targets(
            train_df,
            min_local_refine_ratio=effective_train_min_local_refine_ratio,
            rebalance_mode=train_turn3_rebalance_mode,
        )
    elif sft_stage_mode == SFT_STAGE_MODE_FULL:
        train_df = rebalance_train_turn3_targets(
            train_df,
            min_local_refine_ratio=effective_train_min_local_refine_ratio,
            rebalance_mode=train_turn3_rebalance_mode,
        )
    train_df = filter_train_routing_records_by_confidence(
        train_df,
        min_tier=train_routing_confidence_min_tier,
    )
    train_high_confidence_routing_repeat_factor = int(
        getattr(
            args,
            "train_high_confidence_routing_repeat_factor",
            DEFAULT_TRAIN_HIGH_CONFIDENCE_ROUTING_REPEAT_FACTOR,
        )
    )
    train_df = repeat_high_confidence_routing_rows(
        train_df,
        repeat_factor=train_high_confidence_routing_repeat_factor,
    )

    if sft_stage_mode in {SFT_STAGE_MODE_FULL, SFT_STAGE_MODE_ROUTING_ONLY}:
        train_df = rebalance_train_routing_model_records(
            train_df,
            enabled=bool(args.balance_train_routing_models),
        )
    train_df = rebalance_train_stage_records(
        train_df,
        stage_repeat_factors=train_stage_repeat_factors,
    )
    train_priority_validated_keep_repeat_factor = int(
        getattr(
            args,
            "train_priority_validated_keep_repeat_factor",
            DEFAULT_TRAIN_PRIORITY_VALIDATED_KEEP_REPEAT_FACTOR,
        )
    )
    train_local_refine_refinement_repeat_factor = int(
        getattr(
            args,
            "train_local_refine_refinement_repeat_factor",
            DEFAULT_TRAIN_LOCAL_REFINE_REFINEMENT_REPEAT_FACTOR,
        )
    )
    train_df = repeat_local_refine_refinement_rows(
        train_df,
        repeat_factor=train_local_refine_refinement_repeat_factor,
    )
    train_df = repeat_priority_validated_keep_refinement_rows(
        train_df,
        repeat_factor=train_priority_validated_keep_repeat_factor,
    )
    if len(train_df) != len(train_df_raw) or not train_df.equals(train_df_raw):
        _validate_paper_turn3_protocol(
            train_df,
            split_name="train",
            output_path=output_dir / "train.parquet",
            allow_no_refinement=(sft_stage_mode == SFT_STAGE_MODE_ROUTING_ONLY),
        )
        train_df.to_parquet(output_dir / "train.parquet", index=False)
        print(
            f"Rebalanced train.parquet turn3_target_type distribution: "
            f"{distribution_from_series(train_df['turn3_target_type'])}"
        )
        print(
            f"Rebalanced train.parquet turn_stage distribution: "
            f"{distribution_from_series(train_df['turn_stage'])}"
        )
    val_df = convert_jsonl_to_sft_parquet(
        input_path=args.val_jsonl,
        output_path=output_dir / "val.parquet",
        max_samples=args.max_val_samples,
        turn3_target_mode=turn3_target_mode,
        routing_label_source=routing_label_source,
        sft_stage_mode=sft_stage_mode,
    )

    test_count = 0
    test_df: pd.DataFrame | None = None
    test_path = Path(args.test_jsonl)
    if test_path.exists():
        test_df = convert_jsonl_to_sft_parquet(
            input_path=test_path,
            output_path=output_dir / "test.parquet",
            max_samples=args.max_test_samples,
            turn3_target_mode=turn3_target_mode,
            routing_label_source=routing_label_source,
            sft_stage_mode=sft_stage_mode,
        )
        test_count = len(test_df)

    train_protocol_summary = _summarize_paper_turn3_protocol(train_df)
    val_protocol_summary = _summarize_paper_turn3_protocol(val_df)
    test_protocol_summary = _summarize_paper_turn3_protocol(test_df) if test_df is not None and len(test_df) > 0 else {}
    observed_routing_label_sources: set[str] = set()
    for frame in (train_df, val_df, test_df):
        if frame is None or frame.empty or "routing_label_source" not in frame.columns:
            continue
        observed_routing_label_sources.update(
            {
                str(value).strip().lower()
                for value in frame["routing_label_source"].dropna().tolist()
                if str(value).strip()
            }
        )
    effective_metadata_routing_label_source = (
        sorted(observed_routing_label_sources)[0]
        if len(observed_routing_label_sources) == 1
        else routing_label_source
    )

    metadata_kwargs = dict(
        dataset_kind=DATASET_KIND_RUNTIME_SFT_PARQUET,
        pipeline_stage="runtime_stepwise_sft",
        task_type="multivariate time-series forecasting",
        historical_data_protocol=HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
        target_column=ETTH1_TARGET_COLUMN,
        observed_feature_columns=list(ETTH1_FEATURE_COLUMNS),
        observed_covariates=list(ETTH1_COVARIATE_COLUMNS),
        model_input_width=len(ETTH1_FEATURE_COLUMNS),
        lookback_window=DEFAULT_LOOKBACK_WINDOW,
        forecast_horizon=DEFAULT_FORECAST_HORIZON,
        sft_stage_mode=sft_stage_mode,
        turn3_protocol="paper_think_answer_xml",
        turn3_target_mode=turn3_target_mode,
        routing_label_source=effective_metadata_routing_label_source,
        train_samples_before_balance=len(train_df_raw),
        train_samples=len(train_df),
        train_source_samples_before_balance=_source_sample_unique_count(train_df_raw),
        train_source_samples=_source_sample_unique_count(train_df),
        val_samples=len(val_df),
        val_source_samples=_source_sample_unique_count(val_df),
        test_samples=test_count,
        test_source_samples=_source_sample_unique_count(test_df),
        requested_train_min_local_refine_ratio=float(args.train_min_local_refine_ratio),
        train_min_local_refine_ratio=effective_train_min_local_refine_ratio,
        train_turn3_rebalance_mode=train_turn3_rebalance_mode,
        train_stage_repeat_factors=train_stage_repeat_factors,
        balance_train_routing_models=bool(args.balance_train_routing_models),
        train_priority_validated_keep_repeat_factor=train_priority_validated_keep_repeat_factor,
        train_local_refine_refinement_repeat_factor=train_local_refine_refinement_repeat_factor,
        source_train_jsonl=str(Path(args.train_jsonl)),
        source_val_jsonl=str(Path(args.val_jsonl)),
        source_test_jsonl=str(test_path),
        source_curated_metadata_path=str(source_metadata_paths[0]) if source_metadata_paths else "",
        train_turn_stage_distribution=distribution_from_series(train_df["turn_stage"]),
        val_turn_stage_distribution=distribution_from_series(val_df["turn_stage"]),
        train_source_sample_coverage_by_stage=source_sample_coverage_by_stage(train_df),
        val_source_sample_coverage_by_stage=source_sample_coverage_by_stage(val_df),
        turn_stage_loss_weight_summary={
            "stage_repeat_factors": train_stage_repeat_factors,
            "local_refine_refinement_repeat_factor": train_local_refine_refinement_repeat_factor,
            "priority_validated_keep_repeat_factor": train_priority_validated_keep_repeat_factor,
            "routing_model_balance_enabled": bool(args.balance_train_routing_models),
            "routing_confidence_min_tier": train_routing_confidence_min_tier,
            "high_confidence_routing_repeat_factor": train_high_confidence_routing_repeat_factor,
        },
        train_turn3_target_type_distribution_before_balance=distribution_from_series(
            _source_level_frame(train_df_raw)["turn3_target_type"]
        ),
        train_reference_teacher_model_distribution=distribution_from_series(
            _source_level_frame(train_df)["reference_teacher_model"]
        ),
        train_selected_prediction_model_distribution=distribution_from_series(
            _source_level_frame(train_df)["selected_prediction_model"]
        ),
        train_selected_prediction_model_reference_teacher_agreement_ratio=agreement_ratio_from_frame(
            _source_level_frame(train_df),
            left_column="selected_prediction_model",
            right_column="reference_teacher_model",
        ),
        train_routing_row_selected_prediction_model_distribution=distribution_from_series(
            train_df.loc[train_df["turn_stage"] == "routing", "selected_prediction_model"]
        ),
        train_routing_confidence_tier_distribution_before_filter=distribution_from_series(
            train_df_raw.loc[train_df_raw["turn_stage"] == "routing", "routing_confidence_tier"]
            if "routing_confidence_tier" in train_df_raw.columns
            else []
        ),
        train_routing_confidence_tier_distribution=distribution_from_series(
            train_df.loc[train_df["turn_stage"] == "routing", "routing_confidence_tier"]
            if "routing_confidence_tier" in train_df.columns
            else []
        ),
        routing_only_selected_model_distribution=distribution_from_series(
            train_df["selected_prediction_model"]
            if sft_stage_mode == SFT_STAGE_MODE_ROUTING_ONLY and "selected_prediction_model" in train_df.columns
            else []
        ),
        refinement_target_distribution=distribution_from_series(
            train_df.loc[train_df["turn_stage"] == "refinement", "turn3_target_type"]
            if "turn_stage" in train_df.columns and "turn3_target_type" in train_df.columns
            else []
        ),
        train_turn3_target_type_distribution=distribution_from_series(_source_level_frame(train_df)["turn3_target_type"]),
        train_turn3_trigger_reason_distribution=distribution_from_series(_source_level_frame(train_df)["turn3_trigger_reason"]),
        train_refine_ops_signature_distribution=distribution_from_series(_source_level_frame(train_df)["refine_ops_signature"]),
        train_selected_feature_tool_signature_distribution=distribution_from_series(
            _source_level_frame(train_df)["selected_feature_tool_signature"]
        ),
        train_routing_policy_source_distribution=distribution_from_series(
            _source_level_frame(train_df)["routing_policy_source"]
        ),
        train_base_prediction_source_distribution=distribution_from_series(
            _source_level_frame(train_df)["base_prediction_source"]
        ),
        train_turn3_protocol_checked_count=int(train_protocol_summary.get("turn3_protocol_checked_count", 0)),
        train_turn3_protocol_skipped_count=int(train_protocol_summary.get("turn3_protocol_skipped_count", 0)),
        train_turn3_protocol_valid_count=int(train_protocol_summary.get("turn3_protocol_valid_count", 0)),
        train_turn3_protocol_invalid_count=int(train_protocol_summary.get("turn3_protocol_invalid_count", 0)),
        train_turn3_protocol_valid_ratio=float(train_protocol_summary.get("turn3_protocol_valid_ratio", 0.0)),
        train_turn3_protocol_reason_distribution=dict(train_protocol_summary.get("turn3_protocol_reason_distribution", {})),
        val_reference_teacher_model_distribution=distribution_from_series(
            _source_level_frame(val_df)["reference_teacher_model"]
        ),
        val_selected_prediction_model_distribution=distribution_from_series(
            _source_level_frame(val_df)["selected_prediction_model"]
        ),
        val_selected_prediction_model_reference_teacher_agreement_ratio=agreement_ratio_from_frame(
            _source_level_frame(val_df),
            left_column="selected_prediction_model",
            right_column="reference_teacher_model",
        ),
        val_routing_confidence_tier_distribution=distribution_from_series(
            val_df.loc[val_df["turn_stage"] == "routing", "routing_confidence_tier"]
            if "routing_confidence_tier" in val_df.columns
            else []
        ),
        val_turn3_target_type_distribution=distribution_from_series(_source_level_frame(val_df)["turn3_target_type"]),
        val_turn3_trigger_reason_distribution=distribution_from_series(_source_level_frame(val_df)["turn3_trigger_reason"]),
        val_refine_ops_signature_distribution=distribution_from_series(_source_level_frame(val_df)["refine_ops_signature"]),
        val_selected_feature_tool_signature_distribution=distribution_from_series(
            _source_level_frame(val_df)["selected_feature_tool_signature"]
        ),
        val_routing_policy_source_distribution=distribution_from_series(
            _source_level_frame(val_df)["routing_policy_source"]
        ),
        val_base_prediction_source_distribution=distribution_from_series(
            _source_level_frame(val_df)["base_prediction_source"]
        ),
        val_turn3_protocol_checked_count=int(val_protocol_summary.get("turn3_protocol_checked_count", 0)),
        val_turn3_protocol_skipped_count=int(val_protocol_summary.get("turn3_protocol_skipped_count", 0)),
        val_turn3_protocol_valid_count=int(val_protocol_summary.get("turn3_protocol_valid_count", 0)),
        val_turn3_protocol_invalid_count=int(val_protocol_summary.get("turn3_protocol_invalid_count", 0)),
        val_turn3_protocol_valid_ratio=float(val_protocol_summary.get("turn3_protocol_valid_ratio", 0.0)),
        val_turn3_protocol_reason_distribution=dict(val_protocol_summary.get("turn3_protocol_reason_distribution", {})),
    )
    if test_df is not None and len(test_df) > 0:
        metadata_kwargs.update(
            test_turn_stage_distribution=distribution_from_series(test_df["turn_stage"]),
            test_source_sample_coverage_by_stage=source_sample_coverage_by_stage(test_df),
            test_reference_teacher_model_distribution=distribution_from_series(
                _source_level_frame(test_df)["reference_teacher_model"]
            ),
            test_selected_prediction_model_distribution=distribution_from_series(
                _source_level_frame(test_df)["selected_prediction_model"]
            ),
            test_selected_prediction_model_reference_teacher_agreement_ratio=agreement_ratio_from_frame(
                _source_level_frame(test_df),
                left_column="selected_prediction_model",
                right_column="reference_teacher_model",
            ),
            test_routing_confidence_tier_distribution=distribution_from_series(
                test_df.loc[test_df["turn_stage"] == "routing", "routing_confidence_tier"]
                if "routing_confidence_tier" in test_df.columns
                else []
            ),
            test_turn3_target_type_distribution=distribution_from_series(
                _source_level_frame(test_df)["turn3_target_type"]
            ),
            test_turn3_trigger_reason_distribution=distribution_from_series(
                _source_level_frame(test_df)["turn3_trigger_reason"]
            ),
            test_refine_ops_signature_distribution=distribution_from_series(
                _source_level_frame(test_df)["refine_ops_signature"]
            ),
            test_selected_feature_tool_signature_distribution=distribution_from_series(
                _source_level_frame(test_df)["selected_feature_tool_signature"]
            ),
            test_routing_policy_source_distribution=distribution_from_series(
                _source_level_frame(test_df)["routing_policy_source"]
            ),
            test_base_prediction_source_distribution=distribution_from_series(
                _source_level_frame(test_df)["base_prediction_source"]
            ),
            test_turn3_protocol_checked_count=int(test_protocol_summary.get("turn3_protocol_checked_count", 0)),
            test_turn3_protocol_skipped_count=int(test_protocol_summary.get("turn3_protocol_skipped_count", 0)),
            test_turn3_protocol_valid_count=int(test_protocol_summary.get("turn3_protocol_valid_count", 0)),
            test_turn3_protocol_invalid_count=int(test_protocol_summary.get("turn3_protocol_invalid_count", 0)),
            test_turn3_protocol_valid_ratio=float(test_protocol_summary.get("turn3_protocol_valid_ratio", 0.0)),
            test_turn3_protocol_reason_distribution=dict(test_protocol_summary.get("turn3_protocol_reason_distribution", {})),
        )

    write_metadata_file(output_dir, metadata_kwargs)


if __name__ == "__main__":
    main()
