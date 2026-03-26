from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from recipe.time_series_forecast.config_utils import (
    ETTH1_COVARIATE_COLUMNS,
    ETTH1_FEATURE_COLUMNS,
    ETTH1_TARGET_COLUMN,
)
from recipe.time_series_forecast.dataset_file_utils import load_jsonl_records, write_jsonl_records, write_metadata_file
from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RL_JSONL,
    HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
)
from recipe.time_series_forecast.utils import extract_data_quality


DEFAULT_CSV_PATH = Path("dataset/ETT-small/ETTh1.csv")
DEFAULT_OUTPUT_DIR = Path("dataset/ett_rl_etth1_paper_same2")

# These row counts reproduce the historical split sizes that the rest of the
# repo was built around:
# train windows = 12251 - 96 - 96 + 1 = 12060
# val windows   = 1913  - 96 - 96 + 1 = 1722
# test windows  = 3256  - 96 - 96 + 1 = 3065
DEFAULT_TRAIN_ROWS = 12251
DEFAULT_VAL_ROWS = 1913
DEFAULT_TEST_ROWS = 3256
CURRICULUM_STAGE_DEFINITIONS = {
    "easy": "low teacher error and low entropy",
    "medium": "higher teacher error while entropy stays low to medium",
    "hard": "high-entropy or stochastic windows",
}


@dataclass(frozen=True)
class SplitConfig:
    name: str
    start_row: int
    end_row: int

    @property
    def num_rows(self) -> int:
        return self.end_row - self.start_row


def build_prompt(
    historical_context: pd.DataFrame,
    *,
    lookback_window: int,
    forecast_horizon: int,
    target_column: str,
) -> str:
    if not isinstance(historical_context, pd.DataFrame):
        raise TypeError("ETTh1 paper-aligned RL prompts require a multivariate DataFrame history window.")

    required_columns = ["date", *ETTH1_FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in historical_context.columns]
    if missing_columns:
        raise ValueError(
            f"Historical ETTh1 prompt context is missing required multivariate columns: {missing_columns}"
        )
    if target_column != ETTH1_TARGET_COLUMN:
        raise ValueError(
            f"ETTh1 paper-aligned RL prompts require target_column=`{ETTH1_TARGET_COLUMN}`, got `{target_column}`"
        )

    value_lines = "\n".join(
        f"{row.date} "
        + " ".join(f"{column}={float(getattr(row, column)):.4f}" for column in ETTH1_FEATURE_COLUMNS)
        for row in historical_context.itertuples(index=False)
    )

    return (
        "[Task] Multivariate time-series forecasting.\n"
        f"Target Column: {target_column}\n"
        f"Observed Covariates: {', '.join(ETTH1_COVARIATE_COLUMNS)}\n"
        f"Lookback Window: {lookback_window}\n"
        f"Forecast Horizon: {forecast_horizon}\n"
        "Requirements:\n"
        "1) Extract feature evidence before selecting a forecasting model.\n"
        "2) After diagnostics are complete, choose one model from the enabled experts and then predict.\n"
        "3) After prediction, refine the selected model forecast only if needed and follow the required output protocol with <think>...</think><answer>...</answer>.\n"
        "Historical Data:\n"
        f"{value_lines}"
    )


def build_ground_truth(forecast_df: pd.DataFrame, *, target_column: str) -> str:
    return "\n".join(
        f"{row.date} {float(getattr(row, target_column)):.4f}"
        for row in forecast_df.itertuples(index=False)
    )


def build_split_configs(
    *,
    total_rows: int,
    train_rows: int,
    val_rows: int,
    test_rows: int,
) -> list[SplitConfig]:
    if train_rows + val_rows + test_rows != total_rows:
        raise ValueError(
            f"Split row counts must sum to total rows. "
            f"Got train={train_rows}, val={val_rows}, test={test_rows}, total={total_rows}."
        )

    return [
        SplitConfig(name="train", start_row=0, end_row=train_rows),
        SplitConfig(name="val", start_row=train_rows, end_row=train_rows + val_rows),
        SplitConfig(name="test", start_row=train_rows + val_rows, end_row=total_rows),
    ]


def compute_normalized_permutation_entropy(
    values: Iterable[float],
    *,
    order: int = 3,
    delay: int = 1,
) -> float:
    series = np.asarray(list(values), dtype=float)
    window_size = (order - 1) * delay + 1
    if order < 2 or delay < 1 or len(series) < window_size:
        return 0.0

    pattern_counts: Counter[tuple[int, ...]] = Counter()
    total = 0
    for start in range(len(series) - window_size + 1):
        window = series[start : start + window_size : delay]
        pattern = tuple(np.argsort(window, kind="mergesort"))
        pattern_counts[pattern] += 1
        total += 1

    if total <= 0:
        return 0.0

    probs = np.asarray(list(pattern_counts.values()), dtype=float) / float(total)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = math.log(math.factorial(order))
    if max_entropy <= 0:
        return 0.0
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def _compute_quality_issue_flag(values: list[float]) -> bool:
    quality = extract_data_quality(values)
    return bool(
        float(quality.get("quality_quantization_score", 0.0)) >= 0.25
        or float(quality.get("quality_saturation_ratio", 0.0)) >= 0.08
        or float(quality.get("quality_constant_channel_ratio", 0.0)) > 0.0
        or float(quality.get("quality_dropout_ratio", 0.0)) > 0.0
    )


def _quantile_thresholds(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite_values:
        return (None, None)
    array = np.asarray(finite_values, dtype=float)
    return (float(np.quantile(array, 1.0 / 3.0)), float(np.quantile(array, 2.0 / 3.0)))


def _band_value(value: Optional[float], thresholds: tuple[Optional[float], Optional[float]]) -> str:
    low, high = thresholds
    if value is None or not np.isfinite(value):
        return "unknown"
    if low is None or high is None:
        return "medium"
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def _resolve_difficulty_stage(error_band: str, entropy_band: str) -> str:
    known_error_band = error_band in {"low", "medium", "high"}
    known_entropy_band = entropy_band in {"low", "medium", "high"}
    if not known_error_band and not known_entropy_band:
        return "unknown"
    if entropy_band == "high":
        return "hard"
    if error_band == "low" and entropy_band == "low":
        return "easy"
    if known_error_band and known_entropy_band:
        return "medium"
    return "unknown"


def _resolve_reference_teacher_error(metadata: dict[str, Any]) -> Optional[float]:
    candidate_keys = (
        "reference_teacher_error",
        "best_orig_mse",
        "teacher_eval_best_orig_mse",
        "orig_mse",
    )
    for key in candidate_keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(scalar):
            return scalar
    return None


def load_teacher_metadata(path: str | Path | None) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Teacher metadata jsonl not found: {metadata_path}")

    metadata_by_index: dict[int, dict[str, Any]] = {}
    for record in load_jsonl_records(metadata_path):
        sample_index = int(record.get("sample_index", record.get("index", -1)))
        if sample_index < 0:
            continue
        metadata_by_index[sample_index] = record
    return metadata_by_index


def compute_teacher_metadata_coverage(
    *,
    num_samples: int,
    teacher_metadata_by_index: Optional[dict[int, dict[str, Any]]],
) -> float:
    if num_samples <= 0:
        return 1.0
    metadata = teacher_metadata_by_index or {}
    covered = sum(1 for sample_index in range(num_samples) if sample_index in metadata)
    return float(covered / num_samples)


def iter_split_samples(
    df: pd.DataFrame,
    split: SplitConfig,
    *,
    lookback_window: int,
    forecast_horizon: int,
    target_column: str,
    teacher_metadata_by_index: Optional[dict[int, dict[str, Any]]] = None,
) -> Iterable[dict]:
    split_df = df.iloc[split.start_row : split.end_row].reset_index(drop=True)
    window_span = lookback_window + forecast_horizon
    num_samples = len(split_df) - window_span + 1
    if num_samples <= 0:
        raise ValueError(
            f"Split {split.name} is too short for lookback={lookback_window} and horizon={forecast_horizon}. "
            f"Rows: {len(split_df)}"
        )

    teacher_metadata_by_index = teacher_metadata_by_index or {}
    staged_records: list[dict[str, Any]] = []
    entropy_values: list[float] = []
    reference_teacher_errors: list[float] = []

    for sample_index in range(num_samples):
        hist_end = sample_index + lookback_window
        forecast_end = hist_end + forecast_horizon
        historical = split_df.iloc[sample_index:hist_end]
        forecast = split_df.iloc[hist_end:forecast_end]
        historical_values = historical[target_column].tolist()

        prompt_text = build_prompt(
            historical,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            target_column=target_column,
        )
        ground_truth = build_ground_truth(forecast, target_column=target_column)

        teacher_metadata = dict(teacher_metadata_by_index.get(sample_index, {}))
        reference_teacher_error = _resolve_reference_teacher_error(teacher_metadata)
        normalized_entropy = compute_normalized_permutation_entropy(historical_values)
        quality_issue_flag = _compute_quality_issue_flag(historical_values)

        entropy_values.append(normalized_entropy)
        if reference_teacher_error is not None:
            reference_teacher_errors.append(reference_teacher_error)

        staged_records.append(
            {
                "index": sample_index,
                "uid": f"etth1-{split.name}-{sample_index:05d}",
                "agent_name": "time_series_forecast_agent",
                "data_source": "ETTh1",
                "target_column": target_column,
                "lookback_window": lookback_window,
                "forecast_horizon": forecast_horizon,
                "split": split.name,
                "raw_prompt": [{"role": "user", "content": prompt_text}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "reference_teacher_error": reference_teacher_error,
                "normalized_permutation_entropy": normalized_entropy,
                "offline_best_model": teacher_metadata.get("reference_teacher_model") or teacher_metadata.get("best_model"),
                "offline_margin": teacher_metadata.get("score_margin", teacher_metadata.get("teacher_eval_score_margin")),
                "quality_issue_flag": bool(quality_issue_flag),
            }
        )

    error_thresholds = _quantile_thresholds(reference_teacher_errors)
    entropy_thresholds = _quantile_thresholds(entropy_values)

    for record in staged_records:
        error_band = _band_value(record["reference_teacher_error"], error_thresholds)
        entropy_band = _band_value(record["normalized_permutation_entropy"], entropy_thresholds)
        difficulty_stage = _resolve_difficulty_stage(error_band, entropy_band)
        curriculum_band = f"{error_band}/{entropy_band}"

        record["reference_teacher_error_band"] = error_band
        record["normalized_permutation_entropy_band"] = entropy_band
        record["difficulty_stage"] = difficulty_stage
        record["curriculum_stage"] = difficulty_stage
        record["curriculum_band"] = curriculum_band
        yield record


def build_train_stage_slices(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    stage1 = [record for record in records if record.get("curriculum_stage") == "easy"]
    stage12 = [record for record in records if record.get("curriculum_stage") in {"easy", "medium"}]
    stage123 = list(records)
    return {
        "train_stage1": stage1,
        "train_stage12": stage12,
        "train_stage123": stage123,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild ETTh1 RL jsonl datasets from ETTh1.csv.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH), help="Path to ETTh1.csv.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output RL dataset directory.")
    parser.add_argument("--lookback-window", type=int, default=96, help="Historical window length.")
    parser.add_argument("--forecast-horizon", type=int, default=96, help="Forecast horizon length.")
    parser.add_argument("--target-column", default="OT", help="Forecast target column.")
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS, help="Number of raw rows in train split.")
    parser.add_argument("--val-rows", type=int, default=DEFAULT_VAL_ROWS, help="Number of raw rows in val split.")
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS, help="Number of raw rows in test split.")
    parser.add_argument("--train-teacher-metadata-jsonl", default="", help="Optional teacher-metadata jsonl for train split.")
    parser.add_argument("--val-teacher-metadata-jsonl", default="", help="Optional teacher-metadata jsonl for val split.")
    parser.add_argument("--test-teacher-metadata-jsonl", default="", help="Optional teacher-metadata jsonl for test split.")
    parser.add_argument(
        "--min-teacher-metadata-coverage",
        type=float,
        default=0.95,
        help=(
            "Minimum acceptable teacher-metadata coverage ratio when a teacher-metadata jsonl is provided. "
            "Set to 0 to disable the check."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"ETTh1 csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"date", args.target_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    splits = build_split_configs(
        total_rows=len(df),
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        test_rows=args.test_rows,
    )

    teacher_metadata_paths = {
        "train": args.train_teacher_metadata_jsonl,
        "val": args.val_teacher_metadata_jsonl,
        "test": args.test_teacher_metadata_jsonl,
    }
    teacher_metadata_by_split = {
        split_name: load_teacher_metadata(path) if path else {}
        for split_name, path in teacher_metadata_paths.items()
    }

    metadata: dict[str, object] = {
        "dataset_kind": DATASET_KIND_RL_JSONL,
        "pipeline_stage": "curriculum_rl" if any(teacher_metadata_paths.values()) else "base_rl",
        "curriculum_policy": "teacher_error_entropy_two_axis",
        "curriculum_stage_definitions": CURRICULUM_STAGE_DEFINITIONS,
        "task_type": "multivariate time-series forecasting",
        "historical_data_protocol": HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
        "observed_feature_columns": list(ETTH1_FEATURE_COLUMNS),
        "observed_covariates": list(ETTH1_COVARIATE_COLUMNS),
        "model_input_width": len(ETTH1_FEATURE_COLUMNS),
        "source_csv": str(csv_path),
        "target_column": args.target_column,
        "lookback_window": args.lookback_window,
        "forecast_horizon": args.forecast_horizon,
        "total_rows": len(df),
        "splits": [asdict(split) | {"num_rows": split.num_rows} for split in splits],
        "teacher_metadata_paths": teacher_metadata_paths,
    }

    for split in splits:
        split_df = df.iloc[split.start_row : split.end_row].reset_index(drop=True)
        window_span = args.lookback_window + args.forecast_horizon
        num_samples = len(split_df) - window_span + 1
        metadata_for_split = teacher_metadata_by_split.get(split.name)
        coverage_ratio = compute_teacher_metadata_coverage(
            num_samples=num_samples,
            teacher_metadata_by_index=metadata_for_split,
        )
        coverage_count = sum(
            1 for sample_index in range(max(num_samples, 0)) if sample_index in (metadata_for_split or {})
        )
        if teacher_metadata_paths.get(split.name):
            print(
                f"[RL-DATA] split={split.name} teacher metadata coverage: "
                f"{coverage_count}/{num_samples} ({coverage_ratio:.2%})"
            )
            if args.min_teacher_metadata_coverage > 0 and coverage_ratio < args.min_teacher_metadata_coverage:
                raise ValueError(
                    f"Teacher metadata coverage for split={split.name} is too low: "
                    f"{coverage_count}/{num_samples} ({coverage_ratio:.2%}) < "
                    f"{args.min_teacher_metadata_coverage:.2%}. "
                    f"Provided file: {teacher_metadata_paths.get(split.name)}"
                )

        records = list(
            iter_split_samples(
                df,
                split,
                lookback_window=args.lookback_window,
                forecast_horizon=args.forecast_horizon,
                target_column=args.target_column,
                teacher_metadata_by_index=metadata_for_split,
            )
        )
        output_path = output_dir / f"{split.name}.jsonl"
        count = write_jsonl_records(output_path, records)

        entropy_values = [float(record["normalized_permutation_entropy"]) for record in records]
        reference_teacher_errors = [
            float(record["reference_teacher_error"])
            for record in records
            if record.get("reference_teacher_error") is not None
        ]
        metadata[f"{split.name}_samples"] = count
        metadata[f"{split.name}_teacher_metadata_coverage_ratio"] = coverage_ratio
        metadata[f"{split.name}_teacher_metadata_coverage_count"] = coverage_count
        metadata[f"{split.name}_entropy_thresholds"] = _quantile_thresholds(entropy_values)
        metadata[f"{split.name}_reference_teacher_error_thresholds"] = _quantile_thresholds(reference_teacher_errors)
        metadata[f"{split.name}_curriculum_stage_distribution"] = dict(
            sorted(Counter(str(record.get("curriculum_stage", "unknown")) for record in records).items())
        )
        print(
            f"[RL-DATA] split={split.name} rows={split.num_rows} samples={count} -> {output_path}"
        )

        if split.name == "train":
            staged_slices = build_train_stage_slices(records)
            for stage_name, stage_records in staged_slices.items():
                stage_output_path = output_dir / f"{stage_name}.jsonl"
                stage_count = write_jsonl_records(stage_output_path, stage_records)
                metadata[f"{stage_name}_samples"] = stage_count
                metadata[f"{stage_name}_stage_distribution"] = dict(
                    sorted(Counter(str(record.get("curriculum_stage", "unknown")) for record in stage_records).items())
                )
                print(f"[RL-DATA] {stage_name} samples={stage_count} -> {stage_output_path}")

    metadata_path = write_metadata_file(output_dir, metadata)
    print(f"[RL-DATA] wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
