from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from recipe.time_series_forecast.dataset_file_utils import (
    load_jsonl_records,
    write_jsonl_records,
    write_metadata_file,
)
from recipe.time_series_forecast.config_utils import (
    ETTH1_COVARIATE_COLUMNS,
    ETTH1_FEATURE_COLUMNS,
    ETTH1_TARGET_COLUMN,
    get_default_lengths,
)
from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RL_JSONL,
    DATASET_KIND_TEACHER_CURATED_SFT,
    HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
    require_multivariate_etth1_metadata,
    validate_sibling_metadata,
)


DEFAULT_OUTPUT_DIR = Path("dataset/routing_bootstrap_v16")
DEFAULT_TRAIN_JSONL = Path("dataset/ett_rl_etth1_paper_same2/train.jsonl")
DEFAULT_VAL_JSONL = Path("dataset/ett_rl_etth1_paper_same2/val.jsonl")
DEFAULT_TEST_JSONL = Path("dataset/ett_rl_etth1_paper_same2/test.jsonl")
DEFAULT_TRAIN_TEACHER_EVAL_JSONL = Path("dataset/ett_sft_etth1_runtime_teacher200_refteacher_v15/train_teacher_eval.jsonl")
DEFAULT_VAL_TEACHER_EVAL_JSONL = Path("dataset/ett_sft_etth1_runtime_teacher200_refteacher_v15/val_teacher_eval.jsonl")
DEFAULT_TEST_TEACHER_EVAL_JSONL = Path("dataset/ett_sft_etth1_runtime_teacher200_refteacher_v15/test_teacher_eval.jsonl")
SUPPORTED_ROUTING_MODELS = ("patchtst", "itransformer", "arima", "chronos2")
DEFAULT_LOOKBACK_WINDOW, DEFAULT_FORECAST_HORIZON = get_default_lengths()


def _normalize_model_name(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in SUPPORTED_ROUTING_MODELS else ""


def _coerce_score_details(payload: Any) -> dict[str, dict[str, float]]:
    if not isinstance(payload, dict):
        return {}
    output: dict[str, dict[str, float]] = {}
    for model_name, item in payload.items():
        normalized_name = _normalize_model_name(model_name)
        if not normalized_name or not isinstance(item, dict):
            continue
        model_payload: dict[str, float] = {}
        for key in ("score", "orig_mse", "orig_mae", "norm_mse", "norm_mae"):
            try:
                value = float(item.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                model_payload[key] = value
        if model_payload:
            output[normalized_name] = model_payload
    return output


def _sorted_model_errors(evaluation: dict[str, Any]) -> list[tuple[str, float]]:
    score_details = _coerce_score_details(evaluation.get("model_score_details"))
    ranked: list[tuple[str, float]] = []
    for model_name, payload in score_details.items():
        error = payload.get("orig_mse")
        if error is None:
            continue
        ranked.append((model_name, float(error)))
    ranked.sort(key=lambda item: (item[1], item[0]))
    return ranked


def _route_info_from_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    ranked_errors = _sorted_model_errors(evaluation)
    if len(ranked_errors) < 2:
        raise ValueError(
            f"Teacher evaluation is missing comparable per-model orig_mse values for sample_index="
            f"{evaluation.get('sample_index')}"
        )

    best_model, best_error = ranked_errors[0]
    second_best_model, second_best_error = ranked_errors[1]
    margin_abs = float(second_best_error - best_error)
    margin_rel = float(margin_abs / second_best_error) if second_best_error > 0 else 0.0
    return {
        "route_best_model": best_model,
        "route_second_best_model": second_best_model,
        "route_best_error": float(best_error),
        "route_second_best_error": float(second_best_error),
        "route_margin_abs": margin_abs,
        "route_margin_rel": margin_rel,
        "route_top2_models": [model_name for model_name, _ in ranked_errors[:2]],
        "teacher_eval_score_details": evaluation.get("model_score_details") or {},
    }


def _selection_key(item: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(item.get("route_margin_rel") or 0.0),
        float(item.get("route_margin_abs") or 0.0),
        float(item.get("teacher_eval_score_margin") or 0.0),
        -int(item.get("index", item.get("sample_index", -1)) or -1),
    )


def _build_enriched_record(source_record: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    route_info = _route_info_from_evaluation(evaluation)
    record = dict(source_record)
    record.update(
        {
            "reference_teacher_model": route_info["route_best_model"],
            "offline_best_model": route_info["route_best_model"],
            "teacher_eval_best_score": evaluation.get("best_score"),
            "teacher_eval_second_best_model": route_info["route_second_best_model"],
            "teacher_eval_second_best_score": evaluation.get("second_best_score"),
            "teacher_eval_score_margin": evaluation.get("score_margin"),
            "teacher_eval_scores": evaluation.get("model_scores") or {},
            "teacher_eval_score_details": route_info["teacher_eval_score_details"],
            "teacher_eval_best_orig_mse": route_info["route_best_error"],
            "teacher_eval_best_orig_mae": evaluation.get("best_orig_mae"),
            "teacher_eval_best_norm_mse": evaluation.get("best_norm_mse"),
            "teacher_eval_best_norm_mae": evaluation.get("best_norm_mae"),
            "teacher_prediction_source": evaluation.get("teacher_prediction_source") or "reference_teacher",
            "teacher_prediction_text": (
                evaluation.get("teacher_prediction_text")
                or evaluation.get("reference_teacher_prediction_text")
                or ""
            ),
            "selected_prediction_model": route_info["route_best_model"],
            **route_info,
        }
    )
    return record


def _balanced_select(records: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    if target_count <= 0 or not records:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {model_name: [] for model_name in SUPPORTED_ROUTING_MODELS}
    for item in records:
        grouped[_normalize_model_name(item.get("route_best_model"))].append(item)
    for items in grouped.values():
        items.sort(key=_selection_key, reverse=True)

    target_per_model = max(1, target_count // len(SUPPORTED_ROUTING_MODELS))
    remainder = max(0, target_count - target_per_model * len(SUPPORTED_ROUTING_MODELS))
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []

    for model_idx, model_name in enumerate(SUPPORTED_ROUTING_MODELS):
        quota = target_per_model + (1 if model_idx < remainder else 0)
        items = grouped.get(model_name, [])
        selected.extend(items[:quota])
        leftovers.extend(items[quota:])

    if len(selected) < target_count:
        leftovers.sort(key=_selection_key, reverse=True)
        selected.extend(leftovers[: max(0, target_count - len(selected))])

    selected = selected[:target_count]
    selected.sort(key=lambda item: int(item.get("index", item.get("sample_index", -1)) or -1))
    return selected


def _assign_bootstrap_confidence_tier(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[_normalize_model_name(item.get("route_best_model"))].append(item)

    annotated: list[dict[str, Any]] = []
    for model_name, items in grouped.items():
        del model_name
        ranked = sorted(items, key=_selection_key, reverse=True)
        high_cut = max(1, math.ceil(len(ranked) * 0.5))
        for idx, item in enumerate(ranked):
            updated = dict(item)
            updated["route_bootstrap_confidence_tier"] = "high" if idx < high_cut else "mid"
            updated["route_bootstrap_rank_within_model"] = int(idx)
            annotated.append(updated)
    annotated.sort(key=lambda item: int(item.get("index", item.get("sample_index", -1)) or -1))
    return annotated


def _select_split_records(
    *,
    source_records: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    target_count: int,
    top_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_by_index = {int(record.get("index", -1)): record for record in source_records}
    enriched_by_model: dict[str, list[dict[str, Any]]] = {model_name: [] for model_name in SUPPORTED_ROUTING_MODELS}

    for evaluation in evaluations:
        sample_index = int(evaluation.get("sample_index", -1))
        source_record = source_by_index.get(sample_index)
        if source_record is None:
            continue
        enriched = _build_enriched_record(source_record, evaluation)
        winner = _normalize_model_name(enriched.get("route_best_model"))
        if not winner:
            continue
        enriched_by_model[winner].append(enriched)

    candidate_pool: list[dict[str, Any]] = []
    desired_per_model = max(1, math.ceil(target_count / len(SUPPORTED_ROUTING_MODELS)))
    pool_counts: dict[str, int] = {}
    raw_counts: dict[str, int] = {}
    for model_name in SUPPORTED_ROUTING_MODELS:
        ranked = sorted(enriched_by_model.get(model_name, []), key=_selection_key, reverse=True)
        raw_counts[model_name] = len(ranked)
        top_count = min(len(ranked), max(desired_per_model, math.ceil(len(ranked) * float(top_fraction))))
        pool_counts[model_name] = int(top_count)
        candidate_pool.extend(ranked[:top_count])

    selected = _balanced_select(candidate_pool, target_count=target_count)
    selected = _assign_bootstrap_confidence_tier(selected)
    metadata = {
        "winner_distribution_before_filter": raw_counts,
        "candidate_pool_distribution": pool_counts,
        "selected_distribution": dict(Counter(_normalize_model_name(item.get("route_best_model")) for item in selected)),
        "selected_confidence_distribution": dict(
            Counter(str(item.get("route_bootstrap_confidence_tier") or "") for item in selected)
        ),
        "selected_mean_margin_abs": (
            sum(float(item.get("route_margin_abs") or 0.0) for item in selected) / max(len(selected), 1)
        ),
        "selected_mean_margin_rel": (
            sum(float(item.get("route_margin_rel") or 0.0) for item in selected) / max(len(selected), 1)
        ),
    }
    return selected, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced high-confidence ETTh1 routing bootstrap dataset.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--train-teacher-eval-jsonl", type=Path, default=DEFAULT_TRAIN_TEACHER_EVAL_JSONL)
    parser.add_argument("--val-teacher-eval-jsonl", type=Path, default=DEFAULT_VAL_TEACHER_EVAL_JSONL)
    parser.add_argument("--test-teacher-eval-jsonl", type=Path, default=DEFAULT_TEST_TEACHER_EVAL_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-target-samples", type=int, default=512)
    parser.add_argument("--val-target-samples", type=int, default=128)
    parser.add_argument("--test-target-samples", type=int, default=256)
    parser.add_argument("--top-fraction", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = (
        ("train", args.train_jsonl, args.train_teacher_eval_jsonl, int(args.train_target_samples)),
        ("val", args.val_jsonl, args.val_teacher_eval_jsonl, int(args.val_target_samples)),
        ("test", args.test_jsonl, args.test_teacher_eval_jsonl, int(args.test_target_samples)),
    )

    metadata: dict[str, Any] = {
        "dataset_kind": DATASET_KIND_TEACHER_CURATED_SFT,
        "pipeline_stage": "routing_bootstrap_v16",
        "task_type": "multivariate time-series forecasting",
        "historical_data_protocol": HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
        "observed_feature_columns": list(ETTH1_FEATURE_COLUMNS),
        "observed_covariates": list(ETTH1_COVARIATE_COLUMNS),
        "model_input_width": len(ETTH1_FEATURE_COLUMNS),
        "target_column": ETTH1_TARGET_COLUMN,
        "lookback_window": DEFAULT_LOOKBACK_WINDOW,
        "forecast_horizon": DEFAULT_FORECAST_HORIZON,
        "selection_method": "per-winner-top-fraction-balanced-route-bootstrap",
        "top_fraction": float(args.top_fraction),
        "teacher_models": list(SUPPORTED_ROUTING_MODELS),
    }

    for split_name, source_jsonl, _teacher_eval_jsonl, target_count in split_specs:
        if target_count <= 0:
            continue
        source_metadata, source_metadata_path = validate_sibling_metadata(
            source_jsonl,
            expected_kind=DATASET_KIND_RL_JSONL,
        )
        require_multivariate_etth1_metadata(source_metadata, metadata_path=source_metadata_path)
        metadata[f"source_{split_name}_metadata_path"] = str(source_metadata_path)
        metadata[f"source_{split_name}_pipeline_stage"] = str(source_metadata.get("pipeline_stage") or "")

    for split_name, source_jsonl, teacher_eval_jsonl, target_count in split_specs:
        if target_count <= 0:
            continue
        source_records = load_jsonl_records(source_jsonl)
        evaluations = load_jsonl_records(teacher_eval_jsonl)
        selected_records, split_metadata = _select_split_records(
            source_records=source_records,
            evaluations=evaluations,
            target_count=target_count,
            top_fraction=float(args.top_fraction),
        )
        write_jsonl_records(output_dir / f"{split_name}.jsonl", selected_records)
        metadata[f"{split_name}_source_samples"] = len(source_records)
        metadata[f"{split_name}_teacher_eval_records"] = len(evaluations)
        metadata[f"{split_name}_selected_records"] = len(selected_records)
        for key, value in split_metadata.items():
            metadata[f"{split_name}_{key}"] = value
        print(
            f"[routing-bootstrap] split={split_name} selected={len(selected_records)} "
            f"dist={split_metadata['selected_distribution']} "
            f"conf={split_metadata['selected_confidence_distribution']}"
        )

    write_metadata_file(output_dir, metadata)


if __name__ == "__main__":
    main()
