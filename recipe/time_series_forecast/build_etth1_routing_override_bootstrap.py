from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from recipe.time_series_forecast.config_utils import (
    ETTH1_COVARIATE_COLUMNS,
    ETTH1_FEATURE_COLUMNS,
    ETTH1_TARGET_COLUMN,
    get_default_lengths,
)
from recipe.time_series_forecast.dataset_file_utils import (
    load_jsonl_records,
    write_jsonl_records,
    write_metadata_file,
)
from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RL_JSONL,
    DATASET_KIND_TEACHER_CURATED_SFT,
    HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
    require_multivariate_etth1_metadata,
    validate_sibling_metadata,
)


DEFAULT_OUTPUT_DIR = Path("dataset/routing_override_bootstrap_v17")
DEFAULT_TRAIN_JSONL = Path("dataset/ett_rl_etth1_paper_same2/train.jsonl")
DEFAULT_VAL_JSONL = Path("dataset/ett_rl_etth1_paper_same2/val.jsonl")
DEFAULT_TEST_JSONL = Path("dataset/ett_rl_etth1_paper_same2/test.jsonl")
DEFAULT_TRAIN_TEACHER_EVAL_JSONL = Path("dataset/ett_sft_etth1_runtime_teacher200_refteacher_v15/train_teacher_eval.jsonl")
DEFAULT_VAL_TEACHER_EVAL_JSONL = Path("dataset/ett_sft_etth1_runtime_teacher200_refteacher_v15/val_teacher_eval.jsonl")
DEFAULT_TEST_TEACHER_EVAL_JSONL = Path("dataset/ett_sft_etth1_runtime_teacher200_refteacher_v15/test_teacher_eval.jsonl")

SUPPORTED_ROUTING_MODELS = ("patchtst", "itransformer", "arima", "chronos2")
DEFAULT_EXPERT = "itransformer"
DEFAULT_OVERRIDE_TOP_FRACTION = 0.35
DEFAULT_KEEP_DEFAULT_RATIO = 0.70
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
        values: dict[str, float] = {}
        for key in ("score", "orig_mse", "orig_mae", "norm_mse", "norm_mae"):
            try:
                value = float(item.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values[key] = value
        if values:
            output[normalized_name] = values
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


def _selection_key(item: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(item.get("improvement_vs_default_rel") or 0.0),
        float(item.get("improvement_vs_default") or 0.0),
        float(item.get("route_margin_rel") or 0.0),
        -int(item.get("index", item.get("sample_index", -1)) or -1),
    )


def _take_evenly_spaced(items: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    if target_count <= 0 or not items:
        return []
    if len(items) <= target_count:
        return list(items)
    if target_count == 1:
        return [items[len(items) // 2]]
    positions = sorted({round(i * (len(items) - 1) / (target_count - 1)) for i in range(target_count)})
    return [items[position] for position in positions]


def _route_info_from_evaluation(
    evaluation: dict[str, Any],
    *,
    default_expert: str,
) -> dict[str, Any]:
    ranked_errors = _sorted_model_errors(evaluation)
    if len(ranked_errors) < 2:
        raise ValueError(
            "Teacher evaluation is missing comparable per-model orig_mse values for "
            f"sample_index={evaluation.get('sample_index')}"
        )

    error_by_name = {model_name: float(error) for model_name, error in ranked_errors}
    if default_expert not in error_by_name:
        raise ValueError(
            f"default_expert={default_expert!r} is missing from model_score_details for "
            f"sample_index={evaluation.get('sample_index')}"
        )

    best_model, best_error = ranked_errors[0]
    second_best_model, second_best_error = ranked_errors[1]
    default_error = float(error_by_name[default_expert])
    improvement_vs_default = float(default_error - best_error)
    improvement_vs_default_rel = (
        float(improvement_vs_default / default_error) if default_error > 0 else 0.0
    )
    margin_abs = float(second_best_error - best_error)
    margin_rel = float(margin_abs / second_best_error) if second_best_error > 0 else 0.0
    return {
        "best_model": best_model,
        "best_error": float(best_error),
        "route_best_model": best_model,
        "route_best_error": float(best_error),
        "route_second_best_model": second_best_model,
        "route_second_best_error": float(second_best_error),
        "route_margin_abs": margin_abs,
        "route_margin_rel": margin_rel,
        "route_top2_models": [model_name for model_name, _ in ranked_errors[:2]],
        "default_expert": default_expert,
        "default_error": default_error,
        "improvement_vs_default": improvement_vs_default,
        "improvement_vs_default_rel": improvement_vs_default_rel,
        "teacher_eval_score_details": evaluation.get("model_score_details") or {},
    }


def _build_enriched_record(
    source_record: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    default_expert: str,
) -> dict[str, Any]:
    route_info = _route_info_from_evaluation(evaluation, default_expert=default_expert)
    record = dict(source_record)
    record.update(
        {
            "reference_teacher_model": route_info["best_model"],
            "offline_best_model": route_info["best_model"],
            "selected_prediction_model": default_expert,
            "teacher_eval_best_score": evaluation.get("best_score"),
            "teacher_eval_second_best_model": route_info["route_second_best_model"],
            "teacher_eval_second_best_score": evaluation.get("second_best_score"),
            "teacher_eval_score_margin": evaluation.get("score_margin"),
            "teacher_eval_scores": evaluation.get("model_scores") or {},
            "teacher_eval_score_details": route_info["teacher_eval_score_details"],
            "teacher_eval_best_orig_mse": route_info["best_error"],
            "teacher_eval_best_orig_mae": evaluation.get("best_orig_mae"),
            "teacher_eval_best_norm_mse": evaluation.get("best_norm_mse"),
            "teacher_eval_best_norm_mae": evaluation.get("best_norm_mae"),
            "teacher_prediction_source": evaluation.get("teacher_prediction_source") or "reference_teacher",
            "teacher_prediction_text": (
                evaluation.get("teacher_prediction_text")
                or evaluation.get("reference_teacher_prediction_text")
                or ""
            ),
            "routing_label_source": "default_override",
            "route_decision": "keep_default",
            "route_label": "keep_default",
            "route_override_model": "",
            "route_label_confidence": "mid",
            **route_info,
        }
    )
    return record


def _label_records_for_override(
    records: list[dict[str, Any]],
    *,
    default_expert: str,
    override_top_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labeled_records = [dict(record) for record in records]
    overrides_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    override_thresholds: dict[str, float] = {}

    for record in labeled_records:
        best_model = _normalize_model_name(record.get("best_model"))
        if not best_model or best_model == default_expert:
            continue
        if float(record.get("improvement_vs_default") or 0.0) <= 0.0:
            continue
        overrides_by_model[best_model].append(record)

    for model_name, items in overrides_by_model.items():
        ranked = sorted(items, key=_selection_key, reverse=True)
        select_count = max(1, math.ceil(len(ranked) * float(override_top_fraction)))
        selected = ranked[:select_count]
        threshold = float(selected[-1].get("improvement_vs_default_rel") or 0.0) if selected else 0.0
        override_thresholds[model_name] = threshold
        high_cut = max(1, math.ceil(len(selected) * 0.5))
        selected_ids = {id(item) for item in selected}
        for idx, item in enumerate(selected):
            item["route_decision"] = "override"
            item["route_override_model"] = model_name
            item["route_label"] = f"override_to_{model_name}"
            item["route_label_confidence"] = "high" if idx < high_cut else "mid"
            item["selected_prediction_model"] = model_name
        for item in ranked:
            if id(item) in selected_ids:
                continue
            item["route_decision"] = "keep_default"
            item["route_override_model"] = ""
            item["route_label"] = "keep_default"
            item["route_label_confidence"] = "mid"
            item["selected_prediction_model"] = default_expert

    for record in labeled_records:
        if str(record.get("route_label") or "").strip():
            continue
        record["route_decision"] = "keep_default"
        record["route_override_model"] = ""
        record["route_label"] = "keep_default"
        record["route_label_confidence"] = "high" if _normalize_model_name(record.get("best_model")) == default_expert else "mid"
        record["selected_prediction_model"] = default_expert

    metadata = {
        "override_candidate_distribution_before_selection": {
            model_name: int(len(items)) for model_name, items in sorted(overrides_by_model.items())
        },
        "override_threshold_rel_by_model": {
            model_name: float(threshold) for model_name, threshold in sorted(override_thresholds.items())
        },
        "route_label_distribution_before_sampling": dict(
            Counter(str(item.get("route_label") or "") for item in labeled_records)
        ),
    }
    return labeled_records, metadata


def _select_override_records(
    override_records: list[dict[str, Any]],
    *,
    target_count: int,
) -> list[dict[str, Any]]:
    if target_count <= 0 or not override_records:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in override_records:
        grouped[str(item.get("route_override_model") or "")].append(item)
    for items in grouped.values():
        items.sort(key=_selection_key, reverse=True)

    present_models = [model_name for model_name in SUPPORTED_ROUTING_MODELS if grouped.get(model_name)]
    if not present_models:
        return []
    target_per_model = max(1, target_count // len(present_models))
    remainder = max(0, target_count - target_per_model * len(present_models))
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []

    for model_idx, model_name in enumerate(present_models):
        quota = target_per_model + (1 if model_idx < remainder else 0)
        items = grouped[model_name]
        selected.extend(items[:quota])
        leftovers.extend(items[quota:])

    if len(selected) < target_count:
        leftovers.sort(key=_selection_key, reverse=True)
        selected.extend(leftovers[: max(0, target_count - len(selected))])

    selected = selected[:target_count]
    selected.sort(key=lambda item: int(item.get("index", item.get("sample_index", -1)) or -1))
    return selected


def _select_split_records(
    *,
    source_records: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    target_count: int,
    default_expert: str,
    keep_default_ratio: float,
    override_top_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_by_index = {int(record.get("index", -1)): record for record in source_records}
    enriched: list[dict[str, Any]] = []
    skipped_missing_source = 0

    for evaluation in evaluations:
        sample_index = int(evaluation.get("sample_index", -1))
        source_record = source_by_index.get(sample_index)
        if source_record is None:
            skipped_missing_source += 1
            continue
        enriched.append(_build_enriched_record(source_record, evaluation, default_expert=default_expert))

    labeled, label_metadata = _label_records_for_override(
        enriched,
        default_expert=default_expert,
        override_top_fraction=override_top_fraction,
    )
    keep_candidates = [item for item in labeled if str(item.get("route_label") or "") == "keep_default"]
    override_candidates = [item for item in labeled if str(item.get("route_label") or "").startswith("override_to_")]
    keep_candidates.sort(
        key=lambda item: (
            _normalize_model_name(item.get("best_model")) != default_expert,
            float(item.get("improvement_vs_default_rel") or 0.0),
            int(item.get("index", item.get("sample_index", -1)) or -1),
        )
    )

    target_override_count = min(
        len(override_candidates),
        max(
            len({str(item.get("route_override_model") or "") for item in override_candidates}),
            int(round(target_count * max(0.0, 1.0 - float(keep_default_ratio)))),
        ),
    )
    selected_override = _select_override_records(override_candidates, target_count=target_override_count)
    remaining_target = max(target_count - len(selected_override), 0)
    selected_keep = _take_evenly_spaced(keep_candidates, remaining_target)

    selected_keys = {
        (int(item.get("index", item.get("sample_index", -1)) or -1), str(item.get("route_label") or ""))
        for item in selected_override + selected_keep
    }
    leftovers = [
        item for item in labeled
        if (int(item.get("index", item.get("sample_index", -1)) or -1), str(item.get("route_label") or "")) not in selected_keys
    ]
    leftovers.sort(key=lambda item: int(item.get("index", item.get("sample_index", -1)) or -1))
    if len(selected_override) + len(selected_keep) < target_count:
        fill_count = target_count - len(selected_override) - len(selected_keep)
        selected_keep.extend(leftovers[:fill_count])

    selected = selected_keep + selected_override
    selected = selected[:target_count]
    selected.sort(key=lambda item: int(item.get("index", item.get("sample_index", -1)) or -1))

    selected_errors = [
        float(item.get("default_error") or 0.0) - float(item.get("best_error") or 0.0)
        for item in selected
    ]
    metadata = {
        "default_expert": default_expert,
        "default_expert_mean_mse_full": (
            sum(float(item.get("default_error") or 0.0) for item in labeled) / max(len(labeled), 1)
        ),
        "default_expert_mean_regret_full": (
            sum(float(item.get("improvement_vs_default") or 0.0) * -1.0 for item in labeled) / max(len(labeled), 1)
        ),
        "enriched_record_count": len(enriched),
        "skipped_missing_source_count": int(skipped_missing_source),
        "selected_record_count": len(selected),
        "selected_keep_default_ratio": (
            sum(1 for item in selected if str(item.get("route_label") or "") == "keep_default") / max(len(selected), 1)
        ),
        "selected_override_ratio": (
            sum(1 for item in selected if str(item.get("route_label") or "").startswith("override_to_")) / max(len(selected), 1)
        ),
        "selected_route_label_distribution": dict(Counter(str(item.get("route_label") or "") for item in selected)),
        "selected_route_label_confidence_distribution": dict(
            Counter(str(item.get("route_label_confidence") or "") for item in selected)
        ),
        "selected_best_model_distribution": dict(Counter(str(item.get("best_model") or "") for item in selected)),
        "selected_mean_improvement_vs_default": (
            sum(float(item.get("improvement_vs_default") or 0.0) for item in selected) / max(len(selected), 1)
        ),
        "selected_mean_delta_vs_default": (
            sum(
                (
                    float(item.get("best_error") or 0.0) - float(item.get("default_error") or 0.0)
                    if str(item.get("route_decision") or "") == "override"
                    else 0.0
                )
                for item in selected
            )
            / max(len(selected), 1)
        ),
        "selected_mean_default_regret": (
            sum(float(item.get("best_error") or 0.0) - float(item.get("default_error") or 0.0) for item in selected)
            / max(len(selected), 1)
        ),
        "selected_default_minus_best_error_mean": (
            sum(selected_errors) / max(len(selected_errors), 1)
        ),
        **label_metadata,
    }
    return selected, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ETTh1 v17 routing bootstrap with default-expert override labels.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--train-teacher-eval-jsonl", type=Path, default=DEFAULT_TRAIN_TEACHER_EVAL_JSONL)
    parser.add_argument("--val-teacher-eval-jsonl", type=Path, default=DEFAULT_VAL_TEACHER_EVAL_JSONL)
    parser.add_argument("--test-teacher-eval-jsonl", type=Path, default=DEFAULT_TEST_TEACHER_EVAL_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--default-expert", type=str, default=DEFAULT_EXPERT, choices=SUPPORTED_ROUTING_MODELS)
    parser.add_argument("--train-target-samples", type=int, default=768)
    parser.add_argument("--val-target-samples", type=int, default=192)
    parser.add_argument("--test-target-samples", type=int, default=384)
    parser.add_argument("--keep-default-ratio", type=float, default=DEFAULT_KEEP_DEFAULT_RATIO)
    parser.add_argument("--override-top-fraction", type=float, default=DEFAULT_OVERRIDE_TOP_FRACTION)
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
        "pipeline_stage": "routing_override_bootstrap_v17",
        "task_type": "multivariate time-series forecasting",
        "historical_data_protocol": HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
        "observed_feature_columns": list(ETTH1_FEATURE_COLUMNS),
        "observed_covariates": list(ETTH1_COVARIATE_COLUMNS),
        "model_input_width": len(ETTH1_FEATURE_COLUMNS),
        "target_column": ETTH1_TARGET_COLUMN,
        "lookback_window": DEFAULT_LOOKBACK_WINDOW,
        "forecast_horizon": DEFAULT_FORECAST_HORIZON,
        "default_expert": str(args.default_expert),
        "selection_method": "default-expert-override-top-fraction",
        "keep_default_ratio": float(args.keep_default_ratio),
        "override_top_fraction": float(args.override_top_fraction),
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
            default_expert=str(args.default_expert),
            keep_default_ratio=float(args.keep_default_ratio),
            override_top_fraction=float(args.override_top_fraction),
        )
        write_jsonl_records(output_dir / f"{split_name}.jsonl", selected_records)
        metadata[f"{split_name}_source_samples"] = len(source_records)
        metadata[f"{split_name}_teacher_eval_records"] = len(evaluations)
        for key, value in split_metadata.items():
            metadata[f"{split_name}_{key}"] = value
        print(
            f"[routing-override-bootstrap] split={split_name} selected={len(selected_records)} "
            f"labels={split_metadata['selected_route_label_distribution']}",
            flush=True,
        )

    write_metadata_file(output_dir, metadata)


if __name__ == "__main__":
    main()
