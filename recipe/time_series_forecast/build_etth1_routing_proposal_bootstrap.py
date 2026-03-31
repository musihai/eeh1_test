from __future__ import annotations

import argparse
import json
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


DEFAULT_OUTPUT_DIR = Path("dataset/routing_proposal_bootstrap_v18")
DEFAULT_TRAIN_JSONL = Path("dataset/ett_rl_etth1_paper_same2/train.jsonl")
DEFAULT_VAL_JSONL = Path("dataset/ett_rl_etth1_paper_same2/val.jsonl")
DEFAULT_TEST_JSONL = Path("dataset/ett_rl_etth1_paper_same2/test.jsonl")
DEFAULT_TRAIN_TEACHER_EVAL_JSONL = Path("dataset/ett_sft_etth1_runtime_teacher200_refteacher_v15/train_teacher_eval.jsonl")
DEFAULT_VAL_TEACHER_EVAL_JSONL = Path("dataset/ett_sft_etth1_runtime_teacher200_refteacher_v15/val_teacher_eval.jsonl")
DEFAULT_TEST_TEACHER_EVAL_JSONL = Path("dataset/ett_sft_etth1_runtime_teacher200_refteacher_v15/test_teacher_eval.jsonl")

SUPPORTED_ROUTING_MODELS = ("patchtst", "itransformer", "arima", "chronos2")
DEFAULT_EXPERT = "itransformer"
DEFAULT_TRAIN_TARGET_SAMPLES = 768
DEFAULT_VAL_NATURAL_TARGET_SAMPLES = 192
DEFAULT_VAL_BALANCED_TARGET_SAMPLES = 192
DEFAULT_TEST_TARGET_SAMPLES = 384
DEFAULT_TAU_KEEP = 0.05
DEFAULT_TAU_MARGIN = 0.08
DEFAULT_OVERRIDE_QUANTILE = 0.80
DEFAULT_OVERRIDE_FLOOR = 0.35
DEFAULT_LOOKBACK_WINDOW, DEFAULT_FORECAST_HORIZON = get_default_lengths()

ROUTE_BUCKET_MUST_KEEP = "must_keep"
ROUTE_BUCKET_MUST_OVERRIDE = "must_override"
ROUTE_BUCKET_AMBIGUOUS = "ambiguous"
ROUTE_DECISION_KEEP = "keep_default"
ROUTE_DECISION_OVERRIDE = "override"


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
        float(item.get("route_margin_rel") or 0.0),
        float(item.get("improvement_vs_default") or 0.0),
        -int(item.get("index", item.get("sample_index", -1)) or -1),
    )


def _record_identity(item: dict[str, Any]) -> tuple[str, int]:
    uid = str(item.get("uid") or "")
    index_value = int(item.get("index", item.get("sample_index", -1)) or -1)
    return (uid, index_value)


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
    top2_models = [model_name for model_name, _ in ranked_errors[:2]]
    return {
        "best_model": best_model,
        "best_error": float(best_error),
        "route_best_model": best_model,
        "route_best_error": float(best_error),
        "route_second_best_model": second_best_model,
        "route_second_best_error": float(second_best_error),
        "route_margin_abs": margin_abs,
        "route_margin_rel": margin_rel,
        "route_top2_models": top2_models,
        "default_expert": default_expert,
        "default_error": default_error,
        "improvement_vs_default": improvement_vs_default,
        "improvement_vs_default_rel": improvement_vs_default_rel,
        "default_in_top2": default_expert in top2_models,
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
    best_model = str(route_info["best_model"])
    record.update(
        {
            "reference_teacher_model": best_model,
            "offline_best_model": best_model,
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
            "route_bucket": ROUTE_BUCKET_AMBIGUOUS,
            "route_decision": "",
            "route_label": "",
            "route_override_model": "",
            "route_label_confidence": "",
            "route_target_model": "",
            "route_default_path_id": f"{default_expert}__keep",
            "route_alt_path_id": (
                f"{best_model}__keep" if best_model and best_model != default_expert else ""
            ),
            **route_info,
        }
    )
    return record


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(value) for value in values)
    position = (len(sorted_values) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = float(position - lower)
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _compute_override_threshold_rel_by_model(
    records: list[dict[str, Any]],
    *,
    default_expert: str,
    override_quantile: float,
    override_floor: float,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        best_model = _normalize_model_name(record.get("best_model"))
        if not best_model or best_model == default_expert:
            continue
        improvement_rel = float(record.get("improvement_vs_default_rel") or 0.0)
        if improvement_rel <= 0.0:
            continue
        grouped[best_model].append(improvement_rel)

    thresholds: dict[str, float] = {}
    for model_name in SUPPORTED_ROUTING_MODELS:
        if model_name == default_expert:
            continue
        values = grouped.get(model_name, [])
        thresholds[model_name] = max(float(override_floor), _quantile(values, override_quantile))
    return thresholds


def _label_record_v18(
    record: dict[str, Any],
    *,
    default_expert: str,
    tau_keep: float,
    tau_margin: float,
    override_threshold_rel_by_model: dict[str, float],
) -> dict[str, Any]:
    labeled = dict(record)
    best_model = _normalize_model_name(labeled.get("best_model"))
    improvement_rel = float(labeled.get("improvement_vs_default_rel") or 0.0)
    default_in_top2 = bool(labeled.get("default_in_top2"))
    route_margin_rel = float(labeled.get("route_margin_rel") or 0.0)

    route_bucket = ROUTE_BUCKET_AMBIGUOUS
    route_decision = ""
    route_label = ""
    route_override_model = ""
    route_target_model = ""
    confidence = ""

    if not best_model or best_model == default_expert or improvement_rel <= float(tau_keep):
        route_bucket = ROUTE_BUCKET_MUST_KEEP
        route_decision = ROUTE_DECISION_KEEP
        route_label = ROUTE_DECISION_KEEP
        route_target_model = default_expert
        confidence = "high" if best_model == default_expert else "mid"
    else:
        threshold = float(override_threshold_rel_by_model.get(best_model, 1.0))
        override_supported = improvement_rel >= threshold and (
            route_margin_rel >= float(tau_margin) or not default_in_top2
        )
        if override_supported:
            route_bucket = ROUTE_BUCKET_MUST_OVERRIDE
            route_decision = ROUTE_DECISION_OVERRIDE
            route_label = f"override_to_{best_model}"
            route_override_model = best_model
            route_target_model = best_model
            confidence = "high"

    labeled.update(
        {
            "route_bucket": route_bucket,
            "route_decision": route_decision,
            "route_label": route_label,
            "route_override_model": route_override_model,
            "route_target_model": route_target_model,
            "route_label_confidence": confidence,
            "selected_prediction_model": route_target_model or default_expert,
        }
    )
    return labeled


def _take_evenly_spaced(items: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    if target_count <= 0 or not items:
        return []
    if len(items) <= target_count:
        return list(items)
    if target_count == 1:
        return [items[len(items) // 2]]
    positions = sorted({round(i * (len(items) - 1) / (target_count - 1)) for i in range(target_count)})
    return [items[position] for position in positions]


def _select_keep_records(keep_records: list[dict[str, Any]], *, target_count: int) -> list[dict[str, Any]]:
    ranked = sorted(
        keep_records,
        key=lambda item: (
            _normalize_model_name(item.get("best_model")) != _normalize_model_name(item.get("default_expert")),
            float(item.get("improvement_vs_default_rel") or 0.0),
            int(item.get("index", item.get("sample_index", -1)) or -1),
        ),
    )
    selected = _take_evenly_spaced(ranked, target_count)
    selected.sort(key=lambda item: int(item.get("index", item.get("sample_index", -1)) or -1))
    return selected


def _select_override_records_balanced(
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


def _select_non_ambiguous_natural(
    records: list[dict[str, Any]],
    *,
    target_count: int,
) -> list[dict[str, Any]]:
    ranked = sorted(records, key=lambda item: int(item.get("index", item.get("sample_index", -1)) or -1))
    selected = _take_evenly_spaced(ranked, target_count)
    selected.sort(key=lambda item: int(item.get("index", item.get("sample_index", -1)) or -1))
    return selected


def _count_contradictory_keep(
    records: list[dict[str, Any]],
    *,
    default_expert: str,
    override_threshold_rel_by_model: dict[str, float],
) -> int:
    count = 0
    for record in records:
        if str(record.get("route_bucket") or "") != ROUTE_BUCKET_MUST_KEEP:
            continue
        best_model = _normalize_model_name(record.get("best_model"))
        if not best_model or best_model == default_expert:
            continue
        threshold = float(override_threshold_rel_by_model.get(best_model, 1.0))
        if float(record.get("improvement_vs_default_rel") or 0.0) >= threshold:
            count += 1
    return count


def _select_train_records(
    labeled_records: list[dict[str, Any]],
    *,
    target_count: int,
) -> list[dict[str, Any]]:
    must_keep = [item for item in labeled_records if str(item.get("route_bucket") or "") == ROUTE_BUCKET_MUST_KEEP]
    must_override = [
        item for item in labeled_records if str(item.get("route_bucket") or "") == ROUTE_BUCKET_MUST_OVERRIDE
    ]
    keep_target = min(len(must_keep), max(1, target_count // 2))
    override_target = min(len(must_override), max(1, target_count - keep_target))
    selected_keep = _select_keep_records(must_keep, target_count=keep_target)
    selected_override = _select_override_records_balanced(must_override, target_count=override_target)
    selected = selected_keep + selected_override

    if len(selected) < target_count:
        selected_identities = {_record_identity(sel) for sel in selected}
        leftovers = [
            item
            for item in must_keep + must_override
            if _record_identity(item) not in selected_identities
        ]
        leftovers.sort(key=lambda item: int(item.get("index", item.get("sample_index", -1)) or -1))
        selected.extend(leftovers[: max(0, target_count - len(selected))])

    selected = selected[:target_count]
    selected.sort(key=lambda item: int(item.get("index", item.get("sample_index", -1)) or -1))
    return selected


def _select_balanced_eval_records(
    labeled_records: list[dict[str, Any]],
    *,
    target_count: int,
) -> list[dict[str, Any]]:
    must_keep = [item for item in labeled_records if str(item.get("route_bucket") or "") == ROUTE_BUCKET_MUST_KEEP]
    must_override = [
        item for item in labeled_records if str(item.get("route_bucket") or "") == ROUTE_BUCKET_MUST_OVERRIDE
    ]
    keep_target = min(len(must_keep), max(1, target_count // 2))
    override_target = min(len(must_override), max(1, target_count - keep_target))
    selected_keep = _select_keep_records(must_keep, target_count=keep_target)
    selected_override = _select_override_records_balanced(must_override, target_count=override_target)
    selected = selected_keep + selected_override
    selected = selected[:target_count]
    selected.sort(key=lambda item: int(item.get("index", item.get("sample_index", -1)) or -1))
    return selected


def _enrich_records(
    *,
    source_records: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    default_expert: str,
) -> tuple[list[dict[str, Any]], int]:
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
    return enriched, skipped_missing_source


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "route_bucket_distribution": dict(Counter(str(item.get("route_bucket") or "") for item in records)),
        "route_label_distribution": dict(Counter(str(item.get("route_label") or "") for item in records if str(item.get("route_label") or ""))),
        "route_confidence_distribution": dict(
            Counter(str(item.get("route_label_confidence") or "") for item in records if str(item.get("route_label_confidence") or ""))
        ),
        "best_model_distribution": dict(Counter(str(item.get("best_model") or "") for item in records)),
    }


def _select_split_records_v18(
    *,
    source_records: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    default_expert: str,
    tau_keep: float,
    tau_margin: float,
    override_threshold_rel_by_model: dict[str, float],
    train_target_count: int,
    val_natural_target_count: int,
    val_balanced_target_count: int,
    test_target_count: int,
    split_name: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    enriched, skipped_missing_source = _enrich_records(
        source_records=source_records,
        evaluations=evaluations,
        default_expert=default_expert,
    )
    labeled = [
        _label_record_v18(
            record,
            default_expert=default_expert,
            tau_keep=tau_keep,
            tau_margin=tau_margin,
            override_threshold_rel_by_model=override_threshold_rel_by_model,
        )
        for record in enriched
    ]
    contradictory_keep_count = _count_contradictory_keep(
        labeled,
        default_expert=default_expert,
        override_threshold_rel_by_model=override_threshold_rel_by_model,
    )
    non_ambiguous = [item for item in labeled if str(item.get("route_bucket") or "") != ROUTE_BUCKET_AMBIGUOUS]

    selected_train: list[dict[str, Any]] = []
    selected_val_natural: list[dict[str, Any]] = []
    selected_val_balanced: list[dict[str, Any]] = []
    selected_test: list[dict[str, Any]] = []

    if split_name == "train":
        selected_train = _select_train_records(labeled, target_count=train_target_count)
    elif split_name == "val":
        selected_val_balanced = _select_balanced_eval_records(labeled, target_count=val_balanced_target_count)
        balanced_identities = {_record_identity(item) for item in selected_val_balanced}
        non_ambiguous_remaining = [
            item for item in non_ambiguous if _record_identity(item) not in balanced_identities
        ]
        selected_val_natural = _select_non_ambiguous_natural(
            non_ambiguous_remaining,
            target_count=val_natural_target_count,
        )
    elif split_name == "test":
        selected_test = _select_non_ambiguous_natural(non_ambiguous, target_count=test_target_count)

    metadata = {
        "enriched_record_count": len(enriched),
        "skipped_missing_source_count": int(skipped_missing_source),
        "default_expert_mean_mse_full": (
            sum(float(item.get("default_error") or 0.0) for item in labeled) / max(len(labeled), 1)
        ),
        "default_expert_mean_regret_full": (
            sum(float(item.get("default_error") or 0.0) - float(item.get("best_error") or 0.0) for item in labeled)
            / max(len(labeled), 1)
        ),
        "contradictory_keep_count": int(contradictory_keep_count),
        "override_threshold_rel_by_model": {
            model_name: float(value) for model_name, value in sorted(override_threshold_rel_by_model.items())
        },
        "all_records": _summarize_records(labeled),
        "non_ambiguous_records": _summarize_records(non_ambiguous),
    }
    if selected_train:
        metadata["selected_train"] = _summarize_records(selected_train)
    if selected_val_natural:
        metadata["selected_val_natural"] = _summarize_records(selected_val_natural)
    if selected_val_balanced:
        metadata["selected_val_balanced"] = _summarize_records(selected_val_balanced)
    if selected_test:
        metadata["selected_test"] = _summarize_records(selected_test)

    return {
        "train": selected_train,
        "val_natural": selected_val_natural,
        "val_balanced": selected_val_balanced,
        "test": selected_test,
    }, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ETTh1 v18 route-proposal bootstrap with triage labels.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--train-teacher-eval-jsonl", type=Path, default=DEFAULT_TRAIN_TEACHER_EVAL_JSONL)
    parser.add_argument("--val-teacher-eval-jsonl", type=Path, default=DEFAULT_VAL_TEACHER_EVAL_JSONL)
    parser.add_argument("--test-teacher-eval-jsonl", type=Path, default=DEFAULT_TEST_TEACHER_EVAL_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--default-expert", type=str, default=DEFAULT_EXPERT, choices=SUPPORTED_ROUTING_MODELS)
    parser.add_argument("--train-target-samples", type=int, default=DEFAULT_TRAIN_TARGET_SAMPLES)
    parser.add_argument("--val-natural-target-samples", type=int, default=DEFAULT_VAL_NATURAL_TARGET_SAMPLES)
    parser.add_argument("--val-balanced-target-samples", type=int, default=DEFAULT_VAL_BALANCED_TARGET_SAMPLES)
    parser.add_argument("--test-target-samples", type=int, default=DEFAULT_TEST_TARGET_SAMPLES)
    parser.add_argument("--tau-keep", type=float, default=DEFAULT_TAU_KEEP)
    parser.add_argument("--tau-margin", type=float, default=DEFAULT_TAU_MARGIN)
    parser.add_argument("--override-quantile", type=float, default=DEFAULT_OVERRIDE_QUANTILE)
    parser.add_argument("--override-floor", type=float, default=DEFAULT_OVERRIDE_FLOOR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = (
        ("train", args.train_jsonl, args.train_teacher_eval_jsonl),
        ("val", args.val_jsonl, args.val_teacher_eval_jsonl),
        ("test", args.test_jsonl, args.test_teacher_eval_jsonl),
    )

    metadata: dict[str, Any] = {
        "dataset_kind": DATASET_KIND_TEACHER_CURATED_SFT,
        "pipeline_stage": "routing_proposal_bootstrap_v18",
        "task_type": "multivariate time-series forecasting",
        "historical_data_protocol": HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
        "observed_feature_columns": list(ETTH1_FEATURE_COLUMNS),
        "observed_covariates": list(ETTH1_COVARIATE_COLUMNS),
        "model_input_width": len(ETTH1_FEATURE_COLUMNS),
        "target_column": ETTH1_TARGET_COLUMN,
        "lookback_window": DEFAULT_LOOKBACK_WINDOW,
        "forecast_horizon": DEFAULT_FORECAST_HORIZON,
        "default_expert": str(args.default_expert),
        "selection_method": "triage-proposal-threshold-plus-ambiguous",
        "tau_keep": float(args.tau_keep),
        "tau_margin": float(args.tau_margin),
        "override_quantile": float(args.override_quantile),
        "override_floor": float(args.override_floor),
        "teacher_models": list(SUPPORTED_ROUTING_MODELS),
        "train_target_samples": int(args.train_target_samples),
        "val_natural_target_samples": int(args.val_natural_target_samples),
        "val_balanced_target_samples": int(args.val_balanced_target_samples),
        "test_target_samples": int(args.test_target_samples),
    }

    split_inputs: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for split_name, source_jsonl, teacher_eval_jsonl in split_specs:
        source_metadata, source_metadata_path = validate_sibling_metadata(
            source_jsonl,
            expected_kind=DATASET_KIND_RL_JSONL,
        )
        require_multivariate_etth1_metadata(source_metadata, metadata_path=source_metadata_path)
        metadata[f"source_{split_name}_metadata_path"] = str(source_metadata_path)
        metadata[f"source_{split_name}_pipeline_stage"] = str(source_metadata.get("pipeline_stage") or "")
        split_inputs[split_name] = (
            load_jsonl_records(source_jsonl),
            load_jsonl_records(teacher_eval_jsonl),
        )

    train_source_records, train_evaluations = split_inputs["train"]
    train_enriched, _ = _enrich_records(
        source_records=train_source_records,
        evaluations=train_evaluations,
        default_expert=str(args.default_expert),
    )
    override_threshold_rel_by_model = _compute_override_threshold_rel_by_model(
        train_enriched,
        default_expert=str(args.default_expert),
        override_quantile=float(args.override_quantile),
        override_floor=float(args.override_floor),
    )
    metadata["override_threshold_rel_by_model"] = {
        model_name: float(value) for model_name, value in sorted(override_threshold_rel_by_model.items())
    }

    selected_outputs: dict[str, list[dict[str, Any]]] = {}
    for split_name, (source_records, evaluations) in split_inputs.items():
        selected_by_split, split_metadata = _select_split_records_v18(
            source_records=source_records,
            evaluations=evaluations,
            default_expert=str(args.default_expert),
            tau_keep=float(args.tau_keep),
            tau_margin=float(args.tau_margin),
            override_threshold_rel_by_model=override_threshold_rel_by_model,
            train_target_count=int(args.train_target_samples),
            val_natural_target_count=int(args.val_natural_target_samples),
            val_balanced_target_count=int(args.val_balanced_target_samples),
            test_target_count=int(args.test_target_samples),
            split_name=split_name,
        )
        metadata[f"{split_name}_source_samples"] = len(source_records)
        metadata[f"{split_name}_teacher_eval_records"] = len(evaluations)
        for key, value in split_metadata.items():
            metadata[f"{split_name}_{key}"] = value

        if split_name == "train":
            selected_outputs["train.jsonl"] = selected_by_split["train"]
            print(
                f"[routing-proposal-bootstrap] split=train selected={len(selected_by_split['train'])} "
                f"buckets={split_metadata['selected_train']['route_bucket_distribution']}",
                flush=True,
            )
        elif split_name == "val":
            selected_outputs["val.jsonl"] = selected_by_split["val_balanced"]
            selected_outputs["val_balanced.jsonl"] = selected_by_split["val_balanced"]
            selected_outputs["val_natural.jsonl"] = selected_by_split["val_natural"]
            print(
                f"[routing-proposal-bootstrap] split=val balanced={len(selected_by_split['val_balanced'])} "
                f"natural={len(selected_by_split['val_natural'])}",
                flush=True,
            )
        elif split_name == "test":
            selected_outputs["test.jsonl"] = selected_by_split["test"]
            print(
                f"[routing-proposal-bootstrap] split=test selected={len(selected_by_split['test'])} "
                f"buckets={split_metadata['selected_test']['route_bucket_distribution']}",
                flush=True,
            )

    for filename, records in selected_outputs.items():
        write_jsonl_records(output_dir / filename, records)

    write_metadata_file(output_dir, metadata)
    print(json.dumps({"output_dir": str(output_dir), "files": sorted(selected_outputs.keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
