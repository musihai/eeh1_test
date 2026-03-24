from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipe.time_series_forecast.build_etth1_sft_dataset import (
    _heuristic_routing_scores,
    _select_prediction_model_by_heuristic,
    distribution_from_series,
)
from recipe.time_series_forecast.dataset_file_utils import load_jsonl_records, write_jsonl_records, write_metadata_file
from recipe.time_series_forecast.dataset_identity import DATASET_KIND_RL_JSONL, DATASET_KIND_TEACHER_CURATED_SFT, validate_sibling_metadata
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.utils import parse_time_series_string


DEFAULT_INPUT_DIR = Path("dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2")
DEFAULT_OUTPUT_DIR = Path("dataset/ett_sft_etth1_runtime_teacher200_heuristiccurated_paper_same2")
SUPPORTED_HEURISTIC_MODELS = ("patchtst", "itransformer", "arima", "chronos2")


def _normalize_stage_name(sample: dict[str, Any]) -> str:
    stage = str(sample.get("difficulty_stage") or sample.get("curriculum_stage") or "unknown").strip().lower()
    return stage or "unknown"


def _extract_history_values(sample: dict[str, Any]) -> list[float]:
    raw_prompt = sample["raw_prompt"][0]["content"]
    task_spec = parse_task_prompt(raw_prompt, data_source=sample.get("data_source"))
    historical_data = task_spec.historical_data or raw_prompt
    _, values = parse_time_series_string(historical_data, target_column=task_spec.target_column or "OT")
    if not values:
        raise ValueError(f"Sample index={sample.get('index', -1)} does not contain valid historical values.")
    return [float(value) for value in values]


def annotate_heuristic_sample(sample: dict[str, Any]) -> dict[str, Any]:
    values = _extract_history_values(sample)
    selected_model, feature_snapshot, routing_reason = _select_prediction_model_by_heuristic(values)
    heuristic_scores = _heuristic_routing_scores(feature_snapshot)
    selected_score = float(heuristic_scores.get(selected_model, 0.0))
    runner_up_score = max(
        (float(score) for model_name, score in heuristic_scores.items() if model_name != selected_model),
        default=selected_score,
    )
    offline_best_model = str(sample.get("offline_best_model") or "").strip().lower()
    heuristic_agrees_with_offline_best = bool(offline_best_model and selected_model == offline_best_model)

    enriched = dict(sample)
    enriched["reference_teacher_model"] = offline_best_model or sample.get("reference_teacher_model")
    enriched["selected_prediction_model"] = selected_model
    enriched["heuristic_selected_prediction_model"] = selected_model
    enriched["heuristic_routing_reason"] = routing_reason
    enriched["heuristic_score_margin"] = float(selected_score - runner_up_score)
    enriched["heuristic_selected_score"] = selected_score
    enriched["heuristic_runner_up_score"] = runner_up_score
    enriched["heuristic_routing_scores"] = {
        str(model_name): float(score) for model_name, score in sorted(heuristic_scores.items())
    }
    enriched["heuristic_offline_best_agreement"] = heuristic_agrees_with_offline_best
    enriched["heuristic_feature_snapshot"] = {
        str(name): float(value) if isinstance(value, (int, float)) else value
        for name, value in feature_snapshot.items()
    }
    enriched["difficulty_stage"] = _normalize_stage_name(sample)
    return enriched


def _selection_key(sample: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        1.0 if bool(sample.get("heuristic_offline_best_agreement")) else 0.0,
        float(sample.get("offline_margin", 0.0) or 0.0),
        float(sample.get("heuristic_score_margin", 0.0) or 0.0),
        -int(sample.get("index", -1) or -1),
    )


def _select_bucketed_records(records: Sequence[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    if target_count <= 0 or len(records) <= target_count:
        return sorted(records, key=lambda item: int(item.get("index", -1)))

    ordered = sorted(records, key=lambda item: int(item.get("index", -1)))
    selected_by_index: dict[int, dict[str, Any]] = {}

    for bucket_idx in range(target_count):
        start = int(bucket_idx * len(ordered) / target_count)
        end = int((bucket_idx + 1) * len(ordered) / target_count)
        bucket = ordered[start:max(end, start + 1)]
        best = max(bucket, key=_selection_key)
        selected_by_index[int(best.get("index", -1))] = best

    if len(selected_by_index) < target_count:
        for item in sorted(ordered, key=_selection_key, reverse=True):
            sample_index = int(item.get("index", -1))
            if sample_index not in selected_by_index:
                selected_by_index[sample_index] = item
            if len(selected_by_index) >= target_count:
                break

    return sorted(selected_by_index.values(), key=lambda item: int(item.get("index", -1)))[:target_count]


def allocate_proportional_quotas(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    positive_groups = {name: int(size) for name, size in group_sizes.items() if int(size) > 0}
    if total <= 0 or not positive_groups:
        return {name: 0 for name in group_sizes}

    if total <= len(positive_groups):
        ordered = sorted(positive_groups.items(), key=lambda item: (-item[1], str(item[0])))
        selected_names = {name for name, _ in ordered[:total]}
        return {name: (1 if name in selected_names else 0) for name in group_sizes}

    quotas = {name: 1 for name in positive_groups}
    remaining = total - len(positive_groups)
    remaining_capacity = {name: positive_groups[name] - 1 for name in positive_groups}
    total_remaining_capacity = sum(capacity for capacity in remaining_capacity.values() if capacity > 0)
    if total_remaining_capacity <= 0:
        return {name: quotas.get(name, 0) for name in group_sizes}

    fractional_additions: dict[str, float] = {}
    for name, capacity in remaining_capacity.items():
        if capacity <= 0:
            fractional_additions[name] = 0.0
            continue
        raw_addition = remaining * capacity / total_remaining_capacity
        integer_addition = min(capacity, int(raw_addition))
        quotas[name] += integer_addition
        remaining_capacity[name] -= integer_addition
        fractional_additions[name] = raw_addition - integer_addition

    assigned = sum(quotas.values())
    remaining = total - assigned
    while remaining > 0 and any(capacity > 0 for capacity in remaining_capacity.values()):
        best_name = max(
            (name for name, capacity in remaining_capacity.items() if capacity > 0),
            key=lambda name: (fractional_additions.get(name, 0.0), remaining_capacity[name], positive_groups[name], str(name)),
        )
        quotas[best_name] += 1
        remaining_capacity[best_name] -= 1
        fractional_additions[best_name] = 0.0
        remaining -= 1

    return {name: quotas.get(name, 0) for name in group_sizes}


def allocate_balanced_model_quotas(records_by_model: dict[str, list[dict[str, Any]]], target_count: int) -> dict[str, int]:
    available_models = [model_name for model_name in SUPPORTED_HEURISTIC_MODELS if records_by_model.get(model_name)]
    if target_count <= 0 or not available_models:
        return {model_name: 0 for model_name in SUPPORTED_HEURISTIC_MODELS}

    base_quota = target_count // len(available_models)
    quotas = {
        model_name: min(base_quota, len(records_by_model.get(model_name, [])))
        for model_name in available_models
    }
    assigned = sum(quotas.values())
    remaining = target_count - assigned

    while remaining > 0:
        candidate_models = [
            model_name
            for model_name in available_models
            if quotas[model_name] < len(records_by_model.get(model_name, []))
        ]
        if not candidate_models:
            break
        best_model = max(
            candidate_models,
            key=lambda model_name: (
                len(records_by_model[model_name]) - quotas[model_name],
                -quotas[model_name],
                model_name,
            ),
        )
        quotas[best_model] += 1
        remaining -= 1

    return {model_name: quotas.get(model_name, 0) for model_name in SUPPORTED_HEURISTIC_MODELS}


def select_curated_records(
    annotated_records: Sequence[dict[str, Any]],
    *,
    target_count: int,
) -> list[dict[str, Any]]:
    if target_count <= 0 or len(annotated_records) <= target_count:
        return sorted(annotated_records, key=lambda item: int(item.get("index", -1)))

    records_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in annotated_records:
        records_by_model[str(sample.get("heuristic_selected_prediction_model") or "")].append(sample)

    selected_by_index: dict[int, dict[str, Any]] = {}
    model_quotas = allocate_balanced_model_quotas(records_by_model, target_count)

    for model_name in SUPPORTED_HEURISTIC_MODELS:
        model_records = records_by_model.get(model_name, [])
        model_quota = int(model_quotas.get(model_name, 0))
        if model_quota <= 0 or not model_records:
            continue

        stage_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in model_records:
            stage_groups[str(record.get("difficulty_stage") or "unknown")].append(record)

        stage_quotas = allocate_proportional_quotas(
            {stage_name: len(stage_records) for stage_name, stage_records in stage_groups.items()},
            model_quota,
        )

        model_selected_indices: set[int] = set()
        for stage_name, stage_records in stage_groups.items():
            stage_quota = int(stage_quotas.get(stage_name, 0))
            if stage_quota <= 0:
                continue
            for record in _select_bucketed_records(stage_records, stage_quota):
                sample_index = int(record.get("index", -1))
                selected_by_index[sample_index] = record
                model_selected_indices.add(sample_index)

        if len(model_selected_indices) < model_quota:
            remaining_records = [
                record for record in model_records if int(record.get("index", -1)) not in model_selected_indices
            ]
            for record in _select_bucketed_records(remaining_records, model_quota - len(model_selected_indices)):
                selected_by_index[int(record.get("index", -1))] = record

    if len(selected_by_index) < target_count:
        remaining_records = [
            record
            for record in annotated_records
            if int(record.get("index", -1)) not in selected_by_index
        ]
        for record in _select_bucketed_records(remaining_records, target_count - len(selected_by_index)):
            selected_by_index[int(record.get("index", -1))] = record

    return sorted(selected_by_index.values(), key=lambda item: int(item.get("index", -1)))[:target_count]


def curate_split(records: Sequence[dict[str, Any]], *, target_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotated_records = [annotate_heuristic_sample(record) for record in records]
    selected_records = select_curated_records(annotated_records, target_count=target_count)
    return annotated_records, selected_records


def _agreement_ratio(records: Iterable[dict[str, Any]]) -> float:
    records = list(records)
    if not records:
        return 0.0
    agree_count = sum(1 for record in records if bool(record.get("heuristic_offline_best_agreement")))
    return float(agree_count / len(records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-style heuristic curated ETTh1 SFT jsonl splits from the RL curriculum dataset."
    )
    parser.add_argument("--train-jsonl", default=str(DEFAULT_INPUT_DIR / "train.jsonl"), help="Train RL jsonl path.")
    parser.add_argument("--val-jsonl", default=str(DEFAULT_INPUT_DIR / "val.jsonl"), help="Validation RL jsonl path.")
    parser.add_argument("--test-jsonl", default=str(DEFAULT_INPUT_DIR / "test.jsonl"), help="Test RL jsonl path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for curated jsonl files.")
    parser.add_argument("--train-target-samples", type=int, default=200, help="Curated train sample count.")
    parser.add_argument("--val-target-samples", type=int, default=64, help="Curated val sample count.")
    parser.add_argument("--test-target-samples", type=int, default=128, help="Curated test sample count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_metadata_paths: list[Path] = []
    for split_path in (Path(args.train_jsonl), Path(args.val_jsonl), Path(args.test_jsonl)):
        if not split_path.exists():
            continue
        _, source_metadata_path = validate_sibling_metadata(split_path, expected_kind=DATASET_KIND_RL_JSONL)
        source_metadata_paths.append(source_metadata_path)
    if source_metadata_paths and len({str(path) for path in source_metadata_paths}) != 1:
        raise ValueError(
            "All source jsonl splits must come from the same RL dataset directory. "
            f"Got metadata files: {sorted({str(path) for path in source_metadata_paths})}"
        )

    split_specs = [
        ("train", Path(args.train_jsonl), int(args.train_target_samples)),
        ("val", Path(args.val_jsonl), int(args.val_target_samples)),
        ("test", Path(args.test_jsonl), int(args.test_target_samples)),
    ]

    metadata: dict[str, Any] = {
        "dataset_kind": DATASET_KIND_TEACHER_CURATED_SFT,
        "pipeline_stage": "heuristic_curated_sft",
        "selection_method": "heuristic_rule_coverage_with_model_balance_stage_balance_and_time_coverage",
        "source_metadata_path": str(source_metadata_paths[0]) if source_metadata_paths else "",
    }

    for split_name, split_path, target_count in split_specs:
        if not split_path.exists():
            continue
        source_records = load_jsonl_records(split_path)
        annotated_records, selected_records = curate_split(source_records, target_count=target_count)
        write_jsonl_records(output_dir / f"{split_name}_curated.jsonl", selected_records)

        metadata[f"source_{split_name}_jsonl"] = str(split_path)
        metadata[f"{split_name}_source_samples"] = int(len(source_records))
        metadata[f"{split_name}_target_samples"] = int(target_count)
        metadata[f"{split_name}_selected_samples"] = int(len(selected_records))
        metadata[f"{split_name}_source_heuristic_model_distribution"] = distribution_from_series(
            record.get("heuristic_selected_prediction_model") for record in annotated_records
        )
        metadata[f"{split_name}_selected_heuristic_model_distribution"] = distribution_from_series(
            record.get("heuristic_selected_prediction_model") for record in selected_records
        )
        metadata[f"{split_name}_selected_stage_distribution"] = distribution_from_series(
            record.get("difficulty_stage") for record in selected_records
        )
        metadata[f"{split_name}_selected_reference_teacher_model_distribution"] = distribution_from_series(
            record.get("reference_teacher_model") for record in selected_records
        )
        metadata[f"{split_name}_source_agreement_ratio"] = _agreement_ratio(annotated_records)
        metadata[f"{split_name}_selected_agreement_ratio"] = _agreement_ratio(selected_records)

    metadata_path = write_metadata_file(output_dir, metadata)
    print(f"[HEURISTIC CURATED SFT] wrote dataset metadata to {metadata_path}")


if __name__ == "__main__":
    main()
