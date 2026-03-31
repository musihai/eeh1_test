from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from recipe.time_series_forecast.build_etth1_high_quality_sft import prepare_teacher_sample, score_prediction_text
from recipe.time_series_forecast.build_etth1_sft_dataset import (
    _compute_routing_feature_snapshot,
    build_feature_tool_results,
)
from recipe.time_series_forecast.dataset_file_utils import load_jsonl_records, write_jsonl_records, write_metadata_file
from recipe.time_series_forecast.refinement_support import (
    REFINEMENT_KEEP_DECISION,
    build_refinement_candidate_prediction_text_map,
    filter_refinement_candidates_for_model,
    generate_local_refinement_candidates,
)
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.time_series_io import (
    compact_prediction_tool_output_from_string,
    format_predictions_to_string,
    parse_time_series_string,
    parse_time_series_to_dataframe,
)
from recipe.time_series_forecast.utils import predict_time_series_async


DEFAULT_INPUT_DIR = Path("dataset/ett_sft_etth1_runtime_teacher200_refteacher_v15")
DEFAULT_OUTPUT_DIR = Path("dataset/ett_v19_candidate_bank")
SUPPORTED_MODELS = ("patchtst", "itransformer", "arima", "chronos2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the v19 candidate bank for ETTh1 SFT.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--default-expert", type=str, default="itransformer")
    parser.add_argument("--risk-tau", type=float, default=0.20)
    parser.add_argument("--max-concurrent-samples", type=int, default=4)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def _load_merged_split(input_dir: Path, split: str) -> list[dict[str, Any]]:
    source_rows = load_jsonl_records(input_dir / f"{split}_curated.jsonl")
    teacher_rows = load_jsonl_records(input_dir / f"{split}_teacher_eval_curated.jsonl")
    teacher_by_index = {int(row["sample_index"]): row for row in teacher_rows}
    merged: list[dict[str, Any]] = []
    for source_row in source_rows:
        sample_index = int(source_row["index"])
        teacher_row = teacher_by_index.get(sample_index)
        if teacher_row is None:
            continue
        row = dict(source_row)
        row.update(teacher_row)
        merged.append(row)
    return merged


def _teacher_score_details(row: dict[str, Any]) -> dict[str, dict[str, float]]:
    payload = row.get("model_score_details") or {}
    if not isinstance(payload, dict):
        return {}
    output: dict[str, dict[str, float]] = {}
    for model_name, item in payload.items():
        if not isinstance(item, dict):
            continue
        normalized = str(model_name).strip().lower()
        metrics: dict[str, float] = {}
        for key in ("score", "orig_mse", "orig_mae", "norm_mse", "norm_mae"):
            try:
                value = float(item.get(key))
            except (TypeError, ValueError):
                continue
            metrics[key] = value
        if metrics:
            output[normalized] = metrics
    return output


def _candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, str]:
    score = float((item.get("score_details") or {}).get("score", -1e9))
    mse = float((item.get("score_details") or {}).get("orig_mse", 1e12))
    return (-score, mse, str(item.get("candidate_id") or ""))


async def _predict_baseline_candidate(
    *,
    sample: dict[str, Any],
    prepared_sample: dict[str, Any],
    historical_data: str,
    data_source: str,
    target_column: str,
    forecast_horizon: int,
    model_name: str,
    teacher_score_details: dict[str, dict[str, float]],
) -> dict[str, Any]:
    cached_reference_model = str(sample.get("reference_teacher_model") or "").strip().lower()
    cached_prediction_text = str(
        sample.get("teacher_prediction_text")
        or sample.get("reference_teacher_prediction_text")
        or ""
    ).strip()

    if cached_reference_model == model_name and cached_prediction_text:
        prediction_text = cached_prediction_text
        prediction_source = "teacher_eval_cached"
    else:
        context_df = parse_time_series_to_dataframe(
            historical_data,
            series_id=data_source,
            target_column=target_column,
            include_covariates=True,
        )
        pred_df = await predict_time_series_async(
            context_df,
            prediction_length=forecast_horizon,
            model_name=model_name,
        )
        prediction_text = format_predictions_to_string(pred_df, prepared_sample["last_timestamp"])
        prediction_source = "runtime_model_service"

    score_details = teacher_score_details.get(model_name)
    if score_details is None:
        score_value, computed_details = score_prediction_text(prepared_sample, prediction_text=prediction_text)
        score_details = {"score": score_value, **computed_details}

    return {
        "candidate_id": f"{model_name}__keep",
        "model_name": model_name,
        "path_type": "default" if model_name == prepared_sample["default_expert"] else "alternative",
        "candidate_kind": "baseline",
        "prediction_source": prediction_source,
        "prediction_text": prediction_text,
        "compact_prediction_text": compact_prediction_tool_output_from_string(prediction_text, model_name=model_name),
        "score_details": score_details,
    }


def _refine_candidates_for_default(
    *,
    prepared_sample: dict[str, Any],
    history_values: list[float],
    default_prediction_text: str,
) -> list[dict[str, Any]]:
    _timestamps, default_values = parse_time_series_string(default_prediction_text, target_column=prepared_sample["target_column"])
    candidate_refinements = filter_refinement_candidates_for_model(
        generate_local_refinement_candidates(default_values, history_values),
        prediction_model_used=prepared_sample["default_expert"],
    )
    candidate_prediction_map = build_refinement_candidate_prediction_text_map(
        base_prediction_text=default_prediction_text,
        candidate_refinements=candidate_refinements,
        prediction_model_used=prepared_sample["default_expert"],
    )
    refined_candidates: list[dict[str, Any]] = []
    for decision_name, prediction_text in candidate_prediction_map.items():
        if decision_name == REFINEMENT_KEEP_DECISION:
            continue
        score_value, score_details = score_prediction_text(prepared_sample, prediction_text=prediction_text)
        refined_candidates.append(
            {
                "candidate_id": f"{prepared_sample['default_expert']}__{decision_name}",
                "model_name": prepared_sample["default_expert"],
                "path_type": "default",
                "candidate_kind": "refine",
                "prediction_source": "default_local_refine",
                "prediction_text": prediction_text,
                "compact_prediction_text": compact_prediction_tool_output_from_string(
                    prediction_text,
                    model_name=f"{prepared_sample['default_expert']}__{decision_name}",
                ),
                "score_details": {"score": score_value, **score_details},
            }
        )
    return refined_candidates


async def _build_candidate_bank_record(
    sample: dict[str, Any],
    *,
    default_expert: str,
    risk_tau: float,
) -> dict[str, Any]:
    prepared_sample = prepare_teacher_sample(sample)
    prepared_sample["default_expert"] = default_expert
    historical_data = parse_task_prompt(sample["raw_prompt"][0]["content"], data_source=sample.get("data_source")).historical_data
    if not historical_data:
        historical_data = sample["raw_prompt"][0]["content"]
    target_column = str(prepared_sample["target_column"])
    data_source = str(prepared_sample["data_source"])
    forecast_horizon = int(prepared_sample["forecast_horizon"])
    _timestamps, history_values = parse_time_series_string(historical_data, target_column=target_column)
    routing_feature_snapshot = _compute_routing_feature_snapshot(history_values)
    analysis_history = [result.tool_output for result in build_feature_tool_results(history_values)]
    teacher_score_details = _teacher_score_details(sample)

    baseline_candidates = await asyncio.gather(
        *[
            _predict_baseline_candidate(
                sample=sample,
                prepared_sample=prepared_sample,
                historical_data=historical_data,
                data_source=data_source,
                target_column=target_column,
                forecast_horizon=forecast_horizon,
                model_name=model_name,
                teacher_score_details=teacher_score_details,
            )
            for model_name in SUPPORTED_MODELS
        ]
    )

    baseline_by_id = {candidate["candidate_id"]: candidate for candidate in baseline_candidates}
    default_candidate_id = f"{default_expert}__keep"
    default_prediction_text = str(baseline_by_id[default_candidate_id]["prediction_text"])
    default_refine_candidates = _refine_candidates_for_default(
        prepared_sample=prepared_sample,
        history_values=history_values,
        default_prediction_text=default_prediction_text,
    )

    default_candidates = [baseline_by_id[default_candidate_id], *default_refine_candidates]
    alt_candidates = [
        candidate
        for candidate in baseline_candidates
        if str(candidate["model_name"]) != default_expert
    ]
    all_candidates = [*default_candidates, *alt_candidates]
    best_candidate = min(
        all_candidates,
        key=_candidate_sort_key,
    )
    default_error = float(default_candidates[0]["score_details"]["orig_mse"])
    best_candidate_error = float(best_candidate["score_details"]["orig_mse"])
    risk_value_rel = float((default_error - best_candidate_error) / default_error) if default_error > 0 else 0.0

    return {
        "uid": str(sample.get("uid") or f"sample-{sample.get('index')}"),
        "source_sample_index": int(sample["index"]),
        "split": str(sample.get("split") or ""),
        "data_source": data_source,
        "target_column": target_column,
        "lookback_window": int(sample.get("lookback_window") or prepared_sample.get("lookback_window") or 96),
        "forecast_horizon": forecast_horizon,
        "raw_prompt": sample["raw_prompt"],
        "ground_truth": str(prepared_sample["ground_truth"]),
        "historical_data": historical_data,
        "default_expert": default_expert,
        "default_candidate_id": default_candidate_id,
        "analysis_history": analysis_history,
        "routing_feature_snapshot": routing_feature_snapshot,
        "risk_tau": float(risk_tau),
        "risk_value_rel": risk_value_rel,
        "risk_label": "default_risky" if risk_value_rel >= float(risk_tau) else "default_ok",
        "default_candidates": default_candidates,
        "alt_candidates": alt_candidates,
        "all_candidate_ids": [candidate["candidate_id"] for candidate in all_candidates],
        "candidate_prediction_text_map": {
            candidate["candidate_id"]: candidate["prediction_text"] for candidate in all_candidates
        },
        "candidate_score_details": {
            candidate["candidate_id"]: candidate["score_details"] for candidate in all_candidates
        },
        "final_candidate_label": str(best_candidate["candidate_id"]),
        "final_candidate_score": float(best_candidate["score_details"]["score"]),
        "final_candidate_error": best_candidate_error,
        "default_candidate_error": default_error,
        "final_vs_default_error": float(best_candidate_error - default_error),
    }


async def _build_split_records(
    rows: list[dict[str, Any]],
    *,
    default_expert: str,
    risk_tau: float,
    split_name: str,
    max_concurrent_samples: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, int(max_concurrent_samples or 1)))
    progress_lock = asyncio.Lock()
    output: list[dict[str, Any] | None] = [None] * len(rows)
    processed = 0

    async def _worker(idx: int, row: dict[str, Any]) -> None:
        nonlocal processed
        async with semaphore:
            built = await _build_candidate_bank_record(
                row,
                default_expert=default_expert,
                risk_tau=risk_tau,
            )
        output[idx] = built
        async with progress_lock:
            processed += 1
            if processed % 10 == 0 or processed == len(rows):
                print(f"[v19 candidate bank] split={split_name} processed={processed}/{len(rows)}", flush=True)

    await asyncio.gather(*[_worker(idx, row) for idx, row in enumerate(rows)])
    return [row for row in output if row is not None]


def _split_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    default_candidate_counts = [len(row.get("default_candidates") or []) for row in records]
    alt_candidate_counts = [len(row.get("alt_candidates") or []) for row in records]
    risk_distribution = Counter(str(row.get("risk_label") or "") for row in records)
    final_distribution = Counter(str(row.get("final_candidate_label") or "") for row in records)
    return {
        "source_samples": int(len(records)),
        "risk_label_distribution": {str(k): int(v) for k, v in sorted(risk_distribution.items())},
        "final_candidate_distribution": {str(k): int(v) for k, v in sorted(final_distribution.items())},
        "default_candidate_count_mean": float(sum(default_candidate_counts) / len(default_candidate_counts)) if default_candidate_counts else 0.0,
        "alt_candidate_count_mean": float(sum(alt_candidate_counts) / len(alt_candidate_counts)) if alt_candidate_counts else 0.0,
        "final_vs_default_error_mean": float(
            sum(float(row.get("final_vs_default_error") or 0.0) for row in records) / len(records)
        ) if records else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_records: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] = {
        "dataset_kind": "etth1_v19_candidate_bank",
        "pipeline_stage": "candidate_bank_v19",
        "source_dataset_dir": str(args.input_dir.resolve()),
        "default_expert": str(args.default_expert).strip().lower(),
        "risk_tau": float(args.risk_tau),
    }

    for split_name in args.splits:
        merged_rows = _load_merged_split(args.input_dir, split_name)
        built_rows = asyncio.run(
            _build_split_records(
                merged_rows,
                default_expert=str(args.default_expert).strip().lower(),
                risk_tau=float(args.risk_tau),
                split_name=split_name,
                max_concurrent_samples=int(args.max_concurrent_samples),
            )
        )
        split_records[split_name] = built_rows
        write_jsonl_records(args.output_dir / f"{split_name}.jsonl", built_rows)
        metadata[f"{split_name}_summary"] = _split_metadata(built_rows)

    write_metadata_file(args.output_dir, metadata)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
