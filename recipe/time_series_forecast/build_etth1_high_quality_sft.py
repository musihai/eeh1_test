from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipe.time_series_forecast.build_etth1_sft_dataset import convert_jsonl_to_sft_parquet
from recipe.time_series_forecast.reward import compute_score
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.utils import (
    SUPPORTED_MODELS,
    format_predictions_to_string,
    get_last_timestamp,
    parse_time_series_to_dataframe,
    predict_time_series_async,
)


DEFAULT_OUTPUT_DIR = Path("dataset/ett_sft_etth1_runtime_ot_teacher200")
DEFAULT_TRAIN_JSONL = Path("dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/train.jsonl")
DEFAULT_VAL_JSONL = Path("dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/val.jsonl")
DEFAULT_TEST_JSONL = Path("dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/test.jsonl")
DEFAULT_MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://localhost:8994")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def evenly_spaced_records(records: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or len(records) <= count:
        return list(records)
    if count == 1:
        return [min(records, key=lambda item: int(item.get("index", 0)))]

    sorted_records = sorted(records, key=lambda item: int(item.get("index", 0)))
    selected: list[dict[str, Any]] = []
    used_positions: set[int] = set()
    total = len(sorted_records)

    for bucket in range(count):
        position = round(bucket * (total - 1) / (count - 1))
        if position in used_positions:
            continue
        used_positions.add(position)
        selected.append(sorted_records[position])

    cursor = 0
    while len(selected) < count and cursor < total:
        if cursor not in used_positions:
            selected.append(sorted_records[cursor])
            used_positions.add(cursor)
        cursor += 1

    return sorted(selected, key=lambda item: int(item.get("index", 0)))


def prediction_solution(prediction_text: str) -> str:
    return f"<answer>\n{prediction_text}\n</answer>"


def quality_score(best_score: float, margin: float) -> float:
    return best_score + 0.25 * max(margin, 0.0)


async def ensure_service_ready(models: Sequence[str], model_service_url: str) -> dict[str, Any]:
    import httpx

    service_url = model_service_url.rstrip("/")
    requested_remote_models = [model for model in models if model != "arima"]
    if not requested_remote_models:
        return {
            "service_url": service_url,
            "models_loaded": {},
            "available_models": [],
        }

    async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
        try:
            health_response = await client.get(f"{service_url}/health")
            health_response.raise_for_status()
            health_payload = health_response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Model service is unavailable at {service_url}: {exc}. "
                "Start it first with recipe/time_series_forecast/start_model_server.sh."
            ) from exc

        try:
            models_response = await client.get(f"{service_url}/models")
            models_response.raise_for_status()
            models_payload = models_response.json()
        except httpx.HTTPError:
            models_payload = {}

    models_loaded = health_payload.get("models_loaded", {})
    missing_models = [model for model in requested_remote_models if not bool(models_loaded.get(model))]
    if missing_models:
        raise RuntimeError(
            f"Model service at {service_url} is reachable, but these teacher models are not loaded: {missing_models}. "
            f"Health payload: {health_payload}"
        )

    return {
        "service_url": service_url,
        "models_loaded": models_loaded,
        "available_models": models_payload.get("available_models", []),
    }


class LocalTeacherPredictor:
    def __init__(self, device: str) -> None:
        from recipe.time_series_forecast import model_server

        self.model_server = model_server
        self.device = model_server.resolve_runtime_device(device)
        print(f"[HQ-SFT] Loading local teacher models on device={self.device}...")
        model_server.load_all_models(self.device)
        print("[HQ-SFT] Local teacher models are ready.")

    async def predict(
        self,
        *,
        context_df: pd.DataFrame,
        prediction_length: int,
        model_name: str,
    ) -> pd.DataFrame:
        if model_name == "arima":
            return await predict_time_series_async(
                context_df,
                prediction_length=prediction_length,
                model_name="arima",
            )

        request = self.model_server.PredictRequest(
            timestamps=[
                ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, pd.Timestamp) else str(ts)
                for ts in context_df["timestamp"]
            ],
            values=[float(value) for value in context_df["target"].tolist()],
            series_id=str(context_df["id"].iloc[0]) if "id" in context_df.columns else "series_0",
            prediction_length=prediction_length,
            model_name=model_name,
        )

        loop = asyncio.get_event_loop()

        def _predict_sync() -> pd.DataFrame:
            if model_name == "chronos2":
                response = self.model_server.predict_with_chronos2(request)
            else:
                response = self.model_server.predict_with_pytorch_model(request, model_name)
            return pd.DataFrame(
                {
                    "timestamp": [pd.to_datetime(ts) for ts in response.timestamps],
                    "target_0.5": response.values,
                }
            )

        return await loop.run_in_executor(None, _predict_sync)


async def evaluate_teacher_for_sample(
    sample: dict[str, Any],
    models: Sequence[str],
    predictor_mode: str,
    predictor: LocalTeacherPredictor | None,
    model_service_url: str,
) -> dict[str, Any]:
    raw_prompt = sample["raw_prompt"][0]["content"]
    reward_model = sample.get("reward_model", {}) if isinstance(sample.get("reward_model"), dict) else {}
    ground_truth = str(reward_model.get("ground_truth", "") or "")
    task_spec = parse_task_prompt(raw_prompt, data_source=sample.get("data_source"))
    historical_data = task_spec.historical_data or raw_prompt
    target_column = task_spec.target_column or "OT"
    data_source = task_spec.data_source or str(sample.get("data_source") or "ETTh1")
    forecast_horizon = int(task_spec.forecast_horizon or 96)
    context_df = parse_time_series_to_dataframe(
        historical_data,
        series_id=data_source,
        target_column=target_column,
    )
    last_timestamp = get_last_timestamp(historical_data)

    model_scores: dict[str, float] = {}
    model_predictions: dict[str, str] = {}
    model_errors: dict[str, str] = {}

    for model_name in models:
        try:
            if predictor_mode == "local":
                assert predictor is not None
                pred_df = await predictor.predict(
                    context_df=context_df,
                    prediction_length=forecast_horizon,
                    model_name=model_name,
                )
            else:
                pred_df = await predict_time_series_async(
                    context_df,
                    prediction_length=forecast_horizon,
                    model_name=model_name,
                    model_service_url=model_service_url,
                )
            pred_text = format_predictions_to_string(pred_df, last_timestamp)
            score = compute_score(
                data_source=data_source,
                solution_str=prediction_solution(pred_text),
                ground_truth=ground_truth,
            )
            model_scores[model_name] = float(score["score"] if isinstance(score, dict) else score)
            model_predictions[model_name] = pred_text
        except Exception as exc:  # pragma: no cover - runtime integration path
            model_errors[model_name] = f"{type(exc).__name__}: {exc}"

    if not model_scores:
        raise RuntimeError(
            f"No successful teacher predictions for sample index={sample.get('index')}. "
            f"Errors: {json.dumps(model_errors, ensure_ascii=False, sort_keys=True)}"
        )

    ranked_models = sorted(model_scores.items(), key=lambda item: item[1], reverse=True)
    best_model, best_score = ranked_models[0]
    second_model, second_score = ranked_models[1] if len(ranked_models) > 1 else (best_model, best_score)
    margin = best_score - second_score

    return {
        "sample_index": int(sample.get("index", -1)),
        "best_model": best_model,
        "best_score": float(best_score),
        "second_best_model": second_model,
        "second_best_score": float(second_score),
        "score_margin": float(margin),
        "selection_score": float(quality_score(best_score, margin)),
        "model_scores": model_scores,
        "model_errors": model_errors,
        "teacher_prediction_text": model_predictions[best_model],
        "teacher_prediction_source": "reference_teacher",
        "forecast_horizon": forecast_horizon,
        "curriculum_stage": sample.get("curriculum_stage"),
        "curriculum_band": sample.get("curriculum_band"),
    }


async def evaluate_candidates(
    samples: Sequence[dict[str, Any]],
    models: Sequence[str],
    max_concurrency: int,
    predictor_mode: str,
    predictor: LocalTeacherPredictor | None,
    model_service_url: str,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[dict[str, Any]] = []

    async def _runner(sample: dict[str, Any]) -> None:
        async with semaphore:
            result = await evaluate_teacher_for_sample(
                sample,
                models,
                predictor_mode,
                predictor,
                model_service_url,
            )
            results.append(result)

    await asyncio.gather(*[_runner(sample) for sample in samples])
    return results


def select_curated_evaluations(evaluations: Sequence[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    if target_count <= 0 or len(evaluations) <= target_count:
        return sorted(evaluations, key=lambda item: int(item["sample_index"]))

    ordered = sorted(evaluations, key=lambda item: int(item["sample_index"]))
    selected: list[dict[str, Any]] = []

    for bucket_idx in range(target_count):
        start = math.floor(bucket_idx * len(ordered) / target_count)
        end = math.floor((bucket_idx + 1) * len(ordered) / target_count)
        bucket = ordered[start:max(end, start + 1)]
        best = max(
            bucket,
            key=lambda item: (
                float(item["selection_score"]),
                float(item["best_score"]),
                float(item["score_margin"]),
            ),
        )
        selected.append(best)

    deduped: dict[int, dict[str, Any]] = {}
    for item in selected:
        sample_index = int(item["sample_index"])
        previous = deduped.get(sample_index)
        if previous is None or float(item["selection_score"]) > float(previous["selection_score"]):
            deduped[sample_index] = item

    if len(deduped) < target_count:
        for item in sorted(
            ordered,
            key=lambda current: (
                float(current["selection_score"]),
                float(current["best_score"]),
                float(current["score_margin"]),
            ),
            reverse=True,
        ):
            sample_index = int(item["sample_index"])
            if sample_index not in deduped:
                deduped[sample_index] = item
            if len(deduped) >= target_count:
                break

    final_items = list(deduped.values())[:target_count]
    return sorted(final_items, key=lambda item: int(item["sample_index"]))


def merge_evaluations_into_samples(
    samples: Sequence[dict[str, Any]],
    evaluations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluation_by_index = {int(item["sample_index"]): item for item in evaluations}
    curated: list[dict[str, Any]] = []

    for sample in samples:
        sample_index = int(sample.get("index", -1))
        evaluation = evaluation_by_index[sample_index]
        enriched = dict(sample)
        enriched["reference_teacher_model"] = evaluation["best_model"]
        enriched["teacher_prediction_text"] = evaluation["teacher_prediction_text"]
        enriched["teacher_prediction_source"] = evaluation["teacher_prediction_source"]
        enriched["teacher_eval_best_score"] = evaluation["best_score"]
        enriched["teacher_eval_second_best_model"] = evaluation["second_best_model"]
        enriched["teacher_eval_second_best_score"] = evaluation["second_best_score"]
        enriched["teacher_eval_score_margin"] = evaluation["score_margin"]
        enriched["teacher_eval_scores"] = evaluation["model_scores"]
        curated.append(enriched)

    return curated


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metadata(output_dir: Path, **payload: Any) -> None:
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_split(
    *,
    split_name: str,
    records: Sequence[dict[str, Any]],
    models: Sequence[str],
    candidate_count: int,
    target_count: int,
    max_concurrency: int,
    predictor_mode: str,
    predictor: LocalTeacherPredictor | None,
    model_service_url: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(
        f"[HQ-SFT] split={split_name} candidates={candidate_count} target={target_count} "
        f"models={','.join(models)} mode={predictor_mode}"
    )
    candidate_records = evenly_spaced_records(records, candidate_count)
    evaluations = asyncio.run(
        evaluate_candidates(
            candidate_records,
            models=models,
            max_concurrency=max_concurrency,
            predictor_mode=predictor_mode,
            predictor=predictor,
            model_service_url=model_service_url,
        )
    )
    selected_evaluations = select_curated_evaluations(evaluations, target_count)
    selected_indices = {int(item["sample_index"]) for item in selected_evaluations}
    selected_records = [record for record in candidate_records if int(record.get("index", -1)) in selected_indices]
    selected_records = merge_evaluations_into_samples(selected_records, selected_evaluations)

    eval_path = output_dir / f"{split_name}_teacher_eval.jsonl"
    curated_path = output_dir / f"{split_name}_curated.jsonl"
    write_jsonl(eval_path, selected_evaluations)
    write_jsonl(curated_path, selected_records)
    print(f"[HQ-SFT] split={split_name} selected={len(selected_records)}")
    return selected_records, selected_evaluations


def parse_models(value: str) -> list[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in requested if item not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError(f"Unsupported teacher models: {invalid}. Supported: {SUPPORTED_MODELS}")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a higher-quality ETTh1 SFT dataset with scored teacher models.")
    parser.add_argument("--train-jsonl", default=str(DEFAULT_TRAIN_JSONL), help="Train RL jsonl path.")
    parser.add_argument("--val-jsonl", default=str(DEFAULT_VAL_JSONL), help="Validation RL jsonl path.")
    parser.add_argument("--test-jsonl", default=str(DEFAULT_TEST_JSONL), help="Test RL jsonl path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--models", default="patchtst,chronos2,itransformer,arima", help="Comma-separated teacher model list.")
    parser.add_argument("--train-target-samples", type=int, default=200, help="Curated train sample count.")
    parser.add_argument("--val-target-samples", type=int, default=64, help="Curated val sample count.")
    parser.add_argument("--test-target-samples", type=int, default=128, help="Curated test sample count.")
    parser.add_argument("--train-candidate-samples", type=int, default=600, help="Train candidate pool size.")
    parser.add_argument("--val-candidate-samples", type=int, default=192, help="Validation candidate pool size.")
    parser.add_argument("--test-candidate-samples", type=int, default=256, help="Test candidate pool size.")
    parser.add_argument("--max-concurrency", type=int, default=2, help="Concurrent teacher evaluation count.")
    parser.add_argument(
        "--model-service-url",
        default=DEFAULT_MODEL_SERVICE_URL,
        help="Base URL for the unified teacher model service when predictor-mode=service.",
    )
    parser.add_argument(
        "--predictor-mode",
        choices=["service", "local"],
        default="service",
        help="Use HTTP model service or load teacher models in-process.",
    )
    parser.add_argument("--predictor-device", default="cuda", help="Teacher model device when predictor-mode=local.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = (
        ("train", Path(args.train_jsonl), args.train_candidate_samples, args.train_target_samples),
        ("val", Path(args.val_jsonl), args.val_candidate_samples, args.val_target_samples),
        ("test", Path(args.test_jsonl), args.test_candidate_samples, args.test_target_samples),
    )

    metadata: dict[str, Any] = {
        "selection_method": "teacher_reward_scoring_with_bucketed_time_coverage",
        "teacher_models": models,
        "max_concurrency": args.max_concurrency,
        "predictor_mode": args.predictor_mode,
        "predictor_device": args.predictor_device,
        "model_service_url": args.model_service_url,
    }

    for split_name, path, _candidate_count, target_count in split_specs:
        if target_count > 0 and not path.exists():
            raise FileNotFoundError(
                f"Missing source RL jsonl for split={split_name}: {path}. "
                "Restore the RL dataset or pass the correct split path explicitly."
            )

    predictor = LocalTeacherPredictor(args.predictor_device) if args.predictor_mode == "local" else None
    if args.predictor_mode == "service":
        service_info = asyncio.run(ensure_service_ready(models, args.model_service_url))
        metadata["service_url"] = service_info["service_url"]
        metadata["service_models_loaded"] = service_info["models_loaded"]
        metadata["service_available_models"] = service_info["available_models"]
        print(
            f"[HQ-SFT] Using model service at {service_info['service_url']} "
            f"with loaded models={service_info['models_loaded']}"
        )

    for split_name, path, candidate_count, target_count in split_specs:
        if not path.exists():
            metadata[f"{split_name}_samples"] = 0
            continue
        records = load_jsonl_records(path)
        if not records and target_count > 0:
            raise RuntimeError(
                f"Source RL jsonl for split={split_name} is empty: {path}. "
                "Cannot build teacher-scored SFT data from an empty split."
            )
        curated_records, selected_evaluations = process_split(
            split_name=split_name,
            records=records,
            models=models,
            candidate_count=min(candidate_count, len(records)),
            target_count=min(target_count, len(records)),
            max_concurrency=args.max_concurrency,
            predictor_mode=args.predictor_mode,
            predictor=predictor,
            model_service_url=args.model_service_url,
            output_dir=output_dir,
        )
        if not curated_records and target_count > 0:
            raise RuntimeError(
                f"No curated SFT records were produced for split={split_name}. "
                "Check the RL source data and teacher model service."
            )
        curated_jsonl_path = output_dir / f"{split_name}_curated.jsonl"
        parquet_path = output_dir / f"{split_name}.parquet"
        convert_jsonl_to_sft_parquet(
            input_path=curated_jsonl_path,
            output_path=parquet_path,
            prediction_mode="reference_teacher",
        )
        teacher_distribution: dict[str, int] = {}
        for evaluation in selected_evaluations:
            teacher = str(evaluation["best_model"])
            teacher_distribution[teacher] = teacher_distribution.get(teacher, 0) + 1

        metadata[f"{split_name}_samples"] = len(curated_records)
        metadata[f"{split_name}_candidate_samples"] = min(candidate_count, len(records))
        metadata[f"{split_name}_teacher_distribution"] = teacher_distribution
        metadata[f"source_{split_name}_jsonl"] = str(path)

    write_metadata(output_dir, **metadata)


if __name__ == "__main__":
    main()
