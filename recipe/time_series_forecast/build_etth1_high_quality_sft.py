from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipe.time_series_forecast.build_etth1_sft_dataset import (
    build_sft_record,
    convert_jsonl_to_sft_parquet,
    distribution_from_series,
    rebalance_train_turn3_targets,
)
from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RL_JSONL,
    DATASET_KIND_RUNTIME_SFT_PARQUET,
    DATASET_KIND_TEACHER_CURATED_SFT,
    validate_sibling_metadata,
)
from recipe.time_series_forecast.dataset_file_utils import (
    load_jsonl_records,
    write_jsonl_records,
    write_metadata_file,
)
from recipe.time_series_forecast.reward import compute_score
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.utils import (
    SUPPORTED_MODELS,
    format_predictions_to_string,
    get_last_timestamp,
    parse_time_series_to_dataframe,
    predict_time_series_async,
)


DEFAULT_OUTPUT_DIR = Path("dataset/ett_sft_etth1_runtime_teacher200_paper_same2")
DEFAULT_TRAIN_JSONL = Path("dataset/ett_rl_etth1_paper_same2/train.jsonl")
DEFAULT_VAL_JSONL = Path("dataset/ett_rl_etth1_paper_same2/val.jsonl")
DEFAULT_TEST_JSONL = Path("dataset/ett_rl_etth1_paper_same2/test.jsonl")
DEFAULT_MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://localhost:8994")
PYTORCH_TEACHER_MODELS = {"patchtst", "itransformer"}

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


def chunked(items: Sequence[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [list(items[start:start + chunk_size]) for start in range(0, len(items), chunk_size)]


def prepare_teacher_sample(sample: dict[str, Any]) -> dict[str, Any]:
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
        include_covariates=True,
    )
    timestamps = [
        ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, pd.Timestamp) else str(ts)
        for ts in context_df["timestamp"]
    ]
    values = [float(value) for value in context_df["target"].tolist()]
    feature_columns = list(context_df.attrs.get("feature_columns") or [target_column])
    if len(feature_columns) > 1:
        model_input_values = context_df.loc[:, feature_columns].astype(float).values.tolist()
    else:
        model_input_values = [float(value) for value in context_df[feature_columns[0]].tolist()]
    series_id = str(context_df["id"].iloc[0]) if "id" in context_df.columns else "series_0"
    return {
        "sample_index": int(sample.get("index", -1)),
        "ground_truth": ground_truth,
        "data_source": data_source,
        "forecast_horizon": forecast_horizon,
        "last_timestamp": get_last_timestamp(historical_data),
        "timestamps": timestamps,
        "values": values,
        "model_input_values": model_input_values,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "series_id": series_id,
        "curriculum_stage": sample.get("curriculum_stage"),
        "curriculum_band": sample.get("curriculum_band"),
    }


def build_predict_request_from_prepared(
    prepared_sample: dict[str, Any],
    *,
    model_server: Any,
    model_name: str,
) -> Any:
    return model_server.PredictRequest(
        timestamps=list(prepared_sample["timestamps"]),
        values=list(prepared_sample.get("model_input_values", prepared_sample["values"])),
        series_id=str(prepared_sample.get("series_id") or "series_0"),
        prediction_length=int(prepared_sample["forecast_horizon"]),
        model_name=model_name,
        feature_columns=list(prepared_sample.get("feature_columns") or []),
        target_column=str(prepared_sample.get("target_column") or "OT"),
    )


def score_prediction_text(
    prepared_sample: dict[str, Any],
    *,
    prediction_text: str,
) -> tuple[float, dict[str, float]]:
    # Teacher-eval compares raw forecast strings, not final Turn-3 protocol outputs.
    score = compute_score(
        data_source=str(prepared_sample["data_source"]),
        solution_str=prediction_solution(prediction_text),
        ground_truth=str(prepared_sample["ground_truth"]),
        allow_recovery=True,
    )
    score_value = float(score["score"] if isinstance(score, dict) else score)
    details: dict[str, float] = {}
    if isinstance(score, dict):
        details = {
            "score": score_value,
            "orig_mse": float(score.get("orig_mse", float("nan"))),
            "orig_mae": float(score.get("orig_mae", float("nan"))),
            "norm_mse": float(score.get("norm_mse", float("nan"))),
            "norm_mae": float(score.get("norm_mae", float("nan"))),
        }
    return score_value, details


def _select_reference_teacher_model(
    model_scores: dict[str, float],
    model_score_details: dict[str, dict[str, float]],
) -> str:
    error_ranked_models: list[tuple[float, str]] = []
    for model_name, details in model_score_details.items():
        try:
            orig_mse = float(details.get("orig_mse", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(orig_mse):
            error_ranked_models.append((orig_mse, str(model_name)))

    if error_ranked_models:
        error_ranked_models.sort(key=lambda item: (item[0], item[1]))
        return error_ranked_models[0][1]

    ranked_models = sorted(model_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked_models:
        raise RuntimeError("Cannot select reference teacher model without any successful model scores.")
    return str(ranked_models[0][0])


def _normalize_existing_evaluation_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    model_scores = normalized.get("model_scores")
    model_score_details = normalized.get("model_score_details")
    if not isinstance(model_scores, dict) or not isinstance(model_score_details, dict):
        return normalized
    if not model_scores or not model_score_details:
        return normalized

    reference_teacher_model = _select_reference_teacher_model(model_scores, model_score_details)
    reference_teacher_metrics = model_score_details.get(reference_teacher_model, {})
    normalized["reference_teacher_model"] = reference_teacher_model
    normalized["reference_teacher_score"] = float(model_scores.get(reference_teacher_model, float("nan")))
    normalized["reference_teacher_error"] = reference_teacher_metrics.get("orig_mse")
    normalized.setdefault(
        "reference_teacher_prediction_text",
        normalized.get("teacher_prediction_text"),
    )
    return normalized


def finalize_teacher_evaluations(
    prepared_samples: Sequence[dict[str, Any]],
    sample_state: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for prepared_sample in prepared_samples:
        sample_index = int(prepared_sample["sample_index"])
        state = sample_state[sample_index]
        model_scores = state["model_scores"]
        if not model_scores:
            raise RuntimeError(
                f"No successful teacher predictions for sample index={sample_index}. "
                f"Errors: {json.dumps(state['model_errors'], ensure_ascii=False, sort_keys=True)}"
            )

        ranked_models = sorted(model_scores.items(), key=lambda item: item[1], reverse=True)
        best_model, best_score = ranked_models[0]
        second_model, second_score = ranked_models[1] if len(ranked_models) > 1 else (best_model, best_score)
        margin = best_score - second_score
        best_metrics = state["model_score_details"].get(best_model, {})
        second_metrics = state["model_score_details"].get(second_model, {})
        reference_teacher_model = _select_reference_teacher_model(
            model_scores,
            state["model_score_details"],
        )
        reference_teacher_metrics = state["model_score_details"].get(reference_teacher_model, {})
        reference_teacher_prediction_text = state["model_predictions"].get(
            reference_teacher_model,
            state["model_predictions"][best_model],
        )
        evaluations.append(
            {
                "sample_index": sample_index,
                "best_model": best_model,
                "best_score": float(best_score),
                "second_best_model": second_model,
                "second_best_score": float(second_score),
                "score_margin": float(margin),
                "selection_score": float(quality_score(best_score, margin)),
                "model_scores": model_scores,
                "model_score_details": state["model_score_details"],
                "model_errors": state["model_errors"],
                "reference_teacher_model": reference_teacher_model,
                "reference_teacher_score": float(model_scores.get(reference_teacher_model, best_score)),
                "reference_teacher_prediction_text": reference_teacher_prediction_text,
                "teacher_prediction_text": reference_teacher_prediction_text,
                "teacher_prediction_source": "reference_teacher",
                "forecast_horizon": int(prepared_sample["forecast_horizon"]),
                "reference_teacher_error": reference_teacher_metrics.get("orig_mse"),
                "best_orig_mse": best_metrics.get("orig_mse"),
                "best_orig_mae": best_metrics.get("orig_mae"),
                "best_norm_mse": best_metrics.get("norm_mse"),
                "best_norm_mae": best_metrics.get("norm_mae"),
                "second_best_orig_mse": second_metrics.get("orig_mse"),
                "second_best_orig_mae": second_metrics.get("orig_mae"),
                "second_best_norm_mse": second_metrics.get("norm_mse"),
                "second_best_norm_mae": second_metrics.get("norm_mae"),
                "curriculum_stage": prepared_sample.get("curriculum_stage"),
                "curriculum_band": prepared_sample.get("curriculum_band"),
            }
        )
    return evaluations


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
    def __init__(
        self,
        *,
        models: Sequence[str],
        device: str,
        predictor_devices: Sequence[str] | None = None,
    ) -> None:
        from recipe.time_series_forecast import model_server

        self.model_server = model_server
        self.requested_models = [str(model).strip().lower() for model in models]
        self.model_device_map = build_local_model_device_map(
            self.requested_models,
            default_device=device,
            predictor_devices=predictor_devices,
        )
        self.executors: dict[str, ThreadPoolExecutor] = {}

        print("[HQ-SFT] Loading local teacher models with device map:")
        for model_name in self.requested_models:
            if model_name == "arima":
                continue
            runtime_device = self.model_device_map.get(model_name, "cpu")
            print(f"  - {model_name}: {runtime_device}")
            if model_name == "chronos2":
                model_server.load_chronos2(runtime_device)
            elif model_name == "patchtst":
                model_server.load_patchtst(runtime_device)
            elif model_name == "itransformer":
                model_server.load_itransformer(runtime_device)
            else:
                raise ValueError(f"Unsupported local teacher model: {model_name}")
            self.executors[model_name] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"hqsft-{model_name}")
        print("[HQ-SFT] Local teacher models are ready.")

    @property
    def should_parallelize_models(self) -> bool:
        runtime_devices = {
            self.model_device_map.get(model_name)
            for model_name in self.requested_models
            if model_name != "arima"
        }
        runtime_devices.discard(None)
        return len(runtime_devices) > 1

    def close(self) -> None:
        for executor in self.executors.values():
            executor.shutdown(wait=True)

    def predict_many(
        self,
        *,
        prepared_samples: Sequence[dict[str, Any]],
        model_name: str,
    ) -> list[pd.DataFrame]:
        requests = [
            build_predict_request_from_prepared(
                prepared_sample,
                model_server=self.model_server,
                model_name=model_name,
            )
            for prepared_sample in prepared_samples
        ]

        if model_name == "arima":
            predictions: list[pd.DataFrame] = []
            for prepared_sample in prepared_samples:
                context_df = pd.DataFrame(
                    {
                        "id": [prepared_sample["series_id"]] * len(prepared_sample["timestamps"]),
                        "timestamp": [pd.to_datetime(ts) for ts in prepared_sample["timestamps"]],
                        "target": prepared_sample["values"],
                    }
                )
                predictions.append(
                    asyncio.run(
                        predict_time_series_async(
                            context_df,
                            prediction_length=int(prepared_sample["forecast_horizon"]),
                            model_name="arima",
                        )
                    )
                )
            return predictions

        if model_name in PYTORCH_TEACHER_MODELS:
            try:
                responses = self.model_server.predict_with_pytorch_model_batch(requests, model_name)
            except Exception:
                responses = [self.model_server.predict_with_pytorch_model(request, model_name) for request in requests]
        elif model_name == "chronos2":
            responses = [self.model_server.predict_with_chronos2(request) for request in requests]
        else:
            raise ValueError(f"Unsupported local teacher model: {model_name}")

        return [
            pd.DataFrame(
                {
                    "timestamp": [pd.to_datetime(ts) for ts in response.timestamps],
                    "target_0.5": response.values,
                }
            )
            for response in responses
        ]

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
            values=(
                context_df.loc[:, context_df.attrs["feature_columns"]].astype(float).values.tolist()
                if len(list(context_df.attrs.get("feature_columns") or [])) > 1
                else [float(value) for value in context_df["target"].tolist()]
            ),
            series_id=str(context_df["id"].iloc[0]) if "id" in context_df.columns else "series_0",
            prediction_length=prediction_length,
            model_name=model_name,
            feature_columns=list(context_df.attrs.get("feature_columns") or []),
            target_column=str(context_df.attrs.get("target_column") or "OT"),
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

        return await loop.run_in_executor(self.executors[model_name], _predict_sync)


def _normalize_predictor_devices(
    predictor_devices: Sequence[str] | None,
    *,
    default_device: str,
) -> list[str]:
    if predictor_devices:
        normalized = [str(device).strip() for device in predictor_devices if str(device).strip()]
        if normalized:
            return normalized

    if default_device.startswith("cuda") and torch.cuda.is_available():
        if default_device == "cuda" and torch.cuda.device_count() > 0:
            return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
        return [default_device]
    return [default_device]


def build_local_model_device_map(
    models: Sequence[str],
    *,
    default_device: str,
    predictor_devices: Sequence[str] | None = None,
) -> dict[str, str]:
    from recipe.time_series_forecast import model_server

    normalized_devices = _normalize_predictor_devices(
        predictor_devices,
        default_device=model_server.resolve_runtime_device(default_device),
    )
    non_arima_models = [str(model).strip().lower() for model in models if str(model).strip().lower() != "arima"]
    if not non_arima_models:
        return {}

    device_map: dict[str, str] = {}
    for idx, model_name in enumerate(non_arima_models):
        device_map[model_name] = normalized_devices[idx % len(normalized_devices)]
    return device_map


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
        include_covariates=True,
    )
    last_timestamp = get_last_timestamp(historical_data)

    model_scores: dict[str, float] = {}
    model_score_details: dict[str, dict[str, float]] = {}
    model_predictions: dict[str, str] = {}
    model_errors: dict[str, str] = {}

    async def _evaluate_single_model(model_name: str) -> tuple[str, float, dict[str, float], str]:
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
                allow_recovery=True,
            )
            score_value = float(score["score"] if isinstance(score, dict) else score)
            details = {}
            if isinstance(score, dict):
                details = {
                    "score": score_value,
                    "orig_mse": float(score.get("orig_mse", float("nan"))),
                    "orig_mae": float(score.get("orig_mae", float("nan"))),
                    "norm_mse": float(score.get("norm_mse", float("nan"))),
                    "norm_mae": float(score.get("norm_mae", float("nan"))),
                }
            return (model_name, score_value, details, pred_text)
        except Exception as exc:  # pragma: no cover - runtime integration path
            model_errors[model_name] = f"{type(exc).__name__}: {exc}"
            return (model_name, float("nan"), {}, "")

    if predictor_mode == "local" and predictor is not None and predictor.should_parallelize_models:
        results = await asyncio.gather(*[_evaluate_single_model(model_name) for model_name in models])
    else:
        results = []
        for model_name in models:
            results.append(await _evaluate_single_model(model_name))

    for model_name, score_value, details, pred_text in results:
        if not pd.isna(score_value):
            model_scores[model_name] = score_value
            if details:
                model_score_details[model_name] = details
            model_predictions[model_name] = pred_text

    if not model_scores:
        raise RuntimeError(
            f"No successful teacher predictions for sample index={sample.get('index')}. "
            f"Errors: {json.dumps(model_errors, ensure_ascii=False, sort_keys=True)}"
        )

    ranked_models = sorted(model_scores.items(), key=lambda item: item[1], reverse=True)
    best_model, best_score = ranked_models[0]
    second_model, second_score = ranked_models[1] if len(ranked_models) > 1 else (best_model, best_score)
    margin = best_score - second_score
    best_metrics = model_score_details.get(best_model, {})
    second_metrics = model_score_details.get(second_model, {})
    reference_teacher_model = _select_reference_teacher_model(model_scores, model_score_details)
    reference_teacher_metrics = model_score_details.get(reference_teacher_model, {})
    reference_teacher_prediction_text = model_predictions.get(
        reference_teacher_model,
        model_predictions[best_model],
    )

    return {
        "sample_index": int(sample.get("index", -1)),
        "best_model": best_model,
        "best_score": float(best_score),
        "second_best_model": second_model,
        "second_best_score": float(second_score),
        "score_margin": float(margin),
        "selection_score": float(quality_score(best_score, margin)),
        "model_scores": model_scores,
        "model_score_details": model_score_details,
        "model_errors": model_errors,
        "reference_teacher_model": reference_teacher_model,
        "reference_teacher_score": float(model_scores.get(reference_teacher_model, best_score)),
        "reference_teacher_prediction_text": reference_teacher_prediction_text,
        "teacher_prediction_text": reference_teacher_prediction_text,
        "teacher_prediction_source": "reference_teacher",
        "forecast_horizon": forecast_horizon,
        "reference_teacher_error": reference_teacher_metrics.get("orig_mse"),
        "best_orig_mse": best_metrics.get("orig_mse"),
        "best_orig_mae": best_metrics.get("orig_mae"),
        "best_norm_mse": best_metrics.get("norm_mse"),
        "best_norm_mae": best_metrics.get("norm_mae"),
        "second_best_orig_mse": second_metrics.get("orig_mse"),
        "second_best_orig_mae": second_metrics.get("orig_mae"),
        "second_best_norm_mse": second_metrics.get("norm_mse"),
        "second_best_norm_mae": second_metrics.get("norm_mae"),
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


def evaluate_prepared_chunk_local(
    prepared_samples: Sequence[dict[str, Any]],
    *,
    models: Sequence[str],
    predictor: LocalTeacherPredictor,
) -> list[dict[str, Any]]:
    sample_state: dict[int, dict[str, Any]] = {
        int(prepared_sample["sample_index"]): {
            "model_scores": {},
            "model_score_details": {},
            "model_predictions": {},
            "model_errors": {},
        }
        for prepared_sample in prepared_samples
    }

    for model_name in models:
        try:
            prediction_frames = predictor.predict_many(
                prepared_samples=prepared_samples,
                model_name=model_name,
            )
            if len(prediction_frames) != len(prepared_samples):
                raise RuntimeError(
                    f"Local predictor returned {len(prediction_frames)} rows for model={model_name}, "
                    f"expected {len(prepared_samples)}"
                )
            for prepared_sample, pred_df in zip(prepared_samples, prediction_frames):
                sample_index = int(prepared_sample["sample_index"])
                pred_text = format_predictions_to_string(pred_df, prepared_sample.get("last_timestamp"))
                score_value, details = score_prediction_text(
                    prepared_sample,
                    prediction_text=pred_text,
                )
                sample_state[sample_index]["model_scores"][model_name] = score_value
                if details:
                    sample_state[sample_index]["model_score_details"][model_name] = details
                sample_state[sample_index]["model_predictions"][model_name] = pred_text
        except Exception as exc:  # pragma: no cover - runtime integration path
            error_text = f"{type(exc).__name__}: {exc}"
            for prepared_sample in prepared_samples:
                sample_index = int(prepared_sample["sample_index"])
                sample_state[sample_index]["model_errors"][model_name] = error_text

    return finalize_teacher_evaluations(prepared_samples, sample_state)


def _normalize_process_predictor_devices(
    predictor_devices: Sequence[str] | None,
    *,
    default_device: str,
) -> list[str]:
    normalized = _normalize_predictor_devices(
        predictor_devices,
        default_device=default_device,
    )
    return normalized or [default_device]


def split_predictor_device_groups(
    predictor_devices: Sequence[str] | None,
    *,
    default_device: str,
    num_workers: int,
) -> list[list[str]]:
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
    normalized = _normalize_process_predictor_devices(
        predictor_devices,
        default_device=default_device,
    )
    groups: list[list[str]] = [[] for _ in range(num_workers)]
    for idx, device in enumerate(normalized):
        groups[idx % num_workers].append(device)
    for idx, group in enumerate(groups):
        if not group:
            group.append(normalized[idx % len(normalized)])
    return groups


def shard_samples(samples: Sequence[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    for idx, sample in enumerate(samples):
        shards[idx % num_shards].append(dict(sample))
    return [shard for shard in shards if shard]


def evaluate_local_samples(
    samples: Sequence[dict[str, Any]],
    *,
    models: Sequence[str],
    predictor: LocalTeacherPredictor,
    local_batch_size: int,
) -> list[dict[str, Any]]:
    prepared_samples = [prepare_teacher_sample(sample) for sample in samples]
    evaluations: list[dict[str, Any]] = []
    prepared_chunks = chunked(prepared_samples, local_batch_size)
    total_chunks = len(prepared_chunks)
    progress_every = max(1, total_chunks // 10)
    for chunk_index, prepared_chunk in enumerate(prepared_chunks, start=1):
        evaluations.extend(
            evaluate_prepared_chunk_local(
                prepared_chunk,
                models=models,
                predictor=predictor,
            )
        )
        if (
            chunk_index == 1
            or chunk_index == total_chunks
            or chunk_index % progress_every == 0
        ):
            processed = min(chunk_index * local_batch_size, len(prepared_samples))
            print(
                f"[HQ-SFT local] processed_samples={processed}/{len(prepared_samples)} "
                f"chunks={chunk_index}/{total_chunks}"
            )
    return evaluations


def _evaluate_local_worker(
    worker_id: int,
    samples: Sequence[dict[str, Any]],
    models: Sequence[str],
    predictor_device: str,
    predictor_devices: Sequence[str],
    local_batch_size: int,
) -> list[dict[str, Any]]:
    print(
        f"[HQ-SFT worker {worker_id}] shard_samples={len(samples)} "
        f"devices={','.join(predictor_devices)} batch={local_batch_size}"
    )
    predictor = LocalTeacherPredictor(
        models=models,
        device=predictor_device,
        predictor_devices=predictor_devices,
    )
    try:
        return evaluate_local_samples(
            samples,
            models=models,
            predictor=predictor,
            local_batch_size=local_batch_size,
        )
    finally:
        predictor.close()


def evaluate_local_samples_multiprocess(
    samples: Sequence[dict[str, Any]],
    *,
    models: Sequence[str],
    predictor_device: str,
    predictor_devices: Sequence[str] | None,
    num_workers: int,
    local_batch_size: int,
) -> list[dict[str, Any]]:
    if num_workers <= 1:
        raise ValueError("evaluate_local_samples_multiprocess requires num_workers > 1")

    sample_shards = shard_samples(samples, num_workers)
    device_groups = split_predictor_device_groups(
        predictor_devices,
        default_device=predictor_device,
        num_workers=len(sample_shards),
    )
    ctx = multiprocessing.get_context("spawn")
    evaluations: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(sample_shards), mp_context=ctx) as executor:
        futures = [
            executor.submit(
                _evaluate_local_worker,
                worker_id,
                shard,
                list(models),
                predictor_device,
                device_groups[worker_id],
                local_batch_size,
            )
            for worker_id, shard in enumerate(sample_shards)
        ]
        for future in as_completed(futures):
            evaluations.extend(future.result())
    return sorted(evaluations, key=lambda item: int(item["sample_index"]))


def _select_bucketed_evaluations(
    evaluations: Sequence[dict[str, Any]],
    target_count: int,
    *,
    rank_fn: Callable[[dict[str, Any]], tuple[float, float, float]] | None = None,
) -> list[dict[str, Any]]:
    if target_count <= 0 or len(evaluations) <= target_count:
        return sorted(evaluations, key=lambda item: int(item["sample_index"]))

    effective_rank_fn = rank_fn or (
        lambda item: (
            float(item["selection_score"]),
            float(item["best_score"]),
            float(item["score_margin"]),
        )
    )
    ordered = sorted(evaluations, key=lambda item: int(item["sample_index"]))
    selected: list[dict[str, Any]] = []

    for bucket_idx in range(target_count):
        start = math.floor(bucket_idx * len(ordered) / target_count)
        end = math.floor((bucket_idx + 1) * len(ordered) / target_count)
        bucket = ordered[start:max(end, start + 1)]
        best = max(bucket, key=effective_rank_fn)
        selected.append(best)

    deduped: dict[int, dict[str, Any]] = {}
    for item in selected:
        sample_index = int(item["sample_index"])
        previous = deduped.get(sample_index)
        if previous is None or float(item["selection_score"]) > float(previous["selection_score"]):
            deduped[sample_index] = item

    if len(deduped) < target_count:
        for item in sorted(ordered, key=effective_rank_fn, reverse=True):
            sample_index = int(item["sample_index"])
            if sample_index not in deduped:
                deduped[sample_index] = item
            if len(deduped) >= target_count:
                break

    final_items = list(deduped.values())[:target_count]
    return sorted(final_items, key=lambda item: int(item["sample_index"]))


def _selected_model_support_rank(
    item: dict[str, Any],
) -> tuple[float, float, float]:
    selected_model = str(item.get("selected_prediction_model") or "").strip().lower()
    if not selected_model:
        return (
            float(item.get("selection_score", 0.0)),
            float(item.get("best_score", 0.0)),
            float(item.get("score_margin", 0.0)),
        )

    scores = item.get("teacher_eval_scores") or item.get("model_scores") or {}
    if selected_model not in scores:
        return (
            float(item.get("selection_score", 0.0)),
            float(item.get("best_score", 0.0)),
            float(item.get("score_margin", 0.0)),
        )

    selected_score = float(scores[selected_model])
    runner_up_score = max(
        (
            float(score)
            for model_name, score in scores.items()
            if str(model_name).strip().lower() != selected_model
        ),
        default=selected_score,
    )
    support_margin = selected_score - runner_up_score
    return (
        support_margin,
        selected_score,
        float(item.get("selection_score", 0.0)),
    )


def _curation_rank_tuple(
    item: dict[str, Any],
    *,
    balance_key: str | None,
) -> tuple[float, float, float]:
    if str(balance_key or "").strip().lower() == "selected_prediction_model":
        return _selected_model_support_rank(item)
    return (
        float(item.get("selection_score", 0.0)),
        float(item.get("best_score", 0.0)),
        float(item.get("score_margin", 0.0)),
    )


def _select_curated_evaluations_by_model_balance(
    evaluations: Sequence[dict[str, Any]],
    target_count: int,
    *,
    balance_key: str | None,
) -> list[dict[str, Any]]:
    if target_count <= 0 or len(evaluations) <= target_count:
        return sorted(evaluations, key=lambda item: int(item["sample_index"]))

    if not balance_key:
        return _select_bucketed_evaluations(evaluations, target_count)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in evaluations:
        model_name = str(item.get(balance_key) or "").strip().lower()
        if not model_name:
            return _select_bucketed_evaluations(evaluations, target_count)
        grouped.setdefault(model_name, []).append(item)

    if len(grouped) <= 1:
        return _select_bucketed_evaluations(evaluations, target_count)

    present_models = sorted(grouped)
    base_quota = target_count // len(present_models)
    selected_by_index: dict[int, dict[str, Any]] = {}

    if base_quota > 0:
        for model_name in present_models:
            rank_fn = lambda item, *, _balance_key=balance_key: _curation_rank_tuple(
                item,
                balance_key=_balance_key,
            )
            for item in _select_bucketed_evaluations(
                grouped[model_name],
                base_quota,
                rank_fn=rank_fn,
            ):
                selected_by_index[int(item["sample_index"])] = item

    remaining_target = max(0, target_count - len(selected_by_index))
    if remaining_target > 0:
        remaining_pool = [
            item
            for item in evaluations
            if int(item["sample_index"]) not in selected_by_index
        ]
        rank_fn = lambda item, *, _balance_key=balance_key: _curation_rank_tuple(
            item,
            balance_key=_balance_key,
        )
        for item in _select_bucketed_evaluations(
            remaining_pool,
            remaining_target,
            rank_fn=rank_fn,
        ):
            selected_by_index[int(item["sample_index"])] = item

    final_items = list(selected_by_index.values())[:target_count]
    return sorted(final_items, key=lambda item: int(item["sample_index"]))


def _prediction_tail_run_length(text: str, *, rounded_decimals: int = 4) -> int:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    values: list[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            values.append(f"{float(parts[-1]):.{rounded_decimals}f}")
        except (TypeError, ValueError):
            continue
    if not values:
        return 0
    tail_value = values[-1]
    tail_run = 0
    for value in reversed(values):
        if value != tail_value:
            break
        tail_run += 1
    return tail_run


def _is_arima_validated_keep_plateau(item: dict[str, Any], *, min_tail_run: int) -> bool:
    if str(item.get("selected_prediction_model") or "").strip().lower() != "arima":
        return False
    if str(item.get("turn3_target_type") or "").strip().lower() != "validated_keep":
        return False
    prediction_text = (
        item.get("teacher_prediction_text")
        or item.get("reference_teacher_prediction_text")
        or ""
    )
    return _prediction_tail_run_length(str(prediction_text), rounded_decimals=4) >= int(min_tail_run)


def select_curated_evaluations(
    evaluations: Sequence[dict[str, Any]],
    target_count: int,
    *,
    balance_key: str | None = "best_model",
    min_local_refine_ratio: float = 0.0,
    min_arima_validated_keep_plateau_ratio: float = 0.0,
    min_arima_validated_keep_plateau_tail_run: int = 24,
) -> list[dict[str, Any]]:
    if target_count <= 0 or len(evaluations) <= target_count:
        return sorted(evaluations, key=lambda item: int(item["sample_index"]))

    selected_by_index: dict[int, dict[str, Any]] = {}

    plateau_target = 0
    if min_arima_validated_keep_plateau_ratio > 0:
        plateau_pool = [
            item
            for item in evaluations
            if _is_arima_validated_keep_plateau(
                item,
                min_tail_run=min_arima_validated_keep_plateau_tail_run,
            )
        ]
        plateau_target = int(math.ceil(target_count * min_arima_validated_keep_plateau_ratio))
        plateau_target = min(plateau_target, len(plateau_pool), target_count)
        if plateau_target > 0:
            for item in _select_curated_evaluations_by_model_balance(
                plateau_pool,
                plateau_target,
                balance_key=None,
            ):
                selected_by_index[int(item["sample_index"])] = item

    if min_local_refine_ratio <= 0:
        remaining_target = max(0, target_count - len(selected_by_index))
        if remaining_target <= 0:
            final_items = list(selected_by_index.values())[:target_count]
            return sorted(final_items, key=lambda item: int(item["sample_index"]))
        remaining_pool = [
            item
            for item in evaluations
            if int(item["sample_index"]) not in selected_by_index
        ]
        for item in _select_curated_evaluations_by_model_balance(
            remaining_pool,
            remaining_target,
            balance_key=balance_key,
        ):
            selected_by_index[int(item["sample_index"])] = item
        final_items = list(selected_by_index.values())[:target_count]
        return sorted(final_items, key=lambda item: int(item["sample_index"]))

    local_refine_pool = [
        item
        for item in evaluations
        if str(item.get("turn3_target_type") or "") == "local_refine"
        and int(item["sample_index"]) not in selected_by_index
    ]
    if not local_refine_pool:
        return _select_curated_evaluations_by_model_balance(
            [
                item
                for item in evaluations
                if int(item["sample_index"]) not in selected_by_index
            ],
            max(0, target_count - len(selected_by_index)),
            balance_key=balance_key,
        )

    local_refine_target = int(math.ceil(target_count * min_local_refine_ratio))
    local_refine_target = min(local_refine_target, len(local_refine_pool), max(0, target_count - len(selected_by_index)))
    if local_refine_target <= 0:
        remaining_target = max(0, target_count - len(selected_by_index))
        if remaining_target > 0:
            remaining_pool = [
                item
                for item in evaluations
                if int(item["sample_index"]) not in selected_by_index
            ]
            for item in _select_curated_evaluations_by_model_balance(
                remaining_pool,
                remaining_target,
                balance_key=balance_key,
            ):
                selected_by_index[int(item["sample_index"])] = item
        final_items = list(selected_by_index.values())[:target_count]
        return sorted(final_items, key=lambda item: int(item["sample_index"]))

    for item in _select_curated_evaluations_by_model_balance(
        local_refine_pool,
        local_refine_target,
        balance_key=balance_key,
    ):
        selected_by_index[int(item["sample_index"])] = item

    remaining_target = max(0, target_count - len(selected_by_index))
    if remaining_target > 0:
        remaining_pool = [
            item
            for item in evaluations
            if int(item["sample_index"]) not in selected_by_index
        ]
        for item in _select_curated_evaluations_by_model_balance(
            remaining_pool,
            remaining_target,
            balance_key=balance_key,
        ):
            selected_by_index[int(item["sample_index"])] = item

    final_items = list(selected_by_index.values())[:target_count]
    return sorted(final_items, key=lambda item: int(item["sample_index"]))


def annotate_turn3_targets(
    samples: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotated: list[dict[str, Any]] = []
    annotation_errors: list[dict[str, Any]] = []
    for sample in samples:
        enriched = dict(sample)
        try:
            sft_record = build_sft_record(sample)
        except Exception as exc:
            enriched["turn3_target_type"] = "validated_keep"
            enriched["turn3_trigger_reason"] = "annotation_error"
            enriched["refine_ops_signature"] = "none"
            enriched["refine_gain_mse"] = 0.0
            enriched["refine_gain_mae"] = 0.0
            enriched["turn3_annotation_error"] = f"{type(exc).__name__}: {exc}"
            annotation_errors.append(
                {
                    "sample_index": int(sample.get("index", -1)),
                    "uid": sample.get("uid"),
                    "reference_teacher_model": sample.get("reference_teacher_model"),
                    "teacher_prediction_source": sample.get("teacher_prediction_source"),
                    "error": enriched["turn3_annotation_error"],
                }
            )
            annotated.append(enriched)
            continue

        for key in (
            "turn3_target_type",
            "turn3_trigger_reason",
            "refine_ops_signature",
            "refine_gain_mse",
            "refine_gain_mae",
            "refine_changed_value_count",
            "refine_first_changed_index",
            "refine_last_changed_index",
            "refine_changed_span",
            "refine_mean_abs_delta",
            "refine_max_abs_delta",
            "selected_feature_tool_signature",
            "selected_feature_tool_count",
            "selected_prediction_model",
            "base_prediction_source",
        ):
            enriched[key] = sft_record.get(key)
        annotated.append(enriched)
    return annotated, annotation_errors


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
        enriched["reference_teacher_model"] = evaluation.get("reference_teacher_model") or evaluation["best_model"]
        enriched["teacher_prediction_text"] = (
            evaluation.get("reference_teacher_prediction_text")
            or evaluation.get("teacher_prediction_text")
        )
        enriched["teacher_prediction_source"] = evaluation["teacher_prediction_source"]
        enriched["teacher_eval_best_score"] = evaluation["best_score"]
        enriched["teacher_eval_second_best_model"] = evaluation["second_best_model"]
        enriched["teacher_eval_second_best_score"] = evaluation["second_best_score"]
        enriched["teacher_eval_score_margin"] = evaluation["score_margin"]
        enriched["teacher_eval_scores"] = evaluation["model_scores"]
        enriched["teacher_eval_score_details"] = evaluation.get("model_score_details", {})
        enriched["teacher_eval_best_orig_mse"] = evaluation.get("best_orig_mse")
        enriched["teacher_eval_best_orig_mae"] = evaluation.get("best_orig_mae")
        enriched["teacher_eval_best_norm_mse"] = evaluation.get("best_norm_mse")
        enriched["teacher_eval_best_norm_mae"] = evaluation.get("best_norm_mae")
        enriched["reference_teacher_error"] = evaluation.get("reference_teacher_error")
        curated.append(enriched)

    return curated


def load_existing_evaluations(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[int, dict[str, Any]] = {}
    for record in load_jsonl_records(path):
        normalized = _normalize_existing_evaluation_record(record)
        sample_index = int(record.get("sample_index", -1))
        if sample_index >= 0:
            existing[sample_index] = normalized
    return existing


def process_split(
    *,
    split_name: str,
    records: Sequence[dict[str, Any]],
    models: Sequence[str],
    eval_count: int,
    candidate_count: int,
    target_count: int,
    max_concurrency: int,
    predictor_mode: str,
    predictor: LocalTeacherPredictor | None,
    model_service_url: str,
    output_dir: Path,
    predictor_device: str = "cuda",
    predictor_devices: Sequence[str] | None = None,
    num_workers: int = 1,
    local_batch_size: int = 32,
    resume_eval: bool = True,
    train_min_local_refine_ratio: float = 0.0,
    train_min_arima_validated_keep_plateau_ratio: float = 0.0,
    train_min_arima_validated_keep_plateau_tail_run: int = 24,
    max_turn3_annotation_error_count: int = 0,
    max_turn3_annotation_error_ratio: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if eval_count > 0 and candidate_count > eval_count:
        raise ValueError(
            f"split={split_name}: candidate_count ({candidate_count}) may not exceed eval_count ({eval_count}) "
            "when eval_count is limited, because every curated candidate must have teacher-eval metadata."
        )
    eval_path = output_dir / f"{split_name}_teacher_eval.jsonl"
    curated_eval_path = output_dir / f"{split_name}_teacher_eval_curated.jsonl"
    curated_path = output_dir / f"{split_name}_curated.jsonl"
    print(
        f"[HQ-SFT] split={split_name} eval={eval_count or 'all'} candidates={candidate_count} target={target_count} "
        f"models={','.join(models)} mode={predictor_mode}"
    )
    eval_records = evenly_spaced_records(records, eval_count) if eval_count > 0 else list(records)
    eval_indices = {int(record.get("index", -1)) for record in eval_records}
    existing_evaluations = load_existing_evaluations(eval_path) if resume_eval else {}
    reused_evaluations = {
        sample_index: evaluation
        for sample_index, evaluation in existing_evaluations.items()
        if sample_index in eval_indices
    }
    pending_records = [
        record
        for record in eval_records
        if int(record.get("index", -1)) not in reused_evaluations
    ]
    if reused_evaluations:
        print(
            f"[HQ-SFT] split={split_name} reusing {len(reused_evaluations)} existing teacher eval rows "
            f"from {eval_path}"
        )

    pending_evaluations: list[dict[str, Any]] = []
    if pending_records:
        if predictor_mode == "local":
            if num_workers > 1:
                pending_evaluations = evaluate_local_samples_multiprocess(
                    pending_records,
                    models=models,
                    predictor_device=predictor_device,
                    predictor_devices=predictor_devices,
                    num_workers=num_workers,
                    local_batch_size=local_batch_size,
                )
            else:
                assert predictor is not None
                pending_evaluations = evaluate_local_samples(
                    pending_records,
                    models=models,
                    predictor=predictor,
                    local_batch_size=local_batch_size,
                )
        else:
            pending_evaluations = asyncio.run(
                evaluate_candidates(
                    pending_records,
                    models=models,
                    max_concurrency=max_concurrency,
                    predictor_mode=predictor_mode,
                    predictor=predictor,
                    model_service_url=model_service_url,
                )
            )

    evaluation_by_index = dict(reused_evaluations)
    evaluation_by_index.update({int(item["sample_index"]): item for item in pending_evaluations})
    evaluations = [evaluation_by_index[sample_index] for sample_index in sorted(eval_indices) if sample_index in evaluation_by_index]

    candidate_records = evenly_spaced_records(records, candidate_count) if candidate_count > 0 else list(records)
    candidate_by_index = {int(record.get("index", -1)): record for record in candidate_records}
    candidate_indices = {int(record.get("index", -1)) for record in candidate_records}
    candidate_evaluations = [
        item for item in evaluations if int(item["sample_index"]) in candidate_indices
    ]
    candidate_samples = merge_evaluations_into_samples(candidate_records, candidate_evaluations)
    annotated_candidate_samples, annotation_errors = annotate_turn3_targets(candidate_samples)
    annotation_error_path = output_dir / f"{split_name}_turn3_annotation_errors.jsonl"
    if annotation_errors:
        write_jsonl_records(annotation_error_path, annotation_errors)
    annotation_error_count = len(annotation_errors)
    annotation_error_ratio = float(annotation_error_count / len(candidate_samples)) if candidate_samples else 0.0
    if annotation_error_count > 0:
        print(
            f"[HQ-SFT] split={split_name} turn3 annotation errors: "
            f"{annotation_error_count}/{len(candidate_samples)} ({annotation_error_ratio:.2%})"
        )
    if (
        annotation_error_count > int(max_turn3_annotation_error_count)
        or annotation_error_ratio > float(max_turn3_annotation_error_ratio)
    ):
        preview = "; ".join(
            f"index={item.get('sample_index')} error={item.get('error')}"
            for item in annotation_errors[:3]
        )
        raise RuntimeError(
            f"split={split_name} exceeded turn3 annotation error budget: "
            f"errors={annotation_error_count}, ratio={annotation_error_ratio:.2%}, "
            f"max_count={int(max_turn3_annotation_error_count)}, "
            f"max_ratio={float(max_turn3_annotation_error_ratio):.2%}. "
            f"Examples: {preview}"
        )
    annotated_by_index = {
        int(item.get("index", -1)): item
        for item in annotated_candidate_samples
    }
    annotated_candidate_evaluations: list[dict[str, Any]] = []
    for evaluation in candidate_evaluations:
        sample_index = int(evaluation["sample_index"])
        annotated_sample = annotated_by_index.get(sample_index, {})
        enriched_evaluation = dict(evaluation)
        for key in (
            "turn3_target_type",
            "turn3_trigger_reason",
            "refine_ops_signature",
            "refine_gain_mse",
            "refine_gain_mae",
            "refine_changed_value_count",
            "refine_first_changed_index",
            "refine_last_changed_index",
            "refine_changed_span",
            "refine_mean_abs_delta",
            "refine_max_abs_delta",
            "selected_feature_tool_signature",
            "selected_feature_tool_count",
            "selected_prediction_model",
            "base_prediction_source",
        ):
            if key in annotated_sample:
                enriched_evaluation[key] = annotated_sample[key]
        annotated_candidate_evaluations.append(enriched_evaluation)

    selected_evaluations = select_curated_evaluations(
        annotated_candidate_evaluations,
        target_count,
        balance_key="selected_prediction_model" if split_name == "train" else "best_model",
        min_local_refine_ratio=float(train_min_local_refine_ratio) if split_name == "train" else 0.0,
        min_arima_validated_keep_plateau_ratio=float(train_min_arima_validated_keep_plateau_ratio)
        if split_name == "train"
        else 0.0,
        min_arima_validated_keep_plateau_tail_run=int(train_min_arima_validated_keep_plateau_tail_run),
    )
    selected_indices = {int(item["sample_index"]) for item in selected_evaluations}
    selected_records = [
        annotated_by_index[sample_index]
        for sample_index in sorted(selected_indices)
        if sample_index in annotated_by_index
    ]

    write_jsonl_records(eval_path, evaluations)
    write_jsonl_records(curated_eval_path, selected_evaluations)
    write_jsonl_records(curated_path, selected_records)
    print(
        f"[HQ-SFT] split={split_name} eval_rows={len(evaluations)} reused={len(reused_evaluations)} "
        f"selected={len(selected_records)}"
    )
    annotation_summary = {
        "candidate_samples": int(len(candidate_samples)),
        "turn3_annotation_error_count": int(annotation_error_count),
        "turn3_annotation_error_ratio": float(annotation_error_ratio),
    }
    if annotation_errors:
        annotation_summary["turn3_annotation_errors_path"] = str(annotation_error_path)
    return selected_records, evaluations, annotation_summary


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
    parser.add_argument(
        "--train-eval-samples",
        type=int,
        default=0,
        help="Teacher-eval sample count for train split. 0 means evaluate the full split.",
    )
    parser.add_argument(
        "--val-eval-samples",
        type=int,
        default=0,
        help="Teacher-eval sample count for val split. 0 means evaluate the full split.",
    )
    parser.add_argument(
        "--test-eval-samples",
        type=int,
        default=0,
        help="Teacher-eval sample count for test split. 0 means evaluate the full split.",
    )
    parser.add_argument("--train-candidate-samples", type=int, default=600, help="Train candidate pool size.")
    parser.add_argument("--val-candidate-samples", type=int, default=192, help="Validation candidate pool size.")
    parser.add_argument("--test-candidate-samples", type=int, default=256, help="Test candidate pool size.")
    parser.add_argument(
        "--train-min-local-refine-ratio",
        type=float,
        default=0.25,
        help="Minimum desired local_refine ratio in train parquet. Set <=0 to disable train rebalancing.",
    )
    parser.add_argument(
        "--train-min-arima-validated-keep-plateau-ratio",
        type=float,
        default=0.0,
        help=(
            "Minimum desired ratio of train curated samples reserved for "
            "validated_keep + arima + long rounded plateau-tail forecasts. "
            "Set <=0 to disable this coverage quota."
        ),
    )
    parser.add_argument(
        "--train-min-arima-validated-keep-plateau-tail-run",
        type=int,
        default=24,
        help="Minimum rounded tail run length used to identify plateau-tail arima validated_keep cases.",
    )
    parser.add_argument(
        "--max-turn3-annotation-error-count",
        type=int,
        default=0,
        help="Maximum allowed Turn-3 annotation failures per split before aborting. Default: 0.",
    )
    parser.add_argument(
        "--max-turn3-annotation-error-ratio",
        type=float,
        default=0.0,
        help="Maximum allowed Turn-3 annotation failure ratio per split before aborting. Default: 0.0.",
    )
    parser.add_argument("--max-concurrency", type=int, default=2, help="Concurrent teacher evaluation count.")
    parser.add_argument("--num-workers", type=int, default=1, help="Local teacher-eval worker process count.")
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=32,
        help="Local sample batch size for batched PatchTST/iTransformer inference.",
    )
    parser.add_argument(
        "--resume-teacher-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing *_teacher_eval.jsonl rows instead of recomputing them.",
    )
    parser.add_argument(
        "--model-service-url",
        default=DEFAULT_MODEL_SERVICE_URL,
        help="Base URL for the unified teacher model service when predictor-mode=service.",
    )
    parser.add_argument(
        "--predictor-mode",
        choices=["service", "local"],
        default="local",
        help="Use local in-process teacher models or the HTTP model service.",
    )
    parser.add_argument("--predictor-device", default="cuda", help="Teacher model device when predictor-mode=local.")
    parser.add_argument(
        "--predictor-devices",
        default="",
        help=(
            "Optional comma-separated local devices for predictor-mode=local, e.g. "
            "'cuda:0,cuda:1,cuda:2,cuda:3'. Non-ARIMA models are assigned round-robin."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = (
        ("train", Path(args.train_jsonl), args.train_eval_samples, args.train_candidate_samples, args.train_target_samples),
        ("val", Path(args.val_jsonl), args.val_eval_samples, args.val_candidate_samples, args.val_target_samples),
        ("test", Path(args.test_jsonl), args.test_eval_samples, args.test_candidate_samples, args.test_target_samples),
    )

    metadata: dict[str, Any] = {
        "dataset_kind": DATASET_KIND_RUNTIME_SFT_PARQUET,
        "pipeline_stage": "teacher200_runtime_sft",
        "curated_jsonl_dataset_kind": DATASET_KIND_TEACHER_CURATED_SFT,
        "selection_method": "teacher_reward_scoring_with_bucketed_time_coverage_and_teacher_error_logging",
        "teacher_models": models,
        "max_concurrency": args.max_concurrency,
        "num_workers": args.num_workers,
        "local_batch_size": args.local_batch_size,
        "resume_teacher_eval": bool(args.resume_teacher_eval),
        "predictor_mode": args.predictor_mode,
        "predictor_device": args.predictor_device,
        "predictor_devices": [item.strip() for item in str(args.predictor_devices or "").split(",") if item.strip()],
        "model_service_url": args.model_service_url,
    }

    for split_name, path, _eval_count, _candidate_count, target_count in split_specs:
        if target_count > 0 and not path.exists():
            raise FileNotFoundError(
                f"Missing source RL jsonl for split={split_name}: {path}. "
                "Restore the RL dataset or pass the correct split path explicitly."
            )

    source_metadata_paths: list[Path] = []
    for split_name, path, _eval_count, _candidate_count, target_count in split_specs:
        if target_count <= 0 or not path.exists():
            continue
        source_metadata, source_metadata_path = validate_sibling_metadata(
            path,
            expected_kind=DATASET_KIND_RL_JSONL,
        )
        metadata[f"source_{split_name}_metadata_path"] = str(source_metadata_path)
        metadata[f"source_{split_name}_pipeline_stage"] = str(source_metadata.get("pipeline_stage") or "")
        source_metadata_paths.append(source_metadata_path)
    if source_metadata_paths:
        unique_source_metadata_paths = {str(path) for path in source_metadata_paths}
        if len(unique_source_metadata_paths) != 1:
            raise ValueError(
                "All source RL jsonl splits must come from the same dataset directory. "
                f"Got metadata files: {sorted(unique_source_metadata_paths)}"
            )

    predictor_devices = [item.strip() for item in str(args.predictor_devices or "").split(",") if item.strip()]
    predictor = (
        LocalTeacherPredictor(
            models=models,
            device=args.predictor_device,
            predictor_devices=predictor_devices,
        )
        if args.predictor_mode == "local" and args.num_workers == 1
        else None
    )
    if predictor is not None:
        metadata["local_model_device_map"] = predictor.model_device_map
    elif args.predictor_mode == "local":
        metadata["local_worker_device_groups"] = split_predictor_device_groups(
            predictor_devices,
            default_device=args.predictor_device,
            num_workers=args.num_workers,
        )

    try:
        if args.predictor_mode == "service":
            service_info = asyncio.run(ensure_service_ready(models, args.model_service_url))
            metadata["service_url"] = service_info["service_url"]
            metadata["service_models_loaded"] = service_info["models_loaded"]
            metadata["service_available_models"] = service_info["available_models"]
            print(
                f"[HQ-SFT] Using model service at {service_info['service_url']} "
                f"with loaded models={service_info['models_loaded']}"
            )

        for split_name, path, eval_count, candidate_count, target_count in split_specs:
            if not path.exists():
                metadata[f"{split_name}_samples"] = 0
                continue
            records = load_jsonl_records(path)
            if not records and target_count > 0:
                raise RuntimeError(
                    f"Source RL jsonl for split={split_name} is empty: {path}. "
                    "Cannot build teacher-scored SFT data from an empty split."
                )
            curated_records, full_evaluations, annotation_summary = process_split(
                split_name=split_name,
                records=records,
                models=models,
                eval_count=min(eval_count, len(records)) if eval_count > 0 else 0,
                candidate_count=min(candidate_count, len(records)),
                target_count=min(target_count, len(records)),
                max_concurrency=args.max_concurrency,
                predictor_mode=args.predictor_mode,
                predictor=predictor,
                predictor_device=args.predictor_device,
                predictor_devices=predictor_devices,
                num_workers=args.num_workers,
                local_batch_size=args.local_batch_size,
                resume_eval=bool(args.resume_teacher_eval),
                train_min_local_refine_ratio=float(args.train_min_local_refine_ratio),
                train_min_arima_validated_keep_plateau_ratio=float(
                    args.train_min_arima_validated_keep_plateau_ratio
                ),
                train_min_arima_validated_keep_plateau_tail_run=int(
                    args.train_min_arima_validated_keep_plateau_tail_run
                ),
                max_turn3_annotation_error_count=int(args.max_turn3_annotation_error_count),
                max_turn3_annotation_error_ratio=float(args.max_turn3_annotation_error_ratio),
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
            parquet_df_raw = convert_jsonl_to_sft_parquet(
                input_path=curated_jsonl_path,
                output_path=parquet_path,
            )
            parquet_df = parquet_df_raw
            if split_name == "train":
                parquet_df = rebalance_train_turn3_targets(
                    parquet_df_raw,
                    min_local_refine_ratio=float(args.train_min_local_refine_ratio),
                )
                if len(parquet_df) != len(parquet_df_raw) or not parquet_df.equals(parquet_df_raw):
                    parquet_df.to_parquet(parquet_path, index=False)
                    print(
                        f"Rebalanced {split_name}.parquet turn3_target_type distribution: "
                        f"{distribution_from_series(parquet_df['turn3_target_type'])}"
                    )
            teacher_distribution: dict[str, int] = {}
            for record in curated_records:
                teacher = str(record.get("reference_teacher_model") or "unknown")
                teacher_distribution[teacher] = teacher_distribution.get(teacher, 0) + 1
            selected_prediction_model_distribution: dict[str, int] = {}
            for record in curated_records:
                selected_model = str(record.get("selected_prediction_model") or "unknown")
                selected_prediction_model_distribution[selected_model] = (
                    selected_prediction_model_distribution.get(selected_model, 0) + 1
                )

            metadata[f"{split_name}_samples"] = len(parquet_df)
            metadata[f"{split_name}_teacher_eval_records"] = len(full_evaluations)
            metadata[f"{split_name}_eval_samples"] = min(eval_count, len(records)) if eval_count > 0 else len(records)
            metadata[f"{split_name}_candidate_samples"] = min(candidate_count, len(records))
            metadata[f"{split_name}_teacher_distribution"] = teacher_distribution
            metadata[f"{split_name}_selected_prediction_model_distribution"] = (
                selected_prediction_model_distribution
            )
            metadata[f"source_{split_name}_jsonl"] = str(path)
            metadata[f"{split_name}_turn3_annotation_error_count"] = annotation_summary["turn3_annotation_error_count"]
            metadata[f"{split_name}_turn3_annotation_error_ratio"] = annotation_summary["turn3_annotation_error_ratio"]
            if "turn3_annotation_errors_path" in annotation_summary:
                metadata[f"{split_name}_turn3_annotation_errors_path"] = annotation_summary["turn3_annotation_errors_path"]
            if "turn3_target_type" in parquet_df_raw.columns:
                if split_name == "train":
                    metadata["train_samples_before_balance"] = len(parquet_df_raw)
                    metadata["train_min_local_refine_ratio"] = float(args.train_min_local_refine_ratio)
                    metadata["train_min_arima_validated_keep_plateau_ratio"] = float(
                        args.train_min_arima_validated_keep_plateau_ratio
                    )
                    metadata["train_min_arima_validated_keep_plateau_tail_run"] = int(
                        args.train_min_arima_validated_keep_plateau_tail_run
                    )
                    metadata["train_turn3_target_type_distribution_before_balance"] = distribution_from_series(
                        parquet_df_raw["turn3_target_type"]
                    )
                metadata[f"{split_name}_turn3_target_type_distribution"] = distribution_from_series(
                    parquet_df["turn3_target_type"]
                )
            if "turn3_trigger_reason" in parquet_df.columns:
                metadata[f"{split_name}_turn3_trigger_reason_distribution"] = distribution_from_series(
                    parquet_df["turn3_trigger_reason"]
                )
            if "refine_ops_signature" in parquet_df.columns:
                metadata[f"{split_name}_refine_ops_signature_distribution"] = distribution_from_series(
                    parquet_df["refine_ops_signature"]
                )
            if "selected_feature_tool_signature" in parquet_df.columns:
                metadata[f"{split_name}_selected_feature_tool_signature_distribution"] = distribution_from_series(
                    parquet_df["selected_feature_tool_signature"]
                )

        write_metadata_file(output_dir, metadata)
    finally:
        if predictor is not None:
            predictor.close()


if __name__ == "__main__":
    main()
