from __future__ import annotations

import asyncio
import os
from typing import Optional

import pandas as pd

from recipe.time_series_forecast.config_utils import get_default_lengths
from recipe.time_series_forecast.diagnostic_features import (
    extract_basic_statistics,
    extract_data_quality,
    extract_event_summary,
    extract_forecast_residuals,
    extract_within_channel_dynamics,
    format_basic_statistics,
    format_data_quality,
    format_event_summary,
    format_forecast_residuals,
    format_within_channel_dynamics,
)
from recipe.time_series_forecast.time_series_io import (
    DEFAULT_LOOKBACK_WINDOW,
    SYNTHETIC_TIMESTAMP_ANCHOR,
    compact_prediction_tool_output_from_string,
    format_prediction_tool_output,
    format_predictions_to_string,
    get_last_timestamp,
    infer_frequency,
    parse_time_series_string,
    parse_time_series_to_dataframe,
)


_MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://localhost:8994")

_httpx_client = None
_httpx_client_loop = None

_, DEFAULT_FORECAST_HORIZON = get_default_lengths()
SUPPORTED_MODELS = ["chronos2", "arima", "patchtst", "itransformer"]


async def _get_httpx_client():
    """Get or create an httpx async client bound to the current event loop."""
    global _httpx_client, _httpx_client_loop
    import httpx

    current_loop = asyncio.get_running_loop()
    client_closed = bool(getattr(_httpx_client, "is_closed", False))
    if _httpx_client is None or _httpx_client_loop is not current_loop or client_closed:
        if _httpx_client is not None and not client_closed:
            try:
                await _httpx_client.aclose()
            except Exception:
                pass
        _httpx_client = httpx.AsyncClient(timeout=60.0, trust_env=False)
        _httpx_client_loop = current_loop
    return _httpx_client


def _resolve_model_service_url(model_service_url: Optional[str] = None) -> str:
    return (model_service_url or _MODEL_SERVICE_URL).rstrip("/")


def _format_httpx_error(error: Exception) -> str:
    detail = str(error).strip()
    if detail:
        return f"{type(error).__name__}: {detail}"
    return type(error).__name__


async def predict_time_series_async(
    context_df: pd.DataFrame,
    prediction_length: int = DEFAULT_FORECAST_HORIZON,
    model_name: str = "chronos2",
    model_service_url: Optional[str] = None,
) -> pd.DataFrame:
    """Predict time series using the specified model."""
    model_name = model_name.lower().strip()

    if model_name == "chronos2":
        return await predict_with_chronos_async(
            context_df,
            prediction_length,
            model_service_url=model_service_url,
        )
    if model_name == "arima":
        return await predict_with_arima_async(context_df, prediction_length)
    if model_name == "patchtst":
        return await predict_with_patchtst_async(
            context_df,
            prediction_length,
            model_service_url=model_service_url,
        )
    if model_name == "itransformer":
        return await predict_with_itransformer_async(
            context_df,
            prediction_length,
            model_service_url=model_service_url,
        )
    raise ValueError(f"Unsupported model: {model_name}. Supported models: {SUPPORTED_MODELS}")


def _build_prediction_request(
    context_df: pd.DataFrame,
    prediction_length: int,
    *,
    model_name: Optional[str] = None,
) -> dict:
    timestamps = []
    for ts in context_df["timestamp"]:
        if isinstance(ts, pd.Timestamp):
            timestamps.append(ts.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            timestamps.append(str(ts))

    request_data = {
        "timestamps": timestamps,
        "values": context_df["target"].tolist(),
        "series_id": context_df["id"].iloc[0] if "id" in context_df.columns else "series_0",
        "prediction_length": prediction_length,
    }
    if model_name:
        request_data["model_name"] = model_name
    return request_data


async def _predict_via_model_service(
    context_df: pd.DataFrame,
    prediction_length: int,
    *,
    error_label: str,
    model_name: Optional[str] = None,
    model_service_url: Optional[str] = None,
) -> pd.DataFrame:
    import httpx

    client = await _get_httpx_client()
    service_url = _resolve_model_service_url(model_service_url)
    request_data = _build_prediction_request(
        context_df,
        prediction_length,
        model_name=model_name,
    )
    try:
        response = await client.post(f"{service_url}/predict", json=request_data)
        response.raise_for_status()
        result = response.json()
        return pd.DataFrame(
            {
                "timestamp": [pd.to_datetime(ts) for ts in result["timestamps"]],
                "target_0.5": result["values"],
            }
        )
    except httpx.HTTPError as error:
        raise RuntimeError(
            f"{error_label} service error: {_format_httpx_error(error)}. "
            f"Make sure the model service is running at {service_url}"
        )


async def predict_with_chronos_async(
    context_df: pd.DataFrame,
    prediction_length: int = DEFAULT_FORECAST_HORIZON,
    model_service_url: Optional[str] = None,
) -> pd.DataFrame:
    return await _predict_via_model_service(
        context_df,
        prediction_length,
        error_label="Chronos2",
        model_service_url=model_service_url,
    )


async def predict_with_arima_async(
    context_df: pd.DataFrame,
    prediction_length: int = DEFAULT_FORECAST_HORIZON,
) -> pd.DataFrame:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.stattools import adfuller
    import numpy as np
    import warnings

    loop = asyncio.get_event_loop()

    def _fit_and_predict():
        values = context_df["target"].values
        timestamps = context_df["timestamp"].tolist()

        if len(timestamps) >= 2:
            freq = timestamps[1] - timestamps[0]
        else:
            freq = pd.Timedelta(hours=1)

        def _naive_forecast() -> pd.DataFrame:
            last_value = float(values[-1])
            pred_values = [last_value] * prediction_length
            pred_timestamps = [timestamps[-1] + freq * (i + 1) for i in range(prediction_length)]
            return pd.DataFrame({"timestamp": pred_timestamps, "target_0.5": pred_values})

        if len(values) < 12 or float(np.nanstd(values)) < 1e-6:
            return _naive_forecast()

        def _fit_arima(order: tuple[int, int, int]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
                model = ARIMA(
                    values,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(method_kwargs={"warn_convergence": False})

            mle_retvals = getattr(fitted, "mle_retvals", {}) or {}
            converged = mle_retvals.get("converged", True)
            aic = float(getattr(fitted, "aic", float("inf")))
            if not converged or not np.isfinite(aic):
                return None
            return fitted

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                adf_result = adfuller(values, maxlag=min(10, len(values) // 3))
                d = 0 if adf_result[1] < 0.05 else 1
            except Exception:
                d = 1

        best_model = None
        best_aic = float("inf")
        configs = [
            (1, d, 1),
            (2, d, 1),
            (1, d, 2),
            (2, d, 2),
            (3, d, 1),
            (1, d, 3),
            (0, d, 1),
            (1, d, 0),
            (5, d, 0),
            (0, d, 5),
        ]

        for p, d_val, q in configs:
            try:
                fitted = _fit_arima((p, d_val, q))
                if fitted is not None and fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
            except Exception:
                continue

        if best_model is None:
            try:
                best_model = _fit_arima((1, 1, 1))
            except Exception:
                best_model = None

        if best_model is None:
            try:
                max_lag = max(1, min(24, len(values) // 8))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ar_model = AutoReg(values, lags=max_lag, old_names=False)
                    ar_fit = ar_model.fit()
                forecast = ar_fit.predict(start=len(values), end=len(values) + prediction_length - 1)
                pred_timestamps = [timestamps[-1] + freq * (i + 1) for i in range(prediction_length)]
                return pd.DataFrame(
                    {
                        "timestamp": pred_timestamps,
                        "target_0.5": forecast.tolist() if hasattr(forecast, "tolist") else list(forecast),
                    }
                )
            except Exception:
                return _naive_forecast()

        try:
            forecast = best_model.forecast(steps=prediction_length)
        except Exception:
            return _naive_forecast()

        last_timestamp = timestamps[-1]
        pred_timestamps = [last_timestamp + freq * (i + 1) for i in range(prediction_length)]
        return pd.DataFrame(
            {
                "timestamp": pred_timestamps,
                "target_0.5": forecast.values if hasattr(forecast, "values") else forecast,
            }
        )

    return await loop.run_in_executor(None, _fit_and_predict)


async def predict_with_patchtst_async(
    context_df: pd.DataFrame,
    prediction_length: int = DEFAULT_FORECAST_HORIZON,
    model_service_url: Optional[str] = None,
) -> pd.DataFrame:
    return await _predict_via_model_service(
        context_df,
        prediction_length,
        error_label="PatchTST",
        model_name="patchtst",
        model_service_url=model_service_url,
    )


async def predict_with_itransformer_async(
    context_df: pd.DataFrame,
    prediction_length: int = DEFAULT_FORECAST_HORIZON,
    model_service_url: Optional[str] = None,
) -> pd.DataFrame:
    return await _predict_via_model_service(
        context_df,
        prediction_length,
        error_label="iTransformer",
        model_name="itransformer",
        model_service_url=model_service_url,
    )


__all__ = [
    "DEFAULT_FORECAST_HORIZON",
    "DEFAULT_LOOKBACK_WINDOW",
    "SUPPORTED_MODELS",
    "SYNTHETIC_TIMESTAMP_ANCHOR",
    "compact_prediction_tool_output_from_string",
    "extract_basic_statistics",
    "extract_data_quality",
    "extract_event_summary",
    "extract_forecast_residuals",
    "extract_within_channel_dynamics",
    "format_basic_statistics",
    "format_data_quality",
    "format_event_summary",
    "format_forecast_residuals",
    "format_prediction_tool_output",
    "format_predictions_to_string",
    "format_within_channel_dynamics",
    "get_last_timestamp",
    "infer_frequency",
    "parse_time_series_string",
    "parse_time_series_to_dataframe",
    "predict_time_series_async",
    "predict_with_arima_async",
    "predict_with_chronos_async",
    "predict_with_itransformer_async",
    "predict_with_patchtst_async",
]
