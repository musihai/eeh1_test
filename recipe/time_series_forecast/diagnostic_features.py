from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_widths
from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf

try:
    import ruptures as rpt
except ImportError:
    rpt = None


def _sanitize_value(value: Any) -> Any:
    """Sanitize a single value to ensure no NaN/Inf."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_sanitize_scalar(v) for v in value]
    return _sanitize_scalar(value)


def _sanitize_scalar(value: Any) -> Any:
    """Sanitize a scalar value."""
    if isinstance(value, str):
        return value
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(scalar):
        return 0.0
    return scalar


def extract_basic_statistics(data: List[float], seasonal_period: int = 24) -> Dict[str, Any]:
    """Extract basic statistical features from time series data."""
    if not data:
        return {
            "median": 0.0,
            "mad": 0.0,
            "acf1": 0.0,
            "acf_seasonal": 0.0,
            "peak_freq": 0.0,
            "spec_entropy": 0.0,
            "cusum_max": 0.0,
            "qkurt": 0.0,
            "mean_abs_corr": 0.0,
            "pca_var_ratio1": 0.0,
        }

    arr = np.array(data, dtype=float).reshape(-1, 1)
    features: dict[str, Any] = {}

    features["median"] = float(np.median(arr))
    median_val = np.median(arr, axis=0)
    features["mad"] = float(np.median(np.abs(arr - median_val)))

    try:
        acf_result = acf(arr[:, 0], nlags=1, fft=True)
        features["acf1"] = float(acf_result[1]) if len(acf_result) > 1 else 0.0
    except Exception:
        features["acf1"] = 0.0

    try:
        if len(arr) > seasonal_period:
            acf_result = acf(arr[:, 0], nlags=seasonal_period, fft=True)
            features["acf_seasonal"] = float(acf_result[seasonal_period])
        else:
            features["acf_seasonal"] = 0.0
    except Exception:
        features["acf_seasonal"] = 0.0

    try:
        fft_vals = fft(arr[:, 0])
        freqs = fftfreq(len(arr), d=1.0)
        power = np.abs(fft_vals[: len(fft_vals) // 2]) ** 2
        pos_freqs = freqs[: len(freqs) // 2]
        if len(power) > 0:
            peak_idx = int(np.argmax(power))
            features["peak_freq"] = float(abs(pos_freqs[peak_idx]))
            power_norm = power / np.sum(power)
            power_norm = power_norm[power_norm > 0]
            if len(power_norm) > 1:
                features["spec_entropy"] = float(-np.sum(power_norm * np.log2(power_norm)))
            else:
                features["spec_entropy"] = 0.0
        else:
            features["peak_freq"] = 0.0
            features["spec_entropy"] = 0.0
    except Exception:
        features["peak_freq"] = 0.0
        features["spec_entropy"] = 0.0

    try:
        median_val = np.median(arr[:, 0])
        deviations = arr[:, 0] - median_val
        cusum = np.cumsum(deviations)
        features["cusum_max"] = float(np.max(np.abs(cusum)))
    except Exception:
        features["cusum_max"] = 0.0

    try:
        q975 = np.percentile(arr[:, 0], 97.5)
        q025 = np.percentile(arr[:, 0], 2.5)
        q75 = np.percentile(arr[:, 0], 75)
        q25 = np.percentile(arr[:, 0], 25)
        denom = q75 - q25
        features["qkurt"] = float((q975 - q025) / denom) if denom != 0 else 0.0
    except Exception:
        features["qkurt"] = 0.0

    features["mean_abs_corr"] = 0.0

    try:
        if arr.shape[1] >= 2:
            data_std = (arr - np.mean(arr, axis=0)) / (np.std(arr, axis=0) + 1e-8)
            pca = PCA(n_components=min(arr.shape[1], arr.shape[0]))
            pca.fit(data_std)
            features["pca_var_ratio1"] = float(pca.explained_variance_ratio_[0])
        else:
            features["pca_var_ratio1"] = 1.0
    except Exception:
        features["pca_var_ratio1"] = 0.0

    return {k: _sanitize_value(v) for k, v in features.items()}


def extract_within_channel_dynamics(
    data: List[float],
    changepoint_penalty: float = 5.0,
    changepoint_max: int = 5,
    peak_prominence: float = 1.0,
) -> Dict[str, Any]:
    """Extract within-channel dynamics features."""
    if not data:
        return {
            "changepoint_count": 0.0,
            "changepoint_score": 0.0,
            "slope_max": 0.0,
            "slope_second_diff_max": 0.0,
            "monotone_duration": 0.0,
            "peak_count": 0.0,
            "peak_max_width": 0.0,
            "peak_spacing_cv": 0.0,
        }

    arr = np.array(data, dtype=float)
    features: dict[str, Any] = {}

    if len(arr) < 10 or rpt is None:
        features["changepoint_count"] = 0.0
        features["changepoint_score"] = 0.0
    else:
        try:
            algo = rpt.Pelt(model="rbf").fit(arr)
            bkps = algo.predict(pen=changepoint_penalty)
            count = max(0, len(bkps) - 1)
            features["changepoint_count"] = float(min(count, changepoint_max))
            if count > 0:
                segments = np.split(arr, bkps[:-1])
                seg_means = [np.mean(seg) for seg in segments if len(seg) > 0]
                if len(seg_means) > 1:
                    diff = np.abs(np.diff(seg_means))
                    features["changepoint_score"] = float(np.max(diff))
                else:
                    features["changepoint_score"] = 0.0
            else:
                features["changepoint_score"] = 0.0
        except Exception:
            features["changepoint_count"] = 0.0
            features["changepoint_score"] = 0.0

    if len(arr) < 3:
        features["slope_max"] = 0.0
        features["slope_second_diff_max"] = 0.0
        features["monotone_duration"] = 0.0
    else:
        diff1 = np.diff(arr)
        diff2 = np.diff(arr, n=2)
        features["slope_max"] = float(np.max(np.abs(diff1)))
        features["slope_second_diff_max"] = float(np.max(np.abs(diff2))) if len(diff2) else 0.0

        if len(diff1) == 0:
            features["monotone_duration"] = 0.0
        else:
            signs = np.sign(diff1)
            longest = current = 1
            prev = signs[0]
            for s in signs[1:]:
                if s == 0 or s == prev:
                    current += 1
                else:
                    longest = max(longest, current)
                    current = 1
                if s != 0:
                    prev = s
            longest = max(longest, current)
            features["monotone_duration"] = float(longest / max(len(arr), 1))

    if len(arr) < 3:
        features["peak_count"] = 0.0
        features["peak_max_width"] = 0.0
        features["peak_spacing_cv"] = 0.0
    else:
        try:
            normalized = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
            peaks, _ = find_peaks(normalized, prominence=peak_prominence)
            features["peak_count"] = float(len(peaks))
            if len(peaks) == 0:
                features["peak_max_width"] = 0.0
                features["peak_spacing_cv"] = 0.0
            else:
                widths = peak_widths(normalized, peaks)[0]
                features["peak_max_width"] = float(np.max(widths) if len(widths) else 0.0)
                if len(peaks) > 1:
                    spacing = np.diff(peaks)
                    features["peak_spacing_cv"] = float(np.std(spacing) / (np.mean(spacing) + 1e-8))
                else:
                    features["peak_spacing_cv"] = 0.0
        except Exception:
            features["peak_count"] = 0.0
            features["peak_max_width"] = 0.0
            features["peak_spacing_cv"] = 0.0

    return {k: _sanitize_value(v) for k, v in features.items()}


def extract_forecast_residuals(data: List[float], ar_order: int = 1) -> Dict[str, Any]:
    """Extract forecast residual features using AutoReg."""
    if not data or len(data) <= ar_order + 2:
        return {
            "residual_mean": 0.0,
            "residual_max": 0.0,
            "residual_exceed_ratio": 0.0,
            "residual_acf1": 0.0,
            "residual_concentration": 0.0,
        }

    arr = np.array(data, dtype=float)
    features: dict[str, Any] = {}

    try:
        model = AutoReg(arr, lags=ar_order, old_names=False).fit()
        resid = model.resid
    except Exception:
        resid = arr[ar_order:] - arr[:-ar_order]

    if len(resid) == 0:
        resid = np.zeros(1)

    features["residual_mean"] = float(np.mean(resid))
    features["residual_max"] = float(np.max(np.abs(resid)))

    std_val = np.std(resid) + 1e-8
    features["residual_exceed_ratio"] = float(np.mean(np.abs(resid) > 2 * std_val))

    try:
        acf_val = acf(resid, nlags=1, fft=False)
        features["residual_acf1"] = float(acf_val[1]) if len(acf_val) > 1 else 0.0
    except Exception:
        features["residual_acf1"] = 0.0

    sorted_resid = np.sort(np.abs(resid))
    tail_start = int(0.9 * len(sorted_resid))
    tail_sum = np.sum(sorted_resid[tail_start:])
    total_sum = np.sum(sorted_resid) + 1e-8
    features["residual_concentration"] = float(tail_sum / total_sum)

    return {k: _sanitize_value(v) for k, v in features.items()}


def extract_data_quality(
    data: List[float],
    quantization_bins: int = 25,
    flatline_tol: float = 1e-3,
) -> Dict[str, Any]:
    """Extract data quality features."""
    if not data:
        return {
            "quality_quantization_score": 0.0,
            "quality_saturation_ratio": 0.0,
            "quality_constant_channel_ratio": 0.0,
            "quality_dropout_ratio": 0.0,
        }

    arr = np.array(data, dtype=float).reshape(-1, 1)
    flattened = arr.flatten()
    features: dict[str, Any] = {}

    try:
        hist, _ = np.histogram(flattened, bins=quantization_bins)
        features["quality_quantization_score"] = float(np.max(hist) / len(flattened))
    except Exception:
        features["quality_quantization_score"] = 0.0

    try:
        span = np.max(flattened) - np.min(flattened)
        margin = max(span * 0.01, flatline_tol)
        saturation = np.logical_or(
            flattened <= np.min(flattened) + margin,
            flattened >= np.max(flattened) - margin,
        )
        features["quality_saturation_ratio"] = float(np.mean(saturation))
    except Exception:
        features["quality_saturation_ratio"] = 0.0

    try:
        features["quality_constant_channel_ratio"] = float(np.mean(np.std(arr, axis=0) < flatline_tol))
    except Exception:
        features["quality_constant_channel_ratio"] = 0.0

    try:
        features["quality_dropout_ratio"] = float(np.mean(~np.isfinite(flattened)))
    except Exception:
        features["quality_dropout_ratio"] = 0.0

    return {k: _sanitize_value(v) for k, v in features.items()}


def extract_event_summary(data: List[float]) -> Dict[str, Any]:
    """Extract event summary features by segmenting the series."""
    if not data:
        return {
            "event_segment_count": 0.0,
            "event_counts": [0.0, 0.0, 0.0, 0.0],
            "event_dominant_pattern": 0.0,
        }

    arr = np.array(data, dtype=float)

    if len(arr) < 4:
        segments = [arr]
    else:
        diffs = np.diff(arr)
        threshold = np.std(diffs) * 1.5 + 1e-6
        change_points = [0]
        for idx in range(1, len(diffs)):
            if abs(diffs[idx] - diffs[idx - 1]) > threshold:
                change_points.append(idx)
        change_points.append(len(arr) - 1)
        segments = []
        for start, end in zip(change_points[:-1], change_points[1:]):
            seg = arr[start : end + 1]
            if len(seg) > 0:
                segments.append(seg)
        segments = segments[:8]
        if len(segments) < 3 and len(arr) >= 3:
            split_size = max(1, len(arr) // 3)
            segments = [arr[:split_size], arr[split_size : 2 * split_size], arr[2 * split_size :]]

    slope_threshold = np.std(arr) * 0.05 + 1e-6
    event_counts = Counter({"rise": 0, "fall": 0, "flat": 0, "oscillation": 0})

    for segment in segments:
        if len(segment) < 2:
            continue
        slope = segment[-1] - segment[0]
        segment_std = np.std(segment)
        if slope > slope_threshold:
            event_counts["rise"] += 1
        elif slope < -slope_threshold:
            event_counts["fall"] += 1
        elif segment_std > slope_threshold * 2:
            event_counts["oscillation"] += 1
        else:
            event_counts["flat"] += 1

    order = ["rise", "fall", "flat", "oscillation"]
    counts_list = [float(event_counts[name]) for name in order]
    dominant_idx = int(np.argmax(counts_list)) if counts_list else 0

    features = {
        "event_segment_count": float(len(segments)),
        "event_counts": counts_list,
        "event_dominant_pattern": float(dominant_idx),
    }
    return {k: _sanitize_value(v) for k, v in features.items()}


def format_basic_statistics(features: Dict[str, Any]) -> str:
    lines = ["Basic Statistics:"]
    lines.append(f"  Median: {features.get('median', 0.0):.4f}")
    lines.append(f"  MAD: {features.get('mad', 0.0):.4f}")
    lines.append(f"  ACF(1): {features.get('acf1', 0.0):.4f}")
    lines.append(f"  ACF(seasonal): {features.get('acf_seasonal', 0.0):.4f}")
    lines.append(f"  Peak Frequency: {features.get('peak_freq', 0.0):.4f}")
    lines.append(f"  Spectral Entropy: {features.get('spec_entropy', 0.0):.4f}")
    lines.append(f"  CUSUM Max: {features.get('cusum_max', 0.0):.4f}")
    lines.append(f"  Quantile Kurtosis: {features.get('qkurt', 0.0):.4f}")
    return "\n".join(lines)


def format_within_channel_dynamics(features: Dict[str, Any]) -> str:
    lines = ["Within-Channel Dynamics:"]
    lines.append(f"  Changepoint Count: {features.get('changepoint_count', 0.0):.1f}")
    lines.append(f"  Changepoint Score: {features.get('changepoint_score', 0.0):.4f}")
    lines.append(f"  Max Slope: {features.get('slope_max', 0.0):.4f}")
    lines.append(f"  Max Second Diff: {features.get('slope_second_diff_max', 0.0):.4f}")
    lines.append(f"  Monotone Duration: {features.get('monotone_duration', 0.0):.4f}")
    lines.append(f"  Peak Count: {features.get('peak_count', 0.0):.1f}")
    lines.append(f"  Peak Max Width: {features.get('peak_max_width', 0.0):.4f}")
    lines.append(f"  Peak Spacing CV: {features.get('peak_spacing_cv', 0.0):.4f}")
    return "\n".join(lines)


def format_forecast_residuals(features: Dict[str, Any]) -> str:
    lines = ["Forecast Residuals:"]
    lines.append(f"  Residual Mean: {features.get('residual_mean', 0.0):.4f}")
    lines.append(f"  Residual Max: {features.get('residual_max', 0.0):.4f}")
    lines.append(f"  Exceed Ratio: {features.get('residual_exceed_ratio', 0.0):.4f}")
    lines.append(f"  ACF(1): {features.get('residual_acf1', 0.0):.4f}")
    lines.append(f"  Concentration: {features.get('residual_concentration', 0.0):.4f}")
    return "\n".join(lines)


def format_data_quality(features: Dict[str, Any]) -> str:
    lines = ["Data Quality:"]
    lines.append(f"  Quantization Score: {features.get('quality_quantization_score', 0.0):.4f}")
    lines.append(f"  Saturation Ratio: {features.get('quality_saturation_ratio', 0.0):.4f}")
    lines.append(f"  Constant Channel Ratio: {features.get('quality_constant_channel_ratio', 0.0):.4f}")
    lines.append(f"  Dropout Ratio: {features.get('quality_dropout_ratio', 0.0):.4f}")
    return "\n".join(lines)


def format_event_summary(features: Dict[str, Any]) -> str:
    lines = ["Event Summary:"]
    lines.append(f"  Segment Count: {features.get('event_segment_count', 0.0):.1f}")
    counts = features.get("event_counts", [0.0, 0.0, 0.0, 0.0])
    lines.append(f"  Rise Events: {counts[0]:.1f}")
    lines.append(f"  Fall Events: {counts[1]:.1f}")
    lines.append(f"  Flat Events: {counts[2]:.1f}")
    lines.append(f"  Oscillation Events: {counts[3]:.1f}")
    pattern_names = ["rise", "fall", "flat", "oscillation"]
    dominant_idx = int(features.get("event_dominant_pattern", 0))
    lines.append(f"  Dominant Pattern: {pattern_names[dominant_idx]}")
    return "\n".join(lines)


__all__ = [
    "extract_basic_statistics",
    "extract_data_quality",
    "extract_event_summary",
    "extract_forecast_residuals",
    "extract_within_channel_dynamics",
    "format_basic_statistics",
    "format_data_quality",
    "format_event_summary",
    "format_forecast_residuals",
    "format_within_channel_dynamics",
]
